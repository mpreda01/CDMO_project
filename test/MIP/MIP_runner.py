"""
Sports Tournament Scheduling - MIP Solver Runner using AMPLpy
Runs the AMPL model with multiple solvers and generates JSON results
"""

import os
import sys
import json
import orjson
import time
import math
import pandas as pd
from pathlib import Path
from amplpy import AMPL, Environment

class CircleMethodScheduler:
    """Generate round-robin pairings (circle method) WITHOUT assigning periods."""

    @staticmethod
    def generate_pairings(n):
        weeks = n - 1
        rot = list(range(1, n))  # teams 1..n-1
        fixed = n

        pairings_per_week = []

        for w in range(weeks):
            week_pairs = []

            # Fixed team pairing
            week_pairs.append((rot[w], fixed))

            # Remaining pairs
            # for i, j ∈ N\{w, n}: team i plays team j if i + j ≡ 2r mod (n − 1).
            teams_assigned = [n]
            for k in range(1, n):
                if k not in teams_assigned:
                    for j in range(1, n):
                        if k == rot[w] or j == rot[w] or k == j:
                            continue
                        if j not in teams_assigned:
                            r = (k+j) % weeks
                            div = 2*(w+1) % weeks
                            if r == div:
                                week_pairs.append((k, j))
                                teams_assigned.append(k)
                                teams_assigned.append(j)

            pairings_per_week.append(week_pairs)

        return pairings_per_week

class STSMIPRunner:
    def __init__(self, ampl_path=None, model_file=None, time_limit=300, solvers=None):
        if model_file is None:
            model_file = Path(__file__).parent.parent.parent / "source" / "MIP" / "sts_model.mod"
        self.model_file = model_file
        self.time_limit = time_limit
        self.all_solvers = ["cbc", "highs", "cplex", "gurobi"]
        self.solvers = solvers if solvers else self.all_solvers
        self.results = {}
        self.ampl_path = ampl_path
        
    def load_model(self, ampl, optimization=True, sb_teams=True, sb_weeks=True):
        """Load unified AMPL model and configure for decision/optimization"""
        try:
            # Load the unified model
            ampl.read(self.model_file)
            
            # Set optimization parameter
            optimize_param = 1 if optimization else 0
            ampl.getParameter("optimize_balance").set(optimize_param)
            sb_teams_param = 1 if sb_teams else 0
            ampl.getParameter("sb_teams").set(sb_teams_param)
            sb_weeks_param = 1 if sb_weeks else 0
            ampl.getParameter("sb_weeks").set(sb_weeks_param)
            
            return True
            
        except Exception as e:
            print(f"Error loading model from {self.model_file}: {e}")
            return False
    
    def extract_solution_matrix(self, ampl, n):
        """Extract solution matrix from AMPL variables"""
        try:
            x_var = ampl.getVariable('x')
            x_df = x_var.getValues()
            rows = list(x_df)
            pdf = pd.DataFrame(data=rows, columns=["t1", "t2", "w", "p", "val"])

            weeks = n - 1
            periods = n // 2
            
            # Initialize solution matrix
            solution = []
            for p in range(1, periods + 1):
                period_games = []
                for w in range(1, weeks + 1):
                    # Find the game in this period and week
                    game_found = False
                    for _, row in pdf.iterrows():
                        if (row["val"] > 0.5 and 
                            len(row) >= 5 and
                            int(row["w"]) == w and int(row["p"]) == p):  # week and period match
                            t1, t2 = int(row["t1"]), int(row["t2"])
                            period_games.append([t1, t2])
                            game_found = True
                            break
                    
                    if not game_found:
                        print(f"Warning: No game found for week {w}, period {p}")
                        return None
                
                solution.append(period_games)
            
            solution = str(solution)
            
            return solution
            
        except Exception as e:
            print(f"Error extracting solution: {e}")
            return None

    def run_solver(self, solver, n, optimization=True, sb_teams=True, sb_weeks=True, circle_method=True):
        """Run AMPL with specific solver using AMPLpy"""
        
        try:
            # Create AMPL instance
            if self.ampl_path:
                ampl = AMPL(Environment(self.ampl_path))
            else:
                ampl = AMPL()
            
            start_time = time.time()
            
            if not circle_method:
                # Load model from file
                if not self.load_model(ampl, optimization, sb_teams, sb_weeks):
                                
                    return {
                        "time": self.time_limit,
                        "version": "optimal" if optimization else "decision",
                        "symmetry_breaking": None, 
                        "optimal": False,
                        "stop_reason": None,
                        "obj": None,
                        "sol": None
                    }
            if circle_method:
                pairings_per_week = CircleMethodScheduler.generate_pairings(n)
                circle_model_file = Path(__file__).parent.parent.parent / "source" / "MIP" / "circle_model.mod"
                ampl.read(circle_model_file)
                ampl.getParameter("n").set(n)
                ampl.getParameter("optimize_balance").set(1 if optimization else 0)
                # Fix matches per week, but let AMPL assign periods
                for w, week_pairs in enumerate(pairings_per_week, start=1):
                    for t1, t2 in week_pairs:
                        ampl.eval(f"""
                            subject to FixMatch_{t1}_{t2}_{w}: 
                                sum {{p in PERIODS}} (x[{t1},{t2},{w},p] + x[{t2},{t1},{w},p]) = 1;
                        """)
            else:
                # Set parameter
                ampl.getParameter("n").set(n)
            
            # Configure solver
            ampl.setOption("solver", solver)
            ampl.setOption(f'{solver}_options', f'timelimit={self.time_limit}')
            
            # Solve
            ampl.solve()
            
            end_time = time.time()
            runtime = int(math.floor(end_time - start_time))
            
            # Get solve result
            solve_result = str(ampl.getValue("solve_result")).lower()
            is_optimal = "optimal" in solve_result or "solved" in solve_result
            
            # Determine stop reason
            stop_reason = None
            if is_optimal:
                stop_reason = None  # Solution found
            elif "infeasible" in solve_result:
                stop_reason = "infeasible"
            elif runtime >= self.time_limit or "time" in solve_result or "limit" in solve_result:
                stop_reason = "time_limit"
            else:
                stop_reason = "unknown"  # Default for other failure cases
            
            if not circle_method:
                if sb_teams and sb_weeks:
                        symmbreak = "all"
                elif sb_teams:
                        symmbreak = "symmetry_team1"
                elif sb_weeks:
                        symmbreak = "symmetry_week1"
                else:
                        symmbreak = "None"

                result = {
                    "time": min(runtime, self.time_limit),
                    "method": "classic",
                    "version": "optimal" if optimization else "decision",
                    "symmetry_breaking": symmbreak,
                    "optimal": is_optimal,
                    "stop_reason": stop_reason,
                    "obj": "None",
                    "sol": []
                }
            else:
                result = {
                    "time": min(runtime, self.time_limit),
                    "method": "circle",
                    "version": "optimal" if optimization else "decision",
                    "symmetry_breaking": "None",
                    "optimal": is_optimal,
                    "stop_reason": stop_reason,
                    "obj": "None",
                    "sol": []
                }

            # Get objective value if optimization version and solution found
            if optimization and is_optimal:
                try:
                    obj_value = ampl.getObjective("MaxImbalance").value()
                    result["obj"] = int(obj_value) if obj_value is not None else "None"
                except:
                    result["obj"] = 1
            
            # Extract solution if found
            if is_optimal or "feasible" in solve_result.lower():
                solution_matrix = self.extract_solution_matrix(ampl, n)
                if solution_matrix:
                    result["sol"] = solution_matrix
                    if not optimization:
                        result["obj"] = "None"  # No objective for decision version
            
            ampl.close()
            return result
            
        except Exception as e:
            print(f"Error with solver {solver}: {e}")
            return {
                "time": self.time_limit,
                "version": "optimal" if optimization else "decision",
                "optimal": False,
                "stop_reason": "solver_error",
                "obj": None,
                "sol": None
            }
    
    def check_solver_availability(self, solver):
        """Check if a solver is available"""
        try:
            if self.ampl_path:
                ampl = AMPL(Environment(self.ampl_path))
            else:
                ampl = AMPL()
            
            # Try to set the solver
            ampl.setOption("solver", solver)
            
            # Simple test model
            ampl.eval("""
                var x >= 0;
                minimize obj: x;
                subject to con: x >= 1;
            """)
            
            ampl.solve()
            solve_result = str(ampl.getValue("solve_result"))
            ampl.close()
            
            return "optimal" in solve_result.lower() or "solved" in solve_result.lower()
            
        except Exception as e:
            print(f"Solver {solver} not available: {e}")
            return False
    
    def run_experiments(self, team_sizes, run_decision=True, run_optimization=True, symmetry_combinations=True, sb_teams = True, sb_weeks = True, circle_method = True):
        """Run experiments for different team sizes"""
        mode_str = []
        if run_decision:
            mode_str.append("decision")
        if run_optimization:
            mode_str.append("optimization")
        
        print(f"\n=== Running experiments for {"circle method" if circle_method else "classic method"} ===")
        print(f"Running for {' and '.join(mode_str)}")
        print(f"Time limit: {self.time_limit}s")
        print(f"Solvers to use: {', '.join(self.solvers)}\n")
        
        # Check available solvers
        available_solvers = []
        for solver in self.solvers:
            if self.check_solver_availability(solver):
                available_solvers.append(solver)
                print(f"Solver {solver} is available")
            else:
                print(f"Solver {solver} is not available")
        
        if not available_solvers:
            print("Error: No solvers available!")
            return
        
        if not circle_method:
            for n in team_sizes:
                print(f"\n=== Running experiments for {n} teams ===")
            
                # Combined results structure
                merged_results = {}
            
                for solver in available_solvers:
                    if symmetry_combinations:
                        for s1 in [False,True]:
                            for s2 in [False,True]:
                                # Run decision version if requested
                                if run_decision:
                                    print(f"  Running {solver} (decision) for {n} teams")
                                    decision_result = self.run_solver(solver, n, optimization=False, sb_teams=s1, sb_weeks=s2)
                    
                                    status = "OK" if decision_result["optimal"] else "FAIL"
                                    reason_str = f" ({decision_result['stop_reason']})" if decision_result['stop_reason'] else ""
                                    print(f"    {status} {solver} (decision): {decision_result['time']}s{reason_str}")

                                    comb = "_dec"
                                    comb += "_st1" if s1 else ""
                                    comb += "_sw1" if s2 else ""

                                    merged_results[solver+comb] = decision_result
                
                            # Run optimization version if requested
                                if run_optimization:
                                    print(f"  Running {solver} (optimization) for {n} teams")
                                    optimization_result = self.run_solver(solver, n, optimization=True, sb_teams=s1, sb_weeks=s2)
                    
                                    status = "OK" if optimization_result["optimal"] else "FAIL"
                                    obj_str = f", obj: {optimization_result['obj']}" if optimization_result["obj"] is not None else ""
                                    reason_str = f" ({optimization_result['stop_reason']})" if optimization_result['stop_reason'] else ""
                                    print(f"    {status} {solver} (optimization): {optimization_result['time']}s{obj_str}{reason_str}")

                                    comb = "_opt"
                                    comb += "_st1" if s1 else ""
                                    comb += "_sw1" if s2 else ""
                
                                    merged_results[solver+comb] = optimization_result
                    else:
                        if run_decision:
                            print(f"  Running {solver} (decision) for {n} teams")
                            decision_result = self.run_solver(solver, n, optimization=False, sb_teams=sb_teams, sb_weeks=sb_weeks)
                    
                            status = "OK" if decision_result["optimal"] else "FAIL"
                            reason_str = f" ({decision_result['stop_reason']})" if decision_result['stop_reason'] else ""
                            print(f"    {status} {solver} (decision): {decision_result['time']}s{reason_str}")

                            merged_results[solver+"_dec"] = decision_result
                
                        # Run optimization version if requested
                        if run_optimization:
                            print(f"  Running {solver} (optimization) for {n} teams")
                            optimization_result = self.run_solver(solver, n, optimization=True, sb_teams=sb_teams, sb_weeks=sb_weeks)
                    
                            status = "OK" if optimization_result["optimal"] else "FAIL"
                            obj_str = f", obj: {optimization_result['obj']}" if optimization_result["obj"] is not None else ""
                            reason_str = f" ({optimization_result['stop_reason']})" if optimization_result['stop_reason'] else ""
                            print(f"    {status} {solver} (optimization): {optimization_result['time']}s{obj_str}{reason_str}")
                
                            merged_results[solver+"_opt"] = optimization_result
            
                # Save combined results
                results_dir = (Path(__file__).parent.parent.parent / "res" / "MIP")
                results_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{n}.json"
                with open(results_dir / filename, "wb") as f:
                    f.write(orjson.dumps(merged_results, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
            
                with open(results_dir / filename, "r") as f:
                    lines = f.readlines()

                with open(results_dir / filename, "w") as f:
                    for line in lines:
                        if '"sol": "' in line and "unsat" not in line and "timeout" not in line:
                            # Remove surrounding quotes
                            line = line.replace('\\"', '"') 
                            line = line.replace('"sol": "', '"sol": ').rstrip()
                            if line.endswith('"'):
                                line = line[:-1]  # remove closing quote
                            f.write(line + "\n")
                        else:
                            f.write(line)
            
                print(f"Results saved to {filename}")
        else:
            for n in team_sizes:
                print(f"\n=== Running experiments for {n} teams ===")
                # Combined results structure
                merged_results = {}
                for solver in available_solvers:
                    # Run decision version if requested
                    if run_decision:
                        print(f"  Running {solver} (decision) for {n} teams")
                        decision_result = self.run_solver(solver, n, optimization=False, sb_teams=None, sb_weeks=None, circle_method=True)
                    
                        status = "OK" if decision_result["optimal"] else "FAIL"
                        reason_str = f" ({decision_result['stop_reason']})" if decision_result['stop_reason'] else ""
                        print(f"    {status} {solver} (decision): {decision_result['time']}s{reason_str}")

                        merged_results[solver+"_dec"] = decision_result
                
                    # Run optimization version if requested
                    if run_optimization:
                        print(f"  Running {solver} (optimization) for {n} teams")
                        optimization_result = self.run_solver(solver, n, optimization=True, sb_teams=None, sb_weeks=None, circle_method=True)
                    
                        status = "OK" if optimization_result["optimal"] else "FAIL"
                        obj_str = f", obj: {optimization_result['obj']}" if optimization_result["obj"] is not None else ""
                        reason_str = f" ({optimization_result['stop_reason']})" if optimization_result['stop_reason'] else ""
                        print(f"    {status} {solver} (optimization): {optimization_result['time']}s{obj_str}{reason_str}")
                
                        merged_results[solver+"_opt"] = optimization_result
                        
                # Save combined results
                results_dir = (Path(__file__).parent.parent.parent / "res" / "MIP" / "circle")
                results_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{n}.json"
                with open(results_dir / filename, "wb") as f:
                    f.write(orjson.dumps(merged_results, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
            
                with open(results_dir / filename, "r") as f:
                    lines = f.readlines()

                with open(results_dir / filename, "w") as f:
                    for line in lines:
                        if '"sol": "' in line and "unsat" not in line and "timeout" not in line:
                            # Remove surrounding quotes
                            line = line.replace('\\"', '"') 
                            line = line.replace('"sol": "', '"sol": ').rstrip()
                            if line.endswith('"'):
                                line = line[:-1]  # remove closing quote
                            f.write(line + "\n")
                        else:
                            f.write(line)
            
                print(f"Results saved to {filename}")

def main():
    """Main function to run the MIP solver."""
    if len(sys.argv) < 2:
        team_sizes = [2,4,6]
        params = {
                "sb_weeks":True,
                "sb_teams":True
                }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        circle_method = True
        time_limit = 300
        ampl_path = None
        solvers = None
    else:
        # Parse arguments
        team_sizes = []
        params = {
                "sb_weeks":True,
                "sb_teams":True
                }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        circle_method = True
        time_limit = 300
        ampl_path = None
        solvers = None
        i = 1
        args_input = []
        while i < len(sys.argv):
            arg = sys.argv[i]
            args_input.append(arg.lower())
            if arg.lower() == '--decision-only':
                run_optimization = False
            elif arg.lower() == '--optimization-only':
                run_decision = False
            elif arg.lower() == '--all-optional':
                params = {
                        "sb_weeks":True,
                        "sb_teams":True
                        }
                symmetry_combinations = False
            elif arg.lower() == '--no-optional':
                params = {
                        "sb_weeks":False,
                        "sb_teams":False
                        }
                symmetry_combinations = False
            elif arg.lower() == '--no-combinations':
                symmetry_combinations = False
                if i + 1 < len(sys.argv):
                    symmetry = sys.argv[i + 1]
                    valid_symmetry = ["sb_weeks", "sb_teams"]
                    if symmetry.lower() in ["--decision-only","--optimization-only","--all-optional", "--no-optional", "--no-combinations", "--time-limit","--solvers","--ample-path"]:
                        params = {
                                "sb_weeks":True,
                                "sb_teams":True
                                }
                    else:
                        if symmetry not in valid_symmetry:
                            print(f"Error: {symmetry} is not a valid symmetry breaking. Valid symmetry breaking: {', '.join(valid_symmetry)}")
                            sys.exit(1)
                        else:
                            if symmetry == "sb_teams":
                                params = {
                                        "sb_weeks":False,
                                        "sb_teams":True
                                        }
                            else:
                                params = {
                                        "sb_weeks":True,
                                        "sb_teams":False
                                        }
                            i += 1
            elif arg.lower() == "--no-circle":
                circle_method = False
            elif arg.lower() == '--time-limit':
                if i + 1 < len(sys.argv):
                    try:
                        time_limit = int(sys.argv[i + 1])
                        if time_limit <= 0:
                            print("Error: time limit must be positive")
                            sys.exit(1)
                        i += 1
                    except ValueError:
                        print("Error: time limit must be an integer")
                        sys.exit(1)
                else:
                    print("Error: --time-limit requires a time value in seconds")
                    sys.exit(1)
            elif arg.lower() == '--ampl-path':
                if i + 1 < len(sys.argv):
                    ampl_path = sys.argv[i + 1]
                    i += 1
                else:
                    print("Error: --ampl-path requires a path argument")
                    sys.exit(1)
            elif arg.lower() == '--solvers':
                if i + 1 < len(sys.argv):
                    solver_list = sys.argv[i + 1].split(',')
                    valid_solvers = ["cbc", "highs", "cplex", "gurobi"]
                    solvers = []
                    for solver in solver_list:
                        solver = solver.strip()
                        if solver.lower() in valid_solvers:
                            solvers.append(solver.lower())
                        else:
                            print(f"Error: {solver.lower()} is not a valid solver. Valid solvers: {', '.join(valid_solvers)}")
                            sys.exit(1)
                    i += 1
                else:
                    print("Error: --solvers requires a comma-separated list of solvers")
                    sys.exit(1)
            elif arg.lower() == '--time-limit':
                if i + 1 < len(sys.argv):
                    try:
                        time_limit = int(sys.argv[i + 1])
                        if time_limit <= 0:
                            print("Error: time limit must be positive")
                            sys.exit(1)
                        i += 1
                    except ValueError:
                        print("Error: time limit must be an integer")
                        sys.exit(1)
                else:
                    print("Error: --time-limit requires a time value in seconds")
                    sys.exit(1)
            else:
                try:
                    n = int(arg)
                    if n < 2 or n % 2 != 0:
                        print(f"Error: {n} is not valid (must be even and >= 2)")
                        sys.exit(1)
                    team_sizes.append(n)
                except ValueError:
                    print(f"Error: {arg} is not a valid team size or option")
                    sys.exit(1)
            i += 1
        if not params:
            params = {
                    "sb_match":True,
                    "sb_teams":True,
                    "sb_periods":True
                    }

        if not team_sizes:
            team_sizes = [2,4,6]
    
        if not run_decision and not run_optimization:
            print("Error: Cannot use both --decision-only and --optimization-only")
            sys.exit(1)

        uncomp = ["--no-optional", "--all-optional", "--no-combinations"]
        if sum(s in uncomp for s in args_input) >= 2:
            print("Error: Cannot use both \"--no-optional\", \"--all-optional\" or \"--no-combinations\"")
            sys.exit(1)        
    
    runner = STSMIPRunner(ampl_path=ampl_path, time_limit=time_limit, solvers=solvers)
    runner.run_experiments(team_sizes, run_decision=run_decision, run_optimization=run_optimization, symmetry_combinations=symmetry_combinations, sb_teams=params["sb_teams"], sb_weeks=params["sb_weeks"], circle_method = circle_method)

if __name__ == "__main__":
    main()