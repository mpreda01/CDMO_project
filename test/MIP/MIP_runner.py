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
        
    def load_model(self, ampl, optimization=True):
        """Load unified AMPL model and configure for decision/optimization"""
        try:
            # Load the unified model
            ampl.read(self.model_file)
            
            # Set optimization parameter
            optimize_param = 1 if optimization else 0
            ampl.getParameter("optimize_balance").set(optimize_param)
            
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
    
    def run_solver(self, solver, n, optimization=True):
        """Run AMPL with specific solver using AMPLpy"""
        print(f"Running {solver} for n={n}...")
        
        try:
            # Create AMPL instance
            if self.ampl_path:
                ampl = AMPL(Environment(self.ampl_path))
            else:
                ampl = AMPL()
            
            start_time = time.time()
            
            # Load model from file
            if not self.load_model(ampl, optimization):
                return {
                    "time": self.time_limit,
                    "version": "optimal" if optimization else "decision",
                    "optimal": False,
                    "stop_reason": None,
                    "obj": None,
                    "sol": None
                }
            
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
            
            result = {
                "time": min(runtime, self.time_limit),
                "version": "optimal" if optimization else "decision",
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
    
    def run_experiments(self, team_sizes, run_decision=True, run_optimization=True):
        """Run experiments for different team sizes"""
        mode_str = []
        if run_decision:
            mode_str.append("decision")
        if run_optimization:
            mode_str.append("optimization")
        
        print(f"\n=== Running experiments for {' and '.join(mode_str)} ===")
        print(f"Time limit: {self.time_limit}s")
        print(f"Solvers to use: {', '.join(self.solvers)}")
        
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
        
        for n in team_sizes:
            print(f"\n=== Running experiments for {n} teams ===")
            
            # Combined results structure
            merged_results = {}
            
            for solver in available_solvers:
                
                # Run decision version if requested
                if run_decision:
                    print(f"  Running {solver} (decision)...")
                    decision_result = self.run_solver(solver, n, optimization=False)
                    
                    status = "OK" if decision_result["optimal"] else "FAIL"
                    reason_str = f" ({decision_result['stop_reason']})" if decision_result['stop_reason'] else ""
                    print(f"    {status} {solver} (decision): {decision_result['time']}s{reason_str}")
                    
                    merged_results[solver+"_dec"] = decision_result
                
                # Run optimization version if requested
                if run_optimization:
                    print(f"  Running {solver} (optimization)...")
                    optimization_result = self.run_solver(solver, n, optimization=True)
                    
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

def main():
    if len(sys.argv) < 2:
        # Parse arguments
        team_sizes = [2,4,6,8,10,12]
        ampl_path = None
        solvers = None
        time_limit = 300
        run_decision = True
        run_optimization = True
    else:
        # Parse arguments
        team_sizes = []
        ampl_path = None
        solvers = None
        time_limit = 300
        run_decision = True
        run_optimization = True
    
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == '--ampl-path':
                if i + 1 < len(sys.argv):
                    ampl_path = sys.argv[i + 1]
                    i += 1
                else:
                    print("Error: --ampl-path requires a path argument")
                    sys.exit(1)
            elif arg == '--solvers':
                if i + 1 < len(sys.argv):
                    solver_list = sys.argv[i + 1].split(',')
                    valid_solvers = ["cbc", "highs", "cplex", "gurobi"]
                    solvers = []
                    for solver in solver_list:
                        solver = solver.strip()
                        if solver in valid_solvers:
                            solvers.append(solver)
                        else:
                            print(f"Error: {solver} is not a valid solver. Valid solvers: {', '.join(valid_solvers)}")
                            sys.exit(1)
                    i += 1
                else:
                    print("Error: --solvers requires a comma-separated list of solvers")
                    sys.exit(1)
            elif arg == '--time-limit':
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
            elif arg == '--decision-only':
                run_decision = True
                run_optimization = False
            elif arg == '--optimization-only':
                run_decision = False
                run_optimization = True
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
    
        if not team_sizes:
            print("Error: No valid team sizes provided")
            sys.exit(1)
    
        if not run_decision and not run_optimization:
            print("Error: Cannot use both --decision-only and --optimization-only")
            sys.exit(1)
    
    runner = STSMIPRunner(ampl_path=ampl_path, time_limit=time_limit, solvers=solvers)
    runner.run_experiments(team_sizes, run_decision=run_decision, run_optimization=run_optimization)

if __name__ == "__main__":
    main()