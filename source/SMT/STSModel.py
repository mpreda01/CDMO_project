#!/usr/bin/env python3
import json
import time
import os
import subprocess
from z3 import *

class STSModel:
    def __init__(self, n_teams,
                 sb_weeks=False, sb_periods=False, sb_teams=False,
                 ic_matches_per_team=False, ic_period_count=False):
        self.n = n_teams
        self.n_weeks = n_teams - 1
        self.n_periods = n_teams // 2
        self.sb_weeks = sb_weeks
        self.sb_periods = sb_periods
        self.sb_teams = sb_teams
        self.ic_matches_per_team = ic_matches_per_team
        self.ic_period_count = ic_period_count

        # boolean var x[w][p][h][a]
        self.x = {}
        for w in range(self.n_weeks):
            self.x[w] = {}
            for p in range(self.n_periods):
                self.x[w][p] = {}
                for h in range(self.n):
                    self.x[w][p][h] = {}
                    for a in range(self.n):
                        self.x[w][p][h][a] = Bool(f'x_{w}_{p}_{h}_{a}')

    def create_constraints(self, solver, use_cvc5_encoding=False):
        if use_cvc5_encoding:
            self._create_constraints_cvc5(solver)
        else:
            self._create_constraints_z3(solver)
    
    def _create_constraints_z3(self, solver):
        """Original Z3 constraints using PbEq and PbLe (pseudo-boolean)."""
        # Base constraints 
        for w in range(self.n_weeks):
            for p in range(self.n_periods):
                for t in range(self.n):
                    solver.add(Not(self.x[w][p][t][t]))

        for w in range(self.n_weeks):
            for p in range(self.n_periods):
                game_pairs = [self.x[w][p][h][a] for h in range(self.n) for a in range(self.n) if h != a]
                solver.add(PbEq([(pair, 1) for pair in game_pairs], 1))

        for w in range(self.n_weeks):
            for t in range(self.n):
                plays_in_week = []
                for p in range(self.n_periods):
                    for a in range(self.n):
                        if a != t:
                            plays_in_week.append(self.x[w][p][t][a])
                    for h in range(self.n):
                        if h != t:
                            plays_in_week.append(self.x[w][p][h][t])
                solver.add(PbEq([(game, 1) for game in plays_in_week], 1))

        for t1 in range(self.n):
            for t2 in range(t1 + 1, self.n):
                games_between = []
                for w in range(self.n_weeks):
                    for p in range(self.n_periods):
                        games_between.append(self.x[w][p][t1][t2])
                        games_between.append(self.x[w][p][t2][t1])
                solver.add(PbEq([(game, 1) for game in games_between], 1))

        for t in range(self.n):
            for p in range(self.n_periods):
                period_games = []
                for w in range(self.n_weeks):
                    for a in range(self.n):
                        if a != t:
                            period_games.append(self.x[w][p][t][a])
                    for h in range(self.n):
                        if h != t:
                            period_games.append(self.x[w][p][h][t])
                solver.add(PbLe([(game, 1) for game in period_games], 2))

        
        if self.ic_matches_per_team:
            self.add_ic_matches_per_team(solver)
        if self.ic_period_count:
            self.add_ic_period_count(solver)
        if self.sb_teams or self.sb_weeks or self.sb_periods:
            self.add_symmetry_breaking(solver)
    
    def _create_constraints_cvc5(self, solver):
        """CVC5-compatible constraints using Sum and If instead of PbEq/PbLe."""
        # Base constraints 
        for w in range(self.n_weeks):
            for p in range(self.n_periods):
                for t in range(self.n):
                    solver.add(Not(self.x[w][p][t][t]))

        # Exactly one game per period (using Sum instead of PbEq)
        for w in range(self.n_weeks):
            for p in range(self.n_periods):
                game_sum = Sum([If(self.x[w][p][h][a], 1, 0) 
                               for h in range(self.n) for a in range(self.n) if h != a])
                solver.add(game_sum == 1)

        # Each team plays exactly once per week (using Sum instead of PbEq)
        for w in range(self.n_weeks):
            for t in range(self.n):
                plays_sum = Sum([If(self.x[w][p][t][a], 1, 0) 
                                for p in range(self.n_periods) for a in range(self.n) if a != t] +
                               [If(self.x[w][p][h][t], 1, 0) 
                                for p in range(self.n_periods) for h in range(self.n) if h != t])
                solver.add(plays_sum == 1)

        # Each pair of teams meets exactly once (using Sum instead of PbEq)
        for t1 in range(self.n):
            for t2 in range(t1 + 1, self.n):
                pair_sum = Sum([If(self.x[w][p][t1][t2], 1, 0) 
                               for w in range(self.n_weeks) for p in range(self.n_periods)] +
                              [If(self.x[w][p][t2][t1], 1, 0) 
                               for w in range(self.n_weeks) for p in range(self.n_periods)])
                solver.add(pair_sum == 1)

        # At most 2 games in same period (using Sum instead of PbLe)
        for t in range(self.n):
            for p in range(self.n_periods):
                period_sum = Sum([If(self.x[w][p][t][a], 1, 0) 
                                 for w in range(self.n_weeks) for a in range(self.n) if a != t] +
                                [If(self.x[w][p][h][t], 1, 0) 
                                 for w in range(self.n_weeks) for h in range(self.n) if h != t])
                solver.add(period_sum <= 2)

        
        if self.ic_matches_per_team:
            self.add_ic_matches_per_team_cvc5(solver)
        if self.ic_period_count:
            self.add_ic_period_count_cvc5(solver)
        if self.sb_teams or self.sb_weeks or self.sb_periods:
            self.add_symmetry_breaking(solver)

    def add_ic_matches_per_team(self, solver):
        for t in range(self.n):
            total_games = []
            for w in range(self.n_weeks):
                for p in range(self.n_periods):
                    for a in range(self.n):
                        if a != t:
                            total_games.append(self.x[w][p][t][a])
                            total_games.append(self.x[w][p][a][t])
            solver.add(PbEq([(g, 1) for g in total_games], self.n - 1))
    
    def add_ic_matches_per_team_cvc5(self, solver):
        """CVC5-compatible version using Sum instead of PbEq."""
        for t in range(self.n):
            total_sum = Sum([If(self.x[w][p][t][a], 1, 0) 
                            for w in range(self.n_weeks) 
                            for p in range(self.n_periods) 
                            for a in range(self.n) if a != t] +
                           [If(self.x[w][p][a][t], 1, 0) 
                            for w in range(self.n_weeks) 
                            for p in range(self.n_periods) 
                            for a in range(self.n) if a != t])
            solver.add(total_sum == self.n - 1)

    def add_ic_period_count(self, solver):
        for p in range(self.n_periods):
            total_games = []
            for w in range(self.n_weeks):
                for h in range(self.n):
                    for a in range(self.n):
                        if h != a:
                            total_games.append(self.x[w][p][h][a])
            solver.add(PbEq([(g, 1) for g in total_games], self.n_weeks))
    
    def add_ic_period_count_cvc5(self, solver):
        """CVC5-compatible version using Sum instead of PbEq."""
        for p in range(self.n_periods):
            total_sum = Sum([If(self.x[w][p][h][a], 1, 0) 
                            for w in range(self.n_weeks) 
                            for h in range(self.n) 
                            for a in range(self.n) if h != a])
            solver.add(total_sum == self.n_weeks)

    def add_symmetry_breaking(self, solver):
        if self.sb_teams and self.n >= 2:
            solver.add(self.x[0][0][0][1])

    def optimize_schedule(self, schedule):
        """Optimize schedule using Z3 Optimize solver to minimize max home/away imbalance."""
        opt = Optimize()
        
        # Create swap variables for each match
        # swap[p][w] = True means match in period p, week w should be reversed (home<->away)
        swap = {}
        for p in range(self.n_periods):
            swap[p] = {}
            for w in range(self.n_weeks):
                swap[p][w] = Bool(f'swap_{p}_{w}')
        
        # Create variables for home and away counts for each team
        home_count = [Int(f'home_{t}') for t in range(self.n)]
        away_count = [Int(f'away_{t}') for t in range(self.n)]
        
        # Count home and away games for each team considering swaps
        for t in range(self.n):
            home_expr = 0
            away_expr = 0
            
            for p in range(self.n_periods):
                for w in range(self.n_weeks):
                    h, a = schedule[p][w]
                    if h == 0 or a == 0:  # Invalid match
                        continue
                    
                    # Convert to 0-indexed
                    home_team = h - 1
                    away_team = a - 1
                    
                    if home_team == t:
                        # This team is home in original schedule
                        # If not swapped, it's home (+1), if swapped, it's away (+0)
                        home_expr = home_expr + If(swap[p][w], 0, 1)
                        away_expr = away_expr + If(swap[p][w], 1, 0)
                    elif away_team == t:
                        # This team is away in original schedule
                        # If not swapped, it's away (+1), if swapped, it's home (+0)
                        home_expr = home_expr + If(swap[p][w], 1, 0)
                        away_expr = away_expr + If(swap[p][w], 0, 1)
            
            opt.add(home_count[t] == home_expr)
            opt.add(away_count[t] == away_expr)
        
        # Create variable for maximum imbalance
        max_imbalance = Int('max_imbalance')
        
        # Constrain max_imbalance to be >= all team imbalances
        for t in range(self.n):
            diff = home_count[t] - away_count[t]
            opt.add(max_imbalance >= diff)
            opt.add(max_imbalance >= -diff)
        
        # Minimize the maximum imbalance
        opt.minimize(max_imbalance)
        
        # Solve the optimization problem
        if opt.check() == sat:
            model = opt.model()
            
            # Apply swaps to create optimized schedule
            optimized_schedule = []
            for p in range(self.n_periods):
                period_schedule = []
                for w in range(self.n_weeks):
                    h, a = schedule[p][w]
                    if h == 0 or a == 0:
                        period_schedule.append([h, a])
                    else:
                        # Check if this match should be swapped
                        should_swap = is_true(model.eval(swap[p][w]))
                        if should_swap:
                            period_schedule.append([a, h])  # Reverse home and away
                        else:
                            period_schedule.append([h, a])  # Keep original
                optimized_schedule.append(period_schedule)
            
            # Get the objective value
            obj_val = model.eval(max_imbalance).as_long()
            
            return optimized_schedule, obj_val
        else:
            # If optimization fails, return original schedule
            return schedule, None

    def solve(self, time_limit=300, optimize=False, solver_choice="z3"):
        if solver_choice == "cvc5" and optimize:
            print("ERROR: Optimization is not supported with cvc5 solver.")
            return {
                "time": 0,
                "solve_time": 0,
                "optimize_time": 0,
                "optimal": False,
                "obj": "None",
                "sol": []
            }
        
        if solver_choice == "cvc5":
            return self._solve_with_cvc5(time_limit)
        else:
            return self._solve_with_z3(time_limit, optimize)
    
    def _solve_with_z3(self, time_limit=300, optimize=False):
        solver = Solver()
        solver.set("timeout", time_limit * 1000)
        self.create_constraints(solver)
        start = time.time()
        result = solver.check()
        solve_time = time.time() - start
        optimize_time = 0.0

        if result == sat:
            model = solver.model()
            schedule = self.extract_solution(model)
            obj_val = None
            if optimize:
                opt_start = time.time()
                schedule, obj_val = self.optimize_schedule(schedule)
                optimize_time = time.time() - opt_start
            
            total_time = solve_time + optimize_time
            
            # Print solution details
            print("\n" + "="*60)
            print("SOLUTION FOUND")
            print("="*60)
            print(f"Solve time: {solve_time:.2f} seconds")
            if optimize:
                print(f"Optimize time: {optimize_time:.2f} seconds")
                print(f"Total time: {total_time:.2f} seconds")
            else:
                print(f"Total time: {solve_time:.2f} seconds")
            if optimize and obj_val is not None:
                print(f"Objective value (max imbalance): {obj_val}")
            print("\nSolution matrix (by periods and weeks):")
            for p, period in enumerate(schedule):
                print(f"Period {p+1}:", end=" ")
                print("[", end="")
                for w, match in enumerate(period):
                    print(f"[{match[0]}, {match[1]}]", end="")
                    if w < len(period) - 1:
                        print(", ", end="")
                print("]")
            print("="*60 + "\n")
            
            return {
                "time": int(total_time),
                "solve_time": round(solve_time, 2),
                "optimize_time": round(optimize_time, 2) if optimize else 0,
                "optimal": (obj_val == 1) if optimize else True,
                "obj": obj_val if optimize and obj_val is not None else "None",
                "sol": schedule
            }

        print("\n" + "="*60)
        print("NO SOLUTION FOUND")
        print("="*60)
        print(f"Computation time: {solve_time:.2f} seconds")
        print("="*60 + "\n")
        
        # For decision version (optimize=False), UNSAT is still "solved" → optimal=True
        # For optimization version (optimize=True), UNSAT means not solved → optimal=False
        return {
            "time": int(solve_time), 
            "solve_time": round(solve_time, 2), 
            "optimize_time": 0, 
            "optimal": not optimize,  # True for decision (optimize=False), False for optimization (optimize=True)
            "obj": "None", 
            "sol": []
        }

    def _solve_with_cvc5(self, time_limit=300):
        """Solve using cvc5 by exporting to SMT-LIB2 and calling cvc5 binary."""
        solver = Solver()
        self.create_constraints(solver, use_cvc5_encoding=True)
        
        start = time.time()
        
        # Export to SMT-LIB2
        smt2_string = solver.to_smt2()
        smt2_lines = smt2_string.splitlines()
        # Try QF_UFLIA (Uninterpreted Functions + Linear Integer Arithmetic) which is more general
        # and might handle pseudo-boolean constraints better
        smt2_lines[0] = "(set-logic ALL)"
        smt2_lines.append("(get-model)")
        smt2_string = "\n".join(smt2_lines)
        
        smt2_path = "./sts_stsmodel.smt2"
        with open(smt2_path, 'w') as f:
            f.write(smt2_string)
        
        # Call cvc5 binary - find the correct path relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_cvc5_path = os.path.join(script_dir, "cvc5", "bin", "cvc5.exe")
        if not os.path.exists(default_cvc5_path):
            default_cvc5_path = os.path.join(script_dir, "cvc5", "bin", "cvc5")
        cvc5_bin = os.environ.get("CVC5_BIN", default_cvc5_path)
        cmd = [cvc5_bin, "--lang", "smt2", "--produce-models", smt2_path]
        
        timeout_sec = time_limit
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            solve_time = time.time() - start
            # Clean up temporary file
            if os.path.exists(smt2_path):
                os.remove(smt2_path)
            
            print("\n" + "="*60)
            print("CVC5 TIMEOUT")
            print("="*60)
            print(f"Computation time: {solve_time:.2f} seconds")
            print("="*60 + "\n")
            
            return {
                "time": int(solve_time),
                "solve_time": round(solve_time, 2),
                "optimize_time": 0,
                "optimal": False,
                "obj": "None",
                "sol": []
            }
        
        solve_time = time.time() - start
        stdout = result.stdout.strip()
        
        # Check result
        if "unsat" in stdout:
            # Clean up temporary file
            if os.path.exists(smt2_path):
                os.remove(smt2_path)
            
            print("\n" + "="*60)
            print("NO SOLUTION FOUND (CVC5 UNSAT)")
            print("="*60)
            print(f"Computation time: {solve_time:.2f} seconds")
            print("="*60 + "\n")
            
            return {
                "time": int(solve_time),
                "solve_time": round(solve_time, 2),
                "optimize_time": 0,
                "optimal": True,
                "obj": "None",
                "sol": []
            }
        
        if "sat" not in stdout:
            # Clean up temporary file
            if os.path.exists(smt2_path):
                os.remove(smt2_path)
            
            print("\n" + "="*60)
            print("CVC5 UNKNOWN")
            print("="*60)
            print(f"Computation time: {solve_time:.2f} seconds")
            print("="*60 + "\n")
            
            return {
                "time": int(solve_time),
                "solve_time": round(solve_time, 2),
                "optimize_time": 0,
                "optimal": False,
                "obj": "None",
                "sol": []
            }
        
        # Parse model from cvc5 output
        model = {}
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("(define-fun"):
                line = line[:-1]
                parts = line.split()
                if len(parts) >= 5:
                    var = parts[1]
                    value_str = parts[-1]
                    if value_str.lower() == "true":
                        model[var] = True
                    elif value_str.lower() == "false":
                        model[var] = False
                    else:
                        try:
                            model[var] = int(value_str)
                        except ValueError:
                            pass
        
        # Extract solution from model
        schedule = []
        for p in range(self.n_periods):
            period_schedule = []
            for w in range(self.n_weeks):
                found = False
                for h in range(self.n):
                    for a in range(self.n):
                        if h != a:
                            var_name = f'x_{w}_{p}_{h}_{a}'
                            if model.get(var_name, False) == True:
                                period_schedule.append([h+1, a+1])
                                found = True
                                break
                    if found:
                        break
                if not found:
                    period_schedule.append([0, 0])
            schedule.append(period_schedule)
        
        # Clean up temporary file
        if os.path.exists(smt2_path):
            os.remove(smt2_path)
        
        # Print solution details
        print("\n" + "="*60)
        print("SOLUTION FOUND (CVC5)")
        print("="*60)
        print(f"Solve time: {solve_time:.2f} seconds")
        print(f"Total time: {solve_time:.2f} seconds")
        print("\nSolution matrix (by periods and weeks):")
        for p, period in enumerate(schedule):
            print(f"Period {p+1}:", end=" ")
            print("[", end="")
            for w, match in enumerate(period):
                print(f"[{match[0]}, {match[1]}]", end="")
                if w < len(period) - 1:
                    print(", ", end="")
            print("]")
        print("="*60 + "\n")
        
        return {
            "time": int(solve_time),
            "solve_time": round(solve_time, 2),
            "optimize_time": 0,
            "optimal": True,
            "obj": "None",
            "sol": schedule
        }
    
    def extract_solution(self, model):
        schedule = []
        for p in range(self.n_periods):
            period_schedule = []
            for w in range(self.n_weeks):
                found = False
                for h in range(self.n):
                    for a in range(self.n):
                        if h != a and is_true(model.eval(self.x[w][p][h][a])):
                            period_schedule.append([h+1, a+1])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    period_schedule.append([0, 0])
            schedule.append(period_schedule)
        return schedule
