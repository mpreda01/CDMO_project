"""
STS SAT Solver Runner
Main script to run the STS SAT solver with different configurations.
"""

import sys
import json
import os
import time
from typing import Dict, List, Any
import itertools

# Add source directory to path to import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../source/SAT'))
from sts_sat_model import STSSATSolver


def save_results(n: int, result: Dict[str, Any], 
                 output_dir: str = "res/SAT", symmetry_level: str = "full", 
                 mode: str = "feasible", result_with_meta: dict = {}) -> None:
    """
    Save results to JSON file.
    
    Args:
        n: Number of teams
        result: Dictionary containing solution results
        approach_name: Name of the approach used
        output_dir: Output directory for results
        symmetry_level: Level of symmetry breaking used
        mode: Solving mode (feasible/optimize)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{n}.json")

    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    result_with_meta = {}
    res_name = "z3"
    symmetry_constraints = {"sb_team": False, "sb_match": False, "sb_period": False}
    if mode == "optimize":
        res_name += "_opt"
    else:
        res_name += "_dec"
    if not symmetry_level:
        pass
    else:
        for level in symmetry_level:
            if level == "sb_match":
                symmetry_constraints["sb_match"] = True
                res_name += "_sm1"
            elif level == "sb_team":
                symmetry_constraints["sb_team"] = True
                res_name += "_st1"
            elif level == "sb_period":
                symmetry_constraints["sb_period"] = True
                res_name += "_sp1"
            else:
                pass

    if 'sol' in result:
        sol = result['sol']
        if sol and len(sol) > 0:
            # TRASPOSIZIONE week x period -> period x week per storage JSON
            transposed_sol = [[sol[w][p] for w in range(len(sol))]
                              for p in range(len(sol[0]))]
            result_with_meta['sol'] = transposed_sol
        else:
            result_with_meta['sol'] = []
    if 'time' in result:
        result_with_meta['time'] = int(round(result['time'])) if isinstance(result['time'], (float, int)) else result['time']
    if 'optimal' in result:
        result_with_meta['optimal'] = result['optimal']
    if 'obj' in result:
        result_with_meta['obj'] = result['obj']

    result_with_meta['constraints'] = symmetry_constraints

    key = res_name
    existing_results[key] = result_with_meta

    def format_json_with_compact_solution(data):
        output_lines = ["{"]
        sorted_keys = sorted(data.keys())
        for idx, approach_key in enumerate(sorted_keys):
            approach_data = data[approach_key]
            output_lines.append(f'  "{approach_key}": {{')
            field_lines = []
            for field_key, field_value in approach_data.items():
                if field_key == 'sol' and isinstance(field_value, list):
                    sol_lines = ['    "sol": [']
                    for period_idx, period in enumerate(field_value):
                        period_str = json.dumps(period, separators=(',', ' '))
                        if period_idx < len(field_value) - 1:
                            sol_lines.append(f'      {period_str},')
                        else:
                            sol_lines.append(f'      {period_str}')
                    sol_lines.append('    ]')
                    field_lines.append('\n'.join(sol_lines))
                else:
                    field_str = json.dumps(field_value, indent=4)
                    field_str_indented = '\n'.join(['    ' + line if line else line 
                                                    for line in field_str.split('\n')])
                    field_lines.append(f'    "{field_key}": {field_str_indented.strip()}')
            output_lines.append(',\n'.join(field_lines))
            if idx < len(sorted_keys) - 1:
                output_lines.append('  },')
            else:
                output_lines.append('  }')
        output_lines.append('}')
        return '\n'.join(output_lines)

    with open(output_file, 'w') as f:
        f.write(format_json_with_compact_solution(existing_results))


def main():
    """Main function to run the SAT solver."""
    if len(sys.argv) < 2:
        team_sizes = [2,4,6]
        level = ["sb_match","sb_team","sb_period"]
        mode = "all"
    else:
        i = 1
        level = []
        mode = "all"
        team_sizes = []
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.lower() == "sb_match":
                level.append(arg.lower())
            elif arg.lower() == "sb_team":
                level.append(arg.lower())
            elif arg.lower() == "sb_period":
                level.append(arg.lower())
            elif arg.lower() == "optimization-only":
                mode = "optimization"
            elif arg.lower() == "decision-only":
                mode = "feasible"
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
        if not level:
            level = ["sb_match","sb_team","sb_period"]

    for n in team_sizes:
        # Create solver instance
        solver = STSSATSolver(n, timeout=300)
    
        print("\n" + "-" * 50)
        print(f"Solving STS problem for {n} teams")
        print("-" * 50)
    
        if mode == 'optimization':
            # Run optimization mode
            result_dict = solver.solve_optimization_incremental(symmetry_level=level, output_dir="res/SAT")
            # Don't include level in approach name - save_results will add it
        
            if result_dict['satisfiable']:
            
                json_result = {
                    'time': int(round(result_dict['time'])),
                    'optimal': result_dict['optimal'],
                    'obj': result_dict.get('obj'),
                    'sol': result_dict['solution']
                }

                result_with_meta = {}
                save_results(n, json_result, symmetry_level=level, mode=mode, result_with_meta=result_with_meta)
            else:
                print("No feasible solution found to optimize.")
        elif mode == "feasible":
            # Run feasibility mode
            success, solution, solve_time, model = solver.solve_feasibility(symmetry_level=level)
            if success:
                stats = solver.get_solution_stats(model)
                objective_value = solver._calculate_objective(model)
                        
                json_result = {
                    'time': int(round(solve_time)),
                    'optimal': solve_time < 300,
                    'obj': objective_value,
                    'sol': solution
                }
                result_with_meta = {}
                save_results(n, json_result, symmetry_level=level, mode=mode, result_with_meta=result_with_meta)
            else:
                json_result = {
                    'time': 300,
                    'optimal': False,
                    'obj': None,
                    'sol': []
                }
                result_with_meta = {}
                save_results(n, json_result, symmetry_level=level, mode=mode, result_with_meta=result_with_meta)
        else:
            flag_names = ["sb_team", "sb_match", "sb_period", "optimize"]
            all_flag_combos = list(itertools.product([False, True], repeat=len(flag_names)))
            total_combos = len(all_flag_combos)

            result_with_meta = {}
            for i, combo in enumerate(all_flag_combos, start=1):
                print(f"Solving team={n}, combo={i}/{total_combos}")
                flags_combo = dict(zip(flag_names, combo))
                level = []
                if flags_combo["sb_team"]:
                    level.append("sb_team")
                if flags_combo["sb_match"]:
                    level.append("sb_match")
                if flags_combo["sb_period"]:
                    level.append("sb_period")
                if flags_combo["optimize"]:
                    result_dict = solver.solve_optimization_incremental(symmetry_level=level, output_dir="res/SAT")
                    # Don't include level in approach name - save_results will add it
        
                    if result_dict['satisfiable']:
                                    
                        json_result = {
                            'time': int(round(result_dict['time'])),
                            'optimal': result_dict['optimal'],
                            'obj': result_dict.get('obj'),
                            'sol': result_dict['solution']
                        }
                        save_results(n, json_result, symmetry_level=level, mode="optimize", result_with_meta=result_with_meta)
                    else:
                        json_result = {
                            'time': 300,
                            'optimal': False,
                            'obj': None,
                            'sol': []
                        }
                        result_with_meta = {}
                        save_results(n, json_result, symmetry_level=level, mode=mode, result_with_meta=result_with_meta)
                else:
                    success, solution, solve_time, model = solver.solve_feasibility(symmetry_level=level)
                    if success:
                        stats = solver.get_solution_stats(model)
                        objective_value = solver._calculate_objective(model)
                        
                        json_result = {
                            'time': int(round(solve_time)),
                            'optimal': solve_time < 300,
                            'obj': objective_value,
                            'sol': solution
                        }
                        save_results(n, json_result, symmetry_level=level, mode="decision", result_with_meta=result_with_meta)
                    else:
                        json_result = {
                            'time': 300,
                            'optimal': False,
                            'obj': None,
                            'sol': []
                        }
                        save_results(n, json_result, symmetry_level=level, mode=mode, result_with_meta=result_with_meta)


if __name__ == "__main__":
    main()