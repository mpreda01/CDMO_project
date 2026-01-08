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
from SAT_sts import STSSATSolver


def save_results(n: int, result: Dict[str, Any], 
                 output_dir: str = "res/SAT", symmetry_level: None, 
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
    symmetry_constraints = {"sb_teams": False, "sb_match": False, "sb_periods": False}
    if mode == "optimize":
        res_name += "_opt"
    else:
        res_name += "_dec"
    if not symmetry_level:
        pass
    else:
        if symmetry_level.get("sb_match", False):
            symmetry_constraints["sb_match"] = True
            res_name += "_sm1"
        if symmetry_level.get("sb_teams", False):
            symmetry_constraints["sb_teams"] = True
            res_name += "_st1"
        if symmetry_level.get("sb_periods", False):
            symmetry_constraints["sb_periods"] = True
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
        params = {
                "ic_match":True,
                "ic_teams":True,
                "sb_periods":True
                }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        circle_method = True
        time_limit = 300
    else:
        # Parse arguments
        team_sizes = []
        params = {
                "ic_match":False,
                "ic_teams":False,
                "sb_periods":False
                }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        circle_method = True
        time_limit = 300
        i = 1
        args_input = []
        while i < len(sys.argv):
            arg = sys.argv[i]
            args_input.append(arg.lower())
            if arg.lower() == "--decision-only":
                run_optimization = False
            elif arg.lower() == "--optimization-only":
                run_decision = False
            elif arg.lower() == '--all-optional':
                params = {
                        "ic_match":True,
                        "ic_teams":True,
                        "sb_periods":True
                        }
                symmetry_combinations = False
            elif arg.lower() == '--no-optional':
                params = {
                        "ic_match":False,
                        "ic_teams":False,
                        "sb_periods":False
                        }
                symmetry_combinations = False
            elif arg.lower() == "--no-combinations":
                symmetry_combinations = False
                params = {
                    "ic_match":False,
                    "ic_teams": False,
                    "sb_periods": False,
                    }
                k = i+1
                for j in range(k, len(sys.argv)):
                    i = j
                    symmetry = sys.argv[j]
                    valid_symmetry = ["ic_match", "ic_teams", "sb_periods"]
                    if symmetry.lower() in ["--decision-only","--optimization-only","--time-limit","--no-combinations","--no-optional","--all-optional"]:
                        i -= 1
                        break
                    else:
                        if symmetry.lower() not in valid_symmetry:
                            print(f"Error: {symmetry} is not a valid symmetry breaking. Valid symmetry breaking: {', '.join(valid_symmetry)}")
                            sys.exit(1)
                        else:
                            if symmetry.lower() == "ic_match":
                                params["ic_match"] = True
                            elif symmetry.lower() == "ic_teams":
                                params["ic_teams"] = True
                            else:
                                params["sb_periods"] = True  
            elif arg.lower() == "--no-circle":
                circle_method = False
            elif arg.lower() == "--time-limit":
                if i + 1 < len(sys.argv) and sys.argv[i + 1] not in ["--optimization-only","--decision-only","--no-combinations"]:
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
                    "ic_match":True,
                    "ic_teams":True,
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

    for n in team_sizes:
        # Create solver instance
        solver = STSSATSolver(n=n, use_circle_method = circle_method, timeout=time_limit)
    
        print(f"Solving STS problem for {n} teams")
        if circle_method:
            method = "circle"
        else:
            method = "classic"
        parameters = ""
        if params["ic_match"]:
            parameters += "ic_match"
        if params["ic_teams"]:
            parameters += " ic_teams"
        if params["sb_periods"]:
            parameters += " sb_periods"
        if parameters == "":
            parameters = "None"
        parameters.strip()

        if not symmetry_combinations:
            result_with_meta = {}
            if run_decision:
                # Run feasibility mode
                result = solver.solve_sts(constraints=params)
                if result["satisfiable"]:   
                    json_result = {
                        'time': int(round(result["time"])),
                        'method': method,
                        'version': "decision",
                        'symmetry_breaking': parameters,
                        'optimal': solve_time < 300,
                        'stop_reason': result["stop_reason"],
                        'obj': "None",
                        'sol': result["solution"]
                    }
                    save_results(n, json_result, symmetry_level=params, mode="feasible", result_with_meta=result_with_meta)
                else:
                    json_result = {
                        'time': 300,
                        'method': method,
                        'version': "decision",
                        'symmetry_breaking': parameters,
                        'optimal': False,
                        'stop_reason': result["stop_reason"],
                        'obj': "None",
                        'sol': []
                    }
                    save_results(n, json_result, symmetry_level=params, mode="feasible", result_with_meta=result_with_meta)
            if run_optimization:
                # Run optimization mode
                result = solver.solve_sts_optimize(constraints=params) ## NON ESISTE
                if result['satisfiable']:
                    json_result = {
                        'time': int(round(result["time"])),
                        'method': method,
                        'version': "optimal",
                        'symmetry_breaking': parameters,
                        'optimal': solve_time < 300,
                        'stop_reason': result["stop_reason"],
                        'obj': result["obj"],
                        'sol': result["solution"]
                    }
                    save_results(n, json_result, symmetry_level=params, mode="optimize", result_with_meta=result_with_meta)
                else:
                    json_result = {
                        'time': 300,
                        'method': method,
                        'version': "optimal",
                        'symmetry_breaking': parameters,
                        'optimal': False,
                        'stop_reason': result["stop_reason"],
                        'obj': result["obj"],
                        'sol': []
                    }
                    save_results(n, json_result, symmetry_level=params, mode="optimize", result_with_meta=result_with_meta)
        else:
            flag_names = ["ic_match", "ic_teams", "sb_periods"]
            all_flag_combos = list(itertools.product([False, True], repeat=len(flag_names)))
            total_combos = len(all_flag_combos)
            
            result_with_meta = {}
            for i, combo in enumerate(all_flag_combos, start=1):
                print(f"Solving team={n}, combo={i}/{total_combos}")
                flags_combo = dict(zip(flag_names, combo))
                params = {
                        "ic_match":False,
                        "ic_teams":False,
                        "sb_periods":False
                        }
                if flags_combo["ic_match"]:
                    params["ic_match"] = True
                if flags_combo["ic_teams"]:
                    params["ic_teams"] = True
                if flags_combo["sb_periods"]:
                    params["sb_periods"] = True
                parameters = ""
                if params["ic_match"]:
                    parameters += "ic_match"
                if params["ic_teams"]:
                    parameters += " ic_teams"
                if params["sb_periods"]:
                    parameters += " sb_periods"
                if parameters == "":
                    parameters = "None"
                parameters.strip()

                if run_decision:
                    result = solver.solve_sts(constraints=params)
                    if result["satisfiable"]:   
                        json_result = {
                            'time': int(round(result["time"])),
                            'method': method,
                            'version': "decision",
                            'symmetry_breaking': parameters,
                            'optimal': solve_time < 300,
                            'stop_reason': result["stop_reason"],
                            'obj': "None",
                            'sol': result["solution"]
                        }
                        save_results(n, json_result, symmetry_level=params, mode="feasible", result_with_meta=result_with_meta)
                    else:
                        json_result = {
                            'time': 300,
                            'method': method,
                            'version': "decision",
                            'symmetry_breaking': parameters,
                            'optimal': False,
                            'stop_reason': result["stop_reason"],
                            'obj': "None",
                            'sol': []
                        }
                        save_results(n, json_result, symmetry_level=params, mode="feasible", result_with_meta=result_with_meta)
                if run_optimization:
                    result = solver.solve_sts_optimize(constraints=params)
                    if result['satisfiable']:
                        json_result = {
                            'time': int(round(result["time"])),
                            'method': method,
                            'version': "optimal",
                            'symmetry_breaking': parameters,
                            'optimal': solve_time < 300,
                            'stop_reason': result["stop_reason"],
                            'obj': result["obj"],
                            'sol': result["solution"]
                        }
                        save_results(n, json_result, symmetry_level=params, mode="optimize", result_with_meta=result_with_meta)
                    else:
                        json_result = {
                            'time': 300,
                            'method': method,
                            'version': "optimal",
                            'symmetry_breaking': parameters,
                            'optimal': False,
                            'stop_reason': result["stop_reason"],
                            'obj': result["obj"],
                            'sol': []
                        }
                        save_results(n, json_result, symmetry_level=params, mode="optimize", result_with_meta=result_with_meta)

if __name__ == "__main__":
    main()