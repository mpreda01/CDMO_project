#!/usr/bin/env python3
import json
import os
import sys
import itertools
import re

# Add the source/SMT directory to Python path to import STSModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'SMT'))
from STSModel import STSModel


# -------------------------
# Parse circle2.py output
# -------------------------
def parse_circle2_output(output_text):
    """
    Parse the output from circle2.py to extract specific information.
    
    Args:
        output_text (str): The output text from circle2.py
    
    Returns:
        dict: Dictionary containing extracted values for obj, Total time, optimal, and sol
    """
    result = {
        'obj': None,
        'time': None,
        'solve_time': None,
        'optimize_time': None,
        'optimal': None,
        'sol': None
    }
    
    lines = output_text.split('\n')
    
    # Track multi-line sol parsing
    in_sol_block = False
    sol_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip processing if we're in sol block
        if in_sol_block:
            # Continue collecting sol lines until we find an empty line or reach end
            if stripped == '' or (i == len(lines) - 1):
                # End of sol block
                in_sol_block = False
                if stripped:  # Include last line if not empty
                    sol_lines.append(stripped)
                sol_str = ' '.join(sol_lines)
                try:
                    result['sol'] = json.loads(sol_str)
                except json.JSONDecodeError:
                    result['sol'] = sol_str
                sol_lines = []
            else:
                sol_lines.append(stripped)
            continue
        
        # Extract solve_time value
        if stripped.startswith('solve_time:'):
            match = re.search(r'solve_time:\s*(.+)', stripped)
            if match:
                try:
                    result['solve_time'] = float(match.group(1).strip())
                except ValueError:
                    result['solve_time'] = match.group(1).strip()
        
        # Extract optimize_time value
        elif stripped.startswith('optimize_time:'):
            match = re.search(r'optimize_time:\s*(.+)', stripped)
            if match:
                try:
                    result['optimize_time'] = float(match.group(1).strip())
                except ValueError:
                    result['optimize_time'] = match.group(1).strip()
        
        # Extract Total time value (case-insensitive)
        elif 'total time:' in stripped.lower():
            match = re.search(r'[Tt]otal time:\s*(.+)', stripped)
            if match:
                time_str = match.group(1).strip()
                # Remove "seconds" if present
                time_str = time_str.replace('seconds', '').strip()
                try:
                    result['time'] = float(time_str)
                except ValueError:
                    result['time'] = time_str
        
        # Extract optimal value
        elif stripped.startswith('optimal:'):
            match = re.search(r'optimal:\s*(.+)', stripped)
            if match:
                optimal_str = match.group(1).strip().lower()
                result['optimal'] = optimal_str in ['true', 'yes', '1']
        
        # Extract obj value
        elif stripped.startswith('obj:'):
            match = re.search(r'obj:\s*(.+)', stripped)
            if match:
                obj_str = match.group(1).strip()
                if obj_str.lower() == 'none':
                    result['obj'] = 'None'
                else:
                    try:
                        result['obj'] = int(obj_str)
                    except ValueError:
                        result['obj'] = obj_str
        
        # Extract sol value (handle multi-line JSON)
        elif stripped.startswith('sol:'):
            in_sol_block = True
            # Check if sol is on same line
            remainder = stripped[4:].strip()
            if remainder:
                sol_lines.append(remainder)
    
    # Handle case where sol is the last item and we're still in block
    if in_sol_block and sol_lines:
        sol_str = ' '.join(sol_lines)
        try:
            result['sol'] = json.loads(sol_str)
        except json.JSONDecodeError:
            result['sol'] = sol_str
    
    # Ensure sol is always a list, not a string
    if isinstance(result['sol'], str):
        try:
            result['sol'] = json.loads(result['sol'])
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, set to empty list
            result['sol'] = []
    
    # Ensure sol is [] if None
    if result['sol'] is None:
        result['sol'] = []
    
    return result


# -------------------------
# Input/Output JSON
# -------------------------
def load_json_dict(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    return {}

def load_previous_timeouts(selected_n, method="classic"):
    """Load previous timeout configurations from JSON files for selected n values and all smaller n values"""
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'SMT'))
    timed_out_configs = {}
    
    if not os.path.exists(output_dir):
        return timed_out_configs
    
    print("Loading previous timeout data from JSON files...")
    
    # Determine which n values to check (selected n values + all smaller n values for context)
    valid_teams = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    max_n = max(selected_n) if selected_n else 0
    all_n_to_check = sorted(set([n for n in valid_teams if n <= max_n]))
    
    # Check each n value's JSON file
    for n in all_n_to_check:
        filename = os.path.join(output_dir, f"{n}.json")
        
        if not os.path.exists(filename):
            continue
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Check each result for timeouts (time >= 300 or solve_time >= 300)
            for timestamp, result in existing_data.items():
                time_val = result.get('time', 0)
                solve_time_val = result.get('solve_time', 0)
                
                # Check if it's a timeout (300 seconds)
                is_timeout = (time_val >= 300) or (solve_time_val >= 300)
                
                if is_timeout and 'params' in result:
                    params = result['params']
                    is_circle = params.get('circle', False)
                    
                    # Only track if this method matches our current run
                    if (method == "circle" and not is_circle) or (method == "classic" and is_circle):
                        continue
                    
                    # Create params dict without metadata
                    if method == "classic":
                        # For classic mode: sb_weeks, sb_periods, sb_teams, ic_matches_per_team, ic_period_count, optimize, solver
                        param_dict = {k: v for k, v in params.items() if k not in ['circle', 'solver']}
                    else:
                        # For circle mode: sb_fix_period, sb_lex_periods, implied_constraints, optimize, solver
                        param_dict = {k: v for k, v in params.items() if k not in ['circle', 'solver']}
                    
                    params_tuple = tuple(sorted(param_dict.items()))
                    config_key = params_tuple
                    
                    # Store the smallest n that timed out for this config
                    if config_key not in timed_out_configs or n < timed_out_configs[config_key]:
                        timed_out_configs[config_key] = n
                        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {filename}: {e}")
            continue
    
    if timed_out_configs:
        print(f"Found {len(timed_out_configs)} configurations that previously timed out")
    else:
        print("No previous timeout data found")
        
    return timed_out_configs

def config_exists_in_json(n, params):
    """Check if a configuration already exists in the JSON file for given n"""
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'SMT'))
    filename = os.path.join(output_dir, f"{n}.json")
    
    if not os.path.exists(filename):
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Check if any existing entry has the same params
        for timestamp, result in existing_data.items():
            if result.get('params') == params:
                return True
                
    except (json.JSONDecodeError, IOError):
        return False
    
    return False

def write_compact_json_dict(filename, data):
    param_order = ["sb_weeks", "sb_periods", "sb_teams", "ic_matches_per_team", "ic_period_count", "optimize", "circle"]
    with open(filename, "w", encoding="utf-8") as f:
        f.write('{\n')
        timestamps = sorted(data.keys())
        for idx, timestamp in enumerate(timestamps):
            entry = data[timestamp]
            time_v = json.dumps(entry.get("time"))
            solve_time_v = json.dumps(entry.get("solve_time"))
            optimize_time_v = json.dumps(entry.get("optimize_time"))
            optimal_v = json.dumps(entry.get("optimal"))
            obj_v = json.dumps(entry.get("obj"))
            # Handle sol: use [] if None or missing, otherwise use the actual value
            sol_value = entry.get("sol")
            if sol_value is None:
                sol_compact = "[]"
            else:
                sol_compact = json.dumps(sol_value, separators=(",", ":"), ensure_ascii=False)
            params = entry.get("params", {})
            f.write(f'  "{timestamp}": {{\n')
            f.write(f'    "time": {time_v},\n')
            if entry.get("solve_time") is not None:
                f.write(f'    "solve_time": {solve_time_v},\n')
            f.write(f'    "optimize_time": {optimize_time_v},\n')
            f.write(f'    "optimal": {optimal_v},\n')
            f.write(f'    "obj": {obj_v},\n')
            f.write(f'    "sol": {sol_compact},\n')
            f.write('    "params": {\n')
            printed = []
            for k in param_order:
                if k in params:
                    printed.append(k)
            for k in params:
                if k not in printed:
                    printed.append(k)
            for j, k in enumerate(printed):
                v = params[k]
                comma = ',' if j < len(printed)-1 else ''
                f.write(f'      {json.dumps(k)}: {json.dumps(v)}{comma}\n')
            f.write('    }\n')
            f.write('  }' + (',' if idx < len(timestamps)-1 else '') + '\n')
        f.write('}\n')


if __name__ == "__main__":
    valid_teams = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    team_input = input(f"Insert number of teams ({valid_teams}) or 'all': ").strip()

    if team_input.lower() == "all":
        teams = valid_teams
    else:
        parts = [p.strip() for p in team_input.split(",") if p.strip() != ""]
        teams = []
        for p in parts:
            try:
                v = int(p)
                if v in valid_teams:
                    teams.append(v)
            except Exception:
                pass

    if not teams:
        print("No valid number of teams provided. Terminating.")
        raise SystemExit(1)
    
    valid_methods = ["classic", "circle"]
    method_input = input(f"Select method: (classic/circle) ({valid_methods}): ").strip().lower()

    if method_input not in valid_methods:
        print("Invalid method selected. Terminating.")
        raise SystemExit(1)
    

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'SMT'))
    os.makedirs(output_dir, exist_ok=True)

    if method_input == "classic":

        mode = input("Select mode (manual/all): ").strip().lower()
        
        if mode not in ["manual", "all"]:
            print("Unknown mode. Use 'manual' o 'all'.")
            raise SystemExit(1)

        if mode == "manual":
            solver = input("Select solver (z3/cvc5): ").strip().lower()
            if solver not in ["z3", "cvc5"]:
                print("Invalid solver selected. Terminating.")
                raise SystemExit(1)
            
            optimize = input("Optimize solution? (y/n): ").strip().lower().startswith("y")
            
            if solver == "cvc5" and optimize:
                print("Optimization is not available for cvc5 solver. Terminating.")
                raise SystemExit(1)
            
            for n in teams:
                flags = {
                    "sb_weeks": input("Symmetry break weeks? (y/n): ").strip().lower().startswith("y"),
                    "sb_periods": input("Symmetry break periods? (y/n): ").strip().lower().startswith("y"),
                    "sb_teams": input("Symmetry break teams? (y/n): ").strip().lower().startswith("y"),
                    "ic_matches_per_team": input("Implied matches per team? (y/n): ").strip().lower().startswith("y"),
                    "ic_period_count": input("Implied period count? (y/n): ").strip().lower().startswith("y"),
                }

                # Create params dict for checking
                params_with_meta = {**flags, "optimize": optimize, "solver": solver, "circle": False}
                
                # Check if this configuration already exists
                if config_exists_in_json(n, params_with_meta):
                    print(f"Configuration for n={n} already exists in JSON. Skipping.")
                    continue

                model = STSModel(n, **flags)
                result = model.solve(optimize=optimize, solver_choice=solver)
                result["params"] = params_with_meta

                filename = os.path.join(output_dir, f"{n}.json")
                data = load_json_dict(filename)
                import time
                timestamp = str(time.time())
                data[timestamp] = result
                write_compact_json_dict(filename, data)
                print(f"Risultato salvato in {filename}")

        elif mode == "all":
            solver = input("Select solver (z3/cvc5): ").strip().lower()
            if solver not in ["z3", "cvc5"]:
                print("Invalid solver selected. Terminating.")
                raise SystemExit(1)
            
            flag_names = ["sb_weeks", "sb_periods", "sb_teams", "ic_matches_per_team", "ic_period_count", "optimize"]
            all_flag_combos = list(itertools.product([False, True], repeat=len(flag_names)))
            
            # Filter out combinations with optimize=True if solver is cvc5
            if solver == "cvc5":
                all_flag_combos = [combo for combo in all_flag_combos if not combo[-1]]  # optimize is last
                print("Note: Optimization combinations skipped for cvc5 solver.")
            
            # Load previous timeout data
            teams_sorted = sorted(teams)
            timed_out_configs = load_previous_timeouts(teams_sorted, method="classic")
            
            total_combos = len(all_flag_combos) * len(teams)
            run_count = 0
            skipped_count = 0
            timeout_skipped_count = 0
            
            print(f"\nRunning {total_combos} total combinations...")
            print(f"Note: Configurations that previously timed out (for same or smaller n) will be skipped\n")

            for n in teams_sorted:
                for combo in all_flag_combos:
                    run_count += 1
                    flags_combo = dict(zip(flag_names, combo))
                    
                    # Create params dict for checking
                    params_with_meta = {**flags_combo, "solver": solver, "circle": False}
                    
                    # Create hashable key for timeout checking (without solver)
                    params_tuple = tuple(sorted(flags_combo.items()))
                    config_key = params_tuple
                    
                    # Check if this configuration timed out for a smaller n
                    if config_key in timed_out_configs:
                        timeout_n = timed_out_configs[config_key]
                        if n >= timeout_n:
                            timeout_skipped_count += 1
                            print(f"[{run_count}/{total_combos}] n={n}, solver={solver} - SKIPPED (timed out at n={timeout_n})")
                            continue
                    
                    # Check if this configuration already exists
                    if config_exists_in_json(n, params_with_meta):
                        skipped_count += 1
                        print(f"[{run_count}/{total_combos}] n={n}, solver={solver} - SKIPPED (already exists)")
                        continue
                    
                    print(f"[{run_count}/{total_combos}] n={n}, solver={solver}, optimize={flags_combo['optimize']}")
                    model = STSModel(n,
                                    sb_weeks=flags_combo["sb_weeks"],
                                    sb_periods=flags_combo["sb_periods"],
                                    sb_teams=flags_combo["sb_teams"],
                                    ic_matches_per_team=flags_combo["ic_matches_per_team"],
                                    ic_period_count=flags_combo["ic_period_count"])
                    result = model.solve(optimize=flags_combo["optimize"], solver_choice=solver)
                    result["params"] = params_with_meta

                    filename = os.path.join(output_dir, f"{n}.json")
                    data = load_json_dict(filename)
                    import time
                    timestamp = str(time.time())
                    data[timestamp] = result
                    write_compact_json_dict(filename, data)
                    
                    # Check if this run timed out and add to tracking
                    time_val = result.get('time', 0)
                    solve_time_val = result.get('solve_time', 0)
                    if (time_val >= 300) or (solve_time_val >= 300):
                        timed_out_configs[config_key] = n
                        print(f"⏱ TIMEOUT - Reached 300s limit (will skip for n>{n})")
                    else:
                        print(f"✓ SUCCESS - Time: {time_val:.2f}s")

            if skipped_count > 0:
                print(f"\n⚠ Skipped {skipped_count} configurations that already exist in JSON files")
            if timeout_skipped_count > 0:
                print(f"⏱ Skipped {timeout_skipped_count} configurations due to previous timeouts on smaller n values")
            print(f"\nAll combinations saved in respective JSON files")

        else:
            print("Unknown mode. Use 'manual' o 'all'.")
    
    if method_input == "circle":

        # Default parameters
        sb_fix_period = True
        sb_lex_periods = False
        implied_constraints = True
        
        params = [
            sb_fix_period,
            sb_lex_periods,
            implied_constraints
        ]
        
        param_names = [
            "sb_fix_period",
            "sb_lex_periods",
            "implied_constraints"
        ]

        mode = input("Select mode (manual/all): ").strip().lower()

        if mode not in ["manual", "all"]:
            print("Unknown mode. Use 'manual' o 'all'.")
            raise SystemExit(1)
        
        if mode == "manual":
            solver = input("Select solver (z3/cvc5): ").strip().lower()
            print("Optimization is not available for cvc5 solver.")
            optimize = input("Optimize solution? (y/n): ").strip().lower().startswith("y")

            if solver not in ["z3", "cvc5"]:
                print("Invalid solver selected. Terminating.")
                raise SystemExit(1)
            if solver == "cvc5" and optimize:
                print("Optimization is not available for cvc5 solver. Terminating.")
                raise SystemExit(1)
            
            # Ask for parameters setting
            for i, param_name in enumerate(param_names):
                user_input = input(f"Set parameter '{param_name}'? (y/n): ").strip().lower()
                params[i] = user_input.startswith("y")

            for n in teams:
                # Create params dict for checking
                flags_dict = {param_names[i]: params[i] for i in range(len(params))}
                flags_dict["optimize"] = optimize
                flags_dict["solver"] = solver
                flags_dict["circle"] = True
                
                # Check if this configuration already exists
                if config_exists_in_json(n, flags_dict):
                    print(f"Configuration for n={n} already exists in JSON. Skipping.")
                    continue
                
                # Build params string for command line
                params_str = ", ".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
                print(f"[{params_str}]")

                mode_arg = "opt" if optimize else "sat"
                # Quote the params string to prevent shell from interpreting brackets/spaces
                cmd = f'python "{os.path.join(os.path.dirname(__file__), "..", "..", "source", "SMT", "circle2.py")}" {n} {mode_arg} {solver} "[{params_str}]"'

                print(f"Running command: {cmd}")
                # Use subprocess to capture both stdout and stderr
                import subprocess
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout
                if result.stderr:
                    print("STDERR:", result.stderr)
                if not output.strip():
                    print("WARNING: No output received from circle2.py")
                    print(f"Return code: {result.returncode}")
                print("OUTPUT START")
                print(output)
                print("OUTPUT END")
                r = parse_circle2_output(output)
                
                # Add params to result
                r["params"] = flags_dict
                
                filename = os.path.join(output_dir, f"{n}.json")
                data = load_json_dict(filename)
                import time
                timestamp = str(time.time())
                data[timestamp] = r
                write_compact_json_dict(filename, data)
                print(f"Result saved in {filename}")
        
        elif mode == "all":
            solver = input("Select solver (z3/cvc5): ").strip().lower()
            if solver not in ["z3", "cvc5"]:
                print("Invalid solver selected. Terminating.")
                raise SystemExit(1)
            
            optimize = input("Optimize solution? (y/n): ").strip().lower().startswith("y")
            if solver == "cvc5" and optimize:
                print("Optimization is not available for cvc5 solver. Terminating.")
                raise SystemExit(1)
            
            # Generate all combinations of circle parameters
            circle_param_combos = list(itertools.product([False, True], repeat=len(param_names)))
            
            # Load previous timeout data
            teams_sorted = sorted(teams)
            timed_out_configs = load_previous_timeouts(teams_sorted, method="circle")
            
            total_combos = len(circle_param_combos) * len(teams)
            run_count = 0
            skipped_count = 0
            timeout_skipped_count = 0
            
            print(f"\\nRunning {total_combos} total combinations...")
            print(f"Note: Configurations that previously timed out (for same or smaller n) will be skipped\\n")
            
            for n in teams_sorted:
                for combo in circle_param_combos:
                    run_count += 1
                    params_dict = dict(zip(param_names, combo))
                    
                    # Create params dict for checking
                    params_with_meta = {**params_dict, "optimize": optimize, "solver": solver, "circle": True}
                    
                    # Create hashable key for timeout checking (without solver)
                    params_for_key = {**params_dict, "optimize": optimize}
                    params_tuple = tuple(sorted(params_for_key.items()))
                    config_key = params_tuple
                    
                    # Check if this configuration timed out for a smaller n
                    if config_key in timed_out_configs:
                        timeout_n = timed_out_configs[config_key]
                        if n >= timeout_n:
                            timeout_skipped_count += 1
                            print(f"[{run_count}/{total_combos}] n={n}, solver={solver} - SKIPPED (timed out at n={timeout_n})")
                            continue
                    
                    # Check if this configuration already exists
                    if config_exists_in_json(n, params_with_meta):
                        skipped_count += 1
                        print(f"[{run_count}/{total_combos}] n={n}, solver={solver} - SKIPPED (already exists)")
                        continue
                    
                    print(f"[{run_count}/{total_combos}] n={n}, solver={solver}, optimize={optimize}")
                    
                    # Build params string for command line
                    params_str = ", ".join([f"{param_names[i]}={combo[i]}" for i in range(len(param_names))])
                    
                    mode_arg = "opt" if optimize else "sat"
                    cmd = f'python "{os.path.join(os.path.dirname(__file__), "..", "..", "source", "SMT", "circle2.py")}" {n} {mode_arg} {solver} "[{params_str}]"'
                    
                    import subprocess
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    output = result.stdout
                    
                    r = parse_circle2_output(output)
                    r["params"] = params_with_meta
                    
                    filename = os.path.join(output_dir, f"{n}.json")
                    data = load_json_dict(filename)
                    import time
                    timestamp = str(time.time())
                    data[timestamp] = r
                    write_compact_json_dict(filename, data)
                    
                    # Check if this run timed out and add to tracking
                    time_val = r.get('time', 0)
                    solve_time_val = r.get('solve_time', 0)
                    if (time_val >= 300) or (solve_time_val >= 300):
                        timed_out_configs[config_key] = n
                        print(f"⏱ TIMEOUT - Reached 300s limit (will skip for n>{n})")
                    else:
                        print(f"✓ SUCCESS - Time: {time_val:.2f}s")
            
            if skipped_count > 0:
                print(f"\\n⚠ Skipped {skipped_count} configurations that already exist in JSON files")
            if timeout_skipped_count > 0:
                print(f"⏱ Skipped {timeout_skipped_count} configurations due to previous timeouts on smaller n values")
            print(f"\\nAll combinations saved in respective JSON files")
                


        

