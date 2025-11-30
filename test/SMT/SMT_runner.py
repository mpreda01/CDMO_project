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
        'optimal': None,
        'sol': None
    }
    
    lines = output_text.split('\n')
    
    # Track multi-line sol parsing
    in_sol_block = False
    sol_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Extract obj value
        if stripped.startswith('obj:'):
            match = re.search(r'obj:\s*(.+)', stripped)
            if match:
                try:
                    result['obj'] = int(match.group(1).strip())
                except ValueError:
                    result['obj'] = match.group(1).strip()
        
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
        
        # Extract sol value (handle multi-line JSON)
        elif stripped.startswith('sol:'):
            in_sol_block = True
            # Check if sol is on same line
            remainder = stripped[4:].strip()
            if remainder:
                sol_lines.append(remainder)
        elif in_sol_block:
            # Continue collecting sol lines until we hit the next label or empty meaningful line
            if stripped and not any(stripped.startswith(label) for label in ['obj:', 'Total time:', 'optimal:', 'Using', 'Running', 'Z3', 'Satisfiability']):
                sol_lines.append(stripped)
            elif stripped.startswith('optimal:'):
                # We've reached the next label, parse sol
                in_sol_block = False
                sol_str = ' '.join(sol_lines)
                try:
                    result['sol'] = json.loads(sol_str)
                except json.JSONDecodeError:
                    result['sol'] = sol_str
                sol_lines = []
                # Also process the optimal line
                match = re.search(r'optimal:\s*(.+)', stripped)
                if match:
                    optimal_str = match.group(1).strip().lower()
                    result['optimal'] = optimal_str in ['true', 'yes', '1']
    
    # Handle case where sol is the last item
    if in_sol_block and sol_lines:
        sol_str = ' '.join(sol_lines)
        try:
            result['sol'] = json.loads(sol_str)
        except json.JSONDecodeError:
            result['sol'] = sol_str
    
    return result


# -------------------------
# Input/Output JSON
# -------------------------
def load_json_list(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
    return []

def write_compact_json_list(filename, data):
    flag_order = ["sb_weeks", "sb_periods", "sb_teams", "ic_matches_per_team", "ic_period_count", "optimize"]
    with open(filename, "w", encoding="utf-8") as f:
        f.write('[\n')
        for i, entry in enumerate(data):
            time_v = json.dumps(entry.get("time"))
            optimal_v = json.dumps(entry.get("optimal"))
            obj_v = json.dumps(entry.get("obj"))
            sol_compact = json.dumps(entry["sol"], separators=(",", ":"), ensure_ascii=False) if entry.get("sol") else "null"
            flags = entry.get("flags", {})
            f.write('  {\n')
            f.write(f'    "time": {time_v},\n')
            f.write(f'    "optimal": {optimal_v},\n')
            f.write(f'    "obj": {obj_v},\n')
            f.write(f'    "sol": {sol_compact},\n')
            f.write('    "flags": {\n')
            printed = []
            for k in flag_order:
                if k in flags:
                    printed.append(k)
            for k in flags:
                if k not in printed:
                    printed.append(k)
            for j, k in enumerate(printed):
                v = flags[k]
                comma = ',' if j < len(printed)-1 else ''
                f.write(f'      {json.dumps(k)}: {json.dumps(v)}{comma}\n')
            f.write('    }\n')
            f.write('  }' + (',' if i < len(data)-1 else '') + '\n')
        f.write(']\n')


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
            for n in teams:
                optimize = input("Optimize solution? (y/n): ").strip().lower().startswith("y")
                flags = {
                    "sb_weeks": input("Symmetry break weeks? (y/n): ").strip().lower().startswith("y"),
                    "sb_periods": input("Symmetry break periods? (y/n): ").strip().lower().startswith("y"),
                    "sb_teams": input("Symmetry break teams? (y/n): ").strip().lower().startswith("y"),
                    "ic_matches_per_team": input("Implied matches per team? (y/n): ").strip().lower().startswith("y"),
                    "ic_period_count": input("Implied period count? (y/n): ").strip().lower().startswith("y"),
                }

                model = STSModel(n, **flags)
                result = model.solve(optimize=optimize)
                result["flags"] = {**flags, "optimize": optimize}

                filename = os.path.join(output_dir, f"{n}.json")
                data = load_json_list(filename)
                data.append(result)
                write_compact_json_list(filename, data)
                print(f"Risultato salvato in {filename}")

        elif mode == "all":
            flag_names = ["sb_weeks", "sb_periods", "sb_teams", "ic_matches_per_team", "ic_period_count", "optimize"]
            all_flag_combos = list(itertools.product([False, True], repeat=len(flag_names)))
            total_combos = len(all_flag_combos)

            for n in teams:
                for i, combo in enumerate(all_flag_combos, start=1):
                    print(f"Solving team={n}, combo={i}/{total_combos}")
                    flags_combo = dict(zip(flag_names, combo))
                    model = STSModel(n,
                                    sb_weeks=flags_combo["sb_weeks"],
                                    sb_periods=flags_combo["sb_periods"],
                                    sb_teams=flags_combo["sb_teams"],
                                    ic_matches_per_team=flags_combo["ic_matches_per_team"],
                                    ic_period_count=flags_combo["ic_period_count"])
                    result = model.solve(optimize=flags_combo["optimize"])
                    result["flags"] = flags_combo

                    filename = os.path.join(output_dir, f"{n}.json")
                    data = load_json_list(filename)
                    data.append(result)
                    write_compact_json_list(filename, data)

                print(f"All combination saved in {n}.json")

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
                # Build params string for command line
                params_str = ", ".join([f"{param_names[i]}={params[i]}" for i in range(len(params))])
                print(f"[{params_str}]")

                cmd = f'python "{os.path.join(os.path.dirname(__file__), "..", "..", "source", "SMT", "circle2.py")}" {n} sat {solver} [{params_str}]'

                print(f"Running command: {cmd}")
                output = os.popen(cmd).read()
                print(output)
                r = parse_circle2_output(output)
                print(r)
                


        

