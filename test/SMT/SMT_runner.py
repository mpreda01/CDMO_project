#!/usr/bin/env python3
import json
import os
import sys
import itertools

# Add the source/SMT directory to Python path to import STSModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'SMT'))
from STSModel import STSModel


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
    valid_teams = [2, 4, 6, 8, 10, 12, 14, 16, 18]
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

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'SMT'))
    os.makedirs(output_dir, exist_ok=True)

    mode = input("Select mode (manual/all): ").strip().lower()

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
