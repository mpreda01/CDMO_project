#!/usr/bin/env python3
import json
import os
import sys
import itertools

# Add the source/SMT directory to Python path to import STSModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'SMT'))
from SMTMdl import STSModel


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

def main():
    if len(sys.argv) < 2:
        teams = [2, 4, 6]
        optimize = True
        combinations = False
        flags = {
                "sb_weeks":True,
                "sb_periods": True,
                "sb_teams": True,
                "ic_matches_per_team": True,
                "ic_period_count": True,
            }
    else:
        # Parse arguments
        teams = []
        flags = {
                "sb_weeks":False,
                "sb_periods": False,
                "sb_teams": False,
                "ic_matches_per_team": False,
                "ic_period_count": False,
            }
        optimize = True
        combinations = True
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.lower() == "decision":
                optimize = False
            elif arg.lower() == "all-optional":
                flags = {
                    "sb_weeks":True,
                    "sb_periods": True,
                    "sb_teams": True,
                    "ic_matches_per_team": True,
                    "ic_period_count": True,
                }
                combinations = False
            elif arg.lower() == "no-optional":
                flags = {
                    "sb_weeks":False,
                    "sb_periods": False,
                    "sb_teams": False,
                    "ic_matches_per_team": False,
                    "ic_period_count": False,
                }
                combinations = False
            elif arg.lower() == "sb_weeks":
                flags["sb_weeks"] = True
                combinations = False
            elif arg.lower() == "sb_periods":
                flags["sb_periods"] = True
                combinations = False
            elif arg.lower() == "sb_teams":
                flags["sb_teams"] = True
                combinations = False
            elif arg.lower() == "ic_matches_per_team":
                flags["ic_matches_per_team"] = True
                combinations = False
            elif arg.lower() == "ic_period_count":
                flags["ic_period_count"] = True
                combinations = False
            else:
                try:
                    n = int(arg)
                    if n < 2 or n % 2 != 0:
                        print(f"Error: {n} is not valid (must be even and >= 2)")
                        sys.exit(1)
                    teams.append(n)
                except ValueError:
                    print(f"Error: {arg} is not a valid team size or option")
                    sys.exit(1)
            i += 1
        if not teams:
            teams = [2,4,6]
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'SMT'))
    os.makedirs(output_dir, exist_ok=True)

    if not combinations:
        for n in teams:
            data = {}
            model = STSModel(n, **flags)
            result = model.solve(optimize=optimize)
            result["flags"] = {**flags, "optimize": optimize}

            res_name = "cvc5"
            if result["obj"] == None:
                res_name += "_dec"
            else:
                res_name += "_opt"
            if result["flags"]["sb_weeks"] == True:
                res_name += "_sw1"
            if result["flags"]["sb_periods"] == True:
                res_name += "_sp1"
            if result["flags"]["sb_teams"] == True:
                res_name += "_st1"
            if result["flags"]["ic_matches_per_team"] == True:
                res_name += "_icm"
            if result["flags"]["ic_period_count"] == True:
                res_name += "_icp"

            result["sol"] = str(result["sol"])

            filename = os.path.join(output_dir, f"{n}.json")
            data[res_name] = result
            with open(filename, "w") as f:
                f.write(json.dumps(data, indent=2))
            
            with open(filename, "r") as f:
                lines = f.readlines()

            with open(filename, "w") as f:
                for line in lines:
                    if '"sol": "' in line:
                        # Remove surrounding quotes
                        line = line.replace('\\"', '"') 
                        line = line.replace('"sol": "', '"sol": ').rstrip()
                        if line.endswith('",'):
                            line = line[:-2] + ","  # remove closing quote
                        f.write(line + "\n")
                    else:
                        f.write(line)
            print(f"Results saved in {filename}")
    else:
        flag_names = ["sb_weeks", "sb_periods", "sb_teams", "ic_matches_per_team", "ic_period_count", "optimize"]
        all_flag_combos = list(itertools.product([False, True], repeat=len(flag_names)))
        total_combos = len(all_flag_combos)

        for n in teams:
            data = {}
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

                res_name = "cvc5"
                if result["obj"] == None:
                    res_name += "_dec"
                else:
                    res_name += "_opt"
                if result["flags"]["sb_weeks"] == True:
                    res_name += "_sw1"
                if result["flags"]["sb_periods"] == True:
                    res_name += "_sp1"
                if result["flags"]["sb_teams"] == True:
                    res_name += "_st1"
                if result["flags"]["ic_matches_per_team"] == True:
                    res_name += "_icm"
                if result["flags"]["ic_period_count"] == True:
                    res_name += "_icp"

                result["sol"] = str(result["sol"])

                filename = os.path.join(output_dir, f"{n}.json")
                data[res_name] = result
            with open(filename, "w") as f:
                f.write(json.dumps(data, indent=2))
            
            with open(filename, "r") as f:
                lines = f.readlines()

            with open(filename, "w") as f:
                for line in lines:
                    if '"sol": "' in line:
                        # Remove surrounding quotes
                        line = line.replace('\\"', '"') 
                        line = line.replace('"sol": "', '"sol": ').rstrip()
                        if line.endswith('",'):
                            line = line[:-2] + ","  # remove closing quote
                        f.write(line + "\n")
                    else:
                        f.write(line)

            print(f"All combination saved in {n}.json")

if __name__ == "__main__":
    main()