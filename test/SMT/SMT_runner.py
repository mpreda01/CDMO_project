#!/usr/bin/env python3
import json
import os
import sys
import itertools

# Add the source/SMT directory to Python path to import STSModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'SMT'))
from SMTMdl import STSModel


# Input/Output JSON
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
    """Main function to run the SMT solver."""
    if len(sys.argv) < 2:
        team_sizes = [2, 4, 6]
        params = {
                "sb_weeks":True,
                "sb_periods": True,
                "sb_teams": True,
                "ic_matches_per_team": True,
                "ic_period_count": True,
            }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True ## CAPIRE PERCHE ERA FALSE IN ORIGINE
        time_limit = 300
    else:
        # Parse arguments
        team_sizes = []
        params = {
                "sb_weeks":False,
                "sb_periods": False,
                "sb_teams": False,
                "ic_matches_per_team": False,
                "ic_period_count": False,
            }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        time_limit = 300
        i = 1
        args_input = []
        while i < len(sys.argv):
            arg = sys.argv[i]
            args_input.append(arg.lower())
            if arg.lower() == "--decision-only":
                run_optimization = False
            elif arg.lower() == '--optimization-only':
                run_decision = False
            elif arg.lower() == "--all-optional":
                params = {
                    "sb_weeks":True,
                    "sb_periods": True,
                    "sb_teams": True,
                    "ic_matches_per_team": True,
                    "ic_period_count": True,
                }
                symmetry_combinations = False
            elif arg.lower() == "--no-optional":
                params = {
                    "sb_weeks":False,
                    "sb_periods": False,
                    "sb_teams": False,
                    "ic_matches_per_team": False,
                    "ic_period_count": False,
                }
                symmetry_combinations = False
            elif arg.lower() == "--no-combinations":
                symmetry_combinations = False
                params = {
                    "sb_weeks":False,
                    "sb_periods": False,
                    "sb_teams": False,
                    "ic_matches_per_team": False,
                    "ic_period_count": False,
                }
                k = i+1
                for j in range(k, len(sys.argv)):
                    i = j
                    symmetry = sys.argv[j]
                    valid_symmetry = ["sb_weeks", "sb_periods", "sb_teams", "ic_matches_per_team", "ic_period_count"]
                    if symmetry.lower() in ["--decision-only","--optimization-only","--time-limit","--no-combinations","--no-optional","--all-optional"]:
                        i -= 1
                        break
                    else:
                        if symmetry.lower() not in valid_symmetry:
                            print(f"Error: {symmetry} is not a valid symmetry breaking. Valid symmetry breaking: {', '.join(valid_symmetry)}")
                            sys.exit(1)
                        else:
                            if symmetry.lower() == "sb_weeks":
                                params["sb_weeks"] = True
                            elif symmetry.lower() == "sb_periods":
                                params["sb_periods"] = True
                            elif symmetry.lower() == "sb_teams":
                                params["sb_teams"] = True
                            elif symmetry.lower() == "ic_matches_per_team":
                                params["ic_matches_per_team"] = True
                            else:
                                params["ic_period_count"] = True
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
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'SMT'))
    os.makedirs(output_dir, exist_ok=True)

    if not symmetry_combinations:
        for n in team_sizes:
            if run_optimization:
                data = {}
                model = STSModel(n, **params)
                result = model.solve(time_limit = time_limit, optimize=True)
                result["params"] = {**params, "optimize": True}

                res_name = "cvc5"
                if result["obj"] == None:
                    res_name += "_dec"
                else:
                    res_name += "_opt"
                if result["params"]["sb_weeks"] == True:
                    res_name += "_sw1"
                if result["params"]["sb_periods"] == True:
                    res_name += "_sp1"
                if result["params"]["sb_teams"] == True:
                    res_name += "_st1"
                if result["params"]["ic_matches_per_team"] == True:
                    res_name += "_icm"
                if result["params"]["ic_period_count"] == True:
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
            if run_decision:
                data = {}
                model = STSModel(n, **params)
                result = model.solve(time_limit = time_limit, optimize=False)
                result["params"] = {**params, "optimize": False}

                res_name = "cvc5"
                if result["obj"] == None:
                    res_name += "_dec"
                else:
                    res_name += "_opt"
                if result["params"]["sb_weeks"] == True:
                    res_name += "_sw1"
                if result["params"]["sb_periods"] == True:
                    res_name += "_sp1"
                if result["params"]["sb_teams"] == True:
                    res_name += "_st1"
                if result["params"]["ic_matches_per_team"] == True:
                    res_name += "_icm"
                if result["params"]["ic_period_count"] == True:
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
        flag_names = ["sb_weeks", "sb_periods", "sb_teams", "ic_matches_per_team", "ic_period_count"]
        all_flag_combos = list(itertools.product([False, True], repeat=len(flag_names)))
        total_combos = len(all_flag_combos)

        for n in team_sizes:
            if run_optimization:
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
                    result = model.solve(time_limit=time_limit, optimize=True)
                    result["params"] = flags_combo

                    res_name = "cvc5"
                    if dec:
                        res_name += "_dec"
                    else:
                        res_name += "_opt"
                    if result["params"]["sb_weeks"] == True:
                        res_name += "_sw1"
                    if result["params"]["sb_periods"] == True:
                        res_name += "_sp1"
                    if result["params"]["sb_teams"] == True:
                        res_name += "_st1"
                    if result["params"]["ic_matches_per_team"] == True:
                        res_name += "_icm"
                    if result["params"]["ic_period_count"] == True:
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
            if run_decision:
                dec = True
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
                    result = model.solve(time_limit=time_limit, optimize=False)
                    result["params"] = flags_combo

                    res_name = "cvc5"
                    if dec:
                        res_name += "_dec"
                    else:
                        res_name += "_opt"
                    if result["params"]["sb_weeks"] == True:
                        res_name += "_sw1"
                    if result["params"]["sb_periods"] == True:
                        res_name += "_sp1"
                    if result["params"]["sb_teams"] == True:
                        res_name += "_st1"
                    if result["params"]["ic_matches_per_team"] == True:
                        res_name += "_icm"
                    if result["params"]["ic_period_count"] == True:
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