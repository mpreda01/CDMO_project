"""
Plot results from JSON files in res folder
Creates plots for decision and optimization versions with different solver/symmetry combinations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def load_results(results_dir):
    """Load all JSON results from the results directory"""
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        try:
            n = int(json_file.stem)
            if n >= 6:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[n] = data
        except ValueError:
            continue
    
    return results

def load_cp_results(results_dir):
    """Load CP JSON results with specific filtering"""
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        try:
            n = int(json_file.stem)
            if n >= 6:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # Filter CP data based on required params
                    filtered_data = filter_cp_data(data)
                    if filtered_data:
                        results[n] = filtered_data
        except ValueError:
            continue
    
    return results

def filter_cp_data(data):
    """Filter CP data based on required parameters"""
    filtered = {}
    
    for key, entry in data.items():
        if 'params' in entry:
            params = entry['params']
            # Check required conditions
            if (params.get('ic_matches_per_team') == True and
                params.get('ic_period_count') == True and
                params.get('use_int_search') == True and
                params.get('use_restart_luby') == True and
                params.get('use_relax_and_reconstruct') == True):
                
                # Create identifier from relevant params
                sb_weeks = params.get('sb_weeks', False)
                sb_periods = params.get('sb_periods', False)
                sb_teams = params.get('sb_teams', False)
                chuffed = params.get('chuffed', False)
                
                # Create combination key
                combo_key = create_cp_combination_key(sb_weeks, sb_periods, sb_teams, chuffed)
                
                # Store relevant data
                filtered[combo_key] = {
                    'time': entry.get('time', 0),
                    'optimal': entry.get('optimal', False),
                    'obj': entry.get('obj', None),
                    'version': 'optimization',  # CP is always optimization-like
                    'sb_weeks': sb_weeks,
                    'sb_periods': sb_periods,
                    'sb_teams': sb_teams,
                    'chuffed': chuffed
                }
    
    return filtered

def create_cp_combination_key(sb_weeks, sb_periods, sb_teams, chuffed):
    """Create a combination key for CP data"""
    solver = 'chuffed' if chuffed else 'gecode'
    symmetries = []
    if sb_weeks:
        symmetries.append('sbw')
    if sb_periods:
        symmetries.append('sbp')
    if sb_teams:
        symmetries.append('sbt')
    
    if symmetries:
        return f"{solver}_{'_'.join(symmetries)}"
    else:
        return f"{solver}_none"

def extract_cp_combinations(results):
    """Extract all unique CP solver and symmetry breaking combinations"""
    combinations = set()
    
    for n, data in results.items():
        for key in data.keys():
            combinations.add(key)
    
    return sorted(combinations)

def parse_cp_combination_key(key):
    """Parse CP combination key to extract solver and symmetry info"""
    parts = key.split('_')
    solver = parts[0]  # chuffed or gecode
    
    # Extract symmetry breaking info
    symmetry = []
    if 'sbw' in parts:
        symmetry.append('sbw')
    if 'sbp' in parts:
        symmetry.append('sbp')
    if 'sbt' in parts:
        symmetry.append('sbt')
    
    if not symmetry or 'none' in parts:
        symmetry_str = 'None'
    else:
        symmetry_str = '+'.join(symmetry)
    
    return solver, 'optimization', symmetry_str

def extract_solver_combinations(results):
    """Extract all unique solver and symmetry breaking combinations"""
    combinations = set()
    
    for n, data in results.items():
        if isinstance(data, dict):
            # MIP format: dictionary with keys
            for key in data.keys():
                combinations.add(key)
        elif isinstance(data, list):
            # SMT/SAT format: list of entries with flags
            for entry in data:
                if 'flags' in entry:
                    key = create_smt_combination_key(entry['flags'])
                    combinations.add(key)
    
    return sorted(combinations)

def create_smt_combination_key(flags):
    """Create a combination key for SMT/SAT data based on flags"""
    # Determine solver type (assuming SMT for now, can be extended)
    solver = "cvc5"  # Default SMT solver
    
    # Determine version
    version = "opt" if flags.get('optimize', False) else "dec"
    
    # Extract symmetry breaking info
    symmetries = []
    if flags.get('sb_weeks', False):
        symmetries.append('sbw')
    if flags.get('sb_periods', False):
        symmetries.append('sbp')
    if flags.get('sb_teams', False):
        symmetries.append('sbt')
    
    # Combine all parts
    parts = [solver, version]
    if symmetries:
        parts.extend(symmetries)
    if not symmetries:
        parts.append('none')
    

    return '_'.join(parts)

def load_sat_results(results_dir):
    """Load SAT JSON results - special handling for 6 individual configurations"""
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        try:
            n = int(json_file.stem)
            if n >= 6:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # SAT format: dictionary with solver configurations as keys
                    converted_data = {}
                    if isinstance(data, dict):
                        for config_key, entry in data.items():
                            # Each config_key is a different SAT configuration
                            converted_data[config_key] = {
                                'time': entry.get('time', 0),
                                'optimal': entry.get('optimal', False),
                                'obj': entry.get('obj', None),
                                'version': 'decision'
                            }
                        results[n] = converted_data
        except ValueError:
            continue
    
    return results

def extract_sat_combinations(results):
    """Extract all unique SAT solver configurations"""
    combinations = set()
    
    for n, data in results.items():
        for key in data.keys():
            combinations.add(key)
    
    return sorted(combinations)

def parse_sat_combination_key(key):
    """Parse SAT combination key - each key is treated as a unique configuration"""
    # For SAT, we treat each key as a unique solver configuration
    solver = "z3"  # Default SAT solver
    version = 'optimization' if 'opt' in key else 'decision'  # SAT is decision-based
    symmetry_str = key.replace("solver_", "")

    return solver, version, symmetry_str

def get_sat_color_map():
    """Get a unique color for each of the 3 SAT configurations"""
    colors = [
        '#1f77b4',  # blue
        '#2ca02c',  # green
        '#d62728',  # red
    ]
    return colors

def get_smt_color_map():
    """Get a unique color for each of the 8 SAT configurations"""
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
    ]
    return colors


def load_smt_results(results_dir):
    """Load SMT JSON results and convert to comparable format"""
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        try:
            n = int(json_file.stem)
            if n >= 6:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # Convert list format to dictionary format
                    converted_data = {}
                    if isinstance(data, list):
                        for entry in data:
                            if 'flags' in entry:
                                flags = entry['flags']
                                # Filter: only entries with both ic_matches_per_team and ic_period_count true
                                if (flags.get('ic_matches_per_team') == True and 
                                    flags.get('ic_period_count') == True):
                                    key = create_smt_combination_key(entry['flags'])
                                    converted_data[key] = {
                                        'time': entry.get('time', 0),
                                        'optimal': entry.get('optimal', False),
                                        'obj': entry.get('obj', None),
                                        'version': 'optimization' if entry['flags'].get('optimize', False) else 'decision'
                                    }
                        if converted_data:  # Only add if we have filtered data
                            results[n] = converted_data
        except ValueError:
            continue
    return results

def parse_smt_combination_key(key):
    """Parse SMT/SAT combination key to extract solver and configuration info"""
    parts = key.split('_')
    solver = parts[0]  # cvc5, z3, etc.
    version = 'decision' if 'dec' in parts else 'optimization'
    
    # Extract symmetry breaking info
    symmetry = []
    if 'sbw' in parts:
        symmetry.append('sbw')
    if 'sbp' in parts:
        symmetry.append('sbp')
    if 'sbt' in parts:
        symmetry.append('sbt')
    
    # Extract IC flags
    ic_info = []
    if 'icm' in parts:
        ic_info.append('ic_matches')
    if 'icp' in parts:
        ic_info.append('ic_periods')
    
    # Combine symmetry and IC info
    config_parts = symmetry + ic_info
    if not config_parts or 'none' in parts:
        symmetry_str = 'None'
    else:
        symmetry_str = '+'.join(config_parts)
    
    return solver, version, symmetry_str

def parse_combination_key(key):
    """Parse combination key to extract solver, version, and symmetry info"""
    parts = key.split('_')
    solver = parts[0]
    version = 'decision' if 'dec' in parts else 'optimization'
    
    # Extract symmetry breaking info
    symmetry = []
    if 'st1' in parts:
        symmetry.append('st1')
    if 'sw1' in parts:
        symmetry.append('sw1')
    
    if not symmetry:
        symmetry_str = 'None'
    else:
        symmetry_str = '+'.join(symmetry)
    
    return solver, version, symmetry_str

def get_line_style(symmetry_str):
    """Get line style based on symmetry breaking combination"""
    style_map = {
        'None': '-',
        'st1': '--',
        'sw1': '-.',
        'st1+sw1': ':',
        'sbw': '--',
        'sbp': '-.',
        'sbt': ':',
        'sbw+sbp': '--',
        'sbw+sbt': '-.',
        'sbp+sbt': ':',
        'sbw+sbp+sbt': '-'
    }
    return style_map.get(symmetry_str, '-')

def get_color_map():
    """Get color map for different solvers"""
    return {
        'cbc': 'blue',
        'highs': 'red',
        'cplex': 'green',
        'gurobi': 'orange',
        'chuffed': 'blue',
        'gecode': 'red'
    }

def create_plots(results_dir, output_dir=None, solver_type="MIP"):
    """Create plots for decision and optimization versions"""
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "plots" / solver_type
    else:
        output_dir = Path(output_dir) / solver_type
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results based on solver type
    if solver_type == "CP":
        results = load_cp_results(results_dir)
        if not results:
            print("No CP results found for n 6-10 with required parameters")
            return
        
        print(f"Loaded CP results for team sizes: {sorted(results.keys())}")
        
        # Get all combinations
        combinations = extract_cp_combinations(results)
        
        # For CP, we only have optimization-like results
        decision_combinations = {}
        optimization_combinations = {}
        
        for combo in combinations:
            solver, version, symmetry = parse_cp_combination_key(combo)
            
            if solver not in optimization_combinations:
                optimization_combinations[solver] = []
            optimization_combinations[solver].append((combo, symmetry))
            
    elif solver_type == "SAT":
        # Load SAT results (special format with 6 configurations)
        results = load_sat_results(results_dir)
        if not results:
            print("No SAT results found for n >= 6")
            return
        
        print(f"Loaded SAT results for team sizes: {sorted(results.keys())}")
        
        # Get all combinations
        combinations = extract_sat_combinations(results)
        # For SAT, treat all configurations as decision problems
        decision_combinations = {"z3": []}
        optimization_combinations = {"z3":[]}
        
        for combo in combinations:
            solver, version, symmetry = parse_sat_combination_key(combo)
            if version == 'decision':
                decision_combinations[solver].append((combo, symmetry))
            else:
                optimization_combinations[solver].append((combo, symmetry))
            
    elif solver_type == "SMT":
        # Load SMT results (list format)
        results = load_smt_results(results_dir)
        if not results:
            print("No SMT results found for n >= 6")
            return
        
        print(f"Loaded SMT results for team sizes: {sorted(results.keys())}")
        
        # Get all combinations
        combinations = extract_solver_combinations(results)
        
        # Group combinations by solver and version
        decision_combinations = {}
        optimization_combinations = {}
        
        for combo in combinations:
            solver, version, symmetry = parse_smt_combination_key(combo)
            
            if version == 'decision':
                if solver not in decision_combinations:
                    decision_combinations[solver] = []
                decision_combinations[solver].append((combo, symmetry))
            else:
                if solver not in optimization_combinations:
                    optimization_combinations[solver] = []
                optimization_combinations[solver].append((combo, symmetry))
            
    else:  # MIP
        results = load_results(results_dir)
        if not results:
            print("No results found for n >= 6")
            return
        
        print(f"Loaded results for team sizes: {sorted(results.keys())}")
        
        # Get all combinations
        combinations = extract_solver_combinations(results)
        
        # Group combinations by solver and version
        decision_combinations = {}
        optimization_combinations = {}
        
        for combo in combinations:
            solver, version, symmetry = parse_combination_key(combo)
            
            if version == 'decision':
                if solver not in decision_combinations:
                    decision_combinations[solver] = []
                decision_combinations[solver].append((combo, symmetry))
            else:
                if solver not in optimization_combinations:
                    optimization_combinations[solver] = []
                optimization_combinations[solver].append((combo, symmetry))
    
    color_map = get_color_map()
    
    # Create summary comparison plots
    create_summary_plots(results, decision_combinations, optimization_combinations, output_dir, color_map, solver_type)

def create_summary_plots(results, decision_combinations, optimization_combinations, output_dir, color_map, solver_type):
    """Create summary plots comparing all solvers"""
    
    # Only create decision plot for non-CP solver types
    if solver_type != "CP" and decision_combinations:
        # Summary plot for decision version
        plt.figure(figsize=(14, 10))
        plt.title(f'{solver_type} Decision Version - All Solvers Comparison', fontsize=16, fontweight='bold')
        
        for solver, combos in decision_combinations.items():
            if solver_type == "SAT":
                # Special handling for SAT: use unique colors for each of the 6 configurations
                sat_colors = get_sat_color_map()
                for i, (combo, symmetry) in enumerate(combos):
                    team_sizes = []
                    times = []
                    
                    for n in sorted(results.keys()):
                        if combo in results[n]:
                            result = results[n][combo]
                            if result.get('optimal', False):
                                team_sizes.append(n)
                                times.append(result.get('time', 0))
                    
                    if team_sizes:
                        color = sat_colors[i % len(sat_colors)]  # Use unique color for each config
                        plt.plot(team_sizes, times, 
                                color=color,
                                linestyle='-',  # Use solid line for all SAT configs
                                marker='o',
                                linewidth=2,
                                markersize=4,
                                label=f'{symmetry}')
            else:
                # Regular handling for other solver types
                for i, (combo, symmetry) in enumerate(combos):
                    team_sizes = []
                    times = []
                    
                    for n in sorted(results.keys()):
                        if combo in results[n]:
                            result = results[n][combo]
                            if result.get('optimal', False):
                                team_sizes.append(n)
                                times.append(result.get('time', 0))
                    
                    if team_sizes:
                        if solver_type == "SMT":
                            smt_colors = get_smt_color_map()
                            color_smt = smt_colors[i % len(smt_colors)]  # Use unique color for each config
                            plt.plot(team_sizes, times, 
                                    color=color_smt,
                                    linestyle='-',
                                    marker='o',
                                    linewidth=2,
                                    markersize=4,
                                    label=f'{solver}-{symmetry}')
                        else:
                            plt.plot(team_sizes, times, 
                                    color=color_map.get(solver, 'black'),
                                    linestyle=get_line_style(symmetry),
                                    marker='o',
                                    linewidth=2,
                                    markersize=4,
                                    label=f'{solver}-{symmetry}')
        
        plt.xlabel('Number of Teams', fontsize=12)
        plt.ylabel('Time to Solve (seconds)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=14)
        
        # Set axis ranges and ticks based on solver type
        if solver_type == "CP":
            plt.xlim(5.8, 10.2)
            plt.ylim(-1, 300)
            plt.xticks(range(6, 11, 1))  # 6, 7, 8, 9, 10
        else:
            plt.xlim(5.8, 16.2)
            plt.ylim(-1, 300)
            plt.xticks(range(6, 17, 2))  # 6, 8, 10, 12, 14, 16, 18
        plt.yticks(range(0, 301, 20))  # 0, 20, 40, ..., 300
        
        output_file = output_dir / f'{solver_type}_decision.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved decision summary plot: {output_file}")
    
    # Summary plot for optimization version
    plt.figure(figsize=(14, 10))
    plt.title(f'{solver_type} Optimization Version - All Solvers Comparison', fontsize=16, fontweight='bold')
    
    for solver, combos in optimization_combinations.items():
        if solver_type == "SAT":
                # Special handling for SAT: use unique colors for each of the 6 configurations
                sat_colors = get_sat_color_map()
                for i, (combo, symmetry) in enumerate(combos):
                    team_sizes = []
                    times = []
                    
                    for n in sorted(results.keys()):
                        if combo in results[n]:
                            result = results[n][combo]
                            if result.get('optimal', False):
                                team_sizes.append(n)
                                times.append(result.get('time', 0))
                    
                    if team_sizes:
                        color = sat_colors[i % len(sat_colors)]  # Use unique color for each config
                        plt.plot(team_sizes, times, 
                                color=color,
                                linestyle='-',  # Use solid line for all SAT configs
                                marker='o',
                                linewidth=2,
                                markersize=4,
                                label=f'{symmetry}')
        else:
            for i, (combo, symmetry) in enumerate(combos):
                team_sizes = []
                times = []
            
                for n in sorted(results.keys()):
                    if combo in results[n]:
                        result = results[n][combo]
                        if result.get('optimal', False):
                            team_sizes.append(n)
                            times.append(result.get('time', 0))
            
                if team_sizes:
                    if solver_type == "SMT":
                        smt_colors = get_smt_color_map()
                        color_smt = smt_colors[i % len(smt_colors)]  # Use unique color for each config
                        plt.plot(team_sizes, times, 
                                 color=color_smt,
                                 linestyle='-',
                                 marker='o',
                                 linewidth=2,
                                 markersize=4,
                                 label=f'{solver}-{symmetry}')
                    else:
                        plt.plot(team_sizes, times, 
                                 color=color_map.get(solver, 'black'),
                                 linestyle=get_line_style(symmetry),
                                 marker='o',
                                 linewidth=2,
                                 markersize=4,
                                 label=f'{solver}-{symmetry}')
    
    plt.xlabel('Number of Teams', fontsize=12)
    plt.ylabel('Time to Solve (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=14)
    
    # Set axis ranges and ticks
    plt.xlim(5.8, 16.2)
    plt.ylim(-1, 300)
    plt.xticks(range(6, 17, 2))  # 6, 8, 10, 12, 14, 16, 18
    plt.yticks(range(0, 301, 20))  # 0, 20, 40, ..., 300
    
    output_file = output_dir / f'{solver_type}_optimization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved optimization summary plot: {output_file}")

def main():
    """Main function to create all plots"""
    script_dir = Path(__file__).parent
    res_dir = script_dir / "res"
    
    if not res_dir.exists():
        print(f"Results directory not found: {res_dir}")
        return
    
    # Find all subdirectories in res folder (MIP, CP, SAT, SMT)
    solver_types = []
    for subdir in res_dir.iterdir():
        if subdir.is_dir() and any(subdir.glob("*.json")):
            solver_types.append(subdir.name)
    
    if not solver_types:
        print("No solver directories with JSON files found in res folder")
        return
    
    print(f"Found solver types: {', '.join(solver_types)}")
    
    # Create plots for each solver type
    for solver_type in sorted(solver_types):
        results_dir = res_dir / solver_type
        print(f"\nProcessing {solver_type} results from: {results_dir}")
        
        create_plots(results_dir, solver_type=solver_type)
        print(f"Completed plots for {solver_type}")
    
    print("\nAll plots created successfully!")

if __name__ == "__main__":
    main()
