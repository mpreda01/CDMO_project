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
            if n >= 6:  # Only consider n >= 6
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[n] = data
        except ValueError:
            continue
    
    return results

def extract_solver_combinations(results):
    """Extract all unique solver and symmetry breaking combinations"""
    combinations = set()
    
    for n, data in results.items():
        for key in data.keys():
            combinations.add(key)
    
    return sorted(combinations)

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
        'st1+sw1': ':'
    }
    return style_map.get(symmetry_str, '-')

def get_color_map():
    """Get color map for different solvers"""
    return {
        'cbc': 'blue',
        'highs': 'red',
        'cplex': 'green',
        'gurobi': 'orange'
    }

def create_plots(results_dir, output_dir=None, solver_type="MIP"):
    """Create plots for decision and optimization versions"""
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "plots" / solver_type
    else:
        output_dir = Path(output_dir) / solver_type
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
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
    
    # Summary plot for decision version
    plt.figure(figsize=(14, 10))
    plt.title(f'{solver_type} Decision Version - All Solvers Comparison', fontsize=16, fontweight='bold')
    
    for solver, combos in decision_combinations.items():
        for combo, symmetry in combos:
            team_sizes = []
            times = []
            
            for n in sorted(results.keys()):
                if combo in results[n]:
                    result = results[n][combo]
                    if result.get('optimal', False):
                        team_sizes.append(n)
                        times.append(result.get('time', 0))
            
            if team_sizes:
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
    
    output_file = output_dir / f'{solver_type}_decision.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved decision summary plot: {output_file}")
    
    # Summary plot for optimization version
    plt.figure(figsize=(14, 10))
    plt.title(f'{solver_type} Optimization Version - All Solvers Comparison', fontsize=16, fontweight='bold')
    
    for solver, combos in optimization_combinations.items():
        for combo, symmetry in combos:
            team_sizes = []
            times = []
            
            for n in sorted(results.keys()):
                if combo in results[n]:
                    result = results[n][combo]
                    if result.get('optimal', False):
                        team_sizes.append(n)
                        times.append(result.get('time', 0))
            
            if team_sizes:
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
