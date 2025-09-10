"""
STS SAT Solver Runner
Main script to run the STS SAT solver with different configurations.
"""

import sys
import json
import os
import time
from typing import Dict, List, Any

# Add source directory to path to import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../source/SAT'))
from sts_sat_model import STSSATSolver


def save_results(n: int, result: Dict[str, Any], approach_name: str, 
                 output_dir: str = "res/SAT", symmetry_level: str = "full", 
                 mode: str = "feasible") -> None:
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

    # Build constraint list
    base_constraints = [
        'weekly_participation: each team plays exactly once per week',
        'period_occupancy: each period has exactly one home and one away team',
        'period_limit: each team plays at most twice in the same period',
        'match_uniqueness: every pair of teams meets exactly once'
    ]

    symmetry_constraints = []
    if symmetry_level == 'basic':
        symmetry_constraints.append('symmetry_basic: fix first match (team 0 home in w0 p0, team1 away)')
    elif symmetry_level == 'moderate':
        symmetry_constraints.append('symmetry_basic: fix first match')
        symmetry_constraints.append('symmetry_week: team 0 meets team i in week i-1 (for i)')
    elif symmetry_level == 'full':
        symmetry_constraints.append('symmetry_basic: fix first match')
        symmetry_constraints.append('symmetry_week: team 0 meets team i in week i-1 (for i)')
        symmetry_constraints.append('symmetry_advanced: additional fixes (periods/home-away ordering/opponent order)')

    all_constraints = base_constraints + symmetry_constraints

    result_with_meta: Dict[str, Any] = {}

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

    result_with_meta['constraints'] = all_constraints

    key = f"{approach_name}_{symmetry_level}"
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

    print(f"Results saved to {output_file} under key '{key}'")


def print_solution(solution: List[List[List[int]]], n_weeks: int, n_periods: int) -> None:
    """
    Print the schedule in a readable format.
    
    Args:
        solution: Schedule in week x period format
        n_weeks: Number of weeks
        n_periods: Number of periods per week
    """
    print("Schedule:")
    for w in range(n_weeks):
        print(f"Week {w + 1}:")
        for p in range(n_periods):
            home_away = solution[w][p]
            if home_away:
                home, away = home_away
                print(f"  Period {p + 1}: Team {home} vs Team {away}")


def main():
    """Main function to run the SAT solver."""
    if len(sys.argv) < 2:
        team_sizes = [2,4,6]
        level = "full"
        mode = "feasible"
    else:
        i = 1
        level = "full"
        mode = "feasible"
        team_sizes = []
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.lower() in ['basic', 'moderate', 'full']:
                level = arg.lower()
            elif arg.lower() in ['feasible','optimize']:
                mode = arg.lower()
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

    for n in team_sizes:
        # Create solver instance
        solver = STSSATSolver(n, timeout=300)
    
        print("\n" + "-" * 50)
        print(f"Solving STS problem for {n} teams")
        print(f"Symmetry breaking level: {level}")
        print(f"Mode: {'Optimization' if mode == 'optimize' else 'Feasibility'}")
        print("-" * 50)
    
        if mode == 'optimize':
            # Run optimization mode
            result_dict = solver.solve_optimization_incremental(symmetry_level=level, output_dir="res/SAT")
            # Don't include level in approach name - save_results will add it
            approach_name = 'z3_solver_opt'
        
            if result_dict['satisfiable']:
                print(f"\nSolution found in {int(round(result_dict['time']))} seconds")
                print(f"Objective value (max imbalance): {result_dict['obj']}")
            
                if result_dict['optimal']:
                    print("Solution is OPTIMAL")
                else:
                    print("Solution may not be optimal (timeout or incomplete search)")
            
                print_solution(result_dict['solution'], solver.n_weeks, solver.n_periods)
            
                json_result = {
                    'time': int(round(result_dict['time'])),
                    'optimal': result_dict['optimal'],
                    'obj': result_dict.get('obj'),
                    'sol': result_dict['solution']
                }
                save_results(n, json_result, approach_name, symmetry_level=level, mode=mode)
            else:
                print("No feasible solution found to optimize.")
        else:
            # Run feasibility mode
            success, solution, solve_time, model = solver.solve_feasibility(symmetry_level=level)
            if success:
                stats = solver.get_solution_stats(model)
                objective_value = solver._calculate_objective(model)
            
                print(f"\nSolution found in {int(round(solve_time))} seconds")
                print(f"Objective value (max imbalance): {objective_value}")
            
                print("\nHome/Away distribution:")
                for team, info in stats['teams'].items():
                    print(f"  Team {team}: {info['home']} home, {info['away']} away (abs diff: {info['abs_diff']})")
            
                if stats['is_balanced']:
                    print("Solution is BALANCED (max abs diff <= 1)")
                else:
                    print(f"Max abs diff: {stats['max_deviation']}")
            
                print_solution(solution, solver.n_weeks, solver.n_periods)
            
                json_result = {
                    'time': int(round(solve_time)),
                    'optimal': solve_time < 300,
                    'obj': objective_value,
                    'sol': solution
                }
                save_results(n, json_result, 'z3_solver', symmetry_level=level, mode=mode)
            else:
                print(f"\nNo solution found within {int(round(solve_time))} seconds")
                json_result = {
                    'time': 300,
                    'optimal': False,
                    'obj': None,
                    'sol': []
                }
                save_results(n, json_result, 'z3_solver', symmetry_level=level, mode=mode)


if __name__ == "__main__":
    main()