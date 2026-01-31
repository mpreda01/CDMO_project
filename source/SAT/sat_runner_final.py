#!/usr/bin/env python3
"""
SAT Solver Runner for STS Problem
Runs all solver/method/constraint combinations and saves results to JSON files.
Includes both decision and optimization versions.
"""

import json
import os
import sys
import math

# Add source/SAT to path for imports
sys.path.insert(0, 'source/SAT')

from SAT_solver_final import solve_sts, solve_sts_dimacs
from solution_checker import check_solution

TIMEOUT = 300
OUTPUT_DIR = "res/SAT"


def get_all_configs():
    """
    Generate all approach configurations.
    Returns list of configs, circle method first, then classic.
    Includes both decision (dec) and optimization (opt) versions for all.
    """
    configs = []
    
    solvers_z3 = [('z3', None)]
    solvers_dimacs = [('minisat', 'minisat'), ('glucose', 'glucose')]
    all_solvers = solvers_z3 + solvers_dimacs
    
    # ===== CIRCLE METHOD CONFIGURATIONS =====
    circle_constraint_combos = [
        # (name_suffix, n_fix_period, matches_per_team, different_match_per_period, params)
        ('', False, False, False, {'sb_fix': False}),
        ('_sbfix', True, False, False, {'sb_fix': True}),
        ('_ic2', False, False, True, {'ic_diff_match': True}),
        ('_sbfix_ic1_ic2', True, True, True, {'sb_fix': True, 'ic_matches': True, 'ic_diff_match': True}),
    ]
    
    for solver_name, dimacs_solver in all_solvers:
        solver_type = 'z3' if dimacs_solver is None else 'dimacs'
        
        for suffix, sb_fix, ic1, ic2, params in circle_constraint_combos:
            constraints = {
                'matches_per_team': ic1,
                'different_match_per_period': ic2,
                'n_fix_period': sb_fix,
            }
            
            # Decision version
            configs.append({
                'name': f'{solver_name}_dec_cir{suffix}',
                'solver_type': solver_type,
                'dimacs_solver': dimacs_solver,
                'use_circle_method': True,
                'optimize': False,
                'constraints': constraints.copy(),
                'params': params.copy()
            })
            
            # Optimization version
            configs.append({
                'name': f'{solver_name}_opt_cir{suffix}',
                'solver_type': solver_type,
                'dimacs_solver': dimacs_solver,
                'use_circle_method': True,
                'optimize': True,
                'constraints': constraints.copy(),
                'params': params.copy()
            })
    
    # ===== CLASSIC METHOD CONFIGURATIONS =====
    classic_constraint_combos = [
        # (name_suffix, n_fix_period, lex_periods, matches_per_team, different_match_per_period, params)
        ('', False, False, False, False, {'sb_fix': False, 'sb_lex': False}),
        ('_sbfix_sblex', True, True, False, False, {'sb_fix': True, 'sb_lex': True}),
        ('_ic1_ic2', False, False, True, True, {'ic_matches': True, 'ic_diff_match': True}),
        ('_sblex_ic1_ic2', False, True, True, True, {'sb_lex': True, 'ic_matches': True, 'ic_diff_match': True}),
        ('_sbfix_sblex_ic1_ic2', True, True, True, True, {'sb_fix': True, 'sb_lex': True, 'ic_matches': True, 'ic_diff_match': True}),
    ]
    
    for solver_name, dimacs_solver in all_solvers:
        solver_type = 'z3' if dimacs_solver is None else 'dimacs'
        
        for suffix, sb_fix, sb_lex, ic1, ic2, params in classic_constraint_combos:
            constraints = {
                'matches_per_team': ic1,
                'different_match_per_period': ic2,
                'n_fix_period': sb_fix,
                'lex_periods': sb_lex,
            }
            
            # Decision version
            configs.append({
                'name': f'{solver_name}_dec_cla{suffix}',
                'solver_type': solver_type,
                'dimacs_solver': dimacs_solver,
                'use_circle_method': False,
                'optimize': False,
                'constraints': constraints.copy(),
                'params': params.copy()
            })
            
            # Optimization version
            configs.append({
                'name': f'{solver_name}_opt_cla{suffix}',
                'solver_type': solver_type,
                'dimacs_solver': dimacs_solver,
                'use_circle_method': False,
                'optimize': True,
                'constraints': constraints.copy(),
                'params': params.copy()
            })
    
    return configs


def run_solver(config, n, timeout=TIMEOUT):
    """Run a single configuration."""
    try:
        if config['solver_type'] == 'z3':
            result = solve_sts(
                n=n,
                constraints=config['constraints'],
                use_circle_method=config['use_circle_method'],
                timeout=timeout,
                optimize=config.get('optimize', False)
            )
        else:
            result = solve_sts_dimacs(
                n=n,
                constraints=config['constraints'],
                use_circle_method=config['use_circle_method'],
                solver=config['dimacs_solver'],
                timeout=timeout,
                optimize=config.get('optimize', False)
            )
        return result
    except Exception as e:
        return {
            'solution': None,
            'time': timeout,
            'satisfiable': False,
            'obj': None,
            'error': str(e)
        }


def format_result(config, result):
    """Format result for JSON output."""
    time_floor = int(math.floor(result['time']))
    is_optimization = config.get('optimize', False)
    
    # time = 300 ⟺ optimal = false
    if not result['satisfiable']:
        if time_floor >= TIMEOUT:
            time_floor = TIMEOUT
        optimal = False
        stop_reason = "time_limit" if time_floor >= TIMEOUT else None
    else:
        optimal = True
        stop_reason = None
    
    # Handle obj field
    if is_optimization and result.get('obj') is not None:
        obj_value = result['obj']
    else:
        obj_value = None
    
    return {
        'time': time_floor,
        'method': 'circle' if config['use_circle_method'] else 'classic',
        'version': 'optimization' if is_optimization else 'decision',
        'params': config['params'],
        'optimal': optimal,
        'stop_reason': stop_reason,
        'obj': obj_value,
        'sol': result['solution'] if result['satisfiable'] else []
    }


def validate_solution(solution, n):
    """Validate using solution checker."""
    if solution is None:
        return False
    try:
        result = check_solution(solution, "None", 0, True)
        return result == "Valid solution"
    except:
        return False


class CompactJSONEncoder(json.JSONEncoder):
    """Custom encoder to keep sol arrays compact."""
    def encode(self, obj):
        return self._encode(obj, level=0)
    
    def _encode(self, obj, level):
        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for k, v in obj.items():
                key_str = json.dumps(k)
                if k == 'sol':
                    # Keep sol compact on one line
                    val_str = json.dumps(v, separators=(',', ': '))
                else:
                    val_str = self._encode(v, level + 1)
                items.append(f'{"  " * (level + 1)}{key_str}: {val_str}')
            return '{\n' + ',\n'.join(items) + '\n' + '  ' * level + '}'
        elif isinstance(obj, list):
            return json.dumps(obj)
        else:
            return json.dumps(obj)


def save_json(data, filepath):
    """Save JSON with compact sol format."""
    with open(filepath, 'w') as f:
        f.write(CompactJSONEncoder().encode(data))


def run_all(max_n=100):
    """Run all configurations for all n values until all fail."""
    configs = get_all_configs()
    failed_configs = set()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Count decision vs optimization configs
    dec_count = sum(1 for c in configs if not c.get('optimize', False))
    opt_count = sum(1 for c in configs if c.get('optimize', False))
    
    print(f"{'='*80}")
    print(f"SAT RUNNER - {len(configs)} configurations ({dec_count} decision, {opt_count} optimization)")
    print(f"{'='*80}")
    
    n = 12
    while n <= max_n:
        print(f"\n{'='*80}")
        print(f"n = {n}")
        print(f"{'='*80}")
        
        results_for_n = {}
        active_configs = [c for c in configs if c['name'] not in failed_configs]
        
        if not active_configs:
            print("All configurations failed. Stopping.")
            break
        
        active_dec = sum(1 for c in active_configs if not c.get('optimize', False))
        active_opt = sum(1 for c in active_configs if c.get('optimize', False))
        print(f"Active: {len(active_configs)} / {len(configs)} ({active_dec} dec, {active_opt} opt)")
        print()
        
        for config in active_configs:
            name = config['name']
            method = 'circle' if config['use_circle_method'] else 'classic'
            version = 'opt' if config.get('optimize', False) else 'dec'
            
            result = run_solver(config, n)
            
            # Determine status
            if result['satisfiable']:
                valid = validate_solution(result['solution'], n)
                if valid:
                    if config.get('optimize', False) and result.get('obj') is not None:
                        status = f"SAT (obj={result['obj']})"
                    else:
                        status = "SAT"
                else:
                    status = "INVALID"
                    failed_configs.add(name)
            else:
                if result['time'] >= TIMEOUT:
                    status = "TIMEOUT"
                else:
                    status = "UNSAT"
                # For n=4, UNSAT is expected - don't mark as failed
                if n != 4:
                    failed_configs.add(name)
            
            time_str = f"{result['time']:.2f}s"
            print(f"  {name:45} | {method:7} | {version:3} | {time_str:10} | {status}")
            
            # Store result
            results_for_n[name] = format_result(config, result)
        
        # Save JSON for this n
        if results_for_n:
            json_path = os.path.join(OUTPUT_DIR, f"{n}.json")
            save_json(results_for_n, json_path)
            print(f"\nSaved: {json_path} ({len(results_for_n)} results)")
        
        n += 2
    
    print(f"\n{'='*80}")
    print(f"DONE - Tested up to n={n-2}")
    print(f"Failed configs: {len(failed_configs)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-n', type=int, default=100)
    args = parser.parse_args()
    run_all(max_n=args.max_n)
