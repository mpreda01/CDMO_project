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

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is two levels up from test/SAT/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Add source/SAT to path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'source', 'SAT'))

from SAT_solver_final import solve_sts, solve_sts_dimacs
from solution_checker import check_solution  # Same folder as runner

TIMEOUT = 300
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'res', 'SAT')


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
    """
    Format result for JSON output.
    
    Assignment rules (Section 2.4):
    - time: integer (floor of actual runtime), max 300
    - optimal: boolean (true iff solved for decision, or solved to optimality for optimization)
    - obj: positive integer OR string "None" (NOT JSON null)
    - sol: list of lists
    - CRITICAL: time = 300 ⟺ optimal = false (biconditional)
    """
    time_floor = int(math.floor(result['time']))
    is_optimization = config.get('optimize', False)
    
    # CRITICAL: Enforce timeout limit FIRST - this affects everything else
    # If time >= TIMEOUT, treat as timeout regardless of satisfiability
    if time_floor >= TIMEOUT:
        # Timeout case: time=300, optimal=false, discard solution
        return {
            'time': TIMEOUT,
            'method': 'circle' if config['use_circle_method'] else 'classic',
            'version': 'optimization' if is_optimization else 'decision',
            'params': config['params'],
            'optimal': False,
            'stop_reason': "time_limit",
            'obj': "None",  # String "None", NOT null
            'sol': []
        }
    
    # Within timeout: check satisfiability
    if not result['satisfiable']:
        # UNSAT case (within timeout)
        return {
            'time': time_floor,
            'method': 'circle' if config['use_circle_method'] else 'classic',
            'version': 'optimization' if is_optimization else 'decision',
            'params': config['params'],
            'optimal': False,
            'stop_reason': None,
            'obj': "None",  # String "None", NOT null
            'sol': []
        }
    
    # SAT case (within timeout): optimal=true, keep solution
    # Handle obj field: positive integer for optimization with result, else "None" string
    if is_optimization and result.get('obj') is not None:
        obj_value = result['obj']
    else:
        obj_value = "None"  # String "None", NOT null
    
    return {
        'time': time_floor,
        'method': 'circle' if config['use_circle_method'] else 'classic',
        'version': 'optimization' if is_optimization else 'decision',
        'params': config['params'],
        'optimal': True,
        'stop_reason': None,
        'obj': obj_value,
        'sol': result['solution'] if result['solution'] else []
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


def run_all(max_n=100, start_n=6):
    """Run all configurations for all n values until all fail."""
    configs = get_all_configs()
    failed_configs = set()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Separate decision and optimization configs
    dec_configs = [c for c in configs if not c.get('optimize', False)]
    opt_configs = [c for c in configs if c.get('optimize', False)]
    
    print(f"{'='*80}")
    print(f"SAT RUNNER - {len(configs)} configurations ({len(dec_configs)} decision, {len(opt_configs)} optimization)")
    print(f"{'='*80}")
    
    n = start_n
    while n <= max_n:
        print(f"\n{'='*80}")
        print(f"n = {n}")
        print(f"{'='*80}")
        
        results_for_n = {}
        
        # Filter active decision configs
        active_dec = [c for c in dec_configs if c['name'] not in failed_configs]
        
        if not active_dec:
            print("All decision configurations failed. Stopping.")
            break
        
        print(f"Active decision: {len(active_dec)} / {len(dec_configs)}")
        print()
        
        # Track which decision configs succeeded for this n
        succeeded_dec = set()
        
        # Run decision configs first
        for config in active_dec:
            name = config['name']
            method = 'circle' if config['use_circle_method'] else 'classic'
            
            result = run_solver(config, n)
            
            # Determine status (for logging purposes)
            timed_out = result['time'] >= TIMEOUT
            
            if timed_out:
                status = "TIMEOUT"
                # For n=4, timeout/UNSAT is expected - don't mark as failed
                if n != 4:
                    failed_configs.add(name)
            elif result['satisfiable']:
                valid = validate_solution(result['solution'], n)
                if valid:
                    status = "SAT"
                    succeeded_dec.add(name)
                else:
                    status = "INVALID"
                    failed_configs.add(name)
            else:
                status = "UNSAT"
                # For n=4, UNSAT is expected - don't mark as failed
                if n != 4:
                    failed_configs.add(name)
            
            time_str = f"{result['time']:.2f}s"
            print(f"  {name:45} | {method:7} | dec | {time_str:10} | {status}")
            
            results_for_n[name] = format_result(config, result)
        
        print()
        
        # Run optimization configs only if corresponding decision succeeded
        opt_run_count = 0
        opt_skip_count = 0
        
        for config in opt_configs:
            name = config['name']
            # Get corresponding decision config name (replace _opt_ with _dec_)
            dec_name = name.replace('_opt_', '_dec_')
            
            # Skip if decision config failed or didn't succeed this round
            if dec_name in failed_configs or dec_name not in succeeded_dec:
                opt_skip_count += 1
                continue
            
            method = 'circle' if config['use_circle_method'] else 'classic'
            
            result = run_solver(config, n)
            opt_run_count += 1
            
            # Determine status (for logging purposes)
            timed_out = result['time'] >= TIMEOUT
            
            if timed_out:
                status = "TIMEOUT"
            elif result['satisfiable']:
                valid = validate_solution(result['solution'], n)
                if valid:
                    status = f"SAT (obj={result.get('obj', '?')})"
                else:
                    status = "INVALID"
            else:
                status = "UNSAT"
            
            time_str = f"{result['time']:.2f}s"
            print(f"  {name:45} | {method:7} | opt | {time_str:10} | {status}")
            
            results_for_n[name] = format_result(config, result)
        
        if opt_skip_count > 0:
            print(f"\n  (Skipped {opt_skip_count} optimization configs due to decision failures)")
        
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
    parser.add_argument('--start-n', type=int, default=2)
    args = parser.parse_args()
    run_all(max_n=args.max_n, start_n=args.start_n)
