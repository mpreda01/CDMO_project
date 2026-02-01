#!/usr/bin/env python3
"""
Simple MiniZinc Runner using subprocess
Usage: python simple_runner.py <mzn_file> <n_value>
Example: python simple_runner.py ../../source/CP/CP_cyrcle_method.mzn 6

Also includes circle_method() function for deterministic schedule generation
"""

import subprocess
import sys
import tempfile
import os
from typing import Tuple, List, Dict, Optional
import itertools
import json
import time
from collections import defaultdict
import psutil


def kill_process_tree(process):
    """Kill a process and all its child processes (like fzn-gecode.exe)"""
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        
        # Terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Terminate parent
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
        
        # Wait for termination
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)
        
        # Force kill if still alive
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"Warning: Error killing process tree: {e}")


def circle_method(n: int) -> Tuple[List[int], List[int]]:
    """
    Generate tournament schedule using the circle method algorithm.
    
    This is a deterministic implementation of the CP_cyrcle_method.mzn algorithm.
    The circle method creates a round-robin tournament schedule where each team
    plays against every other team exactly once.
    
    Args:
        n: Number of teams (must be even)
    
    Returns:
        Tuple of (home, away) where:
        - home: List of home team numbers in schedule order
        - away: List of away team numbers in schedule order
        
    The schedule is organized as:
        n_weeks = n - 1 (number of weeks)
        n_periods = n // 2 (number of periods per week)
        Total matches = n_weeks * n_periods
    
    Example:
        >>> home, away = circle_method(6)
        >>> # Returns schedule for 6 teams, 5 weeks, 3 periods per week (15 matches)
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even, got {n}")
    
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    
    n_weeks = n - 1
    n_periods = n // 2
    
    home = []
    away = []
    
    # Iterate through all periods and weeks following the MiniZinc algorithm
    for p in range(1, n_periods + 1):  # Periods: 1..n_periods
        for r in range(1, n_weeks + 1):  # Weeks: 1..n_weeks
            # Calculate deterministic pair for each (period, week)
            offset = p - 1
            
            # Clockwise position in the circle
            cw_pos = ((r + offset - 1) % (n - 1)) + 1
            
            if p == 1:
                team_a = n
            else:
                team_a = (n - 1) if cw_pos == 0 else cw_pos
            
            # Counterclockwise position in the circle
            # +100*(n-1) ensures positivity as in MiniZinc
            ccw_raw = r - offset - 1 + (n - 1) * 100
            ccw_pos = (ccw_raw % (n - 1)) + 1
            
            if p == 1:
                team_b = r
            else:
                team_b = (n - 1) if ccw_pos == 0 else ccw_pos
            
            home.append(team_a)
            away.append(team_b)
    
    return home, away


def run_n20_optimizer(n: int, 
                      use_int_search: bool = True,
                      use_restart_luby: bool = False,
                      use_relax_and_reconstruct: bool = False,
                      chuffed: bool = False,
                      symm_brake: bool = True,
                      ic_diff_match_in_week: bool = True,
                      timeout: int = 300,
                      use_optimizer: bool = True,
                      return_result: bool = False) -> Tuple[bool, str]:
    """
    Run  optimizer with circle_method generated schedule,
    optionally followed by optimizer_2.0.mzn for further optimization.
    
    Args:
        n: Number of teams (must be even)
        use_int_search: Enable integer search strategy
        use_restart_luby: Use Luby restart strategy
        use_relax_and_reconstruct: Use relax and reconstruct
        chuffed: Use Chuffed solver
        symm_brake: Enable symmetry breaking
        ic_diff_match_in_week: Implied constraint - different matches per week
        timeout: Timeout in seconds
        use_optimizer: If True, run optimizer_2.0.mzn after n20
        return_result: If True, return result dict instead of (success, output)
    
    Returns:
        Tuple of (success, output_string) or result Dict if return_result=True
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even, got {n}")
    
    # Generate initial schedule using circle method
    print(f"Generating initial schedule for {n} teams using circle_method...")
    home, away = circle_method(n)
    print(f"Generated {len(home)} matches\n")
    
    # Find MiniZinc
    minizinc = find_minizinc()
    
    # Create .dzn data file with all parameters
    data_content = f"""
    use_int_search = {str(use_int_search).lower()};
    use_restart_luby = {str(use_restart_luby).lower()};
    use_relax_and_reconstruct = {str(use_relax_and_reconstruct).lower()};
    chuffed = {str(chuffed).lower()};
    symm_brake = {str(symm_brake).lower()};
    ic_diff_match_in_week = {str(ic_diff_match_in_week).lower()};
    home0 = {home};
    away0 = {away};
    n_teams = {n};
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dzn', delete=False) as f:
        f.write(data_content)
        dzn_file = f.name
    
    try:
        print(f"Running circle.mzn optimizer...")
        print(f"Timeout: {timeout}s")
        print("=" * 60)
        
        # Determine solver
        solver = 'Chuffed' if chuffed else 'Gecode'
        
        # Path to circle model (relative to test/CP directory)
        n20_model = os.path.join(os.path.dirname(__file__), 
                                  '..', '..', 'source', 'CP', 'circle.mzn')
        n20_model = os.path.abspath(n20_model)
        
        if not os.path.exists(n20_model):
            print(f"Model file not found: {n20_model}")
            if return_result:
                params = {
                    'use_int_search': use_int_search,
                    'use_restart_luby': use_restart_luby,
                    'use_relax_and_reconstruct': use_relax_and_reconstruct,
                    'chuffed': chuffed,
                    'symm_brake': symm_brake,
                    'ic_diff_match_in_week': ic_diff_match_in_week
                }
                params['circle'] = True
                params['optimized'] = use_optimizer
                return {
                    'n': n,
                    'params': params,
                    'n20_success': False,
                    'n20_time': 0,
                    'optimizer_success': False,
                    'optimizer_time': 0,
                    'time': 0,
                    'timeout_reached': False,
                    'optimal': False,
                    'obj': "None",
                    'sol': []
                }
            return False, "Model file not found"
        
        import time
        start_time = time.time()
        
        process = None
        try:
     
            process = subprocess.Popen(
                [minizinc, n20_model, dzn_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                n20_time = time.time() - start_time
                n20_output = stdout
                
                # Kill any remaining child processes
                kill_process_tree(process)
                
                if stdout:
                    print(stdout)
                
                if stderr:
                    print(f"\nStderr:\n{stderr}")
                
                result_returncode = process.returncode
                result_stdout = stdout
                result_stderr = stderr
                
            except subprocess.TimeoutExpired:
                print(f"Timeout reached ({timeout}s), terminating n20 process...")
                kill_process_tree(process)
                
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing n20 process...")
                    process.kill()
                    process.wait()
                
                n20_time = timeout
                raise subprocess.TimeoutExpired(n20_model, timeout)
        
        except subprocess.TimeoutExpired:
            print(f"\nTimeout after {timeout}s")
            if return_result:
                params = {
                    'use_int_search': use_int_search,
                    'use_restart_luby': use_restart_luby,
                    'use_relax_and_reconstruct': use_relax_and_reconstruct,
                    'chuffed': chuffed,
                    'symm_brake': symm_brake,
                    'ic_diff_match_in_week': ic_diff_match_in_week
                }
                params['optimized'] = use_optimizer
                params['circle'] = True
                return {
                    'n': n,
                    'params': params,
                    'n20_success': False,
                    'n20_time': timeout,
                    'optimizer_success': False,
                    'optimizer_time': 0,
                    'time': int(timeout),
                    'timeout_reached': True,
                    'optimal': True if not use_optimizer else False,
                    'obj': "None",
                    'sol': []
                }
            return False, "Timeout"
        
        if result_returncode != 0:
            print(f"\nn20 Failed (exit code {result_returncode})")
            if return_result:
                params = {
                    'use_int_search': use_int_search,
                    'use_restart_luby': use_restart_luby,
                    'use_relax_and_reconstruct': use_relax_and_reconstruct,
                    'chuffed': chuffed,
                    'symm_brake': symm_brake,
                    'ic_diff_match_in_week': ic_diff_match_in_week
                }
                params['circle'] = True
                params['optimized'] = use_optimizer
                return {
                    'n': n,
                    'params': params,
                    'n20_success': False,
                    'n20_time': round(n20_time, 2),
                    'optimizer_success': False,
                    'optimizer_time': 0,
                    'time': int(n20_time),
                    'timeout_reached': False,
                    'optimal': False,
                    'obj': "None",
                    'sol': []
                }
            return False, result_stderr
        
        print(f"\nn20 Success (Time: {n20_time:.2f}s)")
        
        # If not using optimizer, return n20 result
        if not use_optimizer:
            if return_result:
                params = {
                    'use_int_search': use_int_search,
                    'use_restart_luby': use_restart_luby,
                    'use_relax_and_reconstruct': use_relax_and_reconstruct,
                    'chuffed': chuffed,
                    'symm_brake': symm_brake,
                    'ic_diff_match_in_week': ic_diff_match_in_week
                }
                total_time = time.time() - start_time
                return parse_output_to_result(n, params, n20_output, total_time, n20_time, 0, True, False, use_optimizer=use_optimizer)
            return True, n20_output
        
        # Step 2: Extract data for optimizer_2.0.mzn
        print("\nExtracting data for optimizer_2.0.mzn...")
        optimizer_data = extract_optimizer_data(n20_output, n)
        
        if not optimizer_data:
            print("Failed to extract optimizer data, returning n20 output")
            if return_result:
                params = {
                    'use_int_search': use_int_search,
                    'use_restart_luby': use_restart_luby,
                    'use_relax_and_reconstruct': use_relax_and_reconstruct,
                    'chuffed': chuffed,
                    'symm_brake': symm_brake,
                    'ic_diff_match_in_week': ic_diff_match_in_week
                }
                total_time = time.time() - start_time
                return parse_output_to_result(n, params, n20_output, total_time, n20_time, 0, True, False, use_optimizer=use_optimizer)
            return True, n20_output
        
        # Check remaining time
        remaining_time = timeout - n20_time
        if remaining_time <= 0:
            print("Timeout reached, skipping optimizer")
            if return_result:
                params = {
                    'use_int_search': use_int_search,
                    'use_restart_luby': use_restart_luby,
                    'use_relax_and_reconstruct': use_relax_and_reconstruct,
                    'chuffed': chuffed,
                    'symm_brake': symm_brake,
                    'ic_diff_match_in_week': ic_diff_match_in_week
                }
                params['circle'] = True
                params['optimized'] = use_optimizer
                return {
                    'n': n,
                    'params': params,
                    'n20_success': True,
                    'n20_time': round(n20_time, 2),
                    'optimizer_success': False,
                    'optimizer_time': 0,
                    'time': int(n20_time),
                    'timeout_reached': True,
                    'optimal': True,
                    'obj': "None",
                    'sol': []
                }
            return True, n20_output
        
        # Step 3: Run optimizer_2.0.mzn
        print(f"\nRunning optimizer_2.0.mzn...")
        print(f"Remaining timeout: {remaining_time:.0f}s")
        print("=" * 60)
        
        optimizer_model = os.path.join(os.path.dirname(__file__), 
                                       '..', '..', 'source', 'CP', 'optimizer_2.0.mzn')
        optimizer_model = os.path.abspath(optimizer_model)
        
        if not os.path.exists(optimizer_model):
            print(f"Optimizer file not found: {optimizer_model}")
            if return_result:
                params = {
                    'use_int_search': use_int_search,
                    'use_restart_luby': use_restart_luby,
                    'use_relax_and_reconstruct': use_relax_and_reconstruct,
                    'chuffed': chuffed,
                    'symm_brake': symm_brake,
                    'ic_diff_match_in_week': ic_diff_match_in_week
                }
                total_time = time.time() - start_time
                return parse_output_to_result(n, params, n20_output, total_time, n20_time, 0, True, False, use_optimizer=use_optimizer)
            return True, n20_output
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dzn', delete=False) as f:
            f.write(optimizer_data)
            opt_dzn_file = f.name
        
        opt_process = None
        try:
            opt_process = subprocess.Popen(
                [minizinc, '--solver', solver, optimizer_model, opt_dzn_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = opt_process.communicate(timeout=int(remaining_time))
                opt_time = time.time() - start_time - n20_time
                
                # Kill any remaining child processes
                kill_process_tree(opt_process)
                
                if stdout:
                    print(stdout)
                
                if stderr:
                    print(f"\nStderr:\n{stderr}")
                
                opt_returncode = opt_process.returncode
                opt_stdout = stdout
                opt_stderr = stderr
                
            except subprocess.TimeoutExpired:
                print(f"Optimizer timeout reached, terminating process...")
                kill_process_tree(opt_process)
                
                try:
                    opt_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing optimizer process...")
                    opt_process.kill()
                    opt_process.wait()
                
                opt_time = remaining_time
                raise subprocess.TimeoutExpired(optimizer_model, remaining_time)
            
            if opt_returncode == 0:
                print(f"\nOptimizer Success (Time: {opt_time:.2f}s)")
                total_time = time.time() - start_time
                print(f"Total time: {total_time:.2f}s")
                
                if return_result:
                    params = {
                        'use_int_search': use_int_search,
                        'use_restart_luby': use_restart_luby,
                        'use_relax_and_reconstruct': use_relax_and_reconstruct,
                        'chuffed': chuffed,
                        'symm_brake': symm_brake,
                        'ic_diff_match_in_week': ic_diff_match_in_week
                    }
                    return parse_output_to_result(n, params, opt_stdout, total_time, n20_time, opt_time, True, True, use_optimizer=use_optimizer)
                return True, opt_stdout
            else:
                print(f"\nOptimizer failed, returning n20 output")
                
                if return_result:
                    params = {
                        'use_int_search': use_int_search,
                        'use_restart_luby': use_restart_luby,
                        'use_relax_and_reconstruct': use_relax_and_reconstruct,
                        'chuffed': chuffed,
                        'symm_brake': symm_brake,
                        'ic_diff_match_in_week': ic_diff_match_in_week
                    }
                    total_time = time.time() - start_time
                    return parse_output_to_result(n, params, n20_output, total_time, n20_time, opt_time, True, False, use_optimizer=use_optimizer)
                return True, n20_output
                
        except subprocess.TimeoutExpired:
            print(f"\nOptimizer timeout, returning n20 output")
            
            if return_result:
                params = {
                    'use_int_search': use_int_search,
                    'use_restart_luby': use_restart_luby,
                    'use_relax_and_reconstruct': use_relax_and_reconstruct,
                    'chuffed': chuffed,
                    'symm_brake': symm_brake,
                    'ic_diff_match_in_week': ic_diff_match_in_week
                }
                total_time = time.time() - start_time
                opt_time = total_time - n20_time
                return parse_output_to_result(n, params, n20_output, total_time, n20_time, opt_time, True, False, use_optimizer=use_optimizer)
            return True, n20_output
        finally:
            # Ensure process tree is killed
            if opt_process is not None:
                kill_process_tree(opt_process)
            try:
                os.unlink(opt_dzn_file)
            except:
                pass
            
    except Exception as e:
        print(f"\nError: {e}")
        if return_result:
            params = {
                'use_int_search': use_int_search,
                'use_restart_luby': use_restart_luby,
                'use_relax_and_reconstruct': use_relax_and_reconstruct,
                'chuffed': chuffed,
                'symm_brake': symm_brake,
                'ic_diff_match_in_week': ic_diff_match_in_week
            }
            params['optimized'] = use_optimizer
            params['circle'] = True
            return {
                'n': n,
                'params': params,
                'n20_success': False,
                'n20_time': 0,
                'optimizer_success': False,
                'optimizer_time': 0,
                'time': 0,
                'timeout_reached': False,
                'optimal': True if not use_optimizer else False,
                'obj': "None",
                'sol': []
            }
        return False, str(e)
    finally:
        # Ensure process tree is killed
        if process is not None:
            kill_process_tree(process)
        os.unlink(dzn_file)


def extract_optimizer_data(n20_output: str, n_teams: int) -> Optional[str]:
    """Extract home and away arrays from n20 output for optimizer_2.0.mzn"""
    try:
        lines = n20_output.strip().split('\n')
        home_array = None
        away_array = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('home ='):
                # Extract array from "home = [1, 2, 3, ...];"
                array_str = line.split('=')[1].strip().rstrip(';')
                home_array = eval(array_str)
            elif line.startswith('away ='):
                array_str = line.split('=')[1].strip().rstrip(';')
                away_array = eval(array_str)
        
        if not home_array or not away_array:
            return None
        
        # Create optimizer data file
        data_content = f"""n_teams = {n_teams};
home0 = {home_array};
away0 = {away_array};
"""
        return data_content
        
    except Exception as e:
        print(f"Error extracting optimizer data: {e}")
        return None


def convert_home_away_to_solution_matrix(n20_output: str, n_teams: int) -> Optional[List]:
    """Convert home and away arrays from n20 output to solution matrix format"""
    try:
        lines = n20_output.strip().split('\n')
        home_array = None
        away_array = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('home ='):
                # Extract array from "home = [1, 2, 3, ...];"
                array_str = line.split('=')[1].strip().rstrip(';')
                home_array = eval(array_str)
            elif line.startswith('away ='):
                array_str = line.split('=')[1].strip().rstrip(';')
                away_array = eval(array_str)
        
        if not home_array or not away_array:
            return None
        
        # Convert to solution matrix format grouped by periods
        # For circle method: n_periods = n_teams // 2, n_weeks = n_teams - 1
        # Matches are generated as: for each period, for each week
        # So matches are in order: P1W1, P1W2, ..., P1W7, P2W1, P2W2, ...
        n_periods = n_teams // 2
        n_weeks = n_teams - 1
        
        solution = []
        for period in range(n_periods):
            period_matches = []
            start_idx = period * n_weeks
            end_idx = start_idx + n_weeks
            
            for i in range(start_idx, end_idx):
                if i < len(home_array) and i < len(away_array):
                    period_matches.append([home_array[i], away_array[i]])
            
            if period_matches:
                solution.append(period_matches)
        
        return solution
        
    except Exception as e:
        print(f"Error converting home/away to solution matrix: {e}")
        return None

def parse_output_to_result(n: int, params: Dict[str, bool], output: str, 
                           total_time: float, n20_time: float, opt_time: float,
                           n20_success: bool, opt_success: bool, use_optimizer: bool = False) -> Dict:
    """Parse optimizer output and create result dictionary"""
    # Add circle and optimized parameters to params
    params_with_circle = params.copy()
    params_with_circle['circle'] = True
    params_with_circle['optimized'] = use_optimizer
    
    result = {
        'n': n,
        'params': params_with_circle,
        'n20_success': n20_success,
        'n20_time': round(n20_time, 2),
        'optimizer_success': opt_success,
        'optimizer_time': round(opt_time, 2),
        'time': int(total_time),
        'timeout_reached': False,
        'optimal': False,
        'obj': None,
        'sol': []
    }
    
    # Parse solution from output
    try:
        lines = output.strip().split('\n')
        
        # Handle optimal and obj based on whether optimizer was chosen by user
        if not use_optimizer:
            # Only n20 used - no objective optimization
            result['optimal'] = n20_success
            result['obj'] = None
            # Extract solution from n20 home/away arrays
            if n20_success:
                n20_solution = convert_home_away_to_solution_matrix(output, n)
                if n20_solution:
                    result['sol'] = n20_solution
        else:
            # Optimizer was chosen - parse objective value
            for line in lines:
                if "Optimized Max Imbalance:" in line:
                    try:
                        result['obj'] = int(line.split(": ")[1])
                        if result['obj'] == 1:
                            result['optimal'] = True
                    except:
                        pass
        
        # Parse matrix solution (for optimizer output)
        if use_optimizer:
            matrix_lines = []
            found_matrix = False
            for line in lines:
                line = line.strip()
                if line.startswith('[['):
                    found_matrix = True
                    matrix_lines.append(line)
                elif found_matrix and (line.startswith('===') or line.startswith('---')):
                    break
            
            if matrix_lines:
                solution = []
                for line in matrix_lines:
                    clean_line = line.strip()[2:-2]
                    pairs_str = clean_line.split('] , [')
                    period_matches = []
                    
                    for pair_str in pairs_str:
                        import re
                        numbers = re.findall(r'\d+', pair_str)
                        if len(numbers) >= 2:
                            period_matches.append([int(numbers[0]), int(numbers[1])])
                    
                    if period_matches:
                        solution.append(period_matches)
                
                result['sol'] = solution
    except Exception as e:
        print(f"Warning: Error parsing output: {e}")
    
    # Convert None to string "None" for obj field
    if result['obj'] is None:
        result['obj'] = "None"
    
    return result


def config_exists_in_json(n: int, params: Dict[str, bool]) -> bool:
    """Check if a parameter configuration already exists in the JSON file for given n"""
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'CP'))
    filename = os.path.join(output_dir, f"{n}.json")
    
    if not os.path.exists(filename):
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Check each existing result
        for timestamp, result in existing_data.items():
            if 'params' in result:
                existing_params = result['params']
                # Compare only the boolean strategy parameters (exclude 'circle' and 'optimized')
                match = True
                for key, value in params.items():
                    if key not in ['circle', 'optimized']:
                        if existing_params.get(key) != value:
                            match = False
                            break
                
                # Also check that 'circle' and 'optimized' match if they exist
                if match:
                    if existing_params.get('circle') != params.get('circle'):
                        match = False
                    if existing_params.get('optimized') != params.get('optimized'):
                        match = False
                
                if match:
                    return True
        
        return False
        
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read {filename}: {e}")
        return False


def save_results(results: List[Dict]):
    """Save results to JSON file with timestamp structure in res/CP relative to script"""
    print('\nSaving results...')
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'CP'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Group results by 'n' value
    grouped_results = defaultdict(list)
    for result in results:
        if 'n' in result:
            n_value = result['n']
            grouped_results[n_value].append(result)
    
    # Save each group to its respective file
    for n_value, group_results in grouped_results.items():
        filename = os.path.join(output_dir, f"{n_value}.json")
        
        # Load existing data if file exists
        existing_data = {}
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                print(f"Loaded existing data from {filename}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load existing file {filename}: {e}")
                existing_data = {}
        
        # Add new results with current timestamp
        for result in group_results:
            current_timestamp = str(time.time())
            existing_data[current_timestamp] = result
        
        # Save back to file with custom formatting
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('{\n')
            timestamps = list(existing_data.keys())
            for i, timestamp in enumerate(timestamps):
                f.write(f'  "{timestamp}": {{\n')
                
                result = existing_data[timestamp]
                items = list(result.items())
                for j, (key, value) in enumerate(items):
                    f.write(f'    "{key}": ')
                    if key == 'sol' and isinstance(value, list):
                        if value:
                            f.write('[\n')
                            for k, period in enumerate(value):
                                f.write('      [')
                                for l, match in enumerate(period):
                                    f.write(f'[{match[0]}, {match[1]}]')
                                    if l < len(period) - 1:
                                        f.write(', ')
                                f.write(']')
                                if k < len(value) - 1:
                                    f.write(',')
                                f.write('\n')
                            f.write('    ]')
                        else:
                            f.write('[]')
                    else:
                        json.dump(value, f, ensure_ascii=False)
                    
                    if j < len(items) - 1:
                        f.write(',')
                    f.write('\n')
                
                f.write('  }')
                if i < len(timestamps) - 1:
                    f.write(',')
                f.write('\n')
            
            f.write('}\n')
        
        print(f"Results for n={n_value} saved to: {filename} (Total timestamps: {len(existing_data)})")


def find_minizinc():
    """Find MiniZinc executable"""
    paths = [
        r"D:\Program\MiniZinc\minizinc.exe",
        r"C:\Program Files\MiniZinc\bin\minizinc.exe",
        "minizinc"
    ]
    
    for path in paths:
        try:
            result = subprocess.run([path, "--version"], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"Found MiniZinc: {path}\n")
                return path
        except:
            continue
    
    print("MiniZinc not found!")
    sys.exit(1)


def run_minizinc(mzn_file, n_value, timeout=300):
    """Run MiniZinc file with parameter n"""
    
    if not os.path.exists(mzn_file):
        print(f"File not found: {mzn_file}")
        return False
    
    minizinc = find_minizinc()
    
    # Create .dzn data file
    data_content = f"n = {n_value};\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dzn', delete=False) as f:
        f.write(data_content)
        dzn_file = f.name
    
    try:
        print(f"Running: {mzn_file}")
        print(f"Parameter: n = {n_value}")
        print(f"Timeout: {timeout}s")
        print("=" * 60)
        
        result = subprocess.run(
            [minizinc, '--solver', 'Gecode', mzn_file, dzn_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(f"\nStderr:\n{result.stderr}")
        
        if result.returncode == 0:
            print("\nSuccess")
            return True
        else:
            print(f"\nFailed (exit code {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\nTimeout after {timeout}s")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False
    finally:
        os.unlink(dzn_file)


def get_user_input_n20() -> Tuple[List[int], Dict]:
    """Get user input for n20 optimizer parameters"""
    print("=== N20 Optimizer Runner ===\n")
    
    # Available n values
    n_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Ask about optimizer usage
    use_opt_input = input("Use optimizer_2.0.mzn for optimization? (yes/no/both, default: yes): ").strip().lower()
    if use_opt_input in ['both', 'b', 'all']:
        optimizer_modes = [True, False]  # Both with and without optimization
        print(f"Optimizer modes: both with and without optimization\n")
    elif use_opt_input in ['no', 'n', 'false', 'f', '0', 'without']:
        optimizer_modes = [False]  # Only without optimization
        print(f"Optimizer modes: without optimization\n")
    else:
        optimizer_modes = [True]  # Only with optimization (default)
        print(f"Optimizer modes: with optimization\n")
    
    # Boolean parameters for n20
    bool_params = [
        'use_int_search',
        'use_restart_luby',
        'use_relax_and_reconstruct',
        'chuffed',
        'symm_brake',
        'ic_diff_match_in_week'
    ]
    
    # Get n values
    print("Available n values:", n_values)
    n_input = input("Enter n values (comma-separated, or 'all' for all values): ").strip()
    
    if n_input.lower() == 'all':
        selected_n = n_values
    else:
        try:
            selected_n = [int(x.strip()) for x in n_input.split(',')]
            selected_n = [n for n in selected_n if n in n_values and n % 2 == 0]
            if not selected_n:
                print("No valid n values selected, using all.")
                selected_n = n_values
        except:
            print("Invalid input, using all n values.")
            selected_n = n_values
    
    # Get boolean parameters
    print(f"\nBoolean parameters: {bool_params}")
    mode = input("Enter 'manual' for manual input or 'all' for all combinations: ").strip().lower()
    
    if mode == 'manual':
        params = {}
        for param in bool_params:
            while True:
                value = input(f"{param} (true/false): ").strip().lower()
                if value in ['true', 't', '1', 'yes', 'y']:
                    params[param] = True
                    break
                elif value in ['false', 'f', '0', 'no', 'n']:
                    params[param] = False
                    break
                else:
                    print("Please enter true/false")
        return selected_n, {'manual': params, 'bool_params': bool_params, 'optimizer_modes': optimizer_modes}
    else:
        return selected_n, {'all_combinations': True, 'bool_params': bool_params, 'optimizer_modes': optimizer_modes}


def run_n20_interactive():
    """Run n20 optimizer in interactive mode"""
    selected_n, config = get_user_input_n20()
    results = []
    
    if 'manual' in config:
        # Manual mode - single configuration
        params = config['manual']
        optimizer_modes = config.get('optimizer_modes', [True])
        total_runs = len(selected_n) * len(optimizer_modes)
        print(f"\nRunning {total_runs} configurations with manual parameters...")
        print(f"Parameters: {params}")
        print(f"Optimizer modes: {['with optimization' if m else 'without optimization' for m in optimizer_modes]}\n")
        
        run_count = 0
        for n in selected_n:
            for use_optimizer in optimizer_modes:
                run_count += 1
                print(f"[{run_count}/{total_runs}] Running n={n}, opt={use_optimizer}")
                print("=" * 60)
                
                result = run_n20_optimizer(n, **params, use_optimizer=use_optimizer, return_result=True)
                results.append(result)
                
                if isinstance(result, dict):
                    if use_optimizer:
                        # Check both n20 and optimizer success
                        if result.get('n20_success') and result.get('optimizer_success'):
                            print(f"SUCCESS for n={n}")
                        else:
                            print(f"FAILED for n={n}")
                    else:
                        # Only check n20 success when not using optimizer
                        if result.get('n20_success'):
                            print(f"SUCCESS for n={n}")
                        else:
                            print(f"FAILED for n={n}")
                print("-" * 60 + "\n")
    else:
        # All combinations mode
        bool_params = config['bool_params']
        optimizer_modes = config.get('optimizer_modes', [True])  # Default to [True] for backward compatibility
        bool_combinations = list(itertools.product([True, False], repeat=len(bool_params)))
        total_runs = len(selected_n) * len(bool_combinations) * len(optimizer_modes)
        
        print(f"\nRunning {total_runs} total combinations...")
        print(f"Optimizer modes: {['with optimization' if m else 'without optimization' for m in optimizer_modes]}\n")
        
        run_count = 0
        skipped_count = 0
        for n in selected_n:
            for use_optimizer in optimizer_modes:
                for combo in bool_combinations:
                    run_count += 1
                    params = dict(zip(bool_params, combo))
                    
                    # Add circle and optimized to params for comparison
                    params_with_meta = params.copy()
                    params_with_meta['circle'] = True
                    params_with_meta['optimized'] = use_optimizer
                    
                    # Check if this configuration already exists
                    if config_exists_in_json(n, params_with_meta):
                        skipped_count += 1
                        print(f"[{run_count}/{total_runs}] n={n}, opt={use_optimizer}, params={params} - SKIPPED (already exists)")
                        print("-" * 60 + "\n")
                        continue
                    
                    print(f"[{run_count}/{total_runs}] n={n}, opt={use_optimizer}, params={params}")
                    print("=" * 60)
                    
                    result = run_n20_optimizer(n, **params, use_optimizer=use_optimizer, return_result=True)
                    results.append(result)
                    
                    if isinstance(result, dict):
                        if use_optimizer:
                            # Check both n20 and optimizer success
                            if result.get('n20_success') and result.get('optimizer_success'):
                                print(f"SUCCESS")
                            else:
                                print(f"FAILED")
                        else:
                            # Only check n20 success when not using optimizer
                            if result.get('n20_success'):
                                print(f"SUCCESS")
                            else:
                                print(f"FAILED")
                        
                        # Save result immediately after computation
                        save_results([result])
                        
                    print("-" * 60 + "\n")
        
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} configurations that already exist in JSON files")
                
    # Save all results
    if results:
        save_results(results)
        print(f"\nAll results saved! Total runs: {len(results)}")


def test_circle_method():
    """Test the circle_method function and display results"""
    print("\n" + "=" * 60)
    print("Testing circle_method() function")
    print("=" * 60)
    
    for n in [4, 6, 8]:
        print(f"\n--- Testing with n = {n} teams ---")
        home, away = circle_method(n)
        
        n_weeks = n - 1
        n_periods = n // 2
        
        print(f"Weeks: {n_weeks}, Periods per week: {n_periods}")
        print(f"Total matches: {len(home)}")
        print(f"\nHome: {home}")
        print(f"Away: {away}")
        
        # Display as matrix
        print(f"\nSchedule matrix (Period x Week):")
        idx = 0
        for p in range(n_periods):
            row_home = []
            row_away = []
            for w in range(n_weeks):
                row_home.append(home[idx])
                row_away.append(away[idx])
                idx += 1
            print(f"  Period {p+1}: {' '.join(f'({h}-{a})' for h, a in zip(row_home, row_away))}")
        print()


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        selected_n = [2, 4, 6, 8, 10]
        results = []
        bool_params = [
            'use_int_search',
            'use_restart_luby',
            'use_relax_and_reconstruct',
            'chuffed',
            'symm_brake',
            'ic_diff_match_in_week'
        ]
        config = {'all_combinations': True, 'bool_params': bool_params, 'optimizer_modes': [True]}
        bool_params = config['bool_params']
        optimizer_modes = config.get('optimizer_modes', [True])  # Default to [True] for backward compatibility
        bool_combinations = list(itertools.product([True, False], repeat=len(bool_params)))
        total_runs = len(selected_n) * len(bool_combinations) * len(optimizer_modes)
        
        print(f"\nRunning {total_runs} total combinations...")
        print(f"Optimizer modes: {['with optimization' if m else 'without optimization' for m in optimizer_modes]}\n")
        
        run_count = 0
        skipped_count = 0
        for n in selected_n:
            for use_optimizer in optimizer_modes:
                for combo in bool_combinations:
                    run_count += 1
                    params = dict(zip(bool_params, combo))
                    
                    # Add circle and optimized to params for comparison
                    params_with_meta = params.copy()
                    params_with_meta['circle'] = True
                    params_with_meta['optimized'] = use_optimizer
                    
                    # Check if this configuration already exists
                    if config_exists_in_json(n, params_with_meta):
                        skipped_count += 1
                        print(f"[{run_count}/{total_runs}] n={n}, opt={use_optimizer}, params={params} - SKIPPED (already exists)")
                        print("-" * 60 + "\n")
                        continue
                    
                    print(f"[{run_count}/{total_runs}] n={n}, opt={use_optimizer}, params={params}")
                    print("=" * 60)
                    
                    result = run_n20_optimizer(n, **params, use_optimizer=use_optimizer, return_result=True)
                    results.append(result)
                    
                    if isinstance(result, dict):
                        if use_optimizer:
                            # Check both n20 and optimizer success
                            if result.get('n20_success') and result.get('optimizer_success'):
                                print(f"SUCCESS")
                            else:
                                print(f"FAILED")
                        else:
                            # Only check n20 success when not using optimizer
                            if result.get('n20_success'):
                                print(f"SUCCESS")
                            else:
                                print(f"FAILED")
                        
                        # Save result immediately after computation
                        save_results([result])
                        
                    print("-" * 60 + "\n")
        
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} configurations that already exist in JSON files")
                
        # Save all results
        if results:
            save_results(results)
            print(f"\nAll results saved! Total runs: {len(results)}")
        sys.exit(0)
    # If called with no arguments or --interactive, run interactive mode
    if (len(sys.argv) == 2 and sys.argv[1] in ['--interactive', '-i']):
        run_n20_interactive()
        sys.exit(0)
    
    # If called with --test flag, run tests
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        test_circle_method()
        sys.exit(0)
    
    # If called with --n20 flag, run n20 optimizer
    if len(sys.argv) >= 3 and sys.argv[1] == "--n20":
        try:
            n = int(sys.argv[2])
            if n % 2 != 0:
                print(f"n must be even, got {n}")
                sys.exit(1)
            
            # Parse optional boolean flags
            kwargs = {}
            for i in range(3, len(sys.argv)):
                arg = sys.argv[i]
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.lstrip('--')
                    if value.lower() in ['true', '1', 'yes']:
                        kwargs[key] = True
                    elif value.lower() in ['false', '0', 'no']:
                        kwargs[key] = False
                    else:
                        print(f"Invalid boolean value for {key}: {value}")
            
            success, output = run_n20_optimizer(n, **kwargs)
            sys.exit(0 if success else 1)
            
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # If called with --circle flag and n, use circle_method
    if len(sys.argv) == 3 and sys.argv[1] == "--circle":
        try:
            n = int(sys.argv[2])
            if n % 2 != 0:
                print(f"n must be even, got {n}")
                sys.exit(1)
            
            print(f"Generating schedule using circle_method for {n} teams")
            home, away = circle_method(n)
            
            print(f"\nHome schedule: {home}")
            print(f"Away schedule: {away}")
            
            # Display as formatted output
            n_weeks = n - 1
            n_periods = n // 2
            print(f"\nFormatted schedule:")
            print("=" * 60)
            idx = 0
            for p in range(n_periods):
                matches = []
                for w in range(n_weeks):
                    matches.append(f"({home[idx]}-{away[idx]})")
                    idx += 1
                print(f"Period {p+1}: {' '.join(matches)}")
            print("=" * 60)
            sys.exit(0)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    # Otherwise, run MiniZinc with file
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python simple_runner.py                                - Interactive mode (n20 optimizer)")
        print("  python simple_runner.py --interactive / -i             - Interactive mode (n20 optimizer)")
        print("  python simple_runner.py <mzn_file> <n_value>          - Run MiniZinc file")
        print("  python simple_runner.py --circle <n_value>            - Use circle_method function")
        print("  python simple_runner.py --n20 <n_value> [options]    - Run n20 optimizer")
        print("  python simple_runner.py --test                        - Test circle_method")
        print("\nOptions for --n20:")
        print("  --use_int_search=true/false")
        print("  --use_restart_luby=true/false")
        print("  --use_relax_and_reconstruct=true/false")
        print("  --chuffed=true/false")
        print("  --symm_brake=true/false")
        print("  --ic_diff_match_in_week=true/false")
        
        print("  --use_optimizer=true/false  (default: true, runs optimizer_2.0.mzn)")
        print("\nExamples:")
        print("  python simple_runner.py")
        print("  python simple_runner.py ../../source/CP/CP_cyrcle_method.mzn 6")
        print("  python simple_runner.py --circle 8")
        print("  python simple_runner.py --n20 6")
        print("  python simple_runner.py --n20 8 --chuffed=true --symm_brake=true")
        print("  python simple_runner.py --test")
        sys.exit(1)
    
    mzn_file = sys.argv[1]
    try:
        n_value = int(sys.argv[2])
        if n_value % 2 != 0:
            print(f"Warning: n={n_value} is odd (should be even)")
    except ValueError:
        print(f"Second argument must be integer, got: {sys.argv[2]}")
        sys.exit(1)
    
    success = run_minizinc(mzn_file, n_value)
    sys.exit(0 if success else 1)

