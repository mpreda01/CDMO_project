#!/usr/bin/env python3
"""
MiniZinc Tournament Scheduling Runner
Executes CP_3.0.mzn followed by optimizer_2.0.mzn with configurable parameters
"""

import subprocess
import re
import time
import tempfile
import os
import itertools
import random
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import psutil
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


class MinizincRunner:
    def __init__(self, timeout_seconds: int = 300):
        self.minizinc_path = self._find_minizinc()
        self.timeout_seconds = timeout_seconds
        self.n_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.bool_params = [
            'symmetry_break', 'implied_constraint',
            'use_int_search', 'use_restart_luby', 
            'use_relax_and_reconstruct', 'chuffed'
        ]
    
    def _find_minizinc(self) -> str:
        """Trova il percorso di MiniZinc sul sistema"""
       
        common_paths = [
            "minizinc",  
            "D:\\Program\\MiniZinc\\minizinc.exe",  
            "C:\\Program Files\\MiniZinc\\bin\\minizinc.exe",
            "/usr/bin/minizinc",
            "/usr/local/bin/minizinc"
        ]
        
        for path in common_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"✓ MiniZinc trovato: {path}")
                    return path
            except:
                continue
                
        print("Error: MiniZinc non trovato. Assicurati che sia installato e nel PATH.")
        sys.exit(1)
    def convert_cp_output_to_matrix(self, cp_output: str, n_teams: int) -> Optional[List[List[List[int]]]]:
        """Convert CP_3.0 output to solution matrix format"""
        try:
            lines = cp_output.strip().split('\n')
            home_array = None
            away_array = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('home0 ='):
                    # Extract array from "home0 = [1, 2, 3, ...];"
                    array_str = line.split('=')[1].strip().rstrip(';')
                    home_array = eval(array_str)  
                elif line.startswith('away0 ='):
                    array_str = line.split('=')[1].strip().rstrip(';')
                    away_array = eval(array_str)
            
            if not home_array or not away_array:
                return None
                
            n_periods = n_teams // 2
            n_weeks = n_teams - 1
            
            # Convert flat arrays to matrix format
            solution = []
            for p in range(n_periods):
                period_matches = []
                for w in range(n_weeks):
                    index = p * n_weeks + w
                    if index < len(home_array) and index < len(away_array):
                        period_matches.append([home_array[index], away_array[index]])
                if period_matches:
                    solution.append(period_matches)
            
            return solution
            
        except Exception as e:
            print(f"Error converting CP output to matrix: {e}")
            return None
        
    def create_data_file(self, n: int, params: Dict[str, bool]) -> str:
        """Create a .dzn data file with the given parameters"""
        content = f"n = {n};\n"
        for param, value in params.items():
            content += f"{param} = {'true' if value else 'false'};\n"
        return content
        
    def run_minizinc(self, model_file: str, data_content: str, timeout: int) -> Tuple[bool, str, float, str]:
        """Run a MiniZinc model with given data with robust timeout handling"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dzn', delete=False) as f:
            f.write(data_content)
            data_file = f.name
            
        process = None
        try:
            start_time = time.time()
            process = subprocess.Popen(
                [self.minizinc_path, '--solver', 'Gecode', '-r', '42', model_file, data_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                execution_time = time.time() - start_time
                
                # Kill any remaining child processes
                kill_process_tree(process)
                
                if process.returncode == 0:
                    return True, stdout, execution_time, ""
                else:
                    return False, "", execution_time, stderr
                    
            except subprocess.TimeoutExpired:
                print(f"Timeout reached, terminating process...")
                kill_process_tree(process)
                
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print(f"Force killing process...")
                    process.kill()
                    process.wait()
                
                execution_time = timeout
                return False, "", execution_time, f"Timeout after {timeout} seconds"
                
        except Exception as e:
            execution_time = time.time() - start_time
            if process:
                try:
                    kill_process_tree(process)
                except:
                    pass
            return False, "", execution_time, str(e)
        finally:
            # Ensure process tree is killed
            if process is not None:
                kill_process_tree(process)
            try:
                os.unlink(data_file)
            except:
                pass
                
    def extract_optimizer_data(self, cp_output: str) -> Optional[str]:
        """Extract n_teams, home0, and away0 from CP_3.0.mzn output"""
        lines = cp_output.strip().split('\n')
        optimizer_data = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('n_teams =') or line.startswith('home0 =') or line.startswith('away0 ='):
                optimizer_data.append(line)
                
        if len(optimizer_data) >= 3:
            return '\n'.join(optimizer_data) + '\n'
        return None

    def parse_optimized_matrix_to_solution(self, optimizer_output: str, n_teams: int) -> Tuple[Optional[List[str]], Optional[List[List[List[int]]]]]:
        """Convert optimizer_2.0 output to solution matrix format for json"""
        try:
            start_marker = "=== OPTIMIZED TOURNAMENT MATRIX ==="
            lines = optimizer_output.split('\n')
            
            summary = []
            matrix_lines = []
            found_start = False
            
            if len(lines) >= 4:
                summary.append(lines[0].strip())
                summary.append(lines[1].strip())
                summary.append(lines[2].strip())
                summary.append(lines[3].strip())

            for line in lines:
                if start_marker in line:
                    found_start = True
                    continue
                    
                if found_start and line.strip().startswith('[['):
                    matrix_lines.append(line.strip())
                elif found_start and (line.strip().startswith('===') or line.strip().startswith('---')):
                    break
            
            if not matrix_lines:
                return summary if summary else None, None
            
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
            
            return summary if summary else None, solution if solution else None
            
        except Exception as e:
            print(f"Error while parsing optimized solution matrix: {e}")
            return None, None
    
    def run_pipeline(self, n: int, params: Dict[str, bool], use_optimizer: bool = True) -> Dict:
        """Run the complete pipeline: CP_3.0.mzn with optional optimizer_2.0.mzn"""
        # Add circle and optimized parameters (always False and use_optimizer for CP_runner.py)
        params_with_circle = params.copy()
        params_with_circle['circle'] = False
        params_with_circle['optimized'] = use_optimizer
        
        result = {
            'n': n,
            'params': params_with_circle,
            'cp_success': False,
            'cp_time': 0,
            'optimizer_success': False,
            'optimizer_time': 0,
            'time': 0,
            'timeout_reached': False,
            'optimal': False,
            'obj': 'None',
            'sol': []
        }
        
        # Step 1: Run CP_3.0.mzn
        print(f"Running CP_3.0.mzn for n={n}...")
        data_content = self.create_data_file(n, params)
        
        remaining_time = self.timeout_seconds
        cp_success, cp_output, cp_time, cp_error = self.run_minizinc(
            '../../source/CP/CP_3.0.mzn', data_content, remaining_time
        )
        print(type(cp_output))
        if 'UNSATISFIABLE' in cp_output:
            result.update({
            'cp_success': False,
            'cp_time': round(cp_time, 2)
            })

            result.update({
            'optimizer_success': False,
            'optimizer_time': 0,
            'optimal': True if not use_optimizer else False,
            'time': int(cp_time)
            })

            return result
        
        if result['time'] >= self.timeout_seconds:
            result['timeout_reached'] = True
        else:
            result.update({
                'cp_success': cp_success,
                'cp_time': round(cp_time, 2)
            })
        
        if not cp_success:
            print(f"CP_3.0.mzn failed: {cp_error}")
            result['time'] = cp_time
            result['optimal'] = True if not use_optimizer else False
            remaining_time -= cp_time
            if remaining_time <= 0:
                print(f"Timeout reached after CP_3.0.mzn")
                result['timeout_reached'] = True
                result['time'] = self.timeout_seconds
            return result
            
        # Check remaining time
        remaining_time -= cp_time
        if remaining_time <= 0:
            print(f"Timeout reached after CP_3.0.mzn")
            result['timeout_reached'] = True
            result['time'] = self.timeout_seconds
            # If not using optimizer, CP solution is still valid
            if not use_optimizer:
                cp_solution = self.convert_cp_output_to_matrix(cp_output, n)
                result['optimal'] = True
                result['obj'] = "None"
                result['sol'] = cp_solution if cp_solution else []
            return result
        
        # If not using optimizer, convert CP output to solution and return
        if not use_optimizer:
            print("Optimizer disabled, converting CP solution to matrix format...")
            cp_solution = self.convert_cp_output_to_matrix(cp_output, n)
            result.update({
                'optimizer_success': False,
                'optimizer_time': 0,
                'obj': "None",
                'optimal': True,
                'sol': cp_solution if cp_solution else [],
                'time': int(cp_time)
            })
            return result
            
        # Extract data for optimizer
        optimizer_data = self.extract_optimizer_data(cp_output)
        if not optimizer_data:
            print("Failed to extract optimizer data from CP_3.0.mzn output")
            result['time'] = cp_time
            return result
            
        # Run optimizer_2.0.mzn
        print(f"Running optimizer_2.0.mzn for n={n}...")
        opt_success, opt_output, opt_time, opt_error = self.run_minizinc(
            '../../source/CP/optimizer_2.0.mzn', optimizer_data, int(remaining_time)
        )
        
        # Handle optimizer timeout or failure
        if not opt_success:
            print(f"Optimizer failed or timed out: {opt_error}")
            # Convert CP output to solution matrix
            cp_solution = self.convert_cp_output_to_matrix(cp_output, n)
            
            result.update({
                'optimizer_success': False,
                'optimizer_time': round(opt_time, 2),
                'obj': None,
                'optimal': False,
                'sol': cp_solution if cp_solution else "Failed to parse CP output",
                'time': int(cp_time + opt_time)
            })
            
            if result['time'] >= self.timeout_seconds:
                result['timeout_reached'] = True
            
            return result
        
        # If optimizer succeeded, process its output
        summary, sol = self.parse_optimized_matrix_to_solution(optimizer_output=opt_output, n_teams=n)
        
        value = None
        if summary:
            for item in summary:
                if "Optimized Max Imbalance:" in item:
                    value = int(item.split(": ")[1])
                    break
        
        if value == 1:
            result.update({
                'optimizer_success': opt_success,
                'optimizer_time': round(opt_time, 2),
                'obj': value,
                'optimal': True,
                'sol': sol,
                'time': int(cp_time + opt_time)
            })
        else:
            result.update({
                'optimizer_success': opt_success,
                'optimizer_time': round(opt_time, 2),
                'obj': value,
                'optimal': False,
                'sol': sol,
                'time': int(cp_time + opt_time)
            })
        
        if result['time'] >= self.timeout_seconds:
            result['timeout_reached'] = True
            result['optimal'] = False
            
        return result
        
    def get_user_input(self) -> Tuple[List[int], Dict[str, bool]]:
        """Get user input for parameters"""
        print("=== MiniZinc Tournament Scheduler Runner ===\n")
        
        # Ask about optimization
        use_opt_input = input("Use optimizer_2.0.mzn for optimization? (yes/no/both, default: yes): ").strip().lower()
        if use_opt_input in ['both', 'b', 'all']:
            optimizer_modes = [True, False]  # Both with and without optimization
            print(f"Optimizer modes: both with and without optimization\n")
        elif use_opt_input in ['no', 'n', 'false', 'f', '0', 'without']:
            optimizer_modes = [False]  # Only without optimization
            print(f"Optimizer modes: without optimization\n")
        else:
            optimizer_modes = [True]  # Only with optimization
            print(f"Optimizer modes: with optimization\n")
        
        # Get n values
        print("Available n values:", self.n_values)
        n_input = input("Enter n values (comma-separated, or 'all' for all values): ").strip()
        
        if n_input.lower() == 'all':
            selected_n = self.n_values
        else:
            try:
                selected_n = [int(x.strip()) for x in n_input.split(',')]
                selected_n = [n for n in selected_n if n in self.n_values]
                if not selected_n:
                    print("No valid n values selected, using all.")
                    selected_n = self.n_values
            except:
                print("Invalid input, using all n values.")
                selected_n = self.n_values
                
        # Get boolean parameters
        print(f"\nBoolean parameters: {self.bool_params}")
        mode = input("Enter 'manual' for manual input or 'all' for all combinations: ").strip().lower()
        
        if mode == 'manual':
            params = {}
            for param in self.bool_params:
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
            return selected_n, {'manual': params, 'optimizer_modes': optimizer_modes}
        else:
            return selected_n, {'all_combinations': True, 'optimizer_modes': optimizer_modes}
    
    def config_exists_in_json(self, n: int, params: Dict[str, bool]) -> bool:
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
    
    def load_previous_timeouts(self, selected_n: List[int], optimizer_modes: List[bool]) -> Dict:
        """Load previous timeout configurations from JSON files for selected n values and all smaller n values"""
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'CP'))
        timed_out_configs = {}
        
        if not os.path.exists(output_dir):
            return timed_out_configs
        
        print("Loading previous timeout data from JSON files...")
        
        # Determine which n values to check (selected n values + all smaller n values for context)
        max_n = max(selected_n) if selected_n else 0
        all_n_to_check = sorted(set([n for n in self.n_values if n <= max_n]))
        
        # Check each n value's JSON file
        for n in all_n_to_check:
            filename = os.path.join(output_dir, f"{n}.json")
            
            if not os.path.exists(filename):
                continue
                
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Check each result for timeouts
                for timestamp, result in existing_data.items():
                    if result.get('timeout_reached', False) and 'params' in result:
                        params = result['params']
                        use_optimizer = params.get('optimized', True)
                        
                        # Only track if this optimizer mode is in our current run
                        if use_optimizer not in optimizer_modes:
                            continue
                        
                        # Create params dict without metadata
                        param_dict = {k: v for k, v in params.items() if k not in ['circle', 'optimized']}
                        params_tuple = tuple(sorted(param_dict.items()))
                        config_key = (use_optimizer, params_tuple)
                        
                        # Store the smallest n that timed out for this config
                        if config_key not in timed_out_configs or n < timed_out_configs[config_key]:
                            timed_out_configs[config_key] = n
                            
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {filename}: {e}")
                continue
        
        if timed_out_configs:
            print(f"Found {len(timed_out_configs)} configurations that previously timed out")
        else:
            print("No previous timeout data found")
            
        return timed_out_configs
                
        
        if timed_out_configs:
            print(f"Found {len(timed_out_configs)} configurations that previously timed out")
        else:
            print("No previous timeout data found")
            
        return timed_out_configs
            
    def run_all_combinations(self, selected_n: List[int], optimizer_modes: List[bool] = None) -> List[Dict]:
        """Run all possible combinations of boolean parameters"""
        results = []
        
        if optimizer_modes is None:
            optimizer_modes = [True]  # Default for backward compatibility
        
        # Sort n values to ensure we process from smallest to largest
        selected_n = sorted(selected_n)
        
        # Load previous timeout data from JSON files
        timed_out_configs = self.load_previous_timeouts(selected_n, optimizer_modes)
        
        # Generate all combinations of boolean values
        bool_combinations = list(itertools.product([True, False], repeat=len(self.bool_params)))
        total_runs = len(selected_n) * len(bool_combinations) * len(optimizer_modes)
        
        print(f"\nRunning {total_runs} total combinations...")
        print(f"Optimizer modes: {['with optimization' if m else 'without optimization' for m in optimizer_modes]}")
        print(f"Timeout per combination: {self.timeout_seconds} seconds")
        print(f"Note: Configurations that previously timed out (for same or smaller n) will be skipped\n")
        
        run_count = 0
        skipped_count = 0
        timeout_skipped_count = 0
        
        for n in selected_n:
            for use_optimizer in optimizer_modes:
                for combo in bool_combinations:
                    run_count += 1
                    params = dict(zip(self.bool_params, combo))
                    
                    # Create a hashable key for this configuration
                    params_tuple = tuple(sorted(params.items()))
                    config_key = (use_optimizer, params_tuple)
                    
                    # Check if this configuration timed out for a smaller n
                    if config_key in timed_out_configs:
                        timeout_n = timed_out_configs[config_key]
                        if n >= timeout_n:
                            timeout_skipped_count += 1
                            print(f"[{run_count}/{total_runs}] n={n}, opt={use_optimizer} - SKIPPED (timed out at n={timeout_n})")
                            print("-" * 60)
                            continue
                    
                    # Add circle and optimized to params for comparison
                    params_with_meta = params.copy()
                    params_with_meta['circle'] = False
                    params_with_meta['optimized'] = use_optimizer
                    
                    # Check if this configuration already exists
                    if self.config_exists_in_json(n, params_with_meta):
                        skipped_count += 1
                        print(f"[{run_count}/{total_runs}] n={n}, opt={use_optimizer} - SKIPPED (already exists)")
                        print("-" * 60)
                        continue
                    
                    print(f"[{run_count}/{total_runs}] n={n}, opt={use_optimizer}")
                    print(f"  params={params}")
                    
                    result = self.run_pipeline(n, params, use_optimizer)
                    self.save_results([result])
                    
                    # Check if this run timed out
                    if result.get('timeout_reached', False):
                        timed_out_configs[config_key] = n
                        print(f"TIMEOUT - Reached {self.timeout_seconds}s limit (will skip for n>{n})")
                    else:
                        # Print summary based on optimizer usage
                        if use_optimizer:
                            # Check both CP and optimizer success
                            if result['cp_success'] and result['optimizer_success']:
                                print(f"SUCCESS - Total time: {result['time']:.2f}s")
                            else:
                                print(f"FAILED - CP: {'OK' if result['cp_success'] else 'FAIL'}, "
                                      f"OPT: {'OK' if result['optimizer_success'] else 'FAIL'}")
                        else:
                            # Only check CP success when not using optimizer
                            if result['cp_success']:
                                print(f"SUCCESS - Total time: {result['time']:.2f}s")
                            else:
                                print(f"FAILED - CP: {'OK' if result['cp_success'] else 'FAIL'}")
                    
                    print("-" * 60)
        
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} configurations that already exist in JSON files")
        if timeout_skipped_count > 0:
            print(f"Skipped {timeout_skipped_count} configurations due to previous timeouts on smaller n values")
                
        #return results
        
    def run_manual(self, selected_n: List[int], params: Dict[str, bool], optimizer_modes: List[bool] = None) -> List[Dict]:
        """Run with manually specified parameters"""
        results = []
        
        if optimizer_modes is None:
            optimizer_modes = [True]
        
        total_runs = len(selected_n) * len(optimizer_modes)
        print(f"\nRunning {total_runs} configurations...")
        print(f"Parameters: {params}")
        print(f"Optimizer modes: {['with optimization' if m else 'without optimization' for m in optimizer_modes]}")
        print(f"Timeout per run: {self.timeout_seconds} seconds\n")
        
        run_count = 0
        for n in selected_n:
            for use_optimizer in optimizer_modes:
                run_count += 1
                print(f"[{run_count}/{total_runs}] Running n={n}, opt={use_optimizer}")
                
                result = self.run_pipeline(n, params, use_optimizer)
                results.append(result)
                
                # Print summary based on optimizer usage
                if use_optimizer:
                    # Check both CP and optimizer success
                    if result['cp_success'] and result['optimizer_success']:
                        print(f"SUCCESS - Total time: {result['time']:.2f}s")
                    elif result['timeout_reached']:
                        print(f"TIMEOUT - Reached {self.timeout_seconds}s limit")
                    else:
                        print(f"FAILED - CP: {'OK' if result['cp_success'] else 'FAIL'}, "
                              f"OPT: {'OK' if result['optimizer_success'] else 'FAIL'}")
                else:
                    # Only check CP success when not using optimizer
                    if result['cp_success']:
                        print(f"SUCCESS - Total time: {result['time']:.2f}s")
                    elif result['timeout_reached']:
                        print(f"TIMEOUT - Reached {self.timeout_seconds}s limit")
                    else:
                        print(f"FAILED - CP: {'OK' if result['cp_success'] else 'FAIL'}")
                          
                print("-" * 60)
            
        return results

    def save_results(self, results: List[Dict]):
        """Save results to JSON file with timestamp structure in res/CP relative to script"""
        print('saving...')

        
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


            
        
    def print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        total_runs = len(results)
        successful_runs = sum(1 for r in results if r['cp_success'] and r['optimizer_success'])
        timeout_runs = sum(1 for r in results if r['timeout_reached'])
        
        print(f"\n=== SUMMARY ===")
        print(f"Total runs: {total_runs}")
        print(f"Successful runs: {successful_runs}")
        print(f"Timeout runs: {timeout_runs}")
        print(f"Failed runs: {total_runs - successful_runs}")
        
        if successful_runs > 0:
            avg_time = sum(r['time'] for r in results if r['cp_success'] and r['optimizer_success']) / successful_runs
            print(f"Average execution time (successful): {avg_time:.2f}s")
            
        
        successful_results = [r for r in results if r['cp_success'] and r['optimizer_success']]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x['time'])
            print(f"Fastest successful run: n={fastest['n']}, time={fastest['time']:.2f}s")
            
    def run(self):
        """Main execution method"""
        try:
            selected_n, config = self.get_user_input()
            
            if 'manual' in config:
                optimizer_modes = config.get('optimizer_modes', [True])
                results = self.run_manual(selected_n, config['manual'], optimizer_modes)
            else:
                optimizer_modes = config.get('optimizer_modes', [True])
                results = self.run_all_combinations(selected_n, optimizer_modes)
                
            self.print_summary(results)
            
            # Save results
            self.save_results(results)
                
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    runner = MinizincRunner(timeout_seconds=300)
    runner.run()