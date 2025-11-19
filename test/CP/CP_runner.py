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
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class MinizincRunner:
    def __init__(self, timeout_seconds: int = 300):
        self.minizinc_path = self._find_minizinc()
        self.timeout_seconds = timeout_seconds
        self.n_values = [2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.bool_params = [
            'sb_weeks', 'sb_periods', 'sb_teams', 
            'ic_matches_per_team', 'ic_period_count',
            'use_int_search', 'use_restart_luby', 
            'use_relax_and_reconstruct', 'chuffed'
        ]
    
    def _find_minizinc(self) -> str:
        """Trova il percorso di MiniZinc sul sistema"""
       
        common_paths = [
            "minizinc",  
            "D:\\Program\\MiniZinc\\minizinc.exe",  
            "C:\\Program Files\\MiniZinc\\bin\\minizinc.exe",
            "C:\\Program Files\\MiniZinc\\minizinc.exe",
            "/usr/bin/minizinc",
            "/usr/local/bin/minizinc"
        ]
        
        for path in common_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue
                
        print("Error: MiniZinc not found. Verify it is installed and in the PATH.")
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
                [self.minizinc_path, '--solver', 'Gecode', model_file, data_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                execution_time = time.time() - start_time
                
                if process.returncode == 0:
                    return True, stdout, execution_time, ""
                else:
                    return False, "", execution_time, stderr
                    
            except subprocess.TimeoutExpired:
                print(f"Timeout reached, terminating process...")
                process.terminate()
                
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
                    process.kill()
                except:
                    pass
            return False, "", execution_time, str(e)
        finally:
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
            
    def run_pipeline(self, n: int, params: Dict[str, bool]) -> List[Dict]:
        """Run the complete pipeline: CP_3.0.mzn -> optimizer_2.0.mzn
        Returns a list with two results: [feasible_result, optimized_result]"""
        
        # Base result template
        base_result = {
            'n': n,
            'params': params,
            'cp_success': False,
            'cp_time': 0,
            'cp_error': '',
            'optimizer_success': False,
            'optimizer_time': 0,
            'optimizer_error': '',
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
        source_path = os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'CP')
        cp_path = os.path.join(source_path, "CP_3.0.mzn")
        cp_success, cp_output, cp_time, cp_error = self.run_minizinc(
            cp_path, data_content, remaining_time
        )
        
        # Create feasible result (from CP_3.0.mzn)
        feasible_result = base_result.copy()
        feasible_result['params'] = params.copy()
        feasible_result['params']['phase'] = 'feasible'  # Mark as feasible phase
        
        if 'UNSATISFIABLE' in cp_output:
            feasible_result.update({
                'cp_success': False,
                'cp_time': round(cp_time, 2),
                'cp_error': cp_error,
                'time': int(cp_time),
                'sol': []
            })
            
            # Create optimization result (failed because CP failed)
            optimization_result = base_result.copy()
            optimization_result['params'] = params.copy()
            optimization_result['params']['phase'] = 'optimization'
            optimization_result.update({
                'cp_success': False,
                'cp_time': round(cp_time, 2),
                'cp_error': cp_error,
                'optimizer_success': False,
                'optimizer_time': 0,
                'optimal': False,
                'time': int(cp_time),
                'sol': []
            })

            return [feasible_result, optimization_result]
        
        # Extract feasible solution from CP_3.0.mzn output
        cp_feasible_solution = self.convert_cp_output_to_matrix(cp_output, n)
        
        # Update feasible result
        if cp_time >= self.timeout_seconds:
            feasible_result['timeout_reached'] = True
        
        feasible_result.update({
            'cp_success': cp_success,
            'cp_time': round(cp_time, 2),
            'cp_error': cp_error,
            'time': int(cp_time),
            'sol': cp_feasible_solution if cp_feasible_solution else []
        })
        
        if not cp_success:
            print(f"CP_3.0.mzn failed: {cp_error}")
            
            # Create optimization result (failed because CP failed)
            optimization_result = base_result.copy()
            optimization_result['params'] = params.copy()
            optimization_result['params']['phase'] = 'optimization'
            optimization_result.update({
                'cp_success': False,
                'cp_time': round(cp_time, 2),
                'cp_error': cp_error,
                'optimizer_success': False,
                'optimizer_time': 0,
                'optimal': False,
                'time': int(cp_time),
                'sol': []
            })
            
            if cp_time >= self.timeout_seconds:
                feasible_result['timeout_reached'] = True
                optimization_result['timeout_reached'] = True
                
            return [feasible_result, optimization_result]
            
        # Check remaining time
        remaining_time -= cp_time
        if remaining_time <= 0:
            print(f"Timeout reached after CP_3.0.mzn")
            feasible_result['timeout_reached'] = True
            feasible_result['time'] = self.timeout_seconds
            
            # Create optimization result (failed due to timeout)
            optimization_result = base_result.copy()
            optimization_result['params'] = params.copy()
            optimization_result['params']['phase'] = 'optimization'
            optimization_result.update({
                'cp_success': cp_success,
                'cp_time': round(cp_time, 2),
                'cp_error': cp_error,
                'optimizer_success': False,
                'optimizer_time': 0,
                'optimizer_error': 'Timeout after CP phase',
                'optimal': False,
                'timeout_reached': True,
                'time': self.timeout_seconds,
                'sol': []
            })
            
            return [feasible_result, optimization_result]
            
        # Step 2: Extract data for optimizer
        optimizer_data = self.extract_optimizer_data(cp_output)
        
        if not optimizer_data:
            print("Failed to extract optimizer data from CP_3.0.mzn output")
            
            # Create optimization result (failed to extract data)
            optimization_result = base_result.copy()
            optimization_result['params'] = params.copy()
            optimization_result['params']['phase'] = 'optimization'
            optimization_result.update({
                'cp_success': cp_success,
                'cp_time': round(cp_time, 2),
                'cp_error': cp_error,
                'optimizer_success': False,
                'optimizer_time': 0,
                'optimizer_error': "Failed to extract optimizer data",
                'optimal': False,
                'time': int(cp_time),
                'sol': []
            })
            
            return [feasible_result, optimization_result]
            
        # Step 3: Run optimizer_2.0.mzn
        print(f"Running optimizer_2.0.mzn for n={n}...")
        source_path = os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'CP')
        cp_opt_path = os.path.join(source_path,"optimizer_2.0.mzn")
        opt_success, opt_output, opt_time, opt_error = self.run_minizinc(
            cp_opt_path, optimizer_data, int(remaining_time)
        )
        
        # Create optimization result
        optimization_result = base_result.copy()
        optimization_result['params'] = params.copy()
        optimization_result['params']['phase'] = 'optimization'
        optimization_result.update({
            'cp_success': cp_success,
            'cp_time': round(cp_time, 2),
            'cp_error': cp_error,
            'optimizer_success': opt_success,
            'optimizer_time': round(opt_time, 2),
            'optimizer_error': opt_error,
            'time': int(cp_time + opt_time)
        })
        
        # Handle optimizer timeout or failure
        if not opt_success:
            print(f"Optimizer failed or timed out: {opt_error}")
            
            optimization_result.update({
                'optimal': False,
                'obj': None,
                'sol': []
            })
            
            if optimization_result['time'] >= self.timeout_seconds:
                optimization_result['timeout_reached'] = True
            
            return [feasible_result, optimization_result]
        
        # If optimizer succeeded, process its output
        summary, optimized_sol = self.parse_optimized_matrix_to_solution(optimizer_output=opt_output, n_teams=n)
        
        value = None
        if summary:
            for item in summary:
                if "Optimized Max Imbalance:" in item:
                    value = int(item.split(": ")[1])
                    break
        
        optimization_result.update({
            'optimal': value == 1,
            'obj': value,
            'sol': optimized_sol if optimized_sol else []
        })
        
        if optimization_result['time'] >= self.timeout_seconds:
            optimization_result['timeout_reached'] = True
            optimization_result['optimal'] = False
            
        return [feasible_result, optimization_result]

    def save_results(self, results: List[Dict]):
        """Save results to JSON file with timestamp structure in res/CP relative to script"""
        
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'CP'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
            # Add new results with current timestamp
            for result in group_results:
                # Create result name based on parameters and phase
                res_name = ""
                if result["params"]["chuffed"] == True:
                    res_name += "chuffed"
                else:
                    res_name += "gecode"
                phase = result["params"].get("phase", "unknown")
                if phase == "feasible":
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
                if result["params"]["use_int_search"] == True:
                    res_name += "_usint"
                if result["params"]["use_restart_luby"] == True:
                    res_name += "_usrest"
                if result["params"]["use_relax_and_reconstruct"] == True:
                    res_name += "_usrelax"
                                
                existing_data[res_name] = result  # Direttamente l'oggetto, non un array
            # Save back to file with custom formatting
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('{\n')
                results_key = list(existing_data.keys())
                for i, res_key in enumerate(results_key):
                    f.write(f'  "{res_key}": {{\n')

                    result = existing_data[res_key]
                    items = list(result.items())
                    for j, (key, value) in enumerate(items):
                        f.write(f'    "{key}": ')
                        # Handle 'sol' with custom formatting
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
                    if i < len(results_key) - 1:
                        f.write(',')
                    f.write('\n')

                f.write('}\n')

            print(f"Results for n={n_value} saved to: {filename}")
      
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
            
    def run(self, selected_n, combination, params, summary=False):
        """Main execution method"""
        try:
            if combination == False:
                config = {'manual': params}
                """Run with manually specified parameters"""
                results = []
        
                for i, n in enumerate(selected_n, 1):
                    print(f"[{i}/{len(selected_n)}] Running n={n}")
            
                    pipeline_results = self.run_pipeline(n, params)  # Returns [feasible_result, optimization_result]
                    results.extend(pipeline_results)  # Add both results to the list
            
                    # Print summary for both results
                    feasible_result, optimization_result = pipeline_results
                    
                    print(f"FEASIBLE - CP: {'OK' if feasible_result['cp_success'] else 'FAIL'}, Time: {feasible_result['time']:.2f}s")
                    
                    if optimization_result['optimizer_success']:
                        print(f"OPTIMIZATION - SUCCESS, Total time: {optimization_result['time']:.2f}s, Optimal: {optimization_result['optimal']}")
                    elif optimization_result['timeout_reached']:
                        print(f"OPTIMIZATION - TIMEOUT, Reached {self.timeout_seconds}s limit")
                    else:
                        print(f"OPTIMIZATION - FAILED, Error: {optimization_result.get('optimizer_error', 'Unknown error')}")
                      
                    print("-" * 60)
                if summary:
                    self.print_summary(results)
                # Save results
                self.save_results(results)
            else:
                config = {'all_combinations': True}
                """Run all possible combinations of boolean parameters"""
                results = []
        
                # Generate all combinations of boolean values
                bool_combinations = list(itertools.product([True, False], repeat=len(self.bool_params)))
                total_runs = len(selected_n) * len(bool_combinations)
        
                run_count = 0
                for n in selected_n:
                    for combo in bool_combinations:
                        run_count += 1
                        params = dict(zip(self.bool_params, combo))
                
                        print(f"[{run_count}/{total_runs}] n={n}")
                
                        pipeline_results = self.run_pipeline(n, params)  # Returns [feasible_result, optimization_result]

                        self.save_results(pipeline_results)  # Save both results
                        #results.append(result)
                
                        # Print summary for both results
                        feasible_result, optimization_result = pipeline_results
                        
                        print(f"FEASIBLE - CP: {'OK' if feasible_result['cp_success'] else 'FAIL'}, Time: {feasible_result['time']:.2f}s")
                        
                        if optimization_result['optimizer_success']:
                            print(f"OPTIMIZATION - SUCCESS, Total time: {optimization_result['time']:.2f}s, Optimal: {optimization_result['optimal']}")
                        elif optimization_result['timeout_reached']:
                            print(f"OPTIMIZATION - TIMEOUT, Reached {self.timeout_seconds}s limit")
                        else:
                            print(f"OPTIMIZATION - FAILED, Error: {optimization_result.get('optimizer_error', 'Unknown error')}")
                
                        print("-" * 60)
                
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)

def main():
    """Main function to run the CP solver."""
    if len(sys.argv) < 2:
        team_sizes = [2, 4, 6]
        symmetry_combinations = "all" ## CAPIRE
        params = {
                "sb_weeks": True,
                "sb_periods": True,
                "sb_teams": True,
                "ic_matches_per_team": True,
                "ic_period_count": True,
                "use_int_search": True,
                "use_restart_luby": True,
                "use_relax_and_reconstruct": True,
                "chuffed": True
                }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        time_limit = 300
        summary = False
    else:
        # Parse arguments
        team_sizes = []
        params = {
                "sb_weeks": False,
                "sb_periods": False,
                "sb_teams": False,
                "ic_matches_per_team": False,
                "ic_period_count": False,
                "use_int_search": False,
                "use_restart_luby": False,
                "use_relax_and_reconstruct": False,
                "chuffed": False
                }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        time_limit = 300
        summary = False
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
                "sb_weeks": True,
                "sb_periods": True,
                "sb_teams": True,
                "ic_matches_per_team": True,
                "ic_period_count": True,
                "use_int_search": True,
                "use_restart_luby": True,
                "use_relax_and_reconstruct": True,
                "chuffed": True
                }
                symmetry_combinations = False
            elif arg.lower() == "--no-optional":
                params = {
                    "sb_weeks": False,
                    "sb_periods": False,
                    "sb_teams": False,
                    "ic_matches_per_team": False,
                    "ic_period_count": False,
                    "use_int_search": False,
                    "use_restart_luby": False,
                    "use_relax_and_reconstruct": False,
                    "chuffed": False
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
                    "use_int_search": False,
                    "use_restart_luby": False,
                    "use_relax_and_reconstruct": False,
                    "chuffed": False
                    }
                k = i+1
                for j in range(k, len(sys.argv)):
                    i = j
                    symmetry = sys.argv[j]
                    valid_symmetry = ["sb_weeks", "sb_periods", "sb_teams", "ic_matches_per_team", "ic_period_count", "use_int_search", "use_restart_luby", "use_relax_and_reconstruct", "chuffed"]
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
                            elif symmetry.lower() == "ic_period_count":
                                params["ic_period_count"] = True
                            elif symmetry.lower() == "use_int_search":
                                params["use_int_search"] = True
                            elif symmetry.lower() == "use_restart_luby":
                                params["use_restart_luby"] = True
                            elif symmetry.lower() == "use_relax_and_reconstruct":
                                params["use_relax_and_reconstruct"] = True
                            else:
                                params["chuffed"] = True
            elif arg.lower() == "summary":
                summary = True
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
    
    runner = MinizincRunner(timeout_seconds=300)
    runner.run(team_sizes, symmetry_combinations, params, summary)

if __name__ == "__main__":
    main()