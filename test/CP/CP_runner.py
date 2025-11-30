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
            'sb_weeks', 'sb_periods', 'sb_teams', 'sb_week1_fixed', 'symm_brake', 
            'ic_matches_per_team', 'ic_period_count', 'ic_diff_match_in_week', 'ic_diff_match_in_period',
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
        
    def create_data_file(self, n: int = None, params: Dict[str, bool] = None) -> str:
        """Create a .dzn data file with the given parameters"""
        content = ""
        if n:
            content = f"n = {n};\n"
        if params:
            for param, value in params.items():
                content += f"{param} = {'true' if value else 'false'};\n"
        return content

    def create_circle_file(self, n: int, params: Dict[str, bool], home0: list[int], away0: list[int]) -> str:
        """Create a .dzn data file with the given parameters"""
        content = f"n = {n};\n"
        for param, value in params.items():
            content += f"{param} = {'true' if value else 'false'};\n"
        content += f"home0 = [{', '.join(map(str, home0))}];\n"
        content += f"away0 = [{', '.join(map(str, away0))}];\n"
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
        #print(optimizer_data)  ## DEBUG       
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
            
    def run_pipeline(self, n: int, params: Dict[str, bool], run_decision: bool, run_optimization: bool, circle_method: bool) -> List[Dict]:
        """Run the complete pipeline: CP_3.0.mzn -> optimizer_2.0.mzn
        Returns a list with two results: [feasible_result, optimized_result]"""
        # Base result template
        base_result = {
            'time': 0,
            'method':'classic',
            'version':'decision',
            'params': params,
            'optimal': False,
            'stop_reason': "None",
            'obj': [],
        }
        
        # Run CP_3.0.mzn
        print(f"Running CP_3.0.mzn for n={n}...")
        classic_keys = ["sb_weeks", "sb_periods", "sb_teams", "sb_week1_fixed", "ic_matches_per_team", "ic_period_count", "use_int_search", "use_restart_luby", "use_relax_and_reconstruct", "chuffed"]
        classic_params = {k: params[k] for k in classic_keys}
        data_content = self.create_data_file(n, classic_params)
        remaining_time = self.timeout_seconds
        source_path = os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'CP')    
        cp_path = os.path.join(source_path, "CP_3.0.mzn")
        cp_success, cp_output, cp_time, cp_error = self.run_minizinc(
            cp_path, data_content, remaining_time
        )
        #print(data_content) ## DEBUG
        #print("\n") ## DEBUG
        #print(cp_output) ## DEBUG
        # Create feasible result (from CP_3.0.mzn)
        feasible_result = base_result.copy()
        optimization_result = base_result.copy()
        feasible_result["params"] = classic_params
        optimization_result["params"] = classic_params
        res = []
        if 'UNSATISFIABLE' in cp_output:
            if run_decision:
                feasible_result.update({
                    'optimal': True,
                    'stop_reason': 'infeasible',
                    'time': self.timeout_seconds
                })
                res.append(feasible_result)
            if run_optimization:
                optimization_result.update({
                    'version': 'optimal',
                    'optimal': True,
                    'stop_reason': 'infeasible',
                    'time': self.timeout_seconds
                })
                res.append(optimization_result)
            return res

        if not run_optimization:        
            # Extract feasible solution from CP_3.0.mzn output
            cp_feasible_solution = self.convert_cp_output_to_matrix(cp_output, n)
        
            # Update feasible result
            #if cp_time >= self.timeout_seconds:
            #    feasible_result['timeout_reached'] = True
        
            feasible_result.update({
                'time': int(cp_time),
                'optimal': True if cp_feasible_solution else False,
                'stop_reason': 'None' if cp_feasible_solution else 'unknown',
                'sol': cp_feasible_solution if cp_feasible_solution else []
            })
        
            if not cp_success:
                print(f"CP_3.0.mzn failed: {cp_error}")                
            
            res.append(feasible_result)
            
            # Check remaining time
            remaining_time -= cp_time
            if remaining_time <= 0:
                print(f"Timeout reached after CP_3.0.mzn")
                #feasible_result['timeout_reached'] = True
                feasible_result['time'] = self.timeout_seconds
                feasible_result['stop_reason'] = 'time_limit'
                res[0] = feasible_result
            
        else:
            if run_decision:
                # Extract feasible solution from CP_3.0.mzn output
                cp_feasible_solution = self.convert_cp_output_to_matrix(cp_output, n)
        
                # Update feasible result
                #if cp_time >= self.timeout_seconds:
                #    feasible_result['timeout_reached'] = True
        
                feasible_result.update({
                    'time': int(cp_time),
                    'optimal': True if cp_feasible_solution else False,
                    'stop_reason': 'None' if cp_feasible_solution else 'unknown',
                    'sol': cp_feasible_solution if cp_feasible_solution else []
                })
        
                #if not cp_success:
                #    print(f"CP_3.0.mzn failed: {cp_error}")                
            
                # Check remaining time
                remaining_time -= cp_time
                if remaining_time <= 0:
                    print(f"Timeout reached after CP_3.0.mzn")
                    feasible_result['time'] = self.timeout_seconds
                    feasible_result['stop_reason'] = 'time_limit'

                res.append(feasible_result)

            # Extract data for optimizer
            #print(cp_output) ## DEBUG
            optimizer_data = self.extract_optimizer_data(cp_output)
            #print(optimizer_data) ## DEBUG
            if not optimizer_data:
                print("Failed to extract optimizer data from CP_3.0.mzn output")
            
                # Update optimization result (failed to extract data)
                optimization_result.update({
                    'version': 'optimal',
                    'optimal': False,
                    'time': self.timeout_seconds,
                    'stop_reason': 'unknown'
                })
                res.append(optimization_result)
            
            # Run optimizer_2.0.mzn
            print(f"Running optimizer_2.0.mzn for n={n}...")
            source_path = os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'CP')
            cp_opt_path = os.path.join(source_path,"optimizer_2.0.mzn")
            opt_success, opt_output, opt_time, opt_error = self.run_minizinc(
                cp_opt_path, optimizer_data, int(remaining_time)
            )
            #print(optimizer_data) ## DEBUG
            #print("\n") ## DEBUG
            #print(opt_output) ## DEBUG
            optimization_result.update({
                'version': 'optimal',
                'time': int(cp_time + opt_time)
            })
            # Handle optimizer timeout or failure
            if not opt_success:
                print(f"Optimizer failed or timed out: {opt_error}")
            
                if optimization_result['time'] >= self.timeout_seconds:
                    optimization_result['time'] = self.timeout_seconds
                    optimization_result['stop_reason'] = 'time_limit'
                else:
                    optimization_result['stop_reason'] = 'unknown'
                res.append(optimization_result)
        
            # If optimizer succeeded, process its output
            summary, optimized_sol = self.parse_optimized_matrix_to_solution(optimizer_output=opt_output, n_teams=n)
        
            value = None
            if summary:
                for item in summary:
                    if "Optimized Max Imbalance:" in item:
                        value = int(item.split(": ")[1])
                        break
        
            optimization_result.update({
                'optimal': True,
                'obj': value,
                'sol': optimized_sol if optimized_sol else []
            })
        
            if optimization_result['time'] >= self.timeout_seconds:
                optimization_result['time'] = self.timeout_seconds
                optimization_result['optimal'] = False
            
            res.append(optimization_result)
        
        if not circle_method:
            return res

        # Run CP_cyrcle_method.mzn
        if circle_method:
            print(f"Running CP_circle_method.mzn for n={n}...")
            circle_keys = ["symm_brake", "ic_diff_match_in_week", "ic_diff_match_in_period", "use_int_search", "use_restart_luby", "use_relax_and_reconstruct", "chuffed"]
            circle_params = {k: params[k] for k in circle_keys}
            data_content = self.create_data_file(n)
            #print(data_content) ## DEBUG
            remaining_time = self.timeout_seconds
            source_path = os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'CP')
            cp_path = os.path.join(source_path, "CP_cyrcle_method.mzn")
            cp_success, cp_output, cp_time, cp_error = self.run_minizinc(
                cp_path, data_content, remaining_time
            )
            #print(cp_output) ## DEBUG
            # Create feasible result (from CP_circle_method.mzn)
            feasible_result = base_result.copy()
            optimization_result = base_result.copy()
            feasible_result["params"] = circle_params
            optimization_result["params"] = circle_params

            circle_data = self.extract_optimizer_data(cp_output)
            #print(circle_data) ## DEBUG
            circle_params["to_optimize"] = False
            bool_data = self.create_data_file(params=circle_params)
            final_data = bool_data + (circle_data or "")
            #print(final_data) ## DEBUG
            source_path = os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'CP')
            cp_path2 = os.path.join(source_path, "n20_NON_TOCCARE.mzn")
            cp_success, cp_output2, circle_time, cp_error = self.run_minizinc(
                cp_path2, final_data, int(remaining_time - cp_time)
            )
            #print(cp_success)
            #print(circle_time)
            #print(cp_error)
            #print(cp_output2) ## DEBUG
            if run_optimization:
                circle_params["to_optimize"] = True
                bool_data = self.create_data_file(params=circle_params)
                final_data = bool_data + (circle_data or "")
                #print(final_data) ## DEBUG
                opt_success, opt_output2, circle_time_opt, opt_error = self.run_minizinc(
                cp_path2, final_data, int(remaining_time - cp_time)
                )
            #print("=== RAW OPTIMIZER OUTPUT ===")
            #print(repr(opt_output2))
            #print(opt_success)
            #print(circle_time_opt)
            #print(opt_error)
            #print(opt_output) ## DEBUG
            feasible_result['method'] = 'circle'
            optimization_result['method'] = 'circle'

            if 'UNSATISFIABLE' in cp_output2:
                if run_decision:
                    feasible_result.update({
                        'optimal': True,
                        'stop_reason': 'infeasible',
                        'time': self.timeout_seconds
                    })
                    res.append(feasible_result)
                if run_optimization:
                    optimization_result.update({
                        'version': 'optimal',
                        'optimal': True,
                        'stop_reason': 'infeasible',
                        'time': self.timeout_seconds
                    })
                    res.append(optimization_result)
                return res

            if not run_optimization:        
                # Extract feasible solution from CP_3.0.mzn output
                cp_feasible_solution = self.convert_cp_output_to_matrix(cp_output2, n)
            
                # Update feasible result
                #if cp_time >= self.timeout_seconds:
                #    feasible_result['timeout_reached'] = True
            
                feasible_result.update({
                    'time': int(circle_time),
                    'optimal': True if cp_feasible_solution else False,
                    'stop_reason': 'None' if cp_feasible_solution else 'unknown',
                    'sol': cp_feasible_solution if cp_feasible_solution else []
                })
            
                if not cp_success:
                    print(f"n20_NON_TOCCARE.mzn failed: {cp_error}")                
                
                
                # Check remaining time
                remaining_time -= circle_time
                if remaining_time <= 0:
                    print(f"Timeout reached after n20_NON_TOCCARE.mzn")
                    #feasible_result['timeout_reached'] = True
                    feasible_result['time'] = self.timeout_seconds
                    feasible_result['stop_reason'] = 'time_limit'
                
                res.append(feasible_result)
                
            else:
                if run_decision:
                    # Extract feasible solution from CP_3.0.mzn output
                    cp_feasible_solution = self.convert_cp_output_to_matrix(cp_output, n)
            
                    # Update feasible result
                    #if cp_time >= self.timeout_seconds:
                    #    feasible_result['timeout_reached'] = True
            
                    feasible_result.update({
                        'time': int(circle_time),
                        'optimal': True if cp_feasible_solution else False,
                        'stop_reason': 'None' if cp_feasible_solution else 'unknown',
                        'sol': cp_feasible_solution if cp_feasible_solution else []
                    })
            
                    #if not cp_success:
                    #    print(f"CP_3.0.mzn failed: {cp_error}")                
                    
                    # Check remaining time
                    remaining_time -= circle_time
                    if remaining_time <= 0:
                        print(f"Timeout reached after CP_3.0.mzn")
                        feasible_result['time'] = self.timeout_seconds
                        feasible_result['stop_reason'] = 'time_limit'

                    res.append(feasible_result)
                # Extract data for optimizer
                optimizer_data = self.extract_optimizer_data(opt_output2)
                if not optimizer_data:
                    print("Failed to extract optimizer data from optimizer_2.0.mzn output")
                
                    # Update optimization result (failed to extract data)
                    optimization_result.update({
                        'version': 'optimal',
                        'optimal': False,
                        'time': self.timeout_seconds,
                        'stop_reason': 'unknown'
                    })
                    res.append(optimization_result)
                    
                
                # Run optimizer_2.0.mzn
                print(f"Running optimizer_2.0.mzn for n={n}...")
                source_path = os.path.join(os.path.dirname(__file__), '..', '..', 'source', 'CP')
                cp_opt_path = os.path.join(source_path,"optimizer_2.0.mzn")
                opt_success, opt_output3, opt_time, opt_error = self.run_minizinc(
                    cp_opt_path, optimizer_data, int(remaining_time)
                )
                optimization_result.update({
                    'version': 'optimal',
                    'time': int(circle_time + opt_time)
                })
            
                # Handle optimizer timeout or failure
                if not opt_success:
                    print(f"Optimizer failed or timed out: {opt_error}")
                
                    if optimization_result['time'] >= self.timeout_seconds:
                        optimization_result['time'] = self.timeout_seconds
                        optimization_result['stop_reason'] = 'time_limit'
                    else:
                        optimization_result['stop_reason'] = 'unknown'
                    res.append(optimization_result)
            
                # If optimizer succeeded, process its output
                summary, optimized_sol = self.parse_optimized_matrix_to_solution(optimizer_output=opt_output3, n_teams=n)
            
                value = None
                if summary:
                    for item in summary:
                        if "Optimized Max Imbalance:" in item:
                            value = int(item.split(": ")[1])
                            break
            
                optimization_result.update({
                    'optimal': True,
                    'obj': value,
                    'sol': optimized_sol if optimized_sol else []
                })
            
                if optimization_result['time'] >= self.timeout_seconds:
                    optimization_result['time'] = self.timeout_seconds
                    optimization_result['optimal'] = False
                
                res.append(optimization_result)
        return res


    def save_results(self, results: List[Dict], combinations: bool, run_decision: bool, run_optimization: bool, circle_method: bool):
        """Save results to JSON file with timestamp structure in res/CP relative to script"""
        
        if combinations and run_decision and run_optimization and circle_method:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'res', 'CP'))
        else:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'personalized_res', 'CP'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Group results by 'n' value
        #grouped_results = defaultdict(list)
        #for result in results:
        #    if 'n' in result:
        #        n_value = result['n']
        #        grouped_results[n_value].append(result)

        # Save each group to its respective file
        for n_value, group_results in results.items():
            filename = os.path.join(output_dir, f"{n_value}.json")

            # Load existing data if file exists
            existing_data = {}
            # Add new results with current timestamp
            for result in group_results:
                # Create result name based on parameters and phase
                params = result["params"]
                res_name = ""

                res_name += "chuffed" if params.get("chuffed") else "gecode"

                #phase = params.get("phase", "unknown")
                res_name += "_dec" if result["version"] == "decision" else "_opt"
                res_name += "_cir" if result["method"] == "circle" else "_cla"

                suffix_map = {
                    "sb_weeks": "_sw",
                    "sb_periods": "_sp",
                    "sb_teams": "_st",
                    "sb_week1_fixed": "_sw1f",
                    "symm_brake": "_sb",
                    "ic_matches_per_team": "_icm",
                    "ic_period_count": "_icp",
                    "ic_diff_match_in_week": "_icdw",
                    "ic_diff_match_in_period": "_icdp",
                    "use_int_search": "_usint",
                    "use_restart_luby": "_usrest",
                    "use_relax_and_reconstruct": "_usrelax",
                }

                for key, suffix in suffix_map.items():
                    if params.get(key):
                        res_name += suffix

                                
                existing_data[res_name] = result
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
            
    def run(self, selected_n, combinations, params, run_decision=True, run_optimization=True, circle_method=True):
        """Main execution method"""
        try:
            if not combinations:
                config = {'manual': params}
                """Run with manually specified parameters"""
                results = {}
        
                for i, n in enumerate(selected_n, 1):
                    print(f"[{i}/{len(selected_n)}] Running n={n}")
            
                    pipeline_results = self.run_pipeline(n, params, run_decision, run_optimization, circle_method)
                    #print(pipeline_results) ## DEBUG
                    if n not in results:
                        results[n] = []

                    results[n].extend(pipeline_results)  # Add results to the list
                print(results)
                # Save results
                self.save_results(results, combinations, run_decision, run_optimization, circle_method)
            else:
                config = {'all_combinations': True}
                """Run all possible combinations of boolean parameters"""
                results = {}
        
                # Generate all combinations of boolean values
                bool_combinations = list(itertools.product([True, False], repeat=len(self.bool_params)))
                total_runs = len(selected_n) * len(bool_combinations)
        
                run_count = 0
                for n in selected_n:
                    for combo in bool_combinations:
                        run_count += 1
                        params = dict(zip(self.bool_params, combo))
                
                        print(f"[{run_count}/{total_runs}] n={n}")
                
                        pipeline_results = self.run_pipeline(n, params, run_decision, run_optimization, circle_method)
                        #if n not in results:
                        #    results[n] = []
                        #print(pipeline_results)
                        self.save_results(pipeline_results, combinations, run_decision, run_optimization, circle_method)  # Save results ## VERIFICARE
                
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
        params = {
                "sb_weeks": True,
                "sb_periods": True,
                "sb_teams": True,
                "sb_week1_fixed": True,
                "symm_brake":True,
                "ic_matches_per_team": True,
                "ic_period_count": True,
                "ic_diff_match_in_week": True,
                "ic_diff_match_in_period": True,
                "use_int_search": True,
                "use_restart_luby": True,
                "use_relax_and_reconstruct": True,
                "chuffed": True
                }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        circle_method = True
        time_limit = 300
    else:
        # Parse arguments
        team_sizes = []
        params = {
                "sb_weeks": False,
                "sb_periods": False,
                "sb_teams": False,
                "sb_week1_fixed": False,
                "symm_brake":False,
                "ic_matches_per_team": False,
                "ic_period_count": False,
                "ic_diff_match_in_week": False,
                "ic_diff_match_in_period": False,
                "use_int_search": False,
                "use_restart_luby": False,
                "use_relax_and_reconstruct": False,
                "chuffed": False
                }
        run_decision = True
        run_optimization = True
        symmetry_combinations = True
        circle_method = True
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
                "sb_weeks": True,
                "sb_periods": True,
                "sb_teams": True,
                "sb_week1_fixed": True,
                "symm_brake": True,
                "ic_matches_per_team": True,
                "ic_period_count": True,
                "ic_diff_match_in_week": True,
                "ic_diff_match_in_period": True,
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
                    "sb_week1_fixed": False,
                    "symm_brake": False,
                    "ic_matches_per_team": False,
                    "ic_period_count": False,
                    "ic_diff_match_in_week": False,
                    "ic_diff_match_in_period": False,
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
                    "sb_week1_fixed": False,
                    "symm_brake": False,
                    "ic_matches_per_team": False,
                    "ic_period_count": False,
                    "ic_diff_match_in_week": False,
                    "ic_diff_match_in_period": False,
                    "use_int_search": False,
                    "use_restart_luby": False,
                    "use_relax_and_reconstruct": False,
                    "chuffed": False
                    }
                k = i+1
                for j in range(k, len(sys.argv)):
                    i = j
                    symmetry = sys.argv[j]
                    valid_symmetry = ["sb_weeks", "sb_periods", "sb_teams", "sb_week1_fixed", "symm_brake", "ic_matches_per_team", "ic_period_count", "ic_diff_match_in_week", "ic_diff_match_in_period", "use_int_search", "use_restart_luby", "use_relax_and_reconstruct", "chuffed"]
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
                            elif symmetry.lower() == "sb_week1_fixed":
                                params["sb_week1_fixed"] = True
                            elif symmetry.lower() == "symm_brake":
                                params["symm_brake"] = True
                            elif symmetry.lower() == "ic_matches_per_team":
                                params["ic_matches_per_team"] = True
                            elif symmetry.lower() == "ic_period_count":
                                params["ic_period_count"] = True
                            elif symmetry.lower() == "ic_diff_match_in_week":
                                params["ic_diff_match_in_week"] = True
                            elif symmetry.lower() == "ic_diff_match_in_period":
                                params["ic_diff_match_in_period"] = True
                            elif symmetry.lower() == "use_int_search":
                                params["use_int_search"] = True
                            elif symmetry.lower() == "use_restart_luby":
                                params["use_restart_luby"] = True
                            elif symmetry.lower() == "use_relax_and_reconstruct":
                                params["use_relax_and_reconstruct"] = True
                            else:
                                params["chuffed"] = True
            elif arg.lower() == "--no-circle":
                circle_method = False
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
                "sb_weeks": True,
                "sb_periods": True,
                "sb_teams": True,
                "sb_week1_fixed": True,
                "symm_brake": True,
                "ic_matches_per_team": True,
                "ic_period_count": True,
                "ic_diff_match_in_week": True,
                "ic_diff_match_in_period": True,
                "use_int_search": True,
                "use_restart_luby": True,
                "use_relax_and_reconstruct": True,
                "chuffed": True
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
    
    runner = MinizincRunner(timeout_seconds=time_limit)
    runner.run(team_sizes, symmetry_combinations, params, run_decision, run_optimization, circle_method)

if __name__ == "__main__":
    main()