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
        # Percorsi comuni per MiniZinc
        common_paths = [
            "minizinc",  # Se è nel PATH
            "D:\\Program\\MiniZinc\\minizinc.exe",  # Il tuo percorso
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
                
        print("❌ MiniZinc non trovato. Assicurati che sia installato e nel PATH.")
        sys.exit(1)
        
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
                print(f"🕐 Timeout reached, terminating process...")
                process.terminate()
                
                # Wait a bit for graceful termination
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print(f"🔪 Force killing process...")
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
        """Converte la matrice ottimizzata nel formato richiesto per il JSON"""
        try:
            # Trova la sezione della matrice ottimizzata
            start_marker = "=== OPTIMIZED TOURNAMENT MATRIX ==="
            lines = optimizer_output.split('\n')
            
            summary = []
            matrix_lines = []
            found_start = False
            
            # Estrae le prime 4 linee per il summary
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
            
            # Parse ogni riga della matrice in formato compatto su una sola riga
            solution = []
            for line in matrix_lines:
                # Esempio: [[1 , 2] , [1 , 3] , [2 , 6] , [6 , 3] , [4 , 5]]
                # Rimuove le parentesi quadre esterne
                clean_line = line.strip()[2:-2]  # Rimuove [[ e ]]
                
                # Split per le coppie
                pairs_str = clean_line.split('] , [')
                period_matches = []
                
                for pair_str in pairs_str:
                    # Estrae i due numeri dalla coppia
                    import re
                    numbers = re.findall(r'\d+', pair_str)
                    if len(numbers) >= 2:
                        period_matches.append([int(numbers[0]), int(numbers[1])])
                
                if period_matches:
                    solution.append(period_matches)
            
            return summary if summary else None, solution if solution else None
            
        except Exception as e:
            print(f"❌ Errore nel parsing della matrice ottimizzata: {e}")
            return None, None
    
    def run_pipeline(self, n: int, params: Dict[str, bool]) -> Dict:
        """Run the complete pipeline: CP_3.0.mzn -> optimizer_2.0.mzn"""
        result = {
            'n': n,
            'params': params,
            'cp_success': False,
            'cp_time': 0,
            'cp_error': '',
            'optimizer_success': False,
            'optimizer_time': 0,
            'optimizer_error': '',
            'time': 0,
            'timeout_reached': False
        }
        
        # Step 1: Run CP_3.0.mzn
        print(f"Running CP_3.0.mzn for n={n}...")
        data_content = self.create_data_file(n, params)
        
        remaining_time = self.timeout_seconds
        cp_success, cp_output, cp_time, cp_error = self.run_minizinc(
            'CP_3.0.mzn', data_content, remaining_time
        )
        print(type(cp_output))
        if 'UNSATISFIABLE' in cp_output:
            result.update({
            'cp_success': False,
            'cp_time': round(cp_time, 2),
            'cp_error': cp_error
            })

            result.update({
            'optimizer_success': False,
            'optimizer_time': 0,
            'optimal': False,
            'sol' : '=====UNSATISFIABLE=====',
            'time': int(cp_time)
            })

            return result
        
        if result['time'] >= self.timeout_seconds:
            result['timeout_reached'] = True
        else:
            result.update({
                'cp_success': cp_success,
                'cp_time': round(cp_time, 2),
                'cp_error': cp_error
            })
        
        if not cp_success:
            print(f"CP_3.0.mzn failed: {cp_error}")
            result['time'] = cp_time
            result['opt'] = False
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
            return result
            
        # Step 2: Extract data for optimizer
        optimizer_data = self.extract_optimizer_data(cp_output)
        if not optimizer_data:
            print("Failed to extract optimizer data from CP_3.0.mzn output")
            result['optimizer_error'] = "Failed to extract optimizer data"
            result['time'] = cp_time
            return result
            
        # Step 3: Run optimizer_2.0.mzn
        print(f"Running optimizer_2.0.mzn for n={n}...")
        opt_success, opt_output, opt_time, opt_error = self.run_minizinc(
            'optimizer_2.0.mzn', optimizer_data, int(remaining_time)
        )
        summary, sol = self.parse_optimized_matrix_to_solution(optimizer_output = opt_output, n_teams = n) 

        if summary:
            for item in summary:
                if "Optimized Total Imbalance:" in item:
                    value = int(item.split(": ")[1])        
        
        if value == n:
            result.update({
            'optimizer_success': opt_success,
            'optimizer_time': round(opt_time, 2),
            'summary': summary,
            'obj': value,
            'optimal': True,
            'sol' : sol,
            'optimizer_error': opt_error,
            'time': int(cp_time + opt_time)
            })
        else:
            result.update({
            'optimizer_success': opt_success,
            'optimizer_time': round(opt_time, 2),
            'summary': summary,
            'obj': value,
            'optimal': False,
            'sol' : sol,
            'optimizer_error': opt_error,
            'time': int(cp_time + opt_time)
            })
        
        if result['time'] >= self.timeout_seconds:
            result['timeout_reached'] = True
            result['optimal'] = False
            
        return result
        
    def get_user_input(self) -> Tuple[List[int], Dict[str, bool]]:
        """Get user input for parameters"""
        print("=== MiniZinc Tournament Scheduler Runner ===\n")
        
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
            return selected_n, {'manual': params}
        else:
            return selected_n, {'all_combinations': True}
            
    def run_all_combinations(self, selected_n: List[int]) -> List[Dict]:
        """Run all possible combinations of boolean parameters"""
        results = []
        
        # Generate all combinations of boolean values
        bool_combinations = list(itertools.product([True, False], repeat=len(self.bool_params)))
        total_runs = len(selected_n) * len(bool_combinations)
        
        print(f"\nRunning {total_runs} total combinations...")
        print(f"Timeout per combination: {self.timeout_seconds} seconds\n")
        
        run_count = 0
        for n in selected_n:
            for combo in bool_combinations:
                run_count += 1
                params = dict(zip(self.bool_params, combo))
                
                print(f"[{run_count}/{total_runs}] n={n}, params={params}")
                
                result = self.run_pipeline(n, params)
                self.save_results([result])
                #results.append(result)
                
                # Print summary
                if result['cp_success'] and result['optimizer_success']:
                    print(f"✓ SUCCESS - Total time: {result['time']:.2f}s")
                elif result['timeout_reached']:
                    print(f"⏱ TIMEOUT - Reached {self.timeout_seconds}s limit")
                else:
                    print(f"✗ FAILED - CP: {'OK' if result['cp_success'] else 'FAIL'}, "
                          f"OPT: {'OK' if result['optimizer_success'] else 'FAIL'}")
                
                print("-" * 60)
                
        #return results
        
    def run_manual(self, selected_n: List[int], params: Dict[str, bool]) -> List[Dict]:
        """Run with manually specified parameters"""
        results = []
        
        print(f"\nRunning {len(selected_n)} configurations...")
        print(f"Parameters: {params}")
        print(f"Timeout per run: {self.timeout_seconds} seconds\n")
        
        for i, n in enumerate(selected_n, 1):
            print(f"[{i}/{len(selected_n)}] Running n={n}")
            
            result = self.run_pipeline(n, params)
            results.append(result)
            
            # Print summary
            if result['cp_success'] and result['optimizer_success']:
                print(f"✓ SUCCESS - Total time: {result['time']:.2f}s")
            elif result['timeout_reached']:
                print(f"⏱ TIMEOUT - Reached {self.timeout_seconds}s limit")
            else:
                print(f"✗ FAILED - CP: {'OK' if result['cp_success'] else 'FAIL'}, "
                      f"OPT: {'OK' if result['optimizer_success'] else 'FAIL'}")
                      
            print("-" * 60)
            
        return results

    def save_results(self, results: List[Dict]):
        """Save results to JSON file with timestamp structure"""
        print('saving...')
        results_dir = "res\\CP"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Created directory: {results_dir}")
        
        # Group results by 'n' value
        grouped_results = defaultdict(list)
        for result in results:
            if 'n' in result:
                n_value = result['n']
                grouped_results[n_value].append(result)
        
        # Save each group to its own file
        for n_value, group_results in grouped_results.items():
            filename = os.path.join(results_dir, f"{n_value}.json")
            
            # Load existing results if file exists
            existing_data = {}
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    print(f"Loaded existing data from {filename}")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not load existing file {filename}: {e}")
                    existing_data = {}
            
            # Add each new result with its own timestamp
            for result in group_results:
                current_timestamp = str(time.time())
                existing_data[current_timestamp] = result  # Direttamente l'oggetto, non un array
            
            # Save with custom formatting
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
                            # Special formatting for solution array
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
                            # Regular JSON encoding
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
            
        # Best performing configurations
        successful_results = [r for r in results if r['cp_success'] and r['optimizer_success']]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x['time'])
            print(f"Fastest successful run: n={fastest['n']}, time={fastest['time']:.2f}s")
            
    def run(self):
        """Main execution method"""
        try:
            selected_n, config = self.get_user_input()
            
            if 'manual' in config:
                results = self.run_manual(selected_n, config['manual'])
            else:
                results = self.run_all_combinations(selected_n)
                
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
    runner = MinizincRunner(timeout_seconds=300)
    runner.run()