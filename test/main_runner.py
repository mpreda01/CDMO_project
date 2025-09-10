#!/usr/bin/env python3
"""
Main Test Runner for Sports Tournament Scheduling Project
Executes all solver runners: CP, MIP, SAT, and SMT
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple

class MainRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_runners = [
            ("CP", "test/CP/CP_runner.py"),
            ("MIP", "test/MIP/MIP_runner.py"), 
            ("SAT", "test/SAT/SAT_runner.py"),
            ("SMT", "test/SMT/SMT_runner.py")
        ]
        self.results = {}
        
    def run_solver(self, solver_name: str, script_path: str) -> Tuple[bool, str, float]:
        """
        Run a single solver script and return success status, output, and execution time
        
        Args:
            solver_name: Name of the solver (CP, MIP, SAT, SMT)
            script_path: Relative path to the solver script
            
        Returns:
            Tuple of (success, output, execution_time)
        """
        print(f"\n{'='*60}")
        print(f"Starting {solver_name} solver...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Change to project root directory
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            # Run the solver script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                print(f"{solver_name} solver completed successfully in {execution_time:.2f} seconds")
                if result.stdout:
                    print(f"{solver_name} Output (last 10 lines):")
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines[-10:]:
                        print(f"   {line}")
                return True, result.stdout, execution_time
            else:
                print(f"{solver_name} solver failed with return code {result.returncode}")
                print(f"{solver_name} Error output:")
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-10:]:
                        print(f"   {line}")
                return False, result.stderr, execution_time
                
        except subprocess.TimeoutExpired:
            print(f"{solver_name} solver timed out after 30 minutes")
            return False, "Timeout expired", time.time() - start_time
            
        except Exception as e:
            print(f"{solver_name} solver crashed with exception: {str(e)}")
            return False, str(e), time.time() - start_time
            
        finally:
            # Ensure we're back in the original directory
            try:
                os.chdir(original_cwd)
            except:
                pass
    
    def run_all_solvers(self) -> None:
        """
        Run all solver scripts in sequence
        """
        print("Sports Tournament Scheduling - Main Test Runner")
        print(f"Project root: {self.project_root}")
        print(f"Python executable: {sys.executable}")
        
        overall_start_time = time.time()
        successful_runs = 0
        failed_runs = 0
        
        for solver_name, script_path in self.test_runners:
            success, output, execution_time = self.run_solver(solver_name, script_path)
            
            self.results[solver_name] = {
                'success': success,
                'execution_time': execution_time,
                'output_length': len(output) if output else 0
            }
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
        
        # Print summary
        overall_time = time.time() - overall_start_time
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Successful runs: {successful_runs}")
        print(f"Failed runs: {failed_runs}")
        print(f"Total execution time: {overall_time:.2f} seconds")
        
        for solver_name, result in self.results.items():
            status = "OK" if result['success'] else "FAIL"
            print(f"{status} {solver_name}: {result['execution_time']:.2f}s")
        
        if failed_runs == 0:
            print("\nAll solvers completed successfully!")
        else:
            print(f"\n{failed_runs} solver(s) failed. Check the output above for details.")
            
        print(f"\nResults should be available in:")
        print(f"   - res/CP/")
        print(f"   - res/MIP/")
        print(f"   - res/SAT/")
        print(f"   - res/SMT/")

def main():
    """Main entry point"""
    runner = MainRunner()
    runner.run_all_solvers()

if __name__ == "__main__":
    main()