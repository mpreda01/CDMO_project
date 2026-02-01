import subprocess
import sys

scripts = [
    "test/CP/CPcircle_runner.py",
    "test/SMT/SMT_runner.py",
    "test/MIP/MIP_runner.py",
    #"test/SAT/SAT_runner.py",
]

python_exe = sys.executable

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(
        [python_exe, script],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
