import subprocess

# List of scripts to execute in order
scripts = [
    "test/MIP/MIP_runner.py",
    "test/CP/CP_runner.py",
    "test/SAT/SAT_runner.py",
    "test/SMT/SMT_runner.py"
]

# Execute each script one after the other
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script}:", result.stderr)
