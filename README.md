# Exploration of different Optimization Paradigms for the Sports Tournament Scheduling (STS) problem

Sports Tournament Scheduling is addressed within a unified framework that employs four optimization approaches: **Constraint Programming (CP)**, **Boolean Satisfiability (SAT)**, **Satisfiability Modulo Theories (SMT)** and **Mixed Integer Programming (MIP)**.

## Execution

### Option 1: Docker (Recommended)

1. **Build and run**
   ```bash
   docker build -t sts-image .
   docker-compose up -d
   docker-compose exec sts bash
   ```

2. **Run examples inside the container:**
   ```bash
   # CP solver
   python test/CP/CP_runner.py -n 6
   
   # SAT Solver
   python test/SAT/SAT_runner.py 6

   # SMT Solver
   python test/SMT/SMT_runner.py 6

   # MIP solver  
   python test/MIP/MIP_runner.py 6
   ```

### Option 2: Local Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install MiniZinc** (for CP): Download it from https://www.minizinc.org/

3. **Install CVC5** (for SMT): Download it from https://github.com/cvc5/cvc5/releases

4. **Install AMPL** (for MIP): Download it from https://ampl.com/

4. **Run Examples:**
   ```bash
   # CP solver
   python test/CP/CP_runner.py -n 6
   
   # SAT Solver
   python test/SAT/SAT_runner.py 6

   # SMT Solver
   python test/SMT/SMT_runner.py 6

   # MIP solver  
   python test/MIP/MIP_runner.py 6
   ```


## Project Structure

```
res/
├── CP/        # folder of .JSON CP files
├── MIP/       # folder of .JSON MIP files
├── SAT/       # folder of .JSON SAT files
└── SMT/       # folder of .JSON SMT files

source/
├── CP/          # MiniZinc constraint programming models
├── MIP/         # AMPL model
├── SAT/         # Z3 SAT encoding implementations  
└── SMT/         # Z3 SMT theory solver

test/
├── CP/CP_runner.py         # CP solver runner
├── MIP/MIP_runner.py       # MIP solver runner
├── SAT/SAT_runner.py       # SAT solver runner
└── SMT/SMT_runner.py       # SMT solver runner
```

## Key Constraints

Implied and Symmetry Breaking constraints for the solver performance improvement:
- **Symmetry Breaking:** Fixing team, week or period matchups
- **Implied Constraints:** Match counts per team

Optimization Function for the balance between n° of games at home and away for teams 
- **Optimization:** Minimize home/away imbalances (CP/MIP)

## Results Format

All solvers output JSON contains:
- Solution matrix (list of lists, containing matches in smaller list, grouped by lists by period)
- Solution statistics (used solver, taken time, objective function)
- Confirmation of correct solution found

### Common options:

- `--solvers cplex` &nbsp;&nbsp;&nbsp;&nbsp;Run only selected solvers.
- `--teams 6 8 10` &nbsp;&nbsp;&nbsp;&nbsp;Specify exact team numbers.
- `--time-limit 600` &nbsp;&nbsp;&nbsp;&nbsp;Personalized the time limit (in seconds) per run (default 300s).
- `--no-combination` &nbsp;&nbsp;&nbsp;&nbsp;Execute the solver runner only for configurations with all the symmetry breaking constraints.
- `--decision-only` &nbsp;&nbsp;&nbsp;&nbsp;Run the solvers only for the decision version (no objective function).
- `--optimization-only` &nbsp;&nbsp;&nbsp;&nbsp;Run the solvers only for the optimization version (minimize home/away imbalance).

## Solution Checker

The correctness of any JSON file can be verified by the provided solution checker:

```bash
python test/solution_checker.py <path_to_json_directory>
```

- `<path_to_json_directory>`: directory containing the `.json` files created by the solvers runner files.
- Solution checker script prints the validity status.

Example usage:
```bash
python solution_checker.py test/MIP
```

---

