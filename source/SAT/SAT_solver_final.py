from pysat.formula import CNF
from pysat.solvers import Minisat22, Glucose42, Lingeling
from z3 import *
import time
import math
import multiprocessing
from constraints_encoding import *


def _pysat_solve_worker(clauses, solver_name, result_queue):
    """Worker function for multiprocessing - must be at module level."""
    try:
        if solver_name == "minisat":
            solver_class = Minisat22
        elif solver_name == "glucose":
            solver_class = Glucose42
        elif solver_name == "lingeling":
            solver_class = Lingeling
        else:
            result_queue.put((False, None))
            return
            
        with solver_class(bootstrap_with=clauses) as sat_solver:
            satisfiable = sat_solver.solve()
            model = sat_solver.get_model() if satisfiable else None
        result_queue.put((satisfiable, model))
    except Exception as e:
        result_queue.put((False, None)) 


def circle_method(n):
    """
    Implements the Circle Method for round-robin scheduling with n teams,
    fixing the last team (team n-1) to get an initial schedule 
    Returns a list of rounds, each round is a list of (team_i, team_j) tuples.
    Everything indexed from 0 to n-1.
    """
    rounds = []
    teams = list(range(n))
    for r in range(n - 1):
        round_matches = []
        round_matches.append((n - 1, r))
        for i in teams:
            if i == r or i == n - 1:
                continue
            for j in teams:
                if j == r or j == n - 1 or j <= i:
                    continue
                if (i + j) % (n - 1) == (2 * r) % (n - 1):
                    round_matches.append((i, j))
        rounds.append(round_matches)
    return rounds


def lexic_order_for_periods(home, away, Weeks, Periods, Teams):
    """
    Enforce lexicographic order between period vectors for each week.
    For each week w, for each consecutive periods p, p+1:
    The vector of (home, away) assignments for period p is lex <= that for period p+1.
    """
    constraints = []
    for w in Weeks:
        for p in range(len(Periods) - 1):
            for t in Teams:
                prefix_eq = []
                for j in range(t):
                    prefix_eq.append(home[w][p][j] == home[w][p+1][j])
                    prefix_eq.append(away[w][p][j] == away[w][p+1][j])
                if prefix_eq:
                    constraints.append(Implies(And(*prefix_eq), Or(Not(home[w][p][t]), home[w][p+1][t])))
                else:
                    constraints.append(Or(Not(home[w][p][t]), home[w][p+1][t]))
    return constraints


def fix_period_constraints(home, away, n, n_weeks, n_periods):
    constraints = []
    for w in range(n_weeks - 1):
        p_target = w % n_periods
        p_next = (w + 1) % n_periods
        team_n_period_vars = [home[w][p_target][n - 1], away[w][p_target][n - 1]]
        constraints.append(Or(team_n_period_vars))
        if n >= 14:
            team_n1_period_vars = [home[w][p_next][n - 2], away[w][p_next][n - 2]]
            constraints.append(Or(team_n1_period_vars))
    return constraints


def different_match_per_period_constraints(home, away, Teams, Weeks, Periods):
    constraints = []
    for w in Weeks:
        for p1 in Periods:
            for p2 in Periods:
                if p1 < p2:
                    for i in Teams:
                        for j in Teams:
                            if i != j:
                                constraints.append(
                                    Implies(
                                        And(home[w][p1][i], away[w][p1][j]),
                                        Not(And(home[w][p2][i], away[w][p2][j]))
                                    )
                                )
                                constraints.append(
                                    Implies(
                                        And(home[w][p2][i], away[w][p2][j]),
                                        Not(And(home[w][p1][i], away[w][p1][j]))
                                    )
                                )
    return constraints


def matches_per_team_constraints(home, away, Teams, Weeks, Periods, n_weeks):
    constraints = []
    for t in Teams:
        occ = []
        for w in Weeks:
            for p in Periods:
                occ.append(home[w][p][t])
                occ.append(away[w][p][t])
        constraints.append(exactly_k(occ, n_weeks, name=f"team_{t}_matches"))
    return constraints


def create_solver(n: int, solver_args: dict, constraints: dict = None):
    
    if constraints is None: 
        constraints = {
            'matches_per_team': True,
            'different_match_per_period': True,
            'n_fix_period': False,
            'lex_periods': False,
    }
    ic_matches_per_team = constraints.get('matches_per_team', True)
    ic_different_match_per_period = constraints.get('different_match_per_period', True)
    n_fix_period = constraints.get('n_fix_period', False)
    lex_periods = constraints.get('lex_periods', False)
    
    solver = Solver()
    solver.set("random_seed", 20)

    n_weeks = solver_args['n_weeks']
    n_periods = solver_args['n_periods']
    Teams = solver_args['Teams']
    Weeks = solver_args['Weeks']
    Periods = solver_args['Periods']
    home = solver_args['home']
    away = solver_args['away']

    # CORE CONTRAINTS
    # Each period has exactly one home and one away team per week, and they must be different
    for w in Weeks:
        for p in Periods:
            solver.add(exactly_one([home[w][p][t] for t in Teams], name=f"home_{w}_{p}"))
            solver.add(exactly_one([away[w][p][t] for t in Teams], name=f"away_{w}_{p}"))
            for t in Teams:
                solver.add(Implies(home[w][p][t], Not(away[w][p][t])))
                solver.add(Implies(away[w][p][t], Not(home[w][p][t])))

    # Each team plays exactly one match per week
    for w in Weeks:
        for t in Teams:
            occ = []
            for p in Periods:
                occ.append(home[w][p][t])
                occ.append(away[w][p][t])
            solver.add(exactly_one(occ, name=f"team_{t}_week_{w}"))

    # Each pair of teams plays exactly one match in the tournament
    for i in Teams:
        for j in Teams:
            if i < j:
                pair_games = []
                for w in Weeks:
                    for p in Periods:
                        pair_games.append(And(home[w][p][i], away[w][p][j]))
                        pair_games.append(And(home[w][p][j], away[w][p][i]))
                solver.add(exactly_one(pair_games, name=f"pair_{i}_{j}"))

    # Each team plays at most twice in the same period
    for t in Teams:
        for p in Periods:
            occ = []
            for w in Weeks:
                occ.append(home[w][p][t])
                occ.append(away[w][p][t])
            solver.add(at_most_k(occ, 2, name=f"team_{t}_period_{p}"))

    # IMPLIED CONSTRAINTS
    if ic_matches_per_team:
        matches_constraints = matches_per_team_constraints(home, away, Teams, Weeks, Periods, n_weeks)
        for c in matches_constraints:
            solver.add(c)

    if ic_different_match_per_period:
        diff_constraints = different_match_per_period_constraints(home, away, Teams, Weeks, Periods)
        for c in diff_constraints:
            solver.add(c)

    # SYMMETRY BREAKING
    if n_fix_period:
        fix_constraints = fix_period_constraints(home, away, n, n_weeks, n_periods)
        for c in fix_constraints:
            solver.add(c)

    if lex_periods:
        lex_constraints = lexic_order_for_periods(home, away, Weeks, Periods, Teams)
        for c in lex_constraints:
            solver.add(c)

    return solver

               
def create_solver_with_circle_method(n: int, solver_args: dict, constraints: dict = None):

    if constraints is None: 
        constraints = {
            'matches_per_team': False,
            'different_match_per_period': False,
            'n_fix_period': False, 
    }
    
    ic_matches_per_team = constraints.get('matches_per_team', False)
    ic_different_match_per_period = constraints.get('different_match_per_period', False)
    # n_fix_period it's too restrictive pairing with circle method for n<8  (sperimental results)
    if n >= 8:    
        n_fix_period = constraints.get('n_fix_period', False)
    else: 
        n_fix_period = False

    solver = Solver()
    solver.set("random_seed", 20)

    n_weeks = solver_args['n_weeks']
    n_periods = solver_args['n_periods']
    Teams = solver_args['Teams']
    Weeks = solver_args['Weeks']
    Periods = solver_args['Periods']
    home = solver_args['home']
    away = solver_args['away']

    # Starting from a circle method generated schedule
    initial_schedule = circle_method(n)

    # CORE CONSTRAINTS

    for w, week_matches in enumerate(initial_schedule):
        for p in Periods:
            match_in_period_vars = []
            for (i, j) in week_matches:
                match_var = Bool(f"match_{i}_{j}_week_{w}_period_{p}")
                match_in_period_vars.append(match_var)
                solver.add(Implies(match_var, And(home[w][p][i], away[w][p][j])))
                solver.add(Implies(Not(match_var), And(Not(home[w][p][i]), Not(away[w][p][j]))))
            solver.add(exactly_one(match_in_period_vars, name=f"week_{w}_period_{p}_one_match"))

    for w in Weeks:
        for p in Periods:
            solver.add(exactly_one([home[w][p][t] for t in Teams], name=f"home_{w}_{p}"))
            solver.add(exactly_one([away[w][p][t] for t in Teams], name=f"away_{w}_{p}"))

    for w, week_matches in enumerate(initial_schedule):
        for (i, j) in week_matches:
            match_period_vars = []
            for p in Periods:
                match_var = Bool(f"match_{i}_{j}_week_{w}_period_{p}")
                match_period_vars.append(match_var)
            solver.add(exactly_one(match_period_vars, name=f"match_{i}_{j}_week_{w}_one_period"))

    for t in Teams:
        for p in Periods:
            occ = []
            for w in Weeks:
                occ.append(home[w][p][t])
                occ.append(away[w][p][t])
            solver.add(at_most_k(occ, 2, name=f"team_{t}_max_twice_period_{p}"))

    # Implied constraints
    if ic_matches_per_team:
        matches_constraints = matches_per_team_constraints(home, away, Teams, Weeks, Periods, n_weeks)
        for c in matches_constraints:
            solver.add(c)

    if ic_different_match_per_period:
        diff_constraints = different_match_per_period_constraints(home, away, Teams, Weeks, Periods)
        for c in diff_constraints:
            solver.add(c)

    # Symmetry breaking
    if n_fix_period:
        fix_constraints = fix_period_constraints(home, away, n, n_weeks, n_periods)
        for c in fix_constraints:
            solver.add(c)

    return solver


# ===================== OPTIMIZATION FUNCTION =====================

def optimize_home_away(n: int, feasible_solution: list, timeout: float, seed: int = 20, verbose: bool = False) -> dict:
    """
    Phase 2: Optimize home/away assignments using swap variables.
    """
    n_weeks = n - 1
    n_periods = n // 2
    
    swap = [[Bool(f"swap_{p}_{w}") for w in range(n_weeks)] for p in range(n_periods)]
    
    team_slots = {t: [] for t in range(1, n + 1)}
    for p in range(n_periods):
        for w in range(n_weeks):
            home_t, away_t = feasible_solution[p][w]
            team_slots[home_t].append((p, w, True))
            team_slots[away_t].append((p, w, False))
    
    home_counts = [0] * (n + 1)
    for p in range(n_periods):
        for w in range(n_weeks):
            home_counts[feasible_solution[p][w][0]] += 1
    initial_max_imb = max(abs(2 * hc - n_weeks) for hc in home_counts[1:])
    
    if verbose:
        print(f"  Initial max imbalance (before optimization): {initial_max_imb}")
    
    low, high = 1, initial_max_imb
    best_obj = initial_max_imb
    best_model = None
    start = time.time()
    
    while low <= high and (time.time() - start) < timeout - 1:
        mid = (low + high) // 2
        
        if verbose:
            print(f"  Trying max imbalance = {mid}...", end=" ")
        
        s = Solver()
        s.set("timeout", int((timeout - (time.time() - start) - 1) * 1000))
        s.set("random_seed", seed)
        
        for t in range(1, n + 1):
            home_vars = []
            for (p, w, orig_home) in team_slots[t]:
                if orig_home:
                    home_vars.append(Not(swap[p][w]))
                else:
                    home_vars.append(swap[p][w])
            
            min_home = math.ceil((n_weeks - mid) / 2)
            max_home = math.floor((n_weeks + mid) / 2)
            
            if min_home > 0:
                s.add(at_least_k(home_vars, min_home, f"t{t}_min_{mid}"))
            if max_home < len(home_vars):
                s.add(at_most_k(home_vars, max_home, f"t{t}_max_{mid}"))
        
        s.add(Not(swap[0][0]))
        
        if s.check() == sat:
            if verbose:
                print("SAT")
            best_model = s.model()
            best_obj = mid
            if mid <= 1:
                break
            high = mid - 1
        else:
            if verbose:
                print("UNSAT")
            low = mid + 1
    
    final_sol = []
    for p in range(n_periods):
        period = []
        for w in range(n_weeks):
            h, a = feasible_solution[p][w]
            if best_model and is_true(best_model.evaluate(swap[p][w])):
                period.append([a, h])
            else:
                period.append([h, a])
        final_sol.append(period)
    
    is_optimal = (best_obj == 1)
    
    return {
        'solution': final_sol,
        'obj': best_obj,
        'time': time.time() - start,
        'is_optimal': is_optimal
    }


# ===================== MAIN SOLVE FUNCTIONS =====================

def solve_sts(n: int, constraints=None, use_circle_method: bool = True, timeout: int = 300, optimize: bool = False, verbose: bool = False):
    """
    Solve the STS problem using SAT encoding.
    """
    start_time = time.time()
    
    if verbose and optimize:
        print(f"\n{'='*50}")
        print(f"PHASE 1: Finding feasible schedule (n={n})")
        print(f"{'='*50}")
    
    n_weeks = n - 1
    n_periods = n // 2
    Teams = range(n)
    Weeks = range(n_weeks)
    Periods = range(n_periods)    

    home = [[[Bool(f"home_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]
    away = [[[Bool(f"away_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]

    solver_args = {
        "n_weeks": n_weeks,
        "n_periods": n_periods,
        "Teams": Teams,
        "Weeks": Weeks,
        "Periods": Periods,
        "home": home,
        "away": away
    }
    
    if use_circle_method:
        s = create_solver_with_circle_method(n, solver_args, constraints)
    else:
        s = create_solver(n, solver_args, constraints)

    s.set("timeout", timeout * 1000)  # Timeout applies to s.check() only

    solve_start = time.time()  # Measure only solve time
    result = s.check()
    solve_time = time.time() - solve_start
    total_time = time.time() - start_time
    
    # Enforce timeout: if solve exceeded it, report timeout
    if solve_time >= timeout:
        if verbose and optimize:
            print(f"  Status: TIMEOUT")
            print(f"  Time: {timeout}s")
        return {
            'solution': None,
            'time': timeout,  # Report exactly timeout seconds
            'satisfiable': False,
            'obj': None
        }
    
    if result != sat:
        if verbose and optimize:
            print(f"  Status: NO SOLUTION FOUND")
            print(f"  Time: {total_time:.2f}s")
        return {
            'solution': None,
            'time': total_time,
            'satisfiable': False,
            'obj': None
        }
    
    m = s.model()
    sol = []
    for p in Periods:
        period = []
        for w in Weeks:
            home_team = None
            for t in Teams:
                if m.evaluate(home[w][p][t], model_completion=True):
                    home_team = t + 1
                    break
            away_team = None
            for t in Teams:
                if m.evaluate(away[w][p][t], model_completion=True):
                    away_team = t + 1
                    break
            period.append([home_team, away_team])
        sol.append(period)
    
    if verbose and optimize:
        print(f"  Status: FEASIBLE SOLUTION FOUND")
        print(f"  Time: {solve_time:.2f}s")
    
    if not optimize:
        return {
            'solution': sol,
            'time': total_time,
            'satisfiable': True,
            'obj': None
        }
    
    # For optimization, remaining time is based on solve_time (preprocessing doesn't count)
    remaining_time = timeout - solve_time
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"PHASE 2: Optimizing home/away balance")
        print(f"{'='*50}")
        print(f"  Remaining time: {remaining_time:.2f}s")
    
    if remaining_time <= 1:
        home_counts = [0] * (n + 1)
        for p in range(n_periods):
            for w in range(n_weeks):
                home_counts[sol[p][w][0]] += 1
        obj = max(abs(2 * hc - n_weeks) for hc in home_counts[1:])
        if verbose:
            print(f"  No time remaining for optimization!")
            print(f"  Returning feasible solution with obj={obj}")
        return {
            'solution': sol,
            'time': total_time,
            'satisfiable': True,
            'obj': obj
        }
    
    opt_result = optimize_home_away(n, sol, remaining_time, verbose=verbose)
    final_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"OPTIMIZATION RESULT")
        print(f"{'='*50}")
        print(f"  Phase 1 time: {solve_time:.2f}s")
        print(f"  Phase 2 time: {opt_result['time']:.2f}s")
        print(f"  Total time: {final_time:.2f}s")
        print(f"  Objective (max imbalance): {opt_result['obj']}")
        print(f"  Is optimal: {opt_result['is_optimal']}")
    
    return {
        'solution': opt_result['solution'],
        'time': final_time,
        'satisfiable': True,
        'obj': opt_result['obj']
    }
    

def parse_variable_mappings(dimacs_lines: list[str]) -> dict[str, int]:
    var_mappings = {}
    for line in dimacs_lines:
        if line.startswith('c ') and (' home' in line or ' away' in line):
            parts = line.split()
            if len(parts) >= 3:
                var_num_dimacs = int(parts[1])
                var_name_z3 = parts[2]
                var_mappings[var_name_z3] = var_num_dimacs
    return var_mappings


def solve_sts_dimacs(n: int, constraints: dict = None, use_circle_method: bool = True, solver="minisat", timeout: int = 300, optimize: bool = False, verbose: bool = False):
    """
    Solve the STS problem using SAT encoding with DIMACS format.
    """
    start_time = time.time()
    
    if verbose and optimize:
        print(f"\n{'='*50}")
        print(f"PHASE 1: Finding feasible schedule (n={n}, solver={solver})")
        print(f"{'='*50}")
    
    n_weeks = n - 1
    n_periods = n // 2
    Teams = range(n)
    Weeks = range(n_weeks)
    Periods = range(n_periods)    
    
    home = [[[Bool(f"home_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]
    away = [[[Bool(f"away_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]

    solver_args = {
        "n_weeks": n_weeks,
        "n_periods": n_periods,
        "Teams": Teams,
        "Weeks": Weeks,
        "Periods": Periods,
        "home": home,
        "away": away
    }
    
    if use_circle_method:
        s = create_solver_with_circle_method(n, solver_args, constraints)
    else:
        s = create_solver(n, solver_args, constraints)

    goal = Goal()
    goal.add(s.assertions()) # Get all constraints from solver

    tactic = Then(Tactic("simplify"), Tactic("tseitin-cnf"))
    result_goals = tactic(goal)
    cnf_goal = result_goals[0]
    dimacs_string = cnf_goal.dimacs()
    var_mappings = parse_variable_mappings(dimacs_string.splitlines())

    cnf = CNF(from_string=dimacs_string)

    preprocess_time = time.time() - start_time

    if solver == "minisat":
        solver_name = "minisat"
    elif solver == "glucose":
        solver_name = "glucose"
    elif solver == "lingeling":
        solver_name = "lingeling"
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Use multiprocessing for reliable timeout (processes can be killed, threads cannot)
    # Give the solver the FULL timeout (preprocessing time doesn't count)
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_pysat_solve_worker,
        args=(cnf.clauses, solver_name, result_queue)
    )
    
    solve_start = time.time()
    process.start()
    process.join(timeout=timeout)  # Full timeout for solving
    solve_time = time.time() - solve_start
    
    if process.is_alive():
        # Timeout occurred - kill the process forcefully
        import os
        import signal
        try:
            os.kill(process.pid, signal.SIGKILL)
        except:
            pass
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.5)
        if verbose and optimize:
            print(f"  Status: TIMEOUT")
            print(f"  Time: {timeout}s")
        return {
            'solution': None,
            'time': timeout,  # Report exactly the timeout value
            'satisfiable': False,
            'obj': None
        }
    
    # Process finished - get results
    try:
        satisfiable, model = result_queue.get_nowait()
    except:
        satisfiable, model = False, None
    
    total_time = time.time() - start_time
    
    # Enforce timeout: if solve time exceeded it, report timeout
    if solve_time >= timeout:
        if verbose and optimize:
            print(f"  Status: TIMEOUT")
            print(f"  Time: {timeout}s")
        return {
            'solution': None,
            'time': timeout,
            'satisfiable': False,
            'obj': None
        }

    if not satisfiable or model is None:
        if verbose and optimize:
            print(f"  Status: NO SOLUTION FOUND")
            print(f"  Time: {total_time:.2f}s")
        return {
            'solution': None,
            'time': total_time,
            'satisfiable': False,
            'obj': None
        }

    if verbose and optimize:
        print(f"  Status: FEASIBLE SOLUTION FOUND")
        print(f"  Time: {solve_time:.2f}s")

    model_values = {}
    for val in model:
        var_num = abs(val)
        model_values[var_num] = val > 0

    sol = []
    for p in Periods:
        period = []
        for w in Weeks:
            home_team = None
            away_team = None
            for t in Teams:
                home_var = var_mappings.get(f"home_{w}_{p}_{t}")
                away_var = var_mappings.get(f"away_{w}_{p}_{t}")
                if home_var is not None and home_var in model_values and model_values[home_var]:
                    home_team = t + 1
                if away_var is not None and away_var in model_values and model_values[away_var]:
                    away_team = t + 1
            period.append([home_team, away_team])
        sol.append(period)

    if not optimize:
        return {
            'solution': sol,
            'time': total_time,
            'satisfiable': True,
            'obj': None
        }
    
    # For optimization, remaining time is based on solve_time (preprocessing doesn't count)
    remaining_time = timeout - solve_time
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"PHASE 2: Optimizing home/away balance")
        print(f"{'='*50}")
        print(f"  Remaining time: {remaining_time:.2f}s")
    
    if remaining_time <= 1:
        home_counts = [0] * (n + 1)
        for p in range(n_periods):
            for w in range(n_weeks):
                home_counts[sol[p][w][0]] += 1
        obj = max(abs(2 * hc - n_weeks) for hc in home_counts[1:])
        if verbose:
            print(f"  No time remaining for optimization!")
            print(f"  Returning feasible solution with obj={obj}")
        return {
            'solution': sol,
            'time': total_time,
            'satisfiable': True,
            'obj': obj
        }
    
    opt_result = optimize_home_away(n, sol, remaining_time, verbose=verbose)
    final_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"OPTIMIZATION RESULT")
        print(f"{'='*50}")
        print(f"  Phase 1 time: {solve_time:.2f}s")
        print(f"  Phase 2 time: {opt_result['time']:.2f}s")
        print(f"  Total time: {final_time:.2f}s")
        print(f"  Objective (max imbalance): {opt_result['obj']}")
        print(f"  Is optimal: {opt_result['is_optimal']}")
    
    return {
        'solution': opt_result['solution'],
        'time': final_time,
        'satisfiable': True,
        'obj': opt_result['obj']
    }


def print_solution_formatted(solution):
    if not solution:
        print("[]")
        return
    n_periods = len(solution)
    n_weeks = len(solution[0])
    print(f"[")
    for p in range(n_periods):
        period_matches = []
        for w in range(n_weeks):
            match = solution[p][w]
            period_matches.append(f"[{match[0]}, {match[1]}]")
        if p == n_periods - 1:
            print(f"  [{', '.join(period_matches)}]")
        else:
            print(f"  [{', '.join(period_matches)}],")
    print("]")


def main():
    
    n_teams = 12

    timeout = 300
    use_circle_method = True
    optimize = False 

    constraints = {
        'matches_per_team': True,
        'different_match_per_period': True,
        'n_fix_period': True,
        'lex_periods': True
    }

    if 0:
        result = solve_sts(n=n_teams, constraints=constraints, use_circle_method=use_circle_method, timeout=timeout, optimize=optimize, verbose=True)
    else:
        result = solve_sts_dimacs(n=n_teams, constraints=constraints, use_circle_method=use_circle_method, solver="minisat", timeout=timeout, optimize=optimize, verbose=True)
 
    if result['satisfiable']:
        print(f"\nFinal solution:")
        print_solution_formatted(result['solution'])
    else:
        print(f"\nNo solution found (time taken: {result['time']:.2f} seconds).")


if __name__ == "__main__":
    main()
