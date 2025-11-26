from z3 import *
from pysat.formula import CNF
from pysat.solvers import Minisat22, Glucose42
import sat_encodings
import time
import concurrent.futures

def create_solver(n: int, solver_args: dict[str,], constraints: dict[str, bool] =None, encoding_type: str ="bw"):
    """
    Creates a Z3 solver instance for the STS problem.

    Configures a SAT solver with constraints for the STS problem using various encoding schemes.
    Supports symmetry breaking and implied constraints that can be enabled/disabled.

    Args:
        n (int): Number of teams (must be even)
        solver_args (dict): Dictionary containing problem parameters:
        constraints (dict, optional): Dictionary of constraint flags:
            Defaults to all True if None.
        encoding_type (str, optional): Type of SAT encoding to use ('np', 'seq', 'bw', 'he')

    Returns:
        z3.Solver: Configured Z3 solver instance with all constraints added
    """
    # Default constraints
    if constraints is None:
        constraints = {
            'use_symm_break_weeks': False,
            'use_symm_break_periods': False,
            'use_symm_break_teams': True,
            'use_implied_matches_per_team': True,
            'use_implied_period_count': True
        }
    # Extract constraint flags
    use_symm_break_weeks = constraints.get('use_symm_break_weeks', True)
    use_symm_break_periods = constraints.get('use_symm_break_periods', True)
    use_symm_break_teams = constraints.get('use_symm_break_teams', True)
    use_implied_matches_per_team = constraints.get('use_implied_matches_per_team', True)
    use_implied_period_count = constraints.get('use_implied_period_count', True)

    # Validate encoding type
    valid_encodings = ['np', 'seq', 'bw', 'he']
    if encoding_type not in valid_encodings:
        raise ValueError(f"Invalid encoding type '{encoding_type}'. Must be one of: {valid_encodings}")

    # Select encoding functions based on type
    if encoding_type == 'np':
        exactly_one = sat_encodings.exactly_one_np
        at_most_k = sat_encodings.at_most_k_np
        exactly_k = sat_encodings.exactly_k_np
    elif encoding_type == 'seq':
        exactly_one = sat_encodings.exactly_one_seq
        at_most_k = sat_encodings.at_most_k_seq
        exactly_k = sat_encodings.exactly_k_seq
    elif encoding_type == 'bw':
        exactly_one = sat_encodings.exactly_one_bw
        # Note: bitwise encoding doesn't have at_most_k/exactly_k, use sequential as fallback
        at_most_k = sat_encodings.at_most_k_seq
        exactly_k = sat_encodings.exactly_k_seq
    elif encoding_type == 'he':
        exactly_one = sat_encodings.exactly_one_he
        # Note: heule encoding doesn't have at_most_k/exactly_k, use sequential as fallback
        at_most_k = sat_encodings.at_most_k_seq
        exactly_k = sat_encodings.exactly_k_seq

    s = Solver()

    weeks = solver_args['weeks']
    periods = solver_args['periods']
    Teams = solver_args['Teams']
    Weeks = solver_args['Weeks']
    Periods = solver_args['Periods']
    home = solver_args['home']
    away = solver_args['away']

    # Each slot has exactly one home and one away team, and they are different
    for w in Weeks:
        for p in Periods:
            s.add(exactly_one([home[w][p][t] for t in Teams], name=f"home_{w}_{p}"))
            s.add(exactly_one([away[w][p][t] for t in Teams], name=f"away_{w}_{p}"))
            for t in Teams:
                s.add(Implies(home[w][p][t], Not(away[w][p][t])))
                s.add(Implies(away[w][p][t], Not(home[w][p][t])))

    # Each pair plays once
    for i in Teams:
        for j in Teams:
            if i < j:
                pair_games = []
                for w in Weeks:
                    for p in Periods:
                        pair_games.append(And(home[w][p][i], away[w][p][j]))
                        pair_games.append(And(home[w][p][j], away[w][p][i]))
                s.add(exactly_one(pair_games, name=f"pair_{i}_{j}"))

    # Each team plays once per week
    for w in Weeks:
        for t in Teams:
            occ = []
            for p in Periods:
                occ.append(home[w][p][t])
                occ.append(away[w][p][t])
            s.add(exactly_one(occ, name=f"team_{t}_week_{w}"))

    # Period limit: Each team appears in same period at most twice
    for t in Teams:
        for p in Periods:
            occ = []
            for w in Weeks:
                occ.append(home[w][p][t])
                occ.append(away[w][p][t])
            s.add(at_most_k(occ, 2, name=f"team_{t}_period_{p}"))

    # Implied constraint: number of games per team
    if use_implied_matches_per_team:
        for t in Teams:
            occ = []
            for w in Weeks:
                for p in Periods:
                    occ.append(home[w][p][t])
                    occ.append(away[w][p][t])
            s.add(exactly_k(occ, weeks, name=f"team_{t}_matches"))

    # Implied constraint: total period appearances
    if use_implied_period_count:
        for t in Teams:
            occ = []
            for p in Periods:
                for w in Weeks:
                    occ.append(home[w][p][t])
                    occ.append(away[w][p][t])
            s.add(exactly_k(occ, n - 1, name=f"team_{t}_period_total"))

    # Helper function for lexicographic ordering
    def lex_less_bool(curr, next):
        # curr, next: lists of Bools
        conditions = []
        for i in range(len(curr)):
            if i == 0:
                # At position 0: curr[0] = True and next[0] = False
                condition = And(curr[i], Not(next[i]))
            else:
                # At position i: all previous positions equal, curr[i] = True, next[i] = False
                prefix_equal = [curr[j] == next[j] for j in range(i)]
                condition = And(prefix_equal + [curr[i], Not(next[i])])
            conditions.append(condition)
        return Or(conditions)

    # Symmetry breaking: weeks (but only for the first pair of weeks)
    if use_symm_break_weeks and weeks > 1:
        curr = [home[0][p][t] for p in Periods for t in Teams] + \
            [away[0][p][t] for p in Periods for t in Teams]
        nxt = [home[1][p][t] for p in Periods for t in Teams] + \
            [away[1][p][t] for p in Periods for t in Teams]
        s.add(lex_less_bool(curr, nxt))

    # Symmetry breaking: periods (but only for the first pair of periods)
    if use_symm_break_periods and periods > 1:
        curr = [home[w][0][t] for w in Weeks for t in Teams] + \
            [away[w][0][t] for w in Weeks for t in Teams]
        nxt = [home[w][1][t] for w in Weeks for t in Teams] + \
            [away[w][1][t] for w in Weeks for t in Teams]
        s.add(lex_less_bool(curr, nxt))

    # Symmetry breaking: teams (fix first week)
    if use_symm_break_teams:
        for i in Periods:
            for t in Teams:
                s.add(home[0][i][t] if t == 2 * i else Not(home[0][i][t]))
                s.add(away[0][i][t] if t == 2 * i + 1 else Not(away[0][i][t]))
    return s

def solve_sts(n, constraints=None, encoding_type="bw", timeout: int = 300):
    """
    Solve the STS problem using SAT encoding.
    
    Args:
        n (int): Number of teams (must be even)
        constraints (dict): Dictionary of constraint flags, with keys:
            - use_symm_break_weeks (bool): Apply week symmetry breaking
            - use_symm_break_periods (bool): Apply period symmetry breaking  
            - use_symm_break_teams (bool): Apply team symmetry breaking
            - use_implied_matches_per_team (bool): Add implied constraint for matches per team
            - use_implied_period_count (bool): Add implied constraint for period appearances
        encoding_type (str): Type of SAT encoding to use ('np', 'seq', 'bw', 'he')
        timeout (int): Timeout in seconds
    
    Returns:
        dict: Solution dictionary with 'solution', 'time', and 'satisfiable' keys
    """

    weeks = n - 1
    periods = n // 2
    Teams = range(n)
    Weeks = range(weeks)
    Periods = range(periods)    
    # Boolean variables: home[w][p][t] == True iff team t is home in (w,p)
    home = [[[Bool(f"home_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]
    away = [[[Bool(f"away_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]

    solver_args = {
        "weeks": weeks,
        "periods": periods,
        "Teams": Teams,
        "Weeks": Weeks,
        "Periods": Periods,
        "home": home,
        "away": away
    }
    
    s = create_solver(n, solver_args, constraints, encoding_type)

    s.set("timeout", timeout * 1000)

    # Solve
    start_time = time.time()
    result = s.check()
    solve_time = time.time() - start_time
    
    if result == sat:
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
        
        return {
            'solution': sol,
            'time': solve_time,
            'satisfiable': True
        }
    else:
        return {
            'solution': None,
            'time': solve_time,
            'satisfiable': False
        }

def parse_variable_mappings(dimacs_lines: list[str]) -> dict[str, int]:
    """
    Parse variable mappings from DIMACS format lines.
    Args:
        dimacs_lines (list[str]): List of lines from a DIMACS file
    Returns:
        dict[str, int]: Mapping from variable names to their DIMACS numbers
    """
    var_mappings = {}
    for line in dimacs_lines:
        if line.startswith('c ') and (' home' in line or ' away' in line):
            parts = line.split()
            if len(parts) >= 3:
                var_num_dimacs = int(parts[1])
                var_name_z3 = parts[2]
                var_mappings[var_name_z3] = var_num_dimacs
    return var_mappings

def solve_sts_dimacs(n: int, constraints: dict[str, bool] =None, encoding_type="bw", solver="minisat", timeout: int = 300):
    """
    Solve the STS problem using SAT encoding with DIMACS format.
    
    Args:
        n (int): Number of teams (must be even)
        constraints (dict): Dictionary of constraint flags, with keys:
            - use_symm_break_weeks (bool): Apply week symmetry breaking
            - use_symm_break_periods (bool): Apply period symmetry breaking  
            - use_symm_break_teams (bool): Apply team symmetry breaking
            - use_implied_matches_per_team (bool): Add implied constraint for matches per team
            - use_implied_period_count (bool): Add implied constraint for period appearances
        encoding_type (str): Type of SAT encoding to use ('np', 'seq', 'bw', 'he')
        solver (str): Solver used with dimacs implementation
        timeout (int): Timeout in seconds
    
    Returns:
        dict: Solution dictionary with 'solution', 'time', and 'satisfiable' keys
    """

    weeks = n - 1
    periods = n // 2
    Teams = range(n)
    Weeks = range(weeks)
    Periods = range(periods)    # Boolean variables: home[w][p][t] == True iff team t is home in (w,p)
    home = [[[Bool(f"home_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]
    away = [[[Bool(f"away_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]

    solver_args = {
        "weeks": weeks,
        "periods": periods,
        "Teams": Teams,
        "Weeks": Weeks,
        "Periods": Periods,
        "home": home,
        "away": away
    }
    
    s = create_solver(n, solver_args, constraints, encoding_type)

    start_time = time.time()
    goal = Goal()
    goal.add(s.assertions())

    tactic = Then(Tactic("simplify"), Tactic("tseitin-cnf")) # Combine tactics
    result_goals = tactic(goal)
    cnf_goal = result_goals[0]
    dimacs_string = cnf_goal.dimacs()
    var_mappings = parse_variable_mappings(dimacs_string.splitlines())

    # Parse DIMACS into PySAT CNF object
    cnf = CNF(from_string=dimacs_string)

    # Select solver
    pysat_solver = None
    if solver == "minisat":
        pysat_solver = Minisat22
    elif solver == "glucose":
        pysat_solver = Glucose42
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # Solve with timeout for pysat solvers
    def pysat_solve_with_model():
        with pysat_solver(bootstrap_with=cnf.clauses) as sat_solver:
            satisfiable = sat_solver.solve()
            model = sat_solver.get_model() if satisfiable else None
        return satisfiable, model

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pysat_solve_with_model)
        try:
            satisfiable, model = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            solve_time = time.time() - start_time
            return {
                'solution': None,
                'time': solve_time,
                'satisfiable': False
            }
    solve_time = time.time() - start_time

    if not satisfiable or model is None:
        return {
            'solution': None,
            'time': solve_time,
            'satisfiable': False
        }

    # Parse the model into variable assignments
    model_values = {}
    for val in model:
        var_num = abs(val)
        model_values[var_num] = val > 0

    # Convert solution to same format as solve_sts
    sol = []
    weeks = n - 1
    periods = n // 2
    Teams = range(n)
    Weeks = range(weeks)
    Periods = range(periods)

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

    return {
        'solution': sol,
        'time': solve_time,
        'satisfiable': True
    }
def solve_sts_optimize(n, constraints=None, encoding_type="bw", timeout: int = 300) -> dict:
    """
    Solve the optimization version of STS problem: minimize max difference between home and away games.
    
    Uses binary search to find the minimum possible maximum imbalance between home and away games
    for any team. Since we're using SAT, all constraints are expressed using only boolean variables.
    
    Args:
        n (int): Number of teams (must be even)
        constraints (dict): Dictionary of constraint flags (same as solve_sts)
        encoding_type (str): Type of SAT encoding to use ('np', 'seq', 'bw', 'he')
        timeout (int): Timeout in seconds for each SAT call
    
    Returns:
        dict: Solution dictionary with 'sol', 'time', 'satisfiable', and 'obj' keys
    """

    weeks = n - 1
    periods = n // 2
    Teams = range(n)
    Weeks = range(weeks)
    Periods = range(periods)
    
    # Boolean variables
    home = [[[Bool(f"home_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]
    away = [[[Bool(f"away_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]
    
    solver_args = {
        "weeks": weeks,
        "periods": periods,
        "Teams": Teams,
        "Weeks": Weeks,
        "Periods": Periods,
        "home": home,
        "away": away
    }
    
    # Create solver with basic STS constraints
    s = create_solver(n, solver_args, constraints, encoding_type)

    low = 1
    high = n - 1
    
    best_solution = None
    best_max_diff = None
    
    start_time = time.time()

    while low <= high:
        mid = (low + high) // 2
        time_left = timeout - math.ceil(time.time() - start_time)
        # Try to find a solution with max difference <= mid
        result = solve_sts_with_max_diff(n, s, home, away, mid, time_left)
        if result['solution']:
            best_solution = result['solution']
            best_max_diff = mid
            high = mid - 1  # Try to find better solution
        else:
            # No solution with imbalance <= mid
            low = mid + 1   # Need to allow higher imbalance
        
        # Check if we've exceeded total timeout
        if time.time() - start_time > timeout : 
            break
    
    total_time = time.time() - start_time
    
    return {
        'solution': best_solution,
        'time': total_time if total_time < timeout else timeout,
        'satisfiable': total_time < timeout,
        'obj': best_max_diff
    }


def solve_sts_with_max_diff(n, s, home, away, max_diff, time_left: int = 300):
    """
    Solve STS with constraint that no team has the difference between home and away games > max_diff.
    
    Args:
        n (int): Number of teams (must be even)
        s (z3.Solver): Z3 solver instance with basic STS constraints
        home (list): 3D list of boolean variables for home games
        away (list): 3D list of boolean variables for away games
        max_diff (int): Maximum allowed difference for any team
        time_left (int): Time left in seconds for solving
    
    Returns:
        dict: Solution dictionary with 'sol', 'time'
    """
    weeks = n - 1
    periods = n // 2
    Teams = range(n)
    Weeks = range(weeks)
    Periods = range(periods)

    for t in Teams:
        home_games = []
        for w in Weeks:
            for p in Periods:
                home_games.append(home[w][p][t])
        
        min_home_count = max(0, (weeks - max_diff) // 2)
        max_home_count = min(weeks, (weeks + max_diff) // 2)
        
        if min_home_count > 0:
            s.add(sat_encodings.at_least_k_seq(home_games, min_home_count, name=f"team_{t}_min_home"))
        if max_home_count < weeks:
            s.add(sat_encodings.at_most_k_seq(home_games, max_home_count, name=f"team_{t}_max_home"))
    
    # Solve with timeout
    s.set("timeout", time_left * 1000)
    
    start_time = time.time()
    result = s.check()
    solve_time = time.time() - start_time
    
    if result == sat:
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
        
        return {
            'solution': sol,
            'time': solve_time,
            'satisfiable': True
        }
    else:
        return {
            'solution': None,
            'time': solve_time,
            'satisfiable': False
        }



def main():
    """
    Main function for standalone execution with default parameters.
    This preserves the original behavior when sts.py is run directly.
    """
    n = 8
    constraints = {
        'use_symm_break_weeks': True,
        'use_symm_break_periods': True,
        'use_symm_break_teams': True,
        'use_implied_matches_per_team': True,
        'use_implied_period_count': True
    }
    encoding_type = "bw"  # Default to bitwise encoding
    solver = "z3"  # Default to Z3 solver
    if solver == "z3":
        result = solve_sts(n, constraints, encoding_type)
        #result = solve_sts_optimize(n, constraints, encoding_type)
    else:
        result = solve_sts_dimacs(n, constraints, encoding_type, solver=solver)
    
    if result['satisfiable']:
        print(result['solution'])
    else:
        print("unsat")


if __name__ == "__main__":
    main()
