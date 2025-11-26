from pysat.formula import CNF
from pysat.solvers import Minisat22, Glucose42
from z3 import *
import time
import concurrent.futures
from constraints_encoding import *
from SAT_optimizer import * 


def circle_method(n):
    """
    Implements the Circle Method for round-robin scheduling with n teams,
    fixing the last team (team n-1).
    Returns a list of rounds, each round is a list of (team_i, team_j) tuples.
    Everithing indexed from 0 to n-1.
    """
    rounds = []
    teams = list(range(n))
    for r in range(n - 1):
        round_matches = []
        # team n-1 (the last) plays team r in week r
        round_matches.append((n - 1, r))
        # For i, j in N \ {r, n-1}
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


def create_solver(n: int, solver_args: dict[str,], constraints: dict[str, bool] =None):
    
    # Default constraints
    if constraints is None: 
        constraints = {
            'matches_per_team': True,
            'different_match_per_period': True,
            'n_fix_period': True,
    }
        
    # Extract implied constraints flags
    ic_matches_per_team = constraints.get('matches_per_team', True)
    ic_different_match_per_period = constraints.get('different_match_per_period', True)
    # Extract symmetry breaking flags
    n_fix_period = constraints.get('n_fix_period', True)
    
    solver = Solver()

    n_weeks = solver_args['n_weeks']
    n_periods = solver_args['n_periods']
    Teams = solver_args['Teams']
    Weeks = solver_args['Weeks']
    Periods = solver_args['Periods']
    home = solver_args['home']
    away = solver_args['away']

    #-------------------CORE CONSTRAINTS-----------------------

    # Each pair plays exactly once
    for i in Teams:
        for j in Teams:
            if i < j:
                pair_games = []
                for w in Weeks:
                    for p in Periods:
                        pair_games.append(match[i][j][w][p])
                solver.add(exactly_one(pair_games, name=f"pair_{i}_{j}"))

    # Each team plays once per week
    for w in Weeks:
        for t in Teams:
            occ = [match[h, a, w, p] 
                for p in Periods 
                for opp in Teams if opp != t
                for h, a in [(t, opp), (opp, t)]
                if (h, a, w, p) in match]
            solver.add(exactly_one(occ, name=f"team_{t}_week_{w}"))

    # Each team appears in same period at most twice
    for t in Teams:
        for p in Periods:
            occ = []
            for w in Weeks:
                for opp in Teams:
                    if t != opp:
                        occ.append(match[t][opp][w][p])  # t plays at home
                        occ.append(match[opp][t][w][p])  # t plays away
            solver.add(at_most_k(occ, 2, name=f"team_{t}_max_twice_period_{p}"))
    # No team plays vs. itself
    for i in Teams:
        for w in Weeks:
            for p in Periods:
                # La squadra i non può giocare contro se stessa
                if (i, i, w, p) in match:
                    solver.add(Not(match[i, i, w, p]), name=f"no_self_play_{i}_{w}_{p}")


    #-------------------IMPLIED CONSTRAINTS-----------------------

    if ic_matches_per_team:
        # Total matches per team = n-1
        for t in Teams:
            total_matches = []
            for w in Weeks:
                for p in Periods:
                    for opp in Teams:
                        if t != opp:
                            total_matches.append(match[t, opp, w, p])
                            total_matches.append(match[opp, t, w, p])
            solver.add(at_most_k(total_matches, n - 1, name=f"matches_per_team_{t}"))

    if ic_different_match_per_period:
        # EVERY PERIOD CONTAINS DIFFERENT MATCHES 
        for w in Weeks:
            for p in Periods:
                matches_in_period = []
                for i in Teams:
                    for j in Teams:
                        if i != j:
                            matches_in_period.append(match[i][j][w][p])
                
                solver.add(at_most_one(matches_in_period, name=f"unique_matches_week_{w}_period_{p}"))
    

    #-------------------SYMMETRY BREACKING-----------------------

    if n_fix_period:
        # For each week except the last, fix team n in p_target and team n-1 in p_next
        for w in range(n_weeks - 1):  # Exclude last week
            p_target = w % n_periods
            p_next = (w + 1) % n_periods
            # Team n (index n-1) must play in period p_target in week w
            team_n_matches = [match[n - 1][j][w][p_target] for j in Teams if j != n - 1]
            solver.add(Or(team_n_matches))
            # Team n-1 (index n-2) must play in period p_next in week w
            team_n1_matches = [match[n - 2][j][w][p_next] for j in Teams if j != n - 2]
            solver.add(Or(team_n1_matches))

    return solver



def create_solver_with_circle_method(n: int, solver_args: dict[str,], constraints: dict[str, bool] =None):
    """
    Creates a Z3 solver instance for the STS problem.

    Configures a SAT solver with constraints for the STS problem, starting from 
        an initial schedule generated by the circle method.
    Supports symmetry breaking and implied constraints that can be enabled/disabled.

    Args:
        n: Number of teams (must be even)
        solver_args: Dictionary containing problem parameters:
        constraints: Dictionary of constraint flags:
                    
    Returns:
        z3.Solver: Configured Z3 solver instance with all constraints added
    """
    # Default constraints
    if constraints is None: 
        constraints = {
            'matches_per_team': True,
            'different_match_per_period': True,
            'n_fix_period': True,
    }
    
    # Extract implied constraints flags
    ic_matches_per_team = constraints.get('matches_per_team', True)
    ic_different_match_per_period = constraints.get('different_match_per_period', True)
    # Extract symmetry breaking flags
    n_fix_period = constraints.get('n_fix_period', True)
    
    solver = Solver()

    n_weeks = solver_args['n_weeks']
    n_periods = solver_args['n_periods']
    Teams = solver_args['Teams']
    Weeks = solver_args['Weeks']
    Periods = solver_args['Periods']
    home = solver_args['home']
    away = solver_args['away']


    #-------------------CORE CONSTRAINTS-----------------------

    # INITIAL WEEK MAPPING FROM CIRCLE METHOD

    # Matches from the circle method are fixed to their specific weeks, but can be in any period
    initial_schedule = circle_method(n)

    # EVERY MATCH FROM A WEEK OF THE INITIAL SCHEDULE IS ASSIGNED TO EXACTLY ONE PERIOD IN THAT WEEK
    for w, week_matches in enumerate(initial_schedule):
        for p in Periods:
            match_in_period_vars = []
            for (i, j) in week_matches:
                match_var = Bool(f"match_{i}_{j}_week_{w}_period_{p}")
                match_in_period_vars.append(match_var)
                solver.add(Implies(match_var, And(home[w][p][i], away[w][p][j])))
                solver.add(Implies(Not(match_var), And(Not(home[w][p][i]), Not(away[w][p][j]))))
            # Enforce exactly one match per period
            solver.add(exactly_one(match_in_period_vars, name=f"week_{w}_period_{p}_one_match"))

    # EACH MATCH CANNOT BE ASSIGNED TO MULTIPLE PERIODS
    for w, week_matches in enumerate(initial_schedule):
        for (i, j) in week_matches:
            match_period_vars = []
            for p in Periods:
                match_var = Bool(f"match_{i}_{j}_week_{w}_period_{p}")
                match_period_vars.append(match_var)
            # Enforce that this match is assigned to exactly one period in week w
            solver.add(exactly_one(match_period_vars, name=f"match_{i}_{j}_week_{w}_one_period"))


    # EVERY TEAM PLAYS MAX TWICE PER PERIOD DURING THE WHOLE TOURNAMENT
    for t in Teams:
        for p in Periods:
            occ = []
            for w in Weeks:
                occ.append(home[w][p][t])  # t plays at home in week w, period p
                occ.append(away[w][p][t])  # t plays away in week w, period p
            solver.add(at_most_k(occ, 2, name=f"team_{t}_max_twice_period_{p}"))


    #-------------------IMPLIED CONSTRAINTS-----------------------

    '''if ic_matches_per_team:
        # TOTAL MATCHES PER TEAM = n-1
        for t in Teams:
            total_matches = []
            for w in Weeks:
                for p in Periods:
                    total_matches.append(home[w][p][t])  # t plays at home in week w, period p
                    total_matches.append(away[w][p][t])  # t plays away in week w, period p
            solver.add(exactly_k(total_matches, n - 1, name=f"matches_per_team_{t}"))'''
    

    #-------------------SYMMETRY BREACKING-----------------------

    if n_fix_period:
    
        for w in range(n_weeks - 1):  # Exclude last week
            p_target = w % n_periods
            p_next = (w + 1) % n_periods
            # Team n (index n-1) must play in period p_target in week w (either home or away)
            team_n_period_vars = [home[w][p_target][n - 1], away[w][p_target][n - 1]]
            solver.add(Or(team_n_period_vars))
            if n >= 14:
                # Team n-1 (index n-2) must play in period p_next in week w (either home or away)
                team_n1_period_vars = [home[w][p_next][n - 2], away[w][p_next][n - 2]]
                solver.add(Or(team_n1_period_vars))


    return solver


def solve_sts(n: int, constraints=None, use_circle_method: bool = True, timeout: int = 300):
    """
    Solve the STS problem using SAT encoding.
    -use_circle_method: use the circle method for initial scheduling
    Args:
        n: Number of teams (must be even)
        constraints: Dictionary of constraint flags, with keys:
            -ic_matches_per_team: use implied constraint to set number of matches per team
            -ic_different_match_per_period: use implied constraint to set different matches each period
            -n_fix_period: use symmetry breaking to fix specific periods for certain teams
        c_m_flag: Whether to use the circle method for initial scheduling
        timeout: Timeout in seconds
    
    Returns:
        dict: Solution dictionary with 'solution', 'time', and 'satisfiable' keys
    """

    n_weeks = n - 1
    n_periods = n // 2
    Teams = range(n)
    Weeks = range(n_weeks)
    Periods = range(n_periods)    

     # Boolean variables: home[w][p][t] == True iff team t is home in (w,p)
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
    '''else:
        s = create_solver(n, solver_args, constraints)'''

    s.set("timeout", timeout * 1000)  # Timeout in milliseconds

    # Solve
    start_time = time.time()
    result = s.check()
    solve_time = time.time() - start_time
    
    if result == sat:
        m = s.model()
        sol = []
        for w in Weeks:
            week = []
            for p in Periods:
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

                week.append([home_team, away_team])
            sol.append(week)
        
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
        dimacs_lines: List of strings representing lines in DIMACS format.
    Returns:
        dict[str, int]: Mapping from Z3 variable (str) to DIMACS variable (int).
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



def solve_sts_dimacs(n: int, constraints: dict[str, bool] =None, use_circle_method: bool = True, solver="minisat", timeout: int = 300):
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

    n_weeks = n - 1
    n_periods = n // 2
    Teams = range(n)
    Weeks = range(n_weeks)
    Periods = range(n_periods)    # Boolean variables: home[w][p][t] == True iff team t is home in (w,p)
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







def main():
    # PARAMETRI
    n_teams = 20


    timeout = 300  
    use_circle_method = True  

    # VINCOLI
    constraints = {
        'matches_per_team': True,
        'different_match_per_period': True,
        'n_fix_period': True
    }

    if 1:
        result = solve_sts(n=n_teams, constraints=constraints, use_circle_method=use_circle_method, timeout=timeout)
    else:       
        result = solve_sts_optimize(n=n_teams, constraints=constraints, use_circle_method= use_circle_method, timeout=timeout)

    if result['satisfiable']:
        print(f"Soluzione trovata in {result['time']:.2f} secondi:")
        for week_idx, week in enumerate(result['solution'], 1):
            print(f"Settimana {week_idx}:")
            for period_idx, period in enumerate(week, 1):
                print(f"  Periodo {period_idx}: {period}")
    else:
        print(f"Nessuna soluzione trovata (tempo impiegato: {result['time']:.2f} secondi).")


'''def main():
    n = 10  # You can change this value to test with different numbers of teams
    rounds = circle_method(n)
    for round_num, matches in enumerate(rounds, 1):
        print(f"Round {round_num}:")
        for match in matches:
            print(f"  Team {match[0]} vs Team {match[1]}")
        print()'''

if __name__ == "__main__":
    main()