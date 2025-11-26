from pysat.formula import CNF
from pysat.solvers import Minisat22, Glucose42
from z3 import *
import time
import concurrent.futures
from constraints_encoding import *
from SAT_solver_3 import *

 



def solve_sts_optimize(n, constraints = None, use_circle_method: bool = True, timeout: int = 300) -> dict:
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

    n_weeks = n - 1
    n_periods = n // 2
    Teams = range(n)
    Weeks = range(n_weeks)
    Periods = range(n_periods)
    
    # Boolean variables
    home = [[[Bool(f"home_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]
    away = [[[Bool(f"away_{w}_{p}_{t}") for t in Teams] for p in Periods] for w in Weeks]
    
    
    initial_result = solve_sts(n, constraints, use_circle_method=True, timeout=timeout)
    initial_schedule = initial_result['solution']
    initial_time = initial_result['time']
    timeout -= int(initial_time)

    low = 1
    high = n - 1
    best_solution = None
    best_max_diff = None
    start_time = time.time()

    while low <= high:
        mid = (low + high) // 2
        time_left = timeout - math.ceil(time.time() - start_time)
        if time_left <= 0:
            break
        # Try to find a solution with max difference <= mid
        result = solve_sts_with_max_diff(n, home, away, mid, initial_schedule, time_left)
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
    
    if best_solution is not None:
        return {
            'solution': best_solution,
            'time': total_time if total_time < timeout else timeout,
            'satisfiable': True,
            'obj': best_max_diff
        }
    else:
        return {
            'solution': None,
            'time': total_time if total_time < timeout else timeout,
            'satisfiable': False,
            'obj': None
        }


def solve_sts_with_max_diff(n, home, away, max_diff, initial_schedule, time_left: int = 300):
    
    
    n_weeks = n - 1
    n_periods = n // 2
    Teams = range(n)
    Weeks = range(n_weeks)
    Periods = range(n_periods)

    s = Solver()

    # Enforce only home/away swaps for each match in initial_schedule
    for w in Weeks:
        for p in Periods:
            home_team, away_team = initial_schedule[w][p]
            # Only allow (home_team, away_team) or (away_team, home_team) in this slot
            s.add(
                Or(
                    And(home[w][p][home_team - 1], away[w][p][away_team - 1]),
                    And(home[w][p][away_team - 1], away[w][p][home_team - 1])
                )
            )

    # Add max_diff constraints
    for t in Teams:
        home_games = []
        for w in Weeks:
            for p in Periods:
                home_games.append(home[w][p][t])
        min_home_count = max(0, (n_weeks - max_diff) // 2)
        max_home_count = min(n_weeks, (n_weeks + max_diff) // 2)
        if min_home_count > 0:
            s.add(at_least_k(home_games, min_home_count, name=f"team_{t}_min_home"))
        if max_home_count < n_weeks:
            s.add(at_most_k(home_games, max_home_count, name=f"team_{t}_max_home"))

    s.set("timeout", int(time_left * 1000))
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


def main():
    # PARAMETRI
    n_teams = 6

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