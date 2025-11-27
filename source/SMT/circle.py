#!/usr/bin/env python3
"""
STS schedule generator + SMT check (Z3)

Uses the deterministic Circle Method to build a schedule for n teams (n even),
then encodes and checks the main STS constraints in Z3:

  - every pair of teams meets exactly once
  - every team plays exactly once per week
  - every team plays at most twice in the same period (over the tournament)

Input provenance (uploaded MiniZinc file):
  /mnt/data/CP_cyrcle_method.mzn

Run:
  python sts_circle_z3.py 8     # for n = 8 teams (default)
"""

import sys
#from z3 import Solver, IntVal, Int, Bool, If, Sum, And, Or, sat, unknown
import time
from z3 import * 

# def circle_method_schedule(n_teams):
#     """
#     Deterministic circle method schedule.
#     Returns:
#       weeks = n_teams - 1
#       periods = n_teams // 2
#       schedule[p][w] = (home, away)
#     Format: list of periods, each period is list of weeks, each entry is (home, away)
#     This follows the typical rotation algorithm:
#       - teams array of length n
#       - fix ordering, rotate all except index 0
#     Home/away assignment: deterministic (first listed team is home).
#     """
#     assert n_teams % 2 == 0, "n_teams must be even"
#     teams = list(range(1, n_teams + 1))
#     weeks = n_teams - 1
#     periods = n_teams // 2

#     # We'll use the rotation algorithm:
#     # keep teams list, for each round pair teams[i] with teams[-i-1]
#     # then rotate teams[1:] right by 1
#     schedule_by_week = []  # list of weeks, each is list of pairs (home, away)
#     cur = teams[:]
#     for r in range(weeks):
#         pairs = []
#         for i in range(periods):
#             t1 = cur[i]
#             t2 = cur[-1 - i]
#             pairs.append((t1, t2))  # deterministic: t1 home, t2 away
#         schedule_by_week.append(pairs)
#         # rotate (keep first fixed)
#         cur = [cur[0]] + [cur[-1]] + cur[1:-1]
#     # Now transform to schedule[period][week]
#     schedule = [ [ None for _ in range(weeks) ] for _ in range(periods) ]
#     for w in range(weeks):
#         week_pairs = schedule_by_week[w]
#         for p in range(periods):
#             schedule[p][w] = week_pairs[p]  # (home, away)
#     return schedule, periods, weeks
def circle_method_schedule(n_teams):
    """
    Deterministic circle method schedule with the property that the fixed team
    n_teams plays opponents 1,2,...,n_teams-1 in that order across weeks.

    Returns:
      schedule: list of periods, each period is a list of weeks, each entry is (home, away)
      periods, weeks
    """
    assert n_teams % 2 == 0, "n_teams must be even"
    weeks = n_teams - 1
    periods = n_teams // 2

    # inizializza teams in modo che cur[0] sia la squadra fissata (n_teams)
    # e cur[1:] sia in ordine decrescente: n_teams-1, ..., 1
    cur = [n_teams] + list(range(n_teams - 1, 0, -1))

    # costruisco schedule_by_week con il classico algoritmo di rotazione
    schedule_by_week = []
    for _ in range(weeks):
        pairs = []
        for i in range(periods):
            t1 = cur[i]
            t2 = cur[-1 - i]
            pairs.append((t1, t2))  # deterministico: t1 casa, t2 trasferta
        schedule_by_week.append(pairs)
        # rotate (keep first fixed)
        cur = [cur[0]] + [cur[-1]] + cur[1:-1]

    # converti in schedule[period][week]
    schedule = [[None for _ in range(weeks)] for _ in range(periods)]
    for w in range(weeks):
        week_pairs = schedule_by_week[w]
        for p in range(periods):
            schedule[p][w] = week_pairs[p]

    return schedule, periods, weeks


def pretty_print_schedule(schedule):
    periods = len(schedule)
    weeks = len(schedule[0])
    print("[")
    for p in range(periods):
        row = schedule[p]
        print("  [" + ", ".join(f"[{h} , {a}]" for (h,a) in row) + "]" + ("," if p < periods-1 else ""))
    print("]")


def z3_permutation_constraint(original_schedule, n_teams, seed=42, timeout_ms=300_000):
    """
    Costruisce variabili home/away (Int) per ogni slot (period,week)
    e vincola:
     - ogni slot è una delle coppie originali (unordered) ma SOLO tra le coppie
       previste per quella stessa settimana (non si spostano tra settimane)
     - ogni coppia originale è usata esattamente una volta (permutazione) nella
       sua settimana originale
    Restituisce (solver, home_vars, away_vars, check_result).
    """
    start_time = time.time()

    periods = len(original_schedule)
    weeks = len(original_schedule[0])
    s = Solver()
    s.set("random_seed", seed)
    s.set("timeout", timeout_ms)
   

    # variabili Int per ogni slot
    home = [[ Int(f"home_{p}_{w}") for w in range(weeks)] for p in range(periods)]
    away = [[ Int(f"away_{p}_{w}") for w in range(weeks)] for p in range(periods)]


    # lista delle coppie originali raggruppate per settimana
    orig_pairs_by_week = { w: [] for w in range(weeks) }
    for p in range(periods):
        for w in range(weeks):
            h, a = original_schedule[p][w]
            orig_pairs_by_week[w].append((h, a))

    # per ogni slot: deve essere uguale ad almeno una coppia originale della STESSA settimana (unordered)
    slot_is_pair = {}  # mappa (p,w) -> lista di Bool corrispondenti a ciascuna orig_pair della settimana w
    for p in range(periods):
        for w in range(weeks):
            pair_bools = []
            for (a, b) in orig_pairs_by_week[w]:
                pair_bools.append(Or(And(home[p][w] == a, away[p][w] == b),
                                     And(home[p][w] == b, away[p][w] == a)))
            s.add(Or(*pair_bools))
            slot_is_pair[(p, w)] = pair_bools

    # ogni coppia originale deve essere assegnata ad esattamente uno slot nella sua stessa settimana
    for w in range(weeks):
        pairs = orig_pairs_by_week[w]
        for idx, (a, b) in enumerate(pairs):
            occ_terms = []
            for p in range(periods):
                occ_terms.append(If(slot_is_pair[(p, w)][idx], IntVal(1), IntVal(0)))
            s.add(Sum(occ_terms) == 1)


    # 3) Each team appears at most twice per period (over all weeks)
    for p in range(periods):
        for t in range(1, n_teams+1):
            occur_exprs = []
            for w in range(weeks):
                # usa le variabili Z3 home[p][w] e away[p][w]
                is_play = If(Or(home[p][w] == t, away[p][w] == t), IntVal(1), IntVal(0))
                occur_exprs.append(is_play)
            s.add(Sum(occur_exprs) <= 2)

    
    # matches in the same week must be different
    for w in range(weeks):
        for p1 in range(periods):
            for p2 in range(p1 + 1, periods):
                s.add(Or(
                    home[p1][w] != home[p2][w],
                    away[p1][w] != away[p2][w]
                ))

    # Optional: enforce "fix period" constraint (adapted from MiniZinc n_fix_period)
    # MiniZinc (1-based): p_target = ((w - 1) mod n_periods) + 1
    # In this 0-based Python code we want: p_target = w % periods
    # and p_next = (p_target + 1) % periods
    # If fix_period=True then for each week w:
    #   - require team `n_teams` to play in (p_target, w)
    #   - if w < weeks-1 require team `n_teams-1` to play in (p_next, w)
    
    if n_teams >= 8:
        for w in range(weeks):
            p_target = w % periods
            p_next = (p_target + 1) % periods
            cond_target = Or(home[p_target][w] == n_teams, away[p_target][w] == n_teams)
            s.add(cond_target)
            if n_teams >= 14:
                if w < weeks - 2:
                    cond_next = Or(home[p_next][w] == n_teams - 1, away[p_next][w] == n_teams - 1)
                    # both must hold for this week when fix_period is active
                    s.add(cond_next)

    results = s.check()

    end_time = time.time()
    tot_time = end_time - start_time

    return s, home, away, results, tot_time



def optimization(home, away, n_teams, timeout=300_000):

    periods = len(home)
    weeks = len(home[0])

    opt = Optimize()
    opt.set("timeout", timeout) 

    # flip[p][w] = Bool: se True scambia home/away in quel match
    flip = [[ Bool(f"flip_{p}_{w}") for w in range(weeks)] 
            for p in range(periods) ]

    # nuove variabili per home/away dopo lo swap
    new_home = [[ Int(f"new_home_{p}_{w}") for w in range(weeks) ] 
                for p in range(periods) ]
    new_away = [[ Int(f"new_away_{p}_{w}") for w in range(weeks) ] 
                for p in range(periods) ]

    # keep home/away fixed for the first week
    for p in range(periods):
        opt.add(flip[p][0] == False)


    for p in range(periods):
        for w in range(weeks):
            opt.add(new_home[p][w] >= 1, new_home[p][w] <= n_teams)
            opt.add(new_away[p][w] >= 1, new_away[p][w] <= n_teams)


    # vincoli di swap
    for p in range(periods):
        for w in range(weeks):
            opt.add(Implies(flip[p][w], new_home[p][w] == away[p][w]))
            opt.add(Implies(flip[p][w], new_away[p][w] == home[p][w]))

            opt.add(Implies(Not(flip[p][w]), new_home[p][w] == home[p][w]))
            opt.add(Implies(Not(flip[p][w]), new_away[p][w] == away[p][w]))

    # Conteggio partite
    home_count = [ Int(f"home_count_{t}") for t in range(n_teams) ]
    away_count = [ Int(f"away_count_{t}") for t in range(n_teams) ]

    for t in range(n_teams):

        # ricomincia ogni volta!
        home_games = []
        away_games = []

        for p in range(periods):
            for w in range(weeks):
                home_games.append(If(new_home[p][w] == (t+1), 1, 0))
                away_games.append(If(new_away[p][w] == (t+1), 1, 0))

        opt.add(home_count[t] == Sum(home_games))
        opt.add(away_count[t] == Sum(away_games))

    # Minimizzare l'imbalance massimo
    max_diff = Int("max_diff")
    opt.add(max_diff >= 0)

    for t in range(n_teams):
        diff = Int(f"diff_{t}")
        opt.add(diff >= home_count[t] - away_count[t])
        opt.add(diff >= away_count[t] - home_count[t])
        opt.add(max_diff >= diff)

    opt.minimize(max_diff)
    res = opt.check()

    return res, new_home, new_away, max_diff, opt





# def z3_permutation_constraint(original_schedule, n_teams, seed=42, timeout_ms=300_000):
#     """
#     Costruisce variabili home/away (Int) per ogni slot (period,week)
#     e vincola:
#      - ogni slot è una delle coppie originali (unordered) ma SOLO tra le coppie
#        previste per quella stessa settimana (non si spostano tra settimane)
#      - ogni coppia originale è usata esattamente una volta (permutazione) nella
#        sua settimana originale
#      - ogni match originale non viene modificato, solo scambiato tra periodi della stessa settimana
#     Restituisce (solver, home_vars, away_vars, check_result).
#     """
#     periods = len(original_schedule)
#     weeks = len(original_schedule[0])
#     s = Solver()
#     s.set("random_seed", seed)
#     s.set("timeout", timeout_ms)
   
#     # variabili Int per ogni slot
#     home = [[ Int(f"home_{p}_{w}") for w in range(weeks)] for p in range(periods)]
#     away = [[ Int(f"away_{p}_{w}") for w in range(weeks)] for p in range(periods)]

#     # NEW CONSTRAINT: Each slot must contain one of the original matches from the same week
#     # and each original match must appear exactly once in its original week
#     for w in range(weeks):
#         # Collect all original matches in this week
#         original_matches_in_week = []
#         for p in range(periods):
#             h, a = original_schedule[p][w]
#             original_matches_in_week.append((h, a))
        
#         # Each slot (p, w) must be one of the original matches from week w
#         for p in range(periods):
#             match_constraints = []
#             for orig_h, orig_a in original_matches_in_week:
#                 # The match can appear as (orig_h, orig_a) or swapped as (orig_a, orig_h)
#                 match_constraints.append(
#                     And(home[p][w] == orig_h, away[p][w] == orig_a)
#                 )
#             s.add(Or(match_constraints))
        
#         # Each original match must appear exactly once in the week (permutation constraint)
#         for orig_h, orig_a in original_matches_in_week:
#             appears_count = []
#             for p in range(periods):
#                 # Check if this original match appears in period p of week w
#                 is_this_match = And(home[p][w] == orig_h, away[p][w] == orig_a)
#                 appears_count.append(If(is_this_match, IntVal(1), IntVal(0)))
#             s.add(Sum(appears_count) == 1)

#     # 3) Each team appears at most twice per period (over all weeks)
#     for p in range(periods):
#         for t in range(1, n_teams+1):
#             occur_exprs = []
#             for w in range(weeks):
#                 # usa le variabili Z3 home[p][w] e away[p][w]
#                 is_play = If(Or(home[p][w] == t, away[p][w] == t), IntVal(1), IntVal(0))
#                 occur_exprs.append(is_play)
#             s.add(Sum(occur_exprs) <= 2)

#     # enforce "fix period" constraint (adapted from MiniZinc n_fix_period)
#     # for w in range(weeks):
#     #     p_target = w % periods
#     #     p_next = (p_target + 1) % periods
#     #     cond_target = Or(home[p_target][w] == n_teams, away[p_target][w] == n_teams)
#     #     if w < weeks - 2:  # All weeks except the last
#     #         cond_next = Or(home[p_next][w] == n_teams - 1, away[p_next][w] == n_teams - 1)
#     #         s.add(And(cond_target, cond_next))
#     #     else:  # Last week only
#     #         s.add(cond_target)

#     # lexicografic ordering of periods to reduce symmetries
#     for p in range(periods - 1):
#         for w in range(weeks):
#             s.add(Or(home[p][w] < home[p + 1][w],
#                      And(home[p][w] == home[p + 1][w],
#                          away[p][w] <= away[p + 1][w])))
#             # If home teams are equal, enforce away team ordering
#             if w < weeks - 1:
#                 s.add(If(home[p][w] == home[p + 1][w],
#                          away[p][w] < away[p + 1][w],
#                          True))
    
#     return s, home, away, s.check()



def flatten_to_home_away_vectors(schedule):
    """
    Produce the typical home0/away0 vectors used in many models:
    Format required by some of your earlier code: blocks per period,
    i.e., flatten by period blocks: first all weeks of period1, then period2, ...
    Returns home0_list, away0_list (1-based indexing is not used here; lists start at 0)
    """
    periods = len(schedule)
    weeks = len(schedule[0])
    home0 = []
    away0 = []
    for p in range(periods):
        for w in range(weeks):
            h,a = schedule[p][w]
            home0.append(h)
            away0.append(a)
    return home0, away0


def print_solution_matrix(sol_or_model, home, away):
    """
    Stampa la matrice soluzione (periods x weeks) letta dal model Z3.
    sol_or_model: Solver (dopo check()) oppure Model
    home, away: matrici home[p][w], away[p][w] (Z3 Int expressions)
    """
    # ottieni il Model se è stato passato il Solver
    m = sol_or_model.model() if hasattr(sol_or_model, "model") else sol_or_model

    periods = len(home)
    weeks = len(home[0])
    print("Solution matrix (periods x weeks):")
    print("[")
    for p in range(periods):
        row_elems = []
        for w in range(weeks):
            # valutazione sicura del valore (fallback a str se non possibile as_long)
            h_eval = m.evaluate(home[p][w])
            a_eval = m.evaluate(away[p][w])
            try:
                h_val = h_eval.as_long()
            except Exception:
                h_val = str(h_eval)
            try:
                a_val = a_eval.as_long()
            except Exception:
                a_val = str(a_eval)
            row_elems.append(f"[{h_val} , {a_val}]")
        print("  [" + ", ".join(row_elems) + "]" + ("," if p < periods-1 else ""))
    print("]")
    # opzionale: ritorna la struttura come lista di tuple
    schedule = [[ None for _ in range(weeks) ] for _ in range(periods)]
    for p in range(periods):
        for w in range(weeks):
            schedule[p][w] = (int(m.evaluate(home[p][w]).as_long()),
                              int(m.evaluate(away[p][w]).as_long()))
    return schedule


# if __name__ == "__main__":
#     # read n from argv or default 8
#     if len(sys.argv) >= 2:
#         n = int(sys.argv[1])
#     else:
#         n = 8
    
#     # read from argv 'opt'or 'sat' to choose optimization or simple satisfiability check

#     schedule, periods, weeks = circle_method_schedule(n)

#     print("Generated schedule (periods x weeks):")
#     pretty_print_schedule(schedule)

#     # Flatten to home0/away0 format if needed
#     # home0_vec, away0_vec = flatten_to_home_away_vectors(schedule)
#     # print("\nFlattened home0 (length m=%d):" % len(home0_vec))
#     # print(home0_vec)
#     # print("Flattened away0:")
#     # print(away0_vec)

#     # Check with Z3
#     print("\nRunning Z3 checks on the generated schedule...")
#     s, home, away, res = z3_permutation_constraint(schedule, n)
#     print("Z3 result:", res)
#     if res == sat:
#         print("Schedule satisfies STS constraints.")
#         print_solution_matrix(s, home, away)
#         print(home, away)
#         opt, new_home, new_away, flip, max_imb = optimization(home, away, n)

#         if opt.check() == sat:
#             m = opt.model()
#             print("Maximum imbalance =", m[max_imb])
#             print("Optimized schedule:")
#             print_solution_matrix(opt, new_home, new_away)

#     elif res == unknown:
#         print("Z3 TIMEOUT: solver exceeded time limit and could not find solution.")
#     else:
#         print("Z3 reports UNSAT: schedule violates declared constraints.")

# import sys
# from z3 import *

if __name__ == "__main__":

    # --- 1) Leggi n_teams dal comando ---
    if len(sys.argv) >= 2:
        n_teams = int(sys.argv[1])
    else:
        n_teams = 8

    # --- 2) Leggi la modalità: 'sat' oppure 'opt' ---
    if len(sys.argv) >= 3:
        mode = sys.argv[2].lower()
    else:
        mode = "sat"   # default

    if mode not in ["sat", "opt"]:
        print("ERRORE: il secondo argomento deve essere 'sat' oppure 'opt'")
        sys.exit(1)

    # --- 3) Genera il calendario con il circle method ---
    schedule, periods, weeks = circle_method_schedule(n_teams)

    print("Generated schedule (periods x weeks):")
    pretty_print_schedule(schedule)

    # --- 4) Check di soddisfacibilità ---
    print("\nRunning Z3 checks on the generated schedule...")
    s, home, away, res, sat_time = z3_permutation_constraint(schedule, n_teams)

    print("Z3 result:", res)

    if res == sat:
        print("Schedule satisfies STS constraints.\n")
        print("Satisfiability check time: %.2f seconds\n" % sat_time)
        print_solution_matrix(s, home, away)

        # --- 5) Se richiesto, esegui la fase di ottimizzazione ---
        if mode == "opt":
            print("\nRunning optimization...")

            m = s.model()
            home = [[ m[home[p][w]].as_long() for w in range(weeks) ] 
            for p in range(periods) ]

            away = [[ m[away[p][w]].as_long() for w in range(weeks) ] 
                        for p in range(periods) ]

            opt_stime = time.time()
            result, new_home, new_away, max_imb, opt = optimization(home, away, n_teams, timeout=int(300_000-sat_time))                              
            opt_etime = time.time()

            if result == sat:
                m = opt.model()
                print("Optimization result: SAT")
                print("Maximum imbalance =", m[max_imb])
                print("Optimization time: %.2f seconds\n" % (opt_etime - opt_stime))
                print("Optimized schedule:")
                print_solution_matrix(opt, new_home, new_away)
            elif result == unknown:
                print("Optimization TIMEOUT: solver exceeded time limit.")
            else:
                print("Optimization UNSAT.")

    elif res == unknown:
        print("Z3 TIMEOUT: solver exceeded time limit.")
    else:
        print("Z3 reports UNSAT: schedule violates constraints.")
