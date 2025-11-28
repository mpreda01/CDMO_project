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
from z3 import Solver, IntVal, Int, Bool, If, Sum, And, Or, sat, unknown, Distinct, Implies, BoolVal, Optimize, Not
import time



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


# def z3_permutation_constraint(original_schedule, n_teams, seed=42, timeout_ms=300_000):
#     """
#     Costruisce variabili home/away (Int) per ogni slot (period,week)
#     e vincola:
#      - ogni slot è una delle coppie originali (unordered) ma SOLO tra le coppie
#        previste per quella stessa settimana (non si spostano tra settimane)
#      - ogni coppia originale è usata esattamente una volta (permutazione) nella
#        sua settimana originale
#     Restituisce (solver, home_vars, away_vars, check_result).
#     """
#     start_time = time.time()

#     periods = len(original_schedule)
#     weeks = len(original_schedule[0])
#     s = Solver()
#     s.set("random_seed", seed)
#     s.set("timeout", timeout_ms)
   

#     # variabili Int per ogni slot
#     home = [[ Int(f"home_{p}_{w}") for w in range(weeks)] for p in range(periods)]
#     away = [[ Int(f"away_{p}_{w}") for w in range(weeks)] for p in range(periods)]


#     # lista delle coppie originali raggruppate per settimana
#     orig_pairs_by_week = { w: [] for w in range(weeks) }
#     for p in range(periods):
#         for w in range(weeks):
#             h, a = original_schedule[p][w]
#             orig_pairs_by_week[w].append((h, a))

#     # per ogni slot: deve essere uguale ad almeno una coppia originale della STESSA settimana (unordered)
#     slot_is_pair = {}  # mappa (p,w) -> lista di Bool corrispondenti a ciascuna orig_pair della settimana w
#     for p in range(periods):
#         for w in range(weeks):
#             pair_bools = []
#             for (a, b) in orig_pairs_by_week[w]:
#                 pair_bools.append(Or(And(home[p][w] == a, away[p][w] == b),
#                                      And(home[p][w] == b, away[p][w] == a)))
#             s.add(Or(*pair_bools))
#             slot_is_pair[(p, w)] = pair_bools

#     # ogni coppia originale deve essere assegnata ad esattamente uno slot nella sua stessa settimana
#     for w in range(weeks):
#         pairs = orig_pairs_by_week[w]
#         for idx, (a, b) in enumerate(pairs):
#             occ_terms = []
#             for p in range(periods):
#                 occ_terms.append(If(slot_is_pair[(p, w)][idx], IntVal(1), IntVal(0)))
#             s.add(Sum(occ_terms) == 1)


#     # 3) Each team appears at most twice per period (over all weeks)
#     for p in range(periods):
#         for t in range(1, n_teams+1):
#             occur_exprs = []
#             for w in range(weeks):
#                 # usa le variabili Z3 home[p][w] e away[p][w]
#                 is_play = If(Or(home[p][w] == t, away[p][w] == t), IntVal(1), IntVal(0))
#                 occur_exprs.append(is_play)
#             s.add(Sum(occur_exprs) <= 2)

    
#     # matches in the same week must be different
#     for w in range(weeks):
#         for p1 in range(periods):
#             for p2 in range(p1 + 1, periods):
#                 s.add(Or(
#                     home[p1][w] != home[p2][w],
#                     away[p1][w] != away[p2][w]
#                 ))

#     # Optional: enforce "fix period" constraint (adapted from MiniZinc n_fix_period)
#     # MiniZinc (1-based): p_target = ((w - 1) mod n_periods) + 1
#     # In this 0-based Python code we want: p_target = w % periods
#     # and p_next = (p_target + 1) % periods
#     # If fix_period=True then for each week w:
#     #   - require team `n_teams` to play in (p_target, w)
#     #   - if w < weeks-1 require team `n_teams-1` to play in (p_next, w)
    
#     if n_teams >= 8:
#         for w in range(weeks):
#             p_target = w % periods
#             p_next = (p_target + 1) % periods
#             cond_target = Or(home[p_target][w] == n_teams, away[p_target][w] == n_teams)
#             s.add(cond_target)
#             if n_teams >= 14:
#                 if w < weeks - 2:
#                     cond_next = Or(home[p_next][w] == n_teams - 1, away[p_next][w] == n_teams - 1)
#                     # both must hold for this week when fix_period is active
#                     s.add(cond_next)

#     results = s.check()

#     end_time = time.time()
#     tot_time = end_time - start_time

#     return s, home, away, results, tot_time


def z3_permutation_constraint(original_schedule,
                              n_teams,
                              seed=42,
                              timeout_ms=300_000,
                              ic_fix_period=True,
                              ic_lex_periods=False):
    """
    Encoding basato su `assign[p][w]` (int index della partita della settimana w
    assegnata al periodo p). Le coppie originali rimangono costanti (original_schedule).
    Restituisce: (solver, assign, home_out, away_out, result, elapsed_seconds)

    - original_schedule: list(periods)[ list(weeks)[ (home,away) ] ]
    - n_teams: numero di squadre
    - seed: seed per il solver (Solver accetta random_seed)
    - timeout_ms: timeout in millisecondi per il solver
    - ic_fix_period: applica il vincolo fix_period per n_teams / n_teams-1
    - ic_lex_periods: applica vincolo lessicografico fra vettori assign dei periodi
    """

    start_time = time.time()

    periods = len(original_schedule)
    weeks = len(original_schedule[0])

    s = Solver()
  
    s.set("random_seed", seed)
    s.set("timeout", timeout_ms)

    # ------------------------
    # PRECOMPUTE: orig_pairs_by_week e m_t_w
    # ------------------------
    orig_pairs_by_week = { w: [ original_schedule[p][w] for p in range(periods) ]
                           for w in range(weeks) }

    # m_t_w[(t,w)] = index k in 0..periods-1 of match in week w where team t plays
    m_t_w = {}
    for w in range(weeks):
        for k, (h, a) in enumerate(orig_pairs_by_week[w]):
            m_t_w[(h, w)] = k
            m_t_w[(a, w)] = k

    # sanity check (optional): every team must appear exactly once per week in original_schedule
    # (we assume input valid; you can uncomment check if needed)
    # for t in range(1, n_teams+1):
    #     for w in range(weeks):
    #         assert (t, w) in m_t_w, f"team {t} missing in week {w}"

    # ------------------------
    # VARIABLES
    # ------------------------
    assign = [[ Int(f"assign_{p}_{w}") for w in range(weeks) ] for p in range(periods)]
    # pos[t,w] = period p where team t's match in week w is placed (channeling)
    pos = { (t, w): Int(f"pos_{t}_{w}") for t in range(1, n_teams+1) for w in range(weeks) }

    # ------------------------
    # DOMAIN + PERMUTATION PER SETTIMANA
    # ------------------------
    for w in range(weeks):
        for p in range(periods):
            s.add(assign[p][w] >= 0, assign[p][w] < periods)
        # Distinct => permutation of indices 0..periods-1
        s.add(Distinct(*[ assign[p][w] for p in range(periods) ]))

    # ------------------------
    # CHANNEL pos <-> assign
    # pos encodes for each team t and week w the period where its match has been placed
    # ------------------------
    for t in range(1, n_teams+1):
        for w in range(weeks):
            # domain for pos
            s.add(pos[(t, w)] >= 0, pos[(t, w)] < periods)
            m_idx = m_t_w[(t, w)]
            # if assign[p][w] == m_idx then pos[t,w] == p
            for p in range(periods):
                s.add(Implies(assign[p][w] == m_idx, pos[(t, w)] == p))
            # conversely, if pos[t,w] == p then assign[p][w] == m_idx
            for p in range(periods):
                s.add(Implies(pos[(t, w)] == p, assign[p][w] == m_idx))

    # ------------------------
    # CAPACITY: ogni squadra appare al massimo 2 volte PER PERIODO (sommando tutte le settimane)
    # implementato tramite pos: Sum_w If(pos[t,w] == p,1,0) <= 2
    # ------------------------
    for p in range(periods):
        for t in range(1, n_teams+1):
            occ_terms = [ If(pos[(t, w)] == p, IntVal(1), IntVal(0)) for w in range(weeks) ]
            s.add(Sum(occ_terms) <= 2)

    # ------------------------
    # IMPLIED CONSTRAINTS / BOUNDS / GLOBAL INVARIANTS
    # ------------------------
    total_matches = periods * weeks  # numero di match totali
    # count appearances per team across periods
    home_count = { (t, p): Int(f"count_{t}_{p}") for t in range(1, n_teams+1) for p in range(periods) }
    for t in range(1, n_teams+1):
        for p in range(periods):
            expr = Sum([ If(pos[(t, w)] == p, IntVal(1), IntVal(0)) for w in range(weeks) ])
            s.add(home_count[(t, p)] == expr)
            s.add(home_count[(t, p)] >= 0, home_count[(t, p)] <= weeks)

    # per team: somma su p dei count deve essere settimane (ogni team gioca una partita per settimana)
    for t in range(1, n_teams+1):
        s.add(Sum([ home_count[(t, p)] for p in range(periods) ]) == weeks)

    # totale apparizioni (controllo)
    sum_all = Sum([ home_count[(t, p)] for t in range(1, n_teams+1) for p in range(periods) ])
    # ogni match conta due apparizioni di team, quindi total team appearances = 2 * total_matches
    s.add(sum_all == 2 * total_matches)

    # ------------------------
    # SYMMETRY-BREAKING
    # ------------------------
    # A) canonical choice: fix first slot (period 0, week 0) to first match of week 0 (index 0)
    # s.add(assign[0][0] == 0)

    # B) fix_period style (team N in p_target, optionally team N-1 in p_next)
    if ic_fix_period and n_teams >= 8:
        for w in range(weeks):
            p_target = w % periods
            idx_n = m_t_w.get((n_teams, w), None)
            if idx_n is not None:
                s.add(assign[p_target][w] == idx_n)
            # optional second fix for larger tournaments
            if n_teams >= 14:
                p_next = (p_target + 1) % periods
                idx_n1 = m_t_w.get((n_teams - 1, w), None)
                if idx_n1 is not None and w < weeks - 2:
                    s.add(assign[p_next][w] == idx_n1)

    # C) lexicographic order among period-vectors (assign[p] treated as vector over weeks)
    #    This breaks permutations of entire period labels.
    if ic_lex_periods and periods > 1:
        def lex_leq(vec1, vec2):
            # vec1,vec2 are lists of Int expr length weeks
            ors = []
            for i in range(weeks):
                if i == 0:
                    prefix_eq = BoolVal(True)
                else:
                    prefix_eq = And(*[ vec1[j] == vec2[j] for j in range(i) ])
                ors.append( And(prefix_eq, vec1[i] <= vec2[i]) )
            return Or(*ors)
        for p in range(periods - 1):
            s.add( lex_leq([ assign[p][w] for w in range(weeks) ],
                           [ assign[p+1][w] for w in range(weeks) ]) )

    # ------------------------
    # SAT solve
    # ------------------------
    result = s.check()
    tot_time = time.time() - start_time

    # compute home_out/away_out only if sat
    home_out = None
    away_out = None
    if result == sat:
        m = s.model()
        home_out = [[ None for _ in range(weeks) ] for _ in range(periods)]
        away_out = [[ None for _ in range(weeks) ] for _ in range(periods)]
        for p in range(periods):
            for w in range(weeks):
                k = m[assign[p][w]].as_long()
                h, a = orig_pairs_by_week[w][k]
                home_out[p][w] = h
                away_out[p][w] = a

    return s, assign, home_out, away_out, result, tot_time




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
    for w in range(int(weeks / 2)):
        for p in range(periods):
            opt.add(flip[p][w] == False)


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
    # optimization lower bound
    opt.add(max_diff >= 0)

    for t in range(n_teams):
        diff = Int(f"diff_{t}")
        opt.add(diff >= home_count[t] - away_count[t])
        opt.add(diff >= away_count[t] - home_count[t])
        opt.add(max_diff >= diff)

    # optimize upper bound
    opt.add(max_diff <= 1)

    opt.minimize(max_diff)
    res = opt.check()

    if res == sat:
        m = opt.model()

        new_home_out = [[0]*weeks for _ in range(periods)]
        new_away_out = [[0]*weeks for _ in range(periods)]

        for p in range(periods):
            for w in range(weeks):
                h = m.evaluate(new_home[p][w], model_completion=True).as_long()
                a = m.evaluate(new_away[p][w], model_completion=True).as_long()

                new_home_out[p][w] = h
                new_away_out[p][w] = a



        return res, new_home_out, new_away_out, max_diff, opt
    
    return res, None, None, None, opt









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


# def print_solution_matrix(sol_or_model, home, away):
#     """
#     Stampa la matrice soluzione (periods x weeks) letta dal model Z3.
#     sol_or_model: Solver (dopo check()) oppure Model
#     home, away: matrici home[p][w], away[p][w] (Z3 Int expressions)
#     """
#     # ottieni il Model se è stato passato il Solver
#     m = sol_or_model.model() if hasattr(sol_or_model, "model") else sol_or_model

#     periods = len(home)
#     weeks = len(home[0])
#     print("Solution matrix (periods x weeks):")
#     print("[")
#     for p in range(periods):
#         row_elems = []
#         for w in range(weeks):
#             # valutazione sicura del valore (fallback a str se non possibile as_long)
#             h_eval = m.evaluate(home[p][w])
#             a_eval = m.evaluate(away[p][w])
#             try:
#                 h_val = h_eval.as_long()
#             except Exception:
#                 h_val = str(h_eval)
#             try:
#                 a_val = a_eval.as_long()
#             except Exception:
#                 a_val = str(a_eval)
#             row_elems.append(f"[{h_val} , {a_val}]")
#         print("  [" + ", ".join(row_elems) + "]" + ("," if p < periods-1 else ""))
#     print("]")
#     # opzionale: ritorna la struttura come lista di tuple
#     schedule = [[ None for _ in range(weeks) ] for _ in range(periods)]
#     for p in range(periods):
#         for w in range(weeks):
#             schedule[p][w] = (int(m.evaluate(home[p][w]).as_long()),
#                               int(m.evaluate(away[p][w]).as_long()))
#     return schedule
def print_solution_matrix(home, away):

    periods = len(home)
    weeks = len(home[0])

    print("[")
    for p in range(periods):
        print("  [", end="")
        row_elems = []
        for w in range(weeks):
            h = home[p][w]
            a = away[p][w]
            row_elems.append(f"[{h} , {a}]")
        print(", ".join(row_elems), end="")
        print("]", end="")
        if p < periods - 1:
            print(",")
        else:
            print()
    print("]")



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
    s, _, home, away, res, sat_time = z3_permutation_constraint(schedule, n_teams)

    print("Z3 result:", res)

    if res == sat:
        print("Schedule satisfies STS constraints.\n")
        print("Satisfiability check time: %.2f seconds\n" % sat_time)
        print_solution_matrix(home, away)

        # --- 5) Se richiesto, esegui la fase di ottimizzazione ---
        if mode == "opt":
            print("\nRunning optimization...")

            # m = s.model()
            # home = [[ m[home[p][w]].as_long() for w in range(weeks) ] 
            # for p in range(periods) ]

            # away = [[ m[away[p][w]].as_long() for w in range(weeks) ] 
            #             for p in range(periods) ]

            opt_stime = time.time()
            result, new_home, new_away, max_imb, opt = optimization(home, away, n_teams, timeout=int(300_000-sat_time))                              
            opt_etime = time.time()

            if result == sat:
                m = opt.model()
                print("Optimization result: SAT")
                print("Maximum imbalance =", m[max_imb])
                print("Optimization time: %.2f seconds\n" % (opt_etime - opt_stime))
                print("Optimized schedule:")
                print_solution_matrix(new_home, new_away)
            elif result == unknown:
                print("Optimization TIMEOUT: solver exceeded time limit.")
            else:
                print("Optimization UNSAT.")

    elif res == unknown:
        print("Z3 TIMEOUT: solver exceeded time limit.")
    else:
        print("Z3 reports UNSAT: schedule violates constraints.")
