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
from z3 import Solver, IntVal, Int, Bool, If, Sum, And, Or, sat, unknown, Distinct, Implies, BoolVal, Optimize, Not, Tactic, Then, PbLe
import time
import subprocess
import os
import random



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


def find_good_seed(original_schedule, n_teams, num_trials=10, timeout_per_trial=15000,
                   sb_fix_period=True, sb_lex_periods=False, 
                   implied_constraints=True):
    """
    Try multiple seeds and return the best one based on solve time.
    
    Args:
        original_schedule: schedule to solve
        n_teams: number of teams
        num_trials: number of seeds to try
        timeout_per_trial: timeout in ms for each trial
        
    Returns:
        best_seed: the seed that produced the fastest SAT result
    """
    best_seed = 42
    best_time = float('inf')
    
    # Try default seed + random seeds
    seeds_to_try = [42, 123, 456, 789, 1337] + [random.randint(1, 100000) for _ in range(num_trials-5)]
    
    print(f"\n{'='*60}")
    print(f"SEED SEARCH: Trying {num_trials} different seeds")
    print(f"{'='*60}")
    
    for seed in seeds_to_try:
        print(f"Seed {seed:6d}... ", end="", flush=True)
        _, _, _, _, res, t = z3_permutation_constraint(
            original_schedule, n_teams, 
            seed=seed, 
            timeout_ms=timeout_per_trial,
            sb_fix_period=sb_fix_period,
            sb_lex_periods=sb_lex_periods,
            implied_constraints=implied_constraints
        )
        
        if res == sat:
            print(f"SAT in {t:6.2f}s {'✓ NEW BEST' if t < best_time else ''}")
            if t < best_time:
                best_time = t
                best_seed = seed
                if t < 2:  # Good enough, stop early
                    print(f"Found excellent seed ({t:.2f}s < 2s), stopping search")
                    break
        else:
            print(f"{'TIMEOUT' if res == unknown else 'UNSAT':>8}")
    
    print(f"{'='*60}")
    print(f"Best seed: {best_seed} (solved in {best_time:.2f}s)")
    print(f"{'='*60}\n")
    return best_seed

def z3_permutation_constraint(original_schedule,
                              n_teams,
                              seed=42,
                              timeout_ms=300_000,
                              sb_fix_period=True,
                              sb_lex_periods=False,
                              implied_constraints=True,):
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
    questa viene da git e fa n18
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
    if implied_constraints:
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
    if sb_fix_period and n_teams >= 8:
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
    if sb_lex_periods and periods > 1:
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




def z3_permutation_constraint_cvc5(original_schedule,
                                    n_teams,
                                    seed=42,
                                    timeout_ms=300_000,
                                    sb_fix_period=True,
                                    sb_lex_periods=False,
                                    implied_constraints=True,
                                    ):
    """
    Same as z3_permutation_constraint but exports to SMT-LIB2 and uses cvc5 solver.
    Returns: (solver, assign, home_out, away_out, result, elapsed_seconds)
    """
    start_time = time.time()

    periods = len(original_schedule)
    weeks = len(original_schedule[0])

    s = Solver()
    s.set("random_seed", seed)

    # ------------------------
    # PRECOMPUTE: orig_pairs_by_week e m_t_w
    # ------------------------
    orig_pairs_by_week = { w: [ original_schedule[p][w] for p in range(periods) ]
                           for w in range(weeks) }

    m_t_w = {}
    for w in range(weeks):
        for k, (h, a) in enumerate(orig_pairs_by_week[w]):
            m_t_w[(h, w)] = k
            m_t_w[(a, w)] = k

    # ------------------------
    # VARIABLES
    # ------------------------
    assign = [[ Int(f"assign_{p}_{w}") for w in range(weeks) ] for p in range(periods)]
    pos = { (t, w): Int(f"pos_{t}_{w}") for t in range(1, n_teams+1) for w in range(weeks) }

    # ------------------------
    # DOMAIN + PERMUTATION PER SETTIMANA
    # ------------------------
    for w in range(weeks):
        for p in range(periods):
            s.add(assign[p][w] >= 0, assign[p][w] < periods)
        s.add(Distinct(*[ assign[p][w] for p in range(periods) ]))

    # ------------------------
    # CHANNEL pos <-> assign
    # ------------------------
    for t in range(1, n_teams+1):
        for w in range(weeks):
            s.add(pos[(t, w)] >= 0, pos[(t, w)] < periods)
            m_idx = m_t_w[(t, w)]
            for p in range(periods):
                s.add(Implies(assign[p][w] == m_idx, pos[(t, w)] == p))
            for p in range(periods):
                s.add(Implies(pos[(t, w)] == p, assign[p][w] == m_idx))

    # ------------------------
    # CAPACITY
    # ------------------------
    for p in range(periods):
        for t in range(1, n_teams+1):
            occ_terms = [ If(pos[(t, w)] == p, IntVal(1), IntVal(0)) for w in range(weeks) ]
            # Avoid Sum with single element for SMT-LIB2 compatibility
            if len(occ_terms) == 1:
                s.add(occ_terms[0] <= 2)
            else:
                s.add(Sum(occ_terms) <= 2)

    # ------------------------
    # IMPLIED CONSTRAINTS
    # ------------------------
    if implied_constraints:
        total_matches = periods * weeks
        home_count = { (t, p): Int(f"count_{t}_{p}") for t in range(1, n_teams+1) for p in range(periods) }
        for t in range(1, n_teams+1):
            for p in range(periods):
                terms = [ If(pos[(t, w)] == p, IntVal(1), IntVal(0)) for w in range(weeks) ]
                # Avoid Sum with single element for SMT-LIB2 compatibility
                if len(terms) == 1:
                    expr = terms[0]
                else:
                    expr = Sum(terms)
                s.add(home_count[(t, p)] == expr)
                s.add(home_count[(t, p)] >= 0, home_count[(t, p)] <= weeks)

        for t in range(1, n_teams+1):
            terms = [ home_count[(t, p)] for p in range(periods) ]
            if len(terms) == 1:
                s.add(terms[0] == weeks)
            else:
                s.add(Sum(terms) == weeks)

        all_terms = [ home_count[(t, p)] for t in range(1, n_teams+1) for p in range(periods) ]
        if len(all_terms) == 1:
            s.add(all_terms[0] == 2 * total_matches)
        else:
            s.add(Sum(all_terms) == 2 * total_matches)
    


    # ------------------------
    # SYMMETRY-BREAKING
    # ------------------------
    if sb_fix_period and n_teams >= 8:
        for w in range(weeks):
            p_target = w % periods
            idx_n = m_t_w.get((n_teams, w), None)
            if idx_n is not None:
                s.add(assign[p_target][w] == idx_n)
            if n_teams >= 14:
                p_next = (p_target + 1) % periods
                idx_n1 = m_t_w.get((n_teams - 1, w), None)
                if idx_n1 is not None and w < weeks - 2:
                    s.add(assign[p_next][w] == idx_n1)

    if sb_lex_periods and periods > 1:
        def lex_leq(vec1, vec2):
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
    # Export to SMT-LIB2 and solve with cvc5
    # ------------------------
    smt2_string = s.to_smt2()
    smt2_lines = smt2_string.splitlines()
    smt2_lines[0] = "(set-logic QF_LIA)"
    smt2_lines.append("(get-model)")
    smt2_string = "\n".join(smt2_lines)

    smt2_path = "./sts_smt.smt2"
    with open(smt2_path, 'w') as f:
        f.write(smt2_string)

    # Determine cvc5 binary path based on OS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == 'nt':  # Windows
        cvc5_default = os.path.join(script_dir, "cvc5", "bin", "cvc5.exe")
    else:  # Linux/Mac
        cvc5_default = os.path.join(script_dir, "cvc5", "bin", "cvc5")
    
    cvc5_bin = os.environ.get("CVC5_BIN", cvc5_default)
    cmd = [cvc5_bin, "--lang", "smt2", "--produce-models", smt2_path]

    timeout_sec = timeout_ms / 1000.0
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        tot_time = time.time() - start_time
        # Clean up temporary file
        if os.path.exists(smt2_path):
            os.remove(smt2_path)
        return s, assign, None, None, unknown, tot_time

    tot_time = time.time() - start_time
    stdout = result.stdout.strip()

    if "unsat" in stdout:
        # Clean up temporary file
        if os.path.exists(smt2_path):
            os.remove(smt2_path)
        return s, assign, None, None, "unsat", tot_time
    if "sat" not in stdout:
        # Clean up temporary file
        if os.path.exists(smt2_path):
            os.remove(smt2_path)
        return s, assign, None, None, unknown, tot_time

    # Parse model from cvc5 output
    model = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("(define-fun"):
            line = line[:-1]
            parts = line.split()
            if len(parts) >= 5:
                var = parts[1]
                value = int(parts[-1])
                model[var] = value

    home_out = [[ None for _ in range(weeks) ] for _ in range(periods)]
    away_out = [[ None for _ in range(weeks) ] for _ in range(periods)]
    for p in range(periods):
        for w in range(weeks):
            k = model.get(f"assign_{p}_{w}", None)
            if k is not None:
                h, a = orig_pairs_by_week[w][k]
                home_out[p][w] = h
                away_out[p][w] = a

    # Clean up temporary file
    if os.path.exists(smt2_path):
        os.remove(smt2_path)

    return s, assign, home_out, away_out, sat, tot_time


def optimization(home, away, n_teams, timeout=300_000):
    """
    Z3 optimization that reproduces the MiniZinc optimizer logic in optimizer.txt.

    Args:
      home, away: Python lists [periods][weeks] with initial schedule (home0_matrix / away0_matrix)
      n_teams: number of teams (int)
      timeout: solver timeout in milliseconds

    Returns:
      (opt, model_result, home_out, away_out, max_imb_val, elapsed_seconds)
      - opt: the Optimize() object (after check())
      - model_result: the CheckSatResult (sat/unsat/unknown)
      - home_out, away_out: Python lists (periods x weeks) with optimized orientation (or None if not sat)
      - max_imb_val: integer value of optimized max imbalance (or None)
      - elapsed_seconds: float time spent building+solving
    """

    start_time = time.time()

    # basic dims & checks
    periods = len(home)
    if periods == 0:
        raise ValueError("home matrix empty")
    weeks = len(home[0])

    # convert original schedule to Python int matrices (ensure ints)
    home0 = [[int(home[p][w]) for w in range(weeks)] for p in range(periods)]
    away0 = [[int(away[p][w]) for w in range(weeks)] for p in range(periods)]

    # Create solver
    opt = Optimize()
    opt.set("timeout", int(timeout))  # timeout in ms

    # -------------------------
    # Variables
    # -------------------------
    swap = [[ Bool(f"swap_{p}_{w}") for w in range(weeks)] for p in range(periods)]
    new_home = [[ Int(f"new_home_{p}_{w}") for w in range(weeks)] for p in range(periods)]
    new_away = [[ Int(f"new_away_{p}_{w}") for w in range(weeks)] for p in range(periods)]

    # counts per team
    home_count = [ Int(f"home_count_{t}") for t in range(1, n_teams+1) ]
    away_count = [ Int(f"away_count_{t}") for t in range(1, n_teams+1) ]

    # imbalance and diffs
    diff = [ Int(f"diff_{t}") for t in range(1, n_teams+1) ]
    imbalance = [ Int(f"imbalance_{t}") for t in range(1, n_teams+1) ]
    max_imb = Int("max_imbalance")

    # -------------------------
    # Link new_home/new_away to original using swap
    # new_home[p][w] = If(swap[p][w], away0[p][w], home0[p][w]) etc.
    # -------------------------
    for p in range(periods):
        for w in range(weeks):
            opt.add(new_home[p][w] == If(swap[p][w], IntVal(away0[p][w]), IntVal(home0[p][w])))
            opt.add(new_away[p][w] == If(swap[p][w], IntVal(home0[p][w]), IntVal(away0[p][w])))

            # additional safety: domain (teams numbered 1..n_teams)
            opt.add(new_home[p][w] >= 1, new_home[p][w] <= n_teams)
            opt.add(new_away[p][w] >= 1, new_away[p][w] <= n_teams)

            # no self-play
            opt.add(new_home[p][w] != new_away[p][w])

    # -------------------------
    # Each team plays exactly once per week
    # For each week w and team t: sum_p If(new_home==t or new_away==t,1,0) == 1
    # -------------------------
    for w in range(weeks):
        for t in range(1, n_teams+1):
            occ_terms = []
            for p in range(periods):
                occ_terms.append( If(Or(new_home[p][w] == t, new_away[p][w] == t), IntVal(1), IntVal(0)) )
            opt.add(Sum(occ_terms) == 1)

    # -------------------------
    # home_count and away_count (sums over all slots)
    # home_count[t] = sum_{p,w} bool2int(new_home[p][w] == t)
    # -------------------------
    for t in range(1, n_teams+1):
        h_terms = []
        a_terms = []
        for p in range(periods):
            for w in range(weeks):
                h_terms.append( If(new_home[p][w] == t, IntVal(1), IntVal(0)) )
                a_terms.append( If(new_away[p][w] == t, IntVal(1), IntVal(0)) )
        opt.add(home_count[t-1] == Sum(h_terms))
        opt.add(away_count[t-1] == Sum(a_terms))

        # each team plays weeks matches total (home_count + away_count == weeks)
        opt.add(home_count[t-1] + away_count[t-1] == weeks)

    # -------------------------
    # Compute imbalance (absolute) per team and max_imb
    # Use linearization for Abs
    # -------------------------
    for t in range(1, n_teams+1):
        # diff[t-1] >= home_count - away_count and >= opposite
        opt.add(diff[t-1] >= home_count[t-1] - away_count[t-1])
        opt.add(diff[t-1] >= away_count[t-1] - home_count[t-1])
        # bound diff
        opt.add(diff[t-1] >= 0, diff[t-1] <= weeks)
        # define imbalance = diff (keeps naming like MiniZinc)
        opt.add(imbalance[t-1] == diff[t-1])
        # max_imb bounds
        opt.add(max_imb >= imbalance[t-1])

    # optionally set some lower-bound on max_imb (not required)
    opt.add(max_imb >= 0, max_imb <= 1)

    # -------------------------
    # Constraint "Each team appears at most twice per period" in original SMT model:
    # Here MiniZinc used "Each team appears at most twice per period (over all weeks)"
    # It means for each period p and team t: Sum_w If(new_home[p][w]==t or new_away[p][w]==t,1,0) <= 2
    # -------------------------
    for p in range(periods):
        for t in range(1, n_teams+1):
            occ_terms = [ If(Or(new_home[p][w] == t, new_away[p][w] == t), IntVal(1), IntVal(0)) for w in range(weeks) ]
            opt.add(Sum(occ_terms) <= 2)

    # -------------------------
    # Symmetry-breaking "fix period" (team N in p_target = w%periods and optionally N-1 in p_next)
    # -------------------------
    if n_teams >= 8:
        for w in range(weeks):
            p_target = w % periods
            # find index in original week where team n_teams plays (we must ensure such match exists)
            # But since we don't have match indices here, simply impose that in slot p_target team N must play
            opt.add( Or(new_home[p_target][w] == n_teams, new_away[p_target][w] == n_teams) )
            if n_teams >= 14 and w < weeks - 2:
                p_next = (p_target + 1) % periods
                opt.add( Or(new_home[p_next][w] == n_teams - 1, new_away[p_next][w] == n_teams - 1) )

    # -------------------------
    # Objective: minimize max_imb
    # -------------------------
    opt.minimize(max_imb)

    # Solve
    res = opt.check()
    elapsed = time.time() - start_time

    home_out = None
    away_out = None
    max_imb_val = None
    if res == sat or res == sat:  # check for sat (opt.check() returns sat/unknown/...)
        m = opt.model()
        # build Python lists of ints
        home_out = [[0 for _ in range(weeks)] for _ in range(periods)]
        away_out = [[0 for _ in range(weeks)] for _ in range(periods)]
        for p in range(periods):
            for w in range(weeks):
                # evaluate (model_completion ensures a value if something left)
                hv = m.evaluate(new_home[p][w], model_completion=True)
                av = m.evaluate(new_away[p][w], model_completion=True)
                home_out[p][w] = int(hv.as_long())
                away_out[p][w] = int(av.as_long())

        max_imb_val = int(m.evaluate(max_imb, model_completion=True).as_long())

    return  res, home_out, away_out, max_imb_val, opt


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


if __name__ == "__main__":

    # --- 1) Leggi n_teams dal comando ---
    if len(sys.argv) >= 2:
        n_teams = int(sys.argv[1])
    else:
        n_teams = 8

    # --- 2) Leggi la modalità: 'sat' oppure 'opt' oppure 'test' oppure 'seed' ---
    if len(sys.argv) >= 3:
        mode = sys.argv[2].lower()
    else:
        mode = "sat"   # default

    if mode not in ["sat", "opt", "test", "seed"]:
        print("ERRORE: il secondo argomento deve essere 'sat', 'opt', 'test' oppure 'seed'")
        sys.exit(1)

    # --- 3) Leggi il solver: 'z3' oppure 'cvc5' ---
    if len(sys.argv) >= 4:
        solver_choice = sys.argv[3].lower()
    else:
        solver_choice = "z3"   # default

    if solver_choice not in ["z3", "cvc5"]:
        print("ERRORE: il terzo argomento deve essere 'z3' oppure 'cvc5'")
        sys.exit(1)

    if solver_choice == "cvc5" and mode == "opt":
        print("Note: cvc5 will find a solution, then Z3 will optimize it.")

    # --- 4) Parse optional parameters (if provided) ---
    # Default parameters
    params = {
        'sb_fix_period': True,
        'sb_lex_periods': False,
        'implied_constraints': True
    }
    
    # Check if parameters are specified (argv[4])
    if len(sys.argv) >= 5:
        params_str = sys.argv[4]
        # Remove brackets if present
        params_str = params_str.strip('[]')
        
        # Parse parameter assignments
        try:
            for param_pair in params_str.split(','):
                param_pair = param_pair.strip()
                if '=' in param_pair:
                    key, value = param_pair.split('=')
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert string to boolean
                    if value.lower() == 'true':
                        params[key] = True
                    elif value.lower() == 'false':
                        params[key] = False
                    else:
                        print(f"ERRORE: valore non valido per {key}: {value} (usa True o False)")
                        sys.exit(1)
            
            print(f"\nUsing custom parameters: {params}")
        except Exception as e:
            print(f"ERRORE: formato parametri non valido: {e}")
            print("Formato corretto: [sb_fix_period=True,sb_lex_periods=False,implied_constraints=True]")
            print("oppure: sb_fix_period=True,sb_lex_periods=False,implied_constraints=True")
            sys.exit(1)
    else:
        print(f"\nUsing default parameters: {params}")

    # --- 5) Genera il calendario con il circle method ---
    schedule, periods, weeks = circle_method_schedule(n_teams)


    # Select the solver function
    if solver_choice == "z3":
        solve_func = z3_permutation_constraint
        print(f"\nUsing Z3 solver")
    else:
        solve_func = z3_permutation_constraint_cvc5
        print(f"\nUsing CVC5 solver")

    # --- 6) Modalità SEED: cerca il miglior seed ---
    if mode == "seed":
        if solver_choice != "z3":
            print("ERRORE: la modalità 'seed' è supportata solo con il solver Z3")
            sys.exit(1)
        
        print(f"\n{'='*80}")
        print(f"SEED SEARCH MODE for {n_teams} teams")
        print(f"{'='*80}")
        
        best_seed = find_good_seed(
            schedule, n_teams, 
            num_trials=60, 
            timeout_per_trial=300_000,
            sb_fix_period=params['sb_fix_period'],
            sb_lex_periods=params['sb_lex_periods'],
            implied_constraints=params['implied_constraints']
        )
        
        print(f"\nRunning final solve with best seed {best_seed}...")
        s, _, home, away, res, sat_time = z3_permutation_constraint(
            schedule, n_teams, seed=best_seed,
            sb_fix_period=params['sb_fix_period'],
            sb_lex_periods=params['sb_lex_periods'],
            implied_constraints=params['implied_constraints']
        )
        
        if res == sat:
            print(f"FINAL RESULT: SAT in {sat_time:.2f}s")
            print_solution_matrix(home, away)
        else:
            print(f"FINAL RESULT: {res}")
        
        sys.exit(0)

    # --- 7) Modalità TEST: prova tutte le combinazioni ---
    if mode == "test":
        import itertools
        
        sb_fix_options = [True, False]
        sb_lex_options = [True, False]
        ic_option = [True, False]

        
        print("\n" + "="*80)
        print(f"TEST MODE: Running all combinations of constraints with {solver_choice.upper()}")
        print("="*80)
        
        results = []
        for sb_fix, sb_lex, implied_constraints_val in itertools.product(sb_fix_options, sb_lex_options, ic_option):
            print(f"\n--- Testing combination: sb_fix_period={sb_fix}, sb_lex_periods={sb_lex}, implied_constraints={implied_constraints_val} ---")
            
            test_start = time.time()
            s, _, home, away, res, sat_time = solve_func(
                schedule, n_teams,
                sb_fix_period=sb_fix,
                sb_lex_periods=sb_lex,
                implied_constraints=implied_constraints_val
            )
            test_end = time.time()
            
            result_entry = {
                'sb_fix_period': sb_fix,
                'sb_lex_periods': sb_lex,
                'implied_constraints': implied_constraints_val,
                'result': str(res),
                'time': sat_time
            }
            results.append(result_entry)
            
            print(f"Result: {res}, Time: {sat_time:.2f} seconds")
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        for r in results:
            print(f"sb_fix={r['sb_fix_period']}, sb_lex={r['sb_lex_periods']}, implied_constraints={r['implied_constraints']} {r['result']} ({r['time']:.2f}s)")
        
        sys.exit(0)

    # --- 8) Check di soddisfacibilità ---
    print(f"\nRunning {solver_choice.upper()} checks on the generated schedule...")
    s, _, home, away, res, sat_time = solve_func(
        schedule, n_teams,
        sb_fix_period=params['sb_fix_period'],
        sb_lex_periods=params['sb_lex_periods'],
        implied_constraints=params['implied_constraints']
    )

    print(f"{solver_choice.upper()} result:", res)

    # Handle both Z3 constants (sat, unknown) and cvc5 strings ("sat", "unsat", unknown)
    is_sat = (res == sat) or (res == "sat")
    is_unknown = (res == unknown) or (res == "unknown")
    is_unsat = (res == "unsat") or (hasattr(res, '__name__') and res.__name__ == 'unsat')

    if is_sat:
        print("Schedule satisfies STS constraints.\n")
        print("Satisfiability check time: %.2f seconds\n" % int(sat_time))
        print("solve_time:", round(sat_time, 2))
        
        # --- 9) Se richiesto, esegui la fase di ottimizzazione ---
        if mode == "opt":
            print("\nRunning optimization" + (" (using Z3)..." if solver_choice == "cvc5" else "..."))


            opt_stime = time.time()
            result, new_home, new_away, max_imb, opt = optimization(home, away, n_teams, timeout=int(300_000-sat_time))                              
            opt_etime = time.time()
            opt_time = round(opt_etime - opt_stime, 2)

            if result == sat:
                m = opt.model()
                print("Optimization result: SAT")
                print("obj:", max_imb)
                print("Optimization time: %.2f seconds\n" % opt_time)
                print("optimize_time:", opt_time)
                print("Total time: %.2f seconds\n" % int(opt_time + sat_time))
                print("optimal: True")
                print("Optimized schedule:")
                print("sol:")
                print_solution_matrix(new_home, new_away)
            elif result == unknown:
                print("Optimization TIMEOUT: solver exceeded time limit.")
                print("optimize_time:", opt_time)
                print("Total time: 300 seconds\n")
                print("optimal: False")
                print("obj:", n_teams - 1)
                print("sol:")
                print_solution_matrix(home, away)
            else:
                print("Optimization UNSAT.")
                print("optimize_time:", opt_time)
        
        else:
            print("optimize_time: 0")
            print("Total time: %.2f seconds\n" % int(sat_time))
            print("optimal: True")
            print("obj: None")
            print("sol:")
            print_solution_matrix(home, away)  
    elif is_unknown:
        print("SOLVER TIMEOUT: solver exceeded time limit.")
        print("solve_time:", 300.0)
        print("optimize_time: 0")
        print("Total time: 300 seconds\n")
        print("optimal: False")
        print("obj: None")
        print("sol: []")
    else:
        print("SOLVER reports UNSAT: schedule violates constraints.")
        print("solve_time:", round(sat_time, 2))
        print("optimize_time: 0")
        print("optimal: True")
        print("Total time: %.2f seconds\n" % int(sat_time))
        print("obj: None")
        print("sol: []")

