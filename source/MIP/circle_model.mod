# Sports Tournament Scheduling - MIP model in AMPL using Circle Method
param n >= 2;
check: n mod 2 = 0;
param optimize_balance default 0 binary;

# Sets
set TEAMS := 1..n;
set WEEKS := 1..(n-1);
set PERIODS := 1..(n/2);

# Decision Variables
var x{t1 in TEAMS, t2 in TEAMS, w in WEEKS, p in PERIODS: t1 != t2} binary;

# Optimization variables - count home/away games per team
var home_games{t in TEAMS} >= 0;
var away_games{t in TEAMS} >= 0;
var max_imbalance >= 0, <= n-1;  ## n-1

# Objective function for optimization version
minimize MaxImbalance: 
    if optimize_balance=1 then max_imbalance else 0;

# Unique ID for unordered match (t1<t2)
param match_id{t1 in TEAMS, t2 in TEAMS: t1 < t2} :=
    (t1 - 1) * n + t2;

# Assign variable equal to match_id in period p, week w
# This is linear because assign equals sum of match_id * x where the x's
# for a given p,w select exactly one unordered pair (by constraints below).
var assign{p in PERIODS, w in WEEKS} integer;

subject to DefineAssign{p in PERIODS, w in WEEKS}:
    assign[p,w] =
      sum{t1 in TEAMS, t2 in TEAMS: t1 < t2}
          match_id[t1,t2] * ( x[t1,t2,w,p] + x[t2,t1,w,p] );

# Constraint: Every team plays with every other team exactly once
#subject to PlayOnce{t1 in TEAMS, t2 in TEAMS: t1 < t2}:
#    sum{w in WEEKS, p in PERIODS} (x[t1,t2,w,p] + x[t2,t1,w,p]) = 1;

# Constraint: Every team plays exactly once per week
#subject to OneGamePerWeek{t in TEAMS, w in WEEKS}:
#    sum{t2 in TEAMS, p in PERIODS: t2 != t} (x[t,t2,w,p] + x[t2,t,w,p]) = 1;

# Constraint: Each period has exactly one game
subject to OnePeriodPerWeek{w in WEEKS, p in PERIODS}:
    sum{t1 in TEAMS, t2 in TEAMS: t1 < t2} (x[t1,t2,w,p] + x[t2,t1,w,p]) = 1;

# Constraint: Every team plays at most twice in the same period over the tournament
subject to AtMostTwiceInPeriod{t in TEAMS, p in PERIODS}:
    sum{w in WEEKS, t2 in TEAMS: t2 != t} (x[t,t2,w,p] + x[t2,t,w,p]) <= 2;

## Implied constraints

# For each unordered match (t1<t2) and each week w:
#  the same match cannot appear in more than one period of the same week.
#  (This ensures all periods in a week are different matches.)
subject to MatchOncePerWeek{t1 in TEAMS, t2 in TEAMS: t1 < t2, w in WEEKS}:
    sum{p in PERIODS} ( x[t1,t2,w,p] + x[t2,t1,w,p] ) <= 1;

# For each unordered match (t1<t2) and each period p:
#  the same match cannot appear in more than one week in the same period.
#  (This ensures all weeks in a period are different matches.)
subject to MatchOncePerPeriod{t1 in TEAMS, t2 in TEAMS: t1 < t2, p in PERIODS}:
    sum{w in WEEKS} ( x[t1,t2,w,p] + x[t2,t1,w,p] ) <= 1;

## Symmetry breaking constraints

# Lex ordering of periods (n < 8)
#   Original MiniZinc used lex_lesseq over vectors assign[p,*].
#   lex is not linear; we use a linear surrogate (weaker) that orders
#   the sums of assign values across weeks:
#
#       sum_w assign[p,w] <= sum_w assign[p+1,w]
#
#   This is linear, breaks some symmetry, and is safe for MIP.
subject to SymmLex_SumOrder{p in PERIODS: p < card(PERIODS), n < 8}:
    sum{w in WEEKS} assign[p,w] <= sum{w in WEEKS} assign[p+1,w];

# Impose fixed period for the matches containing team = n (n >= 8)
#   For each week w, let p_target = ((w - 1) mod n_periods) + 1,
#   force that the game scheduled in (p_target, w) includes team n.
#   We compute p_target using integer arithmetic (note: card(PERIODS) = n/2).
subject to SymmTeamN{w in WEEKS: n >= 8}:
    let { int p_target := ((w - 1) mod card(PERIODS)) + 1 } in
    sum{t in TEAMS: t != n} ( x[n,t,w,p_target] + x[t,n,w,p_target] ) = 1;

# Impose fixed period for matches containing team = n-1 (n >= 14)
#   Similar to MiniZinc: team n-1 must appear in p_next for week w (except near end).
subject to SymmTeamNm1{w in WEEKS: n >= 14 and w < n-1}:
    let {
        int p_target := ((w - 1) mod card(PERIODS)) + 1;
        int p_next   := (p_target mod card(PERIODS)) + 1;
    } in
    sum{t in TEAMS: t != n-1} ( x[n-1,t,w,p_next] + x[t,n-1,w,p_next] ) = 1;


# Optimization constraints
# Count home and away games
subject to CountHomeGames{t in TEAMS: optimize_balance=1}:
    home_games[t] = sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} x[t,t2,w,p];

subject to CountAwayGames{t in TEAMS: optimize_balance=1}:
    away_games[t] = sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} x[t2,t,w,p];

# Imbalance constraints
subject to Imbalance1{t in TEAMS: optimize_balance = 1}:
    max_imbalance >= home_games[t] - away_games[t];

subject to Imbalance2{t in TEAMS: optimize_balance = 1}:
    max_imbalance >= away_games[t] - home_games[t];
