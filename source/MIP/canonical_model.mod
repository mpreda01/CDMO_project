# MIP Canonical Model in AMPL

# Parameters
param n >= 2;
check: n mod 2 = 0;

# Parameter to control optimization - default: decision version
param optimize_balance default 0 binary;
param sb_teams default 0 binary;
param sb_weeks default 0 binary;

set TEAMS := 1..n;
set WEEKS := 1..(n-1);
set PERIODS := 1..(n/2);

# Decision Variables
var x{t1 in TEAMS, t2 in TEAMS, w in WEEKS, p in PERIODS: t1 != t2} binary;

# Optimization variables - count home/away games per team
var home_games{t in TEAMS} >= 0;
var away_games{t in TEAMS} >= 0;
var max_imbalance >= 0, <= n-1; ## n-1

# Objective function for optimization version
minimize MaxImbalance: 
    if optimize_balance = 1 then max_imbalance else 0;

# Constraint 1: Every team plays with every other team exactly once
subject to PlayOnce{t1 in TEAMS, t2 in TEAMS: t1 < t2}:
    sum{w in WEEKS, p in PERIODS} (x[t1,t2,w,p] + x[t2,t1,w,p]) = 1;

# Constraint 2: Every team plays exactly once per week
subject to OneGamePerWeek{t in TEAMS, w in WEEKS}:
    sum{t2 in TEAMS, p in PERIODS: t2 != t} (x[t,t2,w,p] + x[t2,t,w,p]) = 1;

# Constraint 3: Each period has exactly one game
subject to OnePeriodPerWeek{w in WEEKS, p in PERIODS}:
    sum{t1 in TEAMS, t2 in TEAMS: t1 < t2} (x[t1,t2,w,p] + x[t2,t1,w,p]) = 1;

# Constraint 4: Every team plays at most twice in the same period over the tournament
subject to AtMostTwiceInPeriod{t in TEAMS, p in PERIODS}:
    sum{w in WEEKS, t2 in TEAMS: t2 != t} (x[t,t2,w,p] + x[t2,t,w,p]) <= 2;

# Implied constraint: Every team plays n-1 matches in the tournament
subject to TotalMatches{t in TEAMS}:
    sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} (x[t,t2,w,p] + x[t2,t,w,p]) = n-1;

# Optimization constraints
# Count home and away games
subject to CountHomeGames{t in TEAMS: optimize_balance = 1}:
    home_games[t] = sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} x[t,t2,w,p];

subject to CountAwayGames{t in TEAMS: optimize_balance = 1}:
    away_games[t] = sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} x[t2,t,w,p];

# Imbalance constraints
subject to Imbalance1{t in TEAMS: optimize_balance = 1}:
    max_imbalance >= home_games[t] - away_games[t];

subject to Imbalance2{t in TEAMS: optimize_balance = 1}:
    max_imbalance >= away_games[t] - home_games[t];

# Symmetry breaking constraints
# Fix team 1 to play against team w+1 in week w
subject to SymBreak_team1{w in WEEKS: w+1 <= n}:
    sb_teams * (sum{p in PERIODS} (x[1,w+1,w,p] + x[w+1,1,w,p])) = sb_teams;

# Fix team (2p-1) to play at home against team 2p in week 1, period p
subject to SymBreak_week1{p in PERIODS: 2*p <= n}:
    sb_weeks * x[2*p-1, 2*p, 1, p] = sb_weeks;
