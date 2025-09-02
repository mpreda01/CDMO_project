# Sports Tournament Scheduling - MIP Model in AMPL
# Decision and Optimization versions

# Parameters
param n >= 4;  # Number of teams
check: n mod 2 = 0;  # Ensure n is even
set TEAMS := 1..n;
set WEEKS := 1..(n-1);
set PERIODS := 1..(n/2);

# Decision Variables
# x[t1,t2,w,p] = 1 if team t1 plays at home against team t2 in week w, period p
var x{t1 in TEAMS, t2 in TEAMS, w in WEEKS, p in PERIODS: t1 != t2} binary;

# For optimization version: count home/away games per team
var home_games{t in TEAMS} >= 0;
var away_games{t in TEAMS} >= 0;
var max_imbalance >= 0;

# Objective function for optimization version: minimize the maximum imbalance between home and away games
minimize MaxImbalance: max_imbalance;

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

# For optimization version: count home and away games
subject to CountHomeGames{t in TEAMS}:
    home_games[t] = sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} x[t,t2,w,p];

subject to CountAwayGames{t in TEAMS}:
    away_games[t] = sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} x[t2,t,w,p];

# Imbalance constraints for optimization
subject to Imbalance1{t in TEAMS}:
    max_imbalance >= home_games[t] - away_games[t];

subject to Imbalance2{t in TEAMS}:
    max_imbalance >= away_games[t] - home_games[t];

# Symmetry breaking constraints
# Fix team 1 to play at home in week 1, period 1
subject to SymBreak1:
    sum{t2 in TEAMS: t2 != 1} x[1,t2,1,1] = 1;

# Team 1 plays against team 2 in week 1
subject to SymBreak2:
    x[1,2,1,1] = 1;

# Team n plays in period 1 of week 1
subject to SymBreak3:
    sum{t2 in TEAMS, w in 1..1: t2 != n} (x[n,t2,w,1] + x[t2,n,w,1]) = 1;