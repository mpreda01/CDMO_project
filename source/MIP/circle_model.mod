# CIRCLE MODEL MIP
param n >= 2;
param optimize_balance default 0 binary;

set TEAMS := 1..n;
set WEEKS := 1..(n-1);
set PERIODS := 1..(n/2);

var x{t1 in TEAMS, t2 in TEAMS, w in WEEKS, p in PERIODS: t1 != t2} binary;
var home_games{t in TEAMS} >= 0;
var away_games{t in TEAMS} >= 0;
var max_imbalance >= 0, <= n-1;

minimize MaxImbalance: if optimize_balance=1 then max_imbalance else 0;

# Each match occurs exactly once (period assignment is decision)
subject to MatchOnce{t1 in TEAMS, t2 in TEAMS: t1 < t2}:
    sum{w in WEEKS, p in PERIODS} (x[t1,t2,w,p] + x[t2,t1,w,p]) = 1;

# Each team plays once per week
subject to OneGamePerWeek{t in TEAMS, w in WEEKS}:
    sum{t2 in TEAMS, p in PERIODS: t2 != t} (x[t,t2,w,p] + x[t2,t,w,p]) = 1;

# Each period has exactly one game per week
subject to OnePeriodPerWeek{w in WEEKS, p in PERIODS}:
    sum{t1 in TEAMS, t2 in TEAMS: t1 < t2} (x[t1,t2,w,p] + x[t2,t1,w,p]) = 1;

# At most 2 games per period for each team
subject to AtMostTwiceInPeriod{t in TEAMS, p in PERIODS}:
    sum{w in WEEKS, t2 in TEAMS: t2 != t} (x[t,t2,w,p] + x[t2,t,w,p]) <= 2;

# Home/away counts
subject to CountHomeGames{t in TEAMS: optimize_balance=1}:
    home_games[t] = sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} x[t,t2,w,p];

subject to CountAwayGames{t in TEAMS: optimize_balance=1}:
    away_games[t] = sum{t2 in TEAMS, w in WEEKS, p in PERIODS: t2 != t} x[t2,t,w,p];

# Imbalance constraints
subject to Imbalance1{t in TEAMS: optimize_balance=1}:
    max_imbalance >= home_games[t] - away_games[t];

subject to Imbalance2{t in TEAMS: optimize_balance=1}:
    max_imbalance >= away_games[t] - home_games[t];
