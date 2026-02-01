# MIP Circle Model in AMPL

# Parameter to control optimization - default: decision version
param n >= 2;
param optimize_balance binary default 0;
param sb_per default 0 binary;

check: n mod 2 = 0;

set TEAMS := 1..n;
set WEEKS := 1..(n-1);
set PERIODS := 1..(n/2);

# Unordered matches
set MATCHES := {t1 in TEAMS, t2 in TEAMS: t1 < t2};

# Given by Python (circle method)
param match_week{MATCHES} integer >= 1 <= (n-1);

# Decision: match (t1,t2) assigned to period p (in its fixed week)
var y{(t1,t2) in MATCHES, p in PERIODS} binary;


var home{(t1,t2) in MATCHES} binary;
var home_games{t in TEAMS} >= 0;
var away_games{t in TEAMS} >= 0;
var max_imbalance >= 0; #<= n-1;


# Core constraints

# Each match assigned to exactly one period
subject to OnePeriodPerMatch{(t1,t2) in MATCHES}:
    sum{p in PERIODS} y[t1,t2,p] = 1;

# At most twice in the same period
subject to AtMostTwiceInPeriod{t in TEAMS, p in PERIODS}:
    sum{(t1,t2) in MATCHES:
        (t1 = t or t2 = t)} y[t1,t2,p] <= 2;

# Implied Constraints

# Each period in each week has exactly one match
subject to OneMatchPerPeriod{w in WEEKS, p in PERIODS}:
    sum{(t1,t2) in MATCHES: match_week[t1,t2] = w} y[t1,t2,p] = 1;


# Home / Away constraints
# home[t1,t2] = 1 means t1 is home, t2 is away
# home[t1,t2] = 0 means t2 is home, t1 is away

subject to CountHome{t in TEAMS: optimize_balance = 1}:
    home_games[t] =
        sum{(t1,t2) in MATCHES: t1 = t} home[t1,t2]
      + sum{(t1,t2) in MATCHES: t2 = t} (1 - home[t1,t2]);

subject to CountAway{t in TEAMS: optimize_balance = 1}:
    away_games[t] =
        sum{(t1,t2) in MATCHES: t1 = t} (1 - home[t1,t2])
      + sum{(t1,t2) in MATCHES: t2 = t} home[t1,t2];

subject to Imb1{t in TEAMS: optimize_balance = 1}:
    max_imbalance >= home_games[t] - away_games[t];

subject to Imb2{t in TEAMS: optimize_balance = 1}:
    max_imbalance >= away_games[t] - home_games[t];

# Symmetry breaking

# Fix the first period of week 1 for the first n/2 matches
param period_index{MATCHES} integer; # from Python

subject to FixWeek1Periods{(t1,t2) in MATCHES: match_week[t1,t2] = 1}:
    sb_per*y[t1,t2,period_index[t1,t2]] = sb_per;


# Objective
minimize Obj:
    if optimize_balance = 1 then max_imbalance else 0;
