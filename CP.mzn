% Sports Tournament Scheduling (STS) Problem - Optimization Version
% MiniZinc Implementation with JSON Output

% =============================================================================
% IMPORTS
% =============================================================================
include "globals.mzn";

% =============================================================================
% PARAMETERS
% =============================================================================
int: n;  % number of teams (must be even)
constraint assert(n mod 2 = 0, "Number of teams must be even");

int: weeks = n - 1;
int: periods = n div 2;
int: total_games = (n * (n - 1)) div 2;

set of int: TEAMS = 1..n;
set of int: WEEKS = 1..weeks;
set of int: PERIODS = 1..periods;
set of int: SLOTS = 1..2;  % 1=home, 2=away

% =============================================================================
% DECISION VARIABLES
% =============================================================================
% schedule[w,p,s] = team playing in week w, period p, slot s
array[WEEKS, PERIODS, SLOTS] of var TEAMS: schedule;

% Alternative representation: games as pairs [home_team, away_team]
array[WEEKS, PERIODS] of var TEAMS: home_team;
array[WEEKS, PERIODS] of var TEAMS: away_team;

% Link the two representations
constraint forall(w in WEEKS, p in PERIODS) (
    home_team[w,p] = schedule[w,p,1] /\
    away_team[w,p] = schedule[w,p,2]
);

% =============================================================================
% CORE CONSTRAINTS
% =============================================================================

% Constraint 1: Each team plays every other team exactly once
constraint forall(t1, t2 in TEAMS where t1 < t2) (
    sum(w in WEEKS, p in PERIODS) (
        bool2int(home_team[w,p] = t1 /\ away_team[w,p] = t2) +
        bool2int(home_team[w,p] = t2 /\ away_team[w,p] = t1)
    ) = 1
);

% Constraint 2: Each team plays exactly once per week
constraint forall(w in WEEKS, t in TEAMS) (
    sum(p in PERIODS, s in SLOTS) (bool2int(schedule[w,p,s] = t)) = 1
);

% Constraint 3: No team plays against itself
constraint forall(w in WEEKS, p in PERIODS) (
    home_team[w,p] != away_team[w,p]
);

% Constraint 4: Each team plays at most twice in same period across tournament
constraint forall(p in PERIODS, t in TEAMS) (
    sum(w in WEEKS, s in SLOTS) (bool2int(schedule[w,p,s] = t)) <= 2
);

% =============================================================================
% SYMMETRY BREAKING CONSTRAINTS
% =============================================================================

% Symmetry breaking 1: Fix team 1 in first position
constraint schedule[1,1,1] = 1;

% Symmetry breaking 3: Order periods within each week by minimum team number
constraint forall(p in 1..periods-1) (
    min(schedule[1,p,1], schedule[1,p,2]) <= 
    min(schedule[1,p+1,1], schedule[1,p+1,2])
);

% =============================================================================
% IMPLIED CONSTRAINTS (for better propagation)
% =============================================================================

% Each period in each week has exactly 2 different teams
constraint forall(w in WEEKS, p in PERIODS) (
    alldifferent([schedule[w,p,s] | s in SLOTS])
);

% =============================================================================
% OPTIMIZATION OBJECTIVE
% =============================================================================

% Count home and away games for each team
array[TEAMS] of var 0..weeks: home_games;
array[TEAMS] of var 0..weeks: away_games;

constraint forall(t in TEAMS) (
    home_games[t] = sum(w in WEEKS, p in PERIODS) (bool2int(home_team[w,p] = t))
);

constraint forall(t in TEAMS) (
    away_games[t] = sum(w in WEEKS, p in PERIODS) (bool2int(away_team[w,p] = t))
);

% Balance constraint: each team plays exactly n-1 games total
constraint forall(t in TEAMS) (
    home_games[t] + away_games[t] = n - 1
);

% Optimization objectives
var 0..weeks: max_imbalance = max(t in TEAMS) (abs(home_games[t] - away_games[t]));
var int: total_imbalance = sum(t in TEAMS) (abs(home_games[t] - away_games[t]));

% Bounds for better search
constraint max_imbalance >= 0;
constraint max_imbalance <= if n mod 2 = 0 then 1 else 0 endif;

% =============================================================================
% SEARCH STRATEGY
% =============================================================================

% Optimization solve
solve 
    :: seq_search([
        int_search([schedule[w,p,s] | w in WEEKS, p in PERIODS, s in SLOTS], 
                  first_fail, indomain_min, complete)
    ])
    minimize max_imbalance;

% =============================================================================
% OUTPUT
% =============================================================================

output [
    "OPTIMAL Schedule for " ++ show(n) ++ " teams:\n"
] ++
[
    "Week " ++ show(w) ++ ": " ++
    concat([
        "Period " ++ show(p) ++ ": " ++ 
        show(home_team[w,p]) ++ " v " ++ show(away_team[w,p]) ++ 
        if p < periods then ", " else "" endif
        | p in PERIODS
    ]) ++ "\n"
    | w in WEEKS
] ++
[
    "\nHome games per team: " ++ show(home_games) ++ "\n",
    "Away games per team: " ++ show(away_games) ++ "\n",
    "Max imbalance: " ++ show(max_imbalance) ++ "\n",
    "Total imbalance: " ++ show(total_imbalance) ++ "\n",
    "Objective value: " ++ show(max_imbalance) ++ "\n"
];

% =============================================================================
% SOLUTION MATRIX OUTPUT
% =============================================================================
output [
    "\n=== SOLUTION MATRIX ===\n",
    "[\n"
] ++
[
    "  [[" ++
    concat([
        show(home_team[w,p]) ++ " , " ++ show(away_team[w,p]) ++
        if w < weeks then "] , [" else "" endif
        | w in WEEKS
    ]) ++
    if p < periods then "]]\n" else "]]\n" endif
    | p in PERIODS
] ++
[
    "]\n"
];