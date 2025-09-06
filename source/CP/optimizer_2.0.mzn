

include "globals.mzn";

% ==========================================
% INPUT PARAMETERS
% ==========================================


int: n_teams;       

% Matrix from firs model
array[int] of Teams: home0;  
array[int] of Teams: away0;  


int: n_periods = n_teams div 2;     
int: n_weeks = n_teams - 1;         

% Sets
set of int: Teams = 1..n_teams;
set of int: Weeks = 1..n_weeks;
set of int: Periods = 1..n_periods;

% Convert input arrays to 2D matrices for easier indexing
array[Periods, Weeks] of Teams: home0_matrix = array2d(Periods, Weeks, home0);
array[Periods, Weeks] of Teams: away0_matrix = array2d(Periods, Weeks, away0);

% ==========================================
% DECISION VARIABLES
% ==========================================

% New tournament matrix with possible swaps
array[Periods, Weeks] of var Teams: home;  
array[Periods, Weeks] of var Teams: away;  

% Binary variable to indicate if a swap is made
array[Periods, Weeks] of var 0..1: swap;

% Count of home and away games for each team
array[Teams] of var 0..n_weeks: home_count;
array[Teams] of var 0..n_weeks: away_count;

% Differecence between home and away games for each team
array[Teams] of var 0..n_weeks: imbalance;

% Objective: Maximum imbalance across all teams
var 0..n_weeks: max_imbalance;

% ==========================================
% CONSTRAINTS
% ==========================================

% Constraint 1: Define home and away based on swap variable
constraint forall(p in Periods, w in Weeks) (
    (swap[p,w] = 0 -> (home[p,w] = home0_matrix[p,w] /\ away[p,w] = away0_matrix[p,w])) /\
    (swap[p,w] = 1 -> (home[p,w] = away0_matrix[p,w] /\ away[p,w] = home0_matrix[p,w]))
);

% Constraint 2: Calculate the number of home games for each team
constraint forall(t in Teams) (
    home_count[t] = sum(p in Periods, w in Weeks) (bool2int(home[p,w] = t))
);

% Constraint 3: Calculate the number of away games for each team
constraint forall(t in Teams) (
    away_count[t] = sum(p in Periods, w in Weeks) (bool2int(away[p,w] = t))
);

% Constraint 4: Each team plays exactly n_teams - 1 games
constraint forall(t in Teams) (
    home_count[t] + away_count[t] = n_teams - 1
);

% Constraint 5: Calculate the imbalance for each team
constraint forall(t in Teams) (
    imbalance[t] = abs(home_count[t] - away_count[t])
);

% Constraint 6: Define the maximum imbalance
constraint max_imbalance = max(t in Teams) (imbalance[t]);

% Constraint 7: A team cannot play against itself
constraint forall(p in Periods, w in Weeks) (
    home[p,w] != away[p,w]
);

% Constraint 8: Each team plays only once per week
constraint forall(w in Weeks) (
    alldifferent([home[p,w] | p in Periods] ++ [away[p,w] | p in Periods])
);



% Minimize the maximum imbalance
constraint max_imbalance = 1;
solve satisfy;

% ==========================================
% AUXILIARY VARIABLES FOR OUTPUT
% ==========================================

array[Teams] of var 0..n_weeks: original_home_count;
array[Teams] of var 0..n_weeks: original_away_count;
array[Teams] of var 0..n_weeks: original_imbalance;
var 0..n_weeks: original_max_imbalance;

% Calculate original home and away counts
constraint forall(t in Teams) (
    original_home_count[t] = sum(p in Periods, w in Weeks) (bool2int(home0_matrix[p,w] = t))
);

constraint forall(t in Teams) (
    original_away_count[t] = sum(p in Periods, w in Weeks) (bool2int(away0_matrix[p,w] = t))
);

constraint forall(t in Teams) (
    original_imbalance[t] = abs(original_home_count[t] - original_away_count[t])
);

constraint original_max_imbalance = max(t in Teams) (original_imbalance[t]);

% ==========================================
% OUTPUT
% ==========================================

output [
    "Original Max Imbalance: " ++ show(original_max_imbalance) ++ "\n",
    "Optimized Max Imbalance: " ++ show(max_imbalance) ++ "\n",
    "Improvement: " ++ show(original_max_imbalance - max_imbalance) ++ "\n",
    "Total Swaps Made: " ++ show(sum(p in Periods, w in Weeks)(swap[p,w])) ++ "\n\n"
];

output [
    "=== OPTIMIZED TOURNAMENT MATRIX ===\n"
] ++
[
    "[[" ++
    concat([
        show(home[p,w]) ++ " , " ++ show(away[p,w]) ++
        if w < n_weeks then "] , [" else "" endif
        | w in Weeks
    ]) ++
    if p < n_periods then "]]\n" else "]]\n" endif
    | p in Periods
];

output [
    "----------\n",
    "==========\n"
];