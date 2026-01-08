

include "globals.mzn";

% ==========================================
% PARAMETERS
% ==========================================

% Number of teams (must be even)
int: n;
constraint n mod 2 = 0;


int: n_weeks = n - 1;
int: n_periods = n div 2;
int: n_games = (n * (n - 1)) div 2;

% Configuration flags 
% Symmetry breaking constraint
bool: sb_weeks;
bool: sb_periods;  
bool: sb_teams;
bool: sb_week1_fixed;
% Implied constraint
bool: ic_matches_per_team;
bool: ic_period_count;
% Search strat
bool: use_int_search; 
bool: use_restart_luby;
bool: use_relax_and_reconstruct;
% Solver
bool: chuffed;  

% Sets for indexing
set of int: Teams = 1..n;
set of int: Weeks = 1..n_weeks;
set of int: Periods = 1..n_periods;

% ==========================================
% DECISION VARIABLES 
% ==========================================

% Main decision variables: 
array[Periods, Weeks] of var Teams: home;
array[Periods, Weeks] of var Teams: away;

% ==========================================
% AUXILIARY VARIABLES 
% ==========================================

% Period count variables
array[Teams, Periods] of var 0..2: period_count;

% Home/away count variables for optimization
set of int: Count = 0..n-1;
array[Teams] of var Count: home_count;
array[Teams] of var Count: away_count;

% Auxiliary variable for team period tracking 
array[Teams, Weeks] of var 0..n_periods: team_period;

% ==========================================
% CORE CONSTRAINTS 
% ==========================================

% CONSTRAINT 1: Each pair plays exactly once
constraint
  forall(i, j in Teams where i < j)(
    sum(p in Periods, w in Weeks) (
      bool2int((home[p,w] = i /\ away[p,w] = j) \/ (home[p,w] = j /\ away[p,w] = i))
    ) = 1
  );

% CONSTRAINT 2: Each team plays once per week 
constraint
  forall(w in Weeks)(
    all_different([home[p,w] | p in Periods] ++ [away[p,w] | p in Periods])
  );

% CONSTRAINT 3: Period limit using global_cardinality 
constraint
  forall(p in Periods)(
    global_cardinality(
      [home[p,w] | w in Weeks] ++ [away[p,w] | w in Weeks],
      Teams,
      [period_count[t,p] | t in Teams]
    )
  );

% CONSTRAINT 4: Home count using global_cardinality
constraint
  global_cardinality(
    [home[p,w] | p in Periods, w in Weeks],
    Teams,
    [home_count[t] | t in Teams]
  );

% CONSTRAINT 5: Away count using global_cardinality
constraint
  global_cardinality(
    [away[p,w] | p in Periods, w in Weeks], 
    Teams,
    [away_count[t] | t in Teams]
  );


% No team plays against itself
constraint forall(p in Periods, w in Weeks) (
    home[p,w] != away[p,w]
);

% ==========================================
% CONDITIONAL IMPLIED CONSTRAINTS 
% ==========================================

% Total matches per team
constraint
  ic_matches_per_team ->
  forall(t in Teams)(
    home_count[t] + away_count[t] = n - 1
  );

% Total period appearances per team
constraint 
  ic_period_count ->
  forall(t in Teams) (
    sum(p in Periods) (period_count[t,p]) = n - 1
  );

% ==========================================
% CONDITIONAL SYMMETRY BREAKING
% ==========================================

% Primary symmetry breaking: home < away (always active for efficiency)
constraint forall(p in Periods, w in Weeks) (
    home[p,w] < away[p,w]
);

% Week lex ordering symmetry breaking (conditional)
constraint 
  sb_weeks ->
  forall(w in 1..n_weeks-1)(
    lex_lesseq(
      [home[p,w] | p in Periods],
      [home[p,w+1] | p in Periods]
    )
  );

% Period lex ordering symmetry breaking (conditional)
constraint 
  sb_periods ->
  forall(p in 1..n_periods-1)(
    lex_lesseq(
      [home[p,w] | w in Weeks] ++ [away[p,w] | w in Weeks],
      [home[p+1,w] | w in Weeks] ++ [away[p+1,w] | w in Weeks]
    )
  );

% Team ordering in first week (conditional)
constraint
  sb_week1_fixed ->
  forall(p in Periods) (
    home[p,1] = p /\ 
    away[p,1] = n - p + 1
  );

% ==========================================
% AUXILIARY VARIABLE LINKS
% ==========================================

% Link team_period variable 
constraint forall(t in Teams, w in Weeks) (
    team_period[t,w] = sum(p in Periods) (
        p * (bool2int(home[p,w] = t) + bool2int(away[p,w] = t))
    )
);

% Additional ordering for team 1 
constraint
  sb_teams ->
  (team_period[1,1] = 1 /\
   forall(w in 1..n_weeks-1) (
     team_period[1,w] <= team_period[1,w+1]
   )
  );

% ==========================================
% REDUNDANT CONSTRAINTS FOR PROPAGATION
% ==========================================

% Weekly sum constraint 
int: sumTeams = n*(n+1) div 2;
constraint forall(w in Weeks) (
    sum(p in Periods) (home[p,w] + away[p,w]) = sumTeams
);

% ==========================================
% SEARCH STRATEGY
% ==========================================

solve 
:: if use_int_search then
    if chuffed then
     int_search(
       [home[p,w] | p in Periods, w in Weeks] ++ [away[p,w] | p in Periods, w in Weeks],
       first_fail,
       indomain_min,
       complete
     )
     else
      int_search(
       [home[p,w] | p in Periods, w in Weeks] ++ [away[p,w] | p in Periods, w in Weeks],
       first_fail,
       indomain_random,
       complete
     )
     endif
   endif
:: if (use_restart_luby /\ chuffed) then restart_luby(10000) endif
:: if use_relax_and_reconstruct then 
     relax_and_reconstruct([home[p,w] | p in Periods, w in Weeks] ++ [away[p,w] | p in Periods, w in Weeks], 85)
   endif
%   minimize max_diff;
satisfy;

% ==========================================
% OUTPUT
% ==========================================

output [
%   "% Objective value (max home/away imbalance): ", show(max_diff), "\n",
%   "% Home games per team: ", show(home_count), "\n", 
%   "% Away games per team: ", show(away_count), "\n",
  "n_teams = ", show(n), ";\n",
  "home0 = ", show(home), ";\n", 
  "away0 = ", show(away), ";\n",
%   "% JSON format for analysis:\n",
%   "{",
%   "\"obj\": ", show(max_diff), "," ,
%   "\"home_count\": ", show(home_count), ",",
%   "\"away_count\": ", show(away_count), ",",
%   "\"sol\": ", show([ [ [home[p,w], away[p,w]] | w in Weeks ] | p in Periods ]),
%   "}"
];