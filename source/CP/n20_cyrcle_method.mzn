include "globals.mzn";
% n20 con use_int_search=true, use_restart_luby=true, use_relax_and_reconstruct=true, chuffed=true, n_fix_period=true, ic_matches_per_team=false, ic_period_count=false, ic_different_match_per_period=false


% Simmetry brake 
bool: n_fix_period;
% Implied constraint
bool: ic_matches_per_team;
bool: ic_different_match_per_period;
% Search strat
bool: use_int_search; 
bool: use_restart_luby;
bool: use_relax_and_reconstruct;
% Solver
bool: chuffed;  

% ================================================================
% INPUT
% ================================================================

array[int] of int: home0;
array[int] of int: away0;

int: m = length(home0);
int: n_teams;
int: n_weeks = n_teams - 1;
int: n_periods = n_teams div 2;

% ================================================================
% WEEK MAPPING (circle method standard)
% Each block of n_weeks games corresponds to one period
% game i belongs to week (i-1 mod n_weeks)+1
% ================================================================

function int: week_of(int: i) = ((i - 1) mod n_weeks) + 1;

% set of games belonging to each week
array[1..n_weeks] of set of int: games_in_week =
  [ { i | i in 1..m where week_of(i) = w } | w in 1..n_weeks ];

% ================================================================
% DECISION VARIABLES
% ================================================================

array[1..n_periods, 1..n_weeks] of var 1..m: assign;

% ================================================================
% CONSTRAINTS
% ================================================================

% ---------------------------------------------------------------
% 1) Matches in a week can't be changed, ONLY swapped
% ---------------------------------------------------------------
constraint
  forall(w in 1..n_weeks)(
    { assign[p,w] | p in 1..n_periods } = games_in_week[w]
  );


% ---------------------------------------------------------------
% 2) Every team plays max twice per period
% ---------------------------------------------------------------
constraint
  forall(p in 1..n_periods, t in 1..n_teams)(
    sum(w in 1..n_weeks)(
      bool2int(
        home0[assign[p,w]] == t \/
        away0[assign[p,w]] == t
      )
    ) <= 2
  );
  
% ================================================================
%  IMPLIED CONSTRAINT
% ================================================================

% Total matches per team 
constraint
  ic_matches_per_team ->
  forall(t in 1..n_teams)(
    sum(w in 1..n_weeks)(
      sum(p in 1..n_periods)(
        bool2int(home0[assign[p,w]] == t \/ away0[assign[p,w]] == t)
      )
    ) = n_teams - 1
  );

  
% Every period contains a different match
constraint
  ic_different_match_per_period ->
  forall(w in 1..n_weeks)(
    alldifferent([assign[p,w] | p in 1..n_periods])
  );
  
% ================================================================
%  SYMMETRY BREAKING
% ================================================================

constraint
  n_fix_period ->
  (forall(w in 1..n_weeks)(
    let {
      int: p_target = ((w - 1) mod n_periods) + 1,
      int: p_next   = (p_target mod n_periods) + 1
    } in
      (
        % squadra n nel periodo target
        (( home0[assign[p_target, w]] == n_teams \/
          away0[assign[p_target, w]] == n_teams ))
        /\
        % squadra n-1 nel periodo successivo
        (w < n_weeks - 1 ->( home0[assign[p_next, w]] == n_teams - 1 \/
          away0[assign[p_next, w]] == n_teams - 1 ))
<<<<<<< HEAD
       )
  ));
% constraint
%   forall(p in 1..n_periods-1)(
%     lex_lesseq(
%       [assign[p, w] | w in 1..n_weeks],
%       [assign[p+1, w] | w in 1..n_weeks]
%     )
%   );
=======
      )
  ));

>>>>>>> b3a04f69d8962e9b09fb1992fe00465c8cd600a8



% ================================================================
% SOLVE
% ================================================================

solve 
:: if use_int_search then
    if chuffed then
      int_search(
        [home0[(p-1)*n_weeks + w] | p in 1..n_periods, w in 1..n_weeks] ++
        [away0[(p-1)*n_weeks + w] | p in 1..n_periods, w in 1..n_weeks],
        first_fail,
        indomain_min,
        complete
      )
    else
      int_search(
        [home0[(p-1)*n_weeks + w] | p in 1..n_periods, w in 1..n_weeks] ++
        [away0[(p-1)*n_weeks + w] | p in 1..n_periods, w in 1..n_weeks],
        first_fail,
        indomain_random,
        complete
      )
    endif
  endif
:: if (use_restart_luby /\ chuffed) then restart_luby(10000) endif
:: if use_relax_and_reconstruct then 
     relax_and_reconstruct(
       [home0[(p-1)*n_weeks + w] | p in 1..n_periods, w in 1..n_weeks] ++
       [away0[(p-1)*n_weeks + w] | p in 1..n_periods, w in 1..n_weeks],
       85
     )
   endif
satisfy;


% ================================================================
% OUTPUT
% ================================================================
% OUTPUT per vettori home away
output [
  "home = [", 
  concat([show(home0[assign[p,w]]) ++ (if (p-1)*n_weeks + w < n_periods*n_weeks then ", " else "" endif)
          | p in 1..n_periods, w in 1..n_weeks]),
  "]\n",
  "away = [",
  concat([show(away0[assign[p,w]]) ++ (if (p-1)*n_weeks + w < n_periods*n_weeks then ", " else "" endif)
          | p in 1..n_periods, w in 1..n_weeks]),
  "]\n"
];


% OUTPUT per matrice della soluzione corretta in formato JSON
% output [
%   "[\n"
% ] ++
% [
%   "  [" ++
%   concat([
%     "[" ++ show(home0[assign[p,w]]) ++ " , " ++ show(away0[assign[p,w]]) ++ "]" ++
%     (if w < n_weeks then " , " else "" endif)
%     | w in 1..n_weeks
%   ]) ++
%   "]" ++ (if p < n_periods then ",\n" else "\n" endif)
%   | p in 1..n_periods
% ] ++
% [
%   "]\n"
% ];

