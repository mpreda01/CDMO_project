% Use this editor as a MiniZinc scratch book

include "globals.mzn";

% ==========================================
% PARAMETERS
% ==========================================

% Number of teams (must be even)
int: n;
constraint n mod 2 = 0;


int: n_weeks = n - 1;
int: n_periods = n div 2;

% Sets for indexing
set of int: Teams = 1..n;
set of int: Weeks = 1..n_weeks;
set of int: Periods = 1..n_periods;


% Main decision variables: 
array[Periods, Weeks] of var Teams: home;
array[Periods, Weeks] of var Teams: away;


% Period count variables
array[Teams, Periods] of var 0..2: period_count;

% Home/away count variables for optimization
set of int: Count = 0..n-1;
array[Teams] of var Count: home_count;
array[Teams] of var Count: away_count;

% Auxiliary variable for team period tracking 
array[Teams, Weeks] of var 0..n_periods: team_period;


  



constraint
  (
    forall(r in Weeks, p in Periods)(
      let {
        % Calcolo deterministico della coppia per ogni (periodo, settimana)
        int: offset = p - 1,
        
        % Posizione clockwise nel cerchio
        int: cw_pos = ((r + offset - 1) mod (n - 1)) + 1,
        int: team_a = if p == 1 then n else (if cw_pos == 0 then n - 1 else cw_pos endif) endif,
        
        % Posizione counterclockwise nel cerchio
        int: ccw_raw = r - offset - 1 + (n - 1) * 100,  % +100*(n-1) garantisce positività
        int: ccw_pos = (ccw_raw mod (n - 1)) + 1,
        int: team_b = if p == 1 then r else (if ccw_pos == 0 then n - 1 else ccw_pos endif) endif
      } in
      % Assegnamento diretto senza exists
      (home[p, r] == team_a /\ away[p, r] == team_b)
    )
  );
  
solve satisfy;

output [
"n_teams = ", show(n), ";\n",
"home0 = ", show(home), ";\n", 
"away0 = ", show(away), ";\n",
];