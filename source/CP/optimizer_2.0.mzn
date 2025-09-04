% Tournament Home/Away Optimizer
% Riceve in input una matrice di partite e ottimizza la distribuzione casa/trasferta
% per minimizzare la differenza tra partite in casa e in trasferta per ogni squadra

include "globals.mzn";

% ==========================================
% PARAMETRI DI INPUT (dal risultato del primo modello)
% ==========================================

% Parametri del torneo
int: n_teams;       % Numero di squadre

% Matrici di input dal primo modello (come array 1D o 2D)
array[int] of Teams: home0;  % Array casa (dal dzn)
array[int] of Teams: away0;  % Array trasferta (dal dzn)

% Parametri derivati
int: n_periods = n_teams div 2;     % Numero di periodi per settimana
int: n_weeks = n_teams - 1;         % Numero di settimane

% Sets
set of int: Teams = 1..n_teams;
set of int: Weeks = 1..n_weeks;
set of int: Periods = 1..n_periods;

% Convertiamo gli array di input in matrici 2D
array[Periods, Weeks] of Teams: home0_matrix = array2d(Periods, Weeks, home0);
array[Periods, Weeks] of Teams: away0_matrix = array2d(Periods, Weeks, away0);

% ==========================================
% VARIABILI DECISIONALI
% ==========================================

% Nuova assegnazione casa/trasferta
array[Periods, Weeks] of var Teams: home;  % Nuova matrice casa
array[Periods, Weeks] of var Teams: away;  % Nuova matrice trasferta

% Variabile binaria: per ogni match, decidiamo se invertire casa/trasferta
array[Periods, Weeks] of var 0..1: swap;

% Conteggio partite in casa e trasferta per ogni squadra
array[Teams] of var 0..n_weeks: home_count;
array[Teams] of var 0..n_weeks: away_count;

% Differenza assoluta tra partite in casa e trasferta per ogni squadra
array[Teams] of var 0..n_weeks: imbalance;

% Variabile obiettivo: somma totale degli sbilanciamenti
var int: total_imbalance;

% ==========================================
% VINCOLI
% ==========================================

% Vincolo 1: Mantieni gli stessi accoppiamenti, ma permetti di invertire casa/trasferta
constraint forall(p in Periods, w in Weeks) (
    (swap[p,w] = 0 -> (home[p,w] = home0_matrix[p,w] /\ away[p,w] = away0_matrix[p,w])) /\
    (swap[p,w] = 1 -> (home[p,w] = away0_matrix[p,w] /\ away[p,w] = home0_matrix[p,w]))
);

% Vincolo 2: Calcola il numero di partite in casa per ogni squadra
constraint forall(t in Teams) (
    home_count[t] = sum(p in Periods, w in Weeks) (bool2int(home[p,w] = t))
);

% Vincolo 3: Calcola il numero di partite in trasferta per ogni squadra
constraint forall(t in Teams) (
    away_count[t] = sum(p in Periods, w in Weeks) (bool2int(away[p,w] = t))
);

% Vincolo 4: Ogni squadra deve giocare esattamente (n_teams-1) partite totali
constraint forall(t in Teams) (
    home_count[t] + away_count[t] = n_teams - 1
);

% Vincolo 5: Calcola lo sbilanciamento per ogni squadra
constraint forall(t in Teams) (
    imbalance[t] = abs(home_count[t] - away_count[t])
);

% Vincolo 6: Calcola lo sbilanciamento totale
constraint total_imbalance = sum(t in Teams) (imbalance[t]);

% Vincolo 7: Nessuna squadra gioca contro se stessa
constraint forall(p in Periods, w in Weeks) (
    home[p,w] != away[p,w]
);

% Vincolo 8: Tutte le squadre in ogni settimana devono essere diverse
constraint forall(w in Weeks) (
    alldifferent([home[p,w] | p in Periods] ++ [away[p,w] | p in Periods])
);

constraint total_imbalance >= 0;

constraint total_imbalance <= n_teams;

% ==========================================
% FUNZIONE OBIETTIVO
% ==========================================

% Minimizza lo sbilanciamento totale
solve minimize total_imbalance;

% ==========================================
% VARIABILI AUSILIARIE PER OUTPUT
% ==========================================

% Per mostrare i miglioramenti rispetto alla soluzione originale
array[Teams] of var 0..n_weeks: original_home_count;
array[Teams] of var 0..n_weeks: original_away_count;
array[Teams] of var 0..n_weeks: original_imbalance;
var int: original_total_imbalance;

% Calcola i valori della soluzione originale
constraint forall(t in Teams) (
    original_home_count[t] = sum(p in Periods, w in Weeks) (bool2int(home0_matrix[p,w] = t))
);

constraint forall(t in Teams) (
    original_away_count[t] = sum(p in Periods, w in Weeks) (bool2int(away0_matrix[p,w] = t))
);

constraint forall(t in Teams) (
    original_imbalance[t] = abs(original_home_count[t] - original_away_count[t])
);

constraint original_total_imbalance = sum(t in Teams) (original_imbalance[t]);

% ==========================================
% OUTPUT
% ==========================================

% output [
%     "=== TOURNAMENT HOME/AWAY OPTIMIZATION RESULTS ===\n\n"
% ];

% output [
%     "Optimized Schedule (n=" ++ show(n_teams) ++ "):\n"
% ] ++
% [
%     "W" ++ show(w) ++ ": " ++
%     concat([
%         show(home[p,w]) ++ "v" ++ show(away[p,w]) ++ 
%         if p < n_periods then " " else "" endif
%         | p in Periods
%     ]) ++ "\n"
%     | w in Weeks
% ];

% output [
%     "\n=== COMPARISON TABLE ===\n",
%     "Team | Original H/A | Optimized H/A | Original Imb | Optimized Imb | Swaps Made\n",
%     "-----+-------------+--------------+-------------+--------------+-----------\n"
% ] ++
% [
%     show(t) ++ "   | " ++
%     show(original_home_count[t]) ++ "/" ++ show(original_away_count[t]) ++ "       | " ++
%     show(home_count[t]) ++ "/" ++ show(away_count[t]) ++ "        | " ++
%     show(original_imbalance[t]) ++ "       | " ++
%     show(imbalance[t]) ++ "      | " ++
%     show(sum(p in Periods, w in Weeks where home0_matrix[p,w] = t \/ away0_matrix[p,w] = t) (swap[p,w])) ++ "\n"
%     | t in Teams
% ];

output [
%     "\n=== SUMMARY ===\n",
    "Original Total Imbalance: " ++ show(original_total_imbalance) ++ "\n",
    "Optimized Total Imbalance: " ++ show(total_imbalance) ++ "\n",
    "Improvement: " ++ show(original_total_imbalance - total_imbalance) ++ "\n",
    "Total Swaps Made: " ++ show(sum(p in Periods, w in Weeks)(swap[p,w])) ++ "\n\n"
];

% output [
%     "=== OPTIMIZED MATRICES ===\n",
%     "home = ", show(home), "\n",
%     "away = ", show(away), "\n\n"
% ];

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

% output [
%     "\n=== SWAP DECISIONS ===\n",
%     "swap = ", show(swap), "\n",
%     "(1 = swapped home/away, 0 = kept original)\n"
% ];