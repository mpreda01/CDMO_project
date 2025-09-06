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

% Variabile obiettivo: massimo sbilanciamento tra tutte le squadre
var 0..n_weeks: max_imbalance;

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

% Vincolo 6: Calcola lo sbilanciamento massimo
constraint max_imbalance = max(t in Teams) (imbalance[t]);

% Vincolo 7: Nessuna squadra gioca contro se stessa
constraint forall(p in Periods, w in Weeks) (
    home[p,w] != away[p,w]
);

% Vincolo 8: Tutte le squadre in ogni settimana devono essere diverse
constraint forall(w in Weeks) (
    alldifferent([home[p,w] | p in Periods] ++ [away[p,w] | p in Periods])
);

% ==========================================
% FUNZIONE OBIETTIVO
% ==========================================

% Minimizza lo sbilanciamento massimo
constraint max_imbalance = 1;
solve satisfy;

% ==========================================
% VARIABILI AUSILIARIE PER OUTPUT
% ==========================================

% Per mostrare i miglioramenti rispetto alla soluzione originale
array[Teams] of var 0..n_weeks: original_home_count;
array[Teams] of var 0..n_weeks: original_away_count;
array[Teams] of var 0..n_weeks: original_imbalance;
var 0..n_weeks: original_max_imbalance;

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