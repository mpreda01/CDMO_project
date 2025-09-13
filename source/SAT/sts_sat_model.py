"""
STS (Social Tournament Scheduling) SAT Solver Model
This module contains the core SAT model and constraint definitions for solving STS problems.
"""

import time
import os
import json
from z3 import *
from typing import Dict, List, Tuple, Optional, Any


class STSSATSolver:
    """
    SAT-based solver for the Social Tournament Scheduling (STS) problem.
    
    The STS problem schedules a round-robin tournament where:
    - n teams (n even) play over n-1 weeks
    - Each week has n/2 periods (time slots)
    - Each team plays exactly once per week
    - Each team plays every other team exactly once
    - Each team plays at most twice in any period across all weeks
    """
    
    def __init__(self, n: int, timeout: int = 300):
        """
        Initialize the STS SAT solver.
        
        Args:
            n: Number of teams (must be even)
            timeout: Solver timeout in seconds
        """
        if n % 2 != 0:
            raise ValueError("Number of teams must be even")
        
        self.n = n
        self.n_weeks = n - 1
        self.n_periods = n // 2
        self.timeout = timeout * 1000  # Convert to milliseconds for Z3
        
        # Ranges for iteration (keep 0-indexed internally)
        self.Teams = range(n)
        self.Weeks = range(self.n_weeks)
        self.Periods = range(self.n_periods)
        
        # Initialize solver and variables
        self.solver = Solver()
        self._create_variables()
    
    def _create_variables(self):
        """Create Boolean variables for game assignments."""
        self.game = {}
        for t in self.Teams:
            self.game[t] = {}
            for w in self.Weeks:
                self.game[t][w] = {}
                for p in self.Periods:
                    self.game[t][w][p] = {}
                    for s in range(2):  # 0=home, 1=away
                        var_name = f'game_{t}_{w}_{p}_{s}'
                        self.game[t][w][p][s] = Bool(var_name)
    
    def add_base_constraints(self):
        """Add all base STS constraints."""
        self._add_weekly_participation_constraints()
        self._add_period_occupancy_constraints()
        self._add_period_limit_constraints()
        self._add_match_uniqueness_constraints()
    
    def _add_weekly_participation_constraints(self):
        """Constraint: Each team plays exactly once per week."""
        for t in self.Teams:
            for w in self.Weeks:
                positions = []
                for p in self.Periods:
                    positions.append(self.game[t][w][p][0]) 
                    positions.append(self.game[t][w][p][1])  
                
                self.solver.add(PbEq([(pos, 1) for pos in positions], 1))
    
    def _add_period_occupancy_constraints(self):
        """Constraint: Each period has exactly one home team and one away team."""
        for w in self.Weeks:
            for p in self.Periods:
                home_teams = [self.game[t][w][p][0] for t in self.Teams]
                away_teams = [self.game[t][w][p][1] for t in self.Teams]
                
                # Exactly one team at home and one away
                self.solver.add(PbEq([(h, 1) for h in home_teams], 1))
                self.solver.add(PbEq([(a, 1) for a in away_teams], 1))
    
    def _add_period_limit_constraints(self):
        """Constraint: Each team plays at most twice in the same period."""
        for t in self.Teams:
            for p in self.Periods:
                period_appearances = []
                for w in self.Weeks:
                    period_appearances.append(self.game[t][w][p][0]) 
                    period_appearances.append(self.game[t][w][p][1]) 
                
                # At most 2 appearances in the same period
                self.solver.add(PbLe([(app, 1) for app in period_appearances], 2))
    
    def _add_match_uniqueness_constraints(self):
        """Constraint: Every team plays with every other team exactly once."""
        for i in self.Teams:
            for j in range(i + 1, self.n):
                matches = []
                for w in self.Weeks:
                    for p in self.Periods:
                        matches.append(And(self.game[i][w][p][0], 
                                         self.game[j][w][p][1]))
                        matches.append(And(self.game[j][w][p][0], 
                                         self.game[i][w][p][1]))
                
                # Exactly one match between each pair
                self.solver.add(PbEq([(m, 1) for m in matches], 1))
    
    def add_symmetry_breaking_constraints(self, level = ["full"]):
        """
        Add symmetry breaking constraints to reduce search space.
        
        Args:
            level: Symmetry breaking level:
                - "sb_match": Only fix first match - basic
                - "sb_team": Fix first week structure - moderate  
                - "sb_period": All symmetry breaking constraints - full
        """
    

        for l in level:
            if l == "sb_match":
                self._add_match_symmetry_breaking()
        
            elif l == "sb_team":
                self._add_team_symmetry_breaking()
        
            elif l == "sb_period":
                self._add_period_symmetry_breaking()
            else:
                self._add_match_symmetry_breaking()
                self._add_team_symmetry_breaking()
                self._add_period_symmetry_breaking()

    def _add_match_symmetry_breaking(self):
        """Match symmetry breaking: Fix first match."""
        # Team 0 plays at home in period 0 of week 0
        self.solver.add(self.game[0][0][0][0])
        
        # Team 1 plays away in period 0 of week 0
        if self.n > 1:
            self.solver.add(self.game[1][0][0][1])
    
    def _add_team_symmetry_breaking(self):
        """Team symmetry: Team 0 meets team i in week i-1."""
        if self.n >= 4:
            for i in range(1, min(self.n // 2 + 1, self.n)):
                week_idx = i - 1
                if week_idx < self.n_weeks:
                    match_constraint = []
                    for p in self.Periods:
                        # Team 0 vs team i (either home or away)
                        match_constraint.append(
                            And(self.game[0][week_idx][p][0], 
                                self.game[i][week_idx][p][1])
                        )
                        match_constraint.append(
                            And(self.game[0][week_idx][p][1], 
                                self.game[i][week_idx][p][0])
                        )
                    self.solver.add(Or(match_constraint))
    
    def _add_period_symmetry_breaking(self):
        """Period symmetry breaking for larger instances."""
        # Fix second period for n >= 6
        if self.n >= 6:
            self.solver.add(self.game[2][0][1][0])  # Team 2 home in period 1
            self.solver.add(self.game[3][0][1][1])  # Team 3 away in period 1
        
        # Home/Away symmetry: Lower index team plays at home in first matches
        for p in range(min(2, self.n_periods)):
            for i in self.Teams:
                for j in range(i + 1, self.n):
                    match_occurs = Or(
                        And(self.game[i][0][p][0], self.game[j][0][p][1]),
                        And(self.game[i][0][p][1], self.game[j][0][p][0])
                    )
                    self.solver.add(Implies(match_occurs, self.game[i][0][p][0]))
        
        # Opponent order: Team 0 meets lower-indexed teams first
        if self.n >= 3:
            self._add_opponent_order_constraints()
    
    def _add_opponent_order_constraints(self):
        """Ensure team 0 meets lower-indexed teams first."""
        for w1 in self.Weeks:
            for w2 in range(w1 + 1, self.n_weeks):
                plays_2_in_w1 = []
                plays_1_in_w2 = []
                
                for p in self.Periods:
                    # Team 0 plays team 2 in week w1
                    plays_2_in_w1.append(Or(
                        And(self.game[0][w1][p][0], self.game[2][w1][p][1]),
                        And(self.game[0][w1][p][1], self.game[2][w1][p][0])
                    ))
                    # Team 0 plays team 1 in week w2
                    plays_1_in_w2.append(Or(
                        And(self.game[0][w2][p][0], self.game[1][w2][p][1]),
                        And(self.game[0][w2][p][1], self.game[1][w2][p][0])
                    ))
                
                self.solver.add(Implies(Or(plays_2_in_w1), Not(Or(plays_1_in_w2))))
    
    def solve_feasibility(self, symmetry_level = ["full"]) -> Tuple[bool, Optional[List], float, Optional[Any]]:
        """
        Solve the STS problem for feasibility.
        
        Args:
            symmetry_level: Level of symmetry breaking constraints
            
        Returns:
            Tuple of (satisfiable, solution, solve_time, model)
        """
        self.add_base_constraints()
        self.add_symmetry_breaking_constraints(level=symmetry_level)
        
        self.solver.set("timeout", self.timeout)
        
        start_time = time.time()
        result = self.solver.check()
        solve_time = time.time() - start_time
        
        if result == sat:
            model = self.solver.model()
            solution = self._extract_solution(model)
            return True, solution, solve_time, model
        else:
            return False, None, solve_time, None
    
    def _load_initial_solution(self, output_dir: str = "res/SAT") -> Tuple[Optional[List[List[List[int]]]], Optional[int]]:
        """
        Load existing feasible solution and its objective value.
        The solution in JSON is in format period x week, but we need week x period.
        """
        json_file = os.path.join(output_dir, f"{self.n}.json")
        
        if not os.path.exists(json_file):
            return None, None
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for approach, result in data.items():
                if 'sol' in result and result['sol'] and 'obj' in result:
                    sol_from_json = result['sol']
                    
                    # Il JSON contiene sol in formato period x week
                    # Dobbiamo trasporlo in week x period per uso interno
                    
                    # Verifica se abbiamo una soluzione valida
                    if not sol_from_json or not isinstance(sol_from_json, list):
                        continue
                        
                    # Determina le dimensioni
                    n_periods_in_json = len(sol_from_json)
                    n_weeks_in_json = len(sol_from_json[0]) if sol_from_json else 0
                    
                    # Crea la struttura week x period (quella che usa internamente il solver)
                    week_x_period_solution = [[[] for _ in range(self.n_periods)] 
                                              for _ in range(self.n_weeks)]
                    
                    # Trasponi da period x week a week x period
                    for p in range(min(n_periods_in_json, self.n_periods)):
                        for w in range(min(n_weeks_in_json, self.n_weeks)):
                            if p < len(sol_from_json) and w < len(sol_from_json[p]):
                                match = sol_from_json[p][w]
                                if match and isinstance(match, list) and len(match) == 2:
                                    week_x_period_solution[w][p] = match
                    
                    return week_x_period_solution, result['obj']
                    
        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
            print(f"Error loading initial solution: {e}")
            pass
        
        return None, None
    
    def solve_optimization_incremental(self, symmetry_level = ["full"], output_dir: str = "res/SAT") -> Dict[str, Any]:
        """
        Solve optimization by iteratively improving from an initial solution.
        Uses binary search on the objective value.
        """
        initial_solution, initial_obj = self._load_initial_solution(output_dir)
        
        if initial_solution is None or initial_obj is None:
            print("No feasible solution found. Please run in 'feasible' mode first.")
            return {
                'satisfiable': False,
                'solution': None,
                'time': 0,
                'obj': None,
                'optimal': False,
                'model': None
            }
        
        
        # Soluzione di partenza (sempre valida e completa)
        best_solution = initial_solution  # Inizializza con la soluzione iniziale
        best_obj = initial_obj
        best_model = None
        
        lower_bound = 0
        upper_bound = initial_obj
        
        start_total = time.time()
        timeout_reached = False
        
        while lower_bound <= upper_bound:
            elapsed = time.time() - start_total
            remaining_time = self.timeout / 1000 - elapsed
            if remaining_time <= 0:
                print("Timeout reached before starting new iteration.")
                timeout_reached = True
                break
            
            mid = (lower_bound + upper_bound) // 2
            
            self.solver = Solver()
            self.solver.set("timeout", int(remaining_time * 1000))
            
            self._create_variables()
            self.add_base_constraints()
            self.add_symmetry_breaking_constraints(level=symmetry_level)
            self._add_objective_constraint(mid)
            
            start_iter = time.time()
            result = self.solver.check()
            iter_time = time.time() - start_iter
            
            if result == sat:
                model = self.solver.model()
                solution = self._extract_solution(model)
                obj_value = self._calculate_objective(model)
                
                
                # Aggiorna solo se troviamo una soluzione migliore
                if obj_value < best_obj:
                    best_solution = solution
                    best_obj = obj_value
                    best_model = model
                
                upper_bound = obj_value - 1
                
            elif result == unsat:
                lower_bound = mid + 1
                
            else:  # result == unknown (timeout durante il check)
                timeout_reached = True
                break
            
            # Controlla se abbiamo superato il timeout totale
            if time.time() - start_total >= self.timeout / 1000:
                print("Total timeout reached.")
                timeout_reached = True
                break
        
        total_time = time.time() - start_total
        
        # Determina se la soluzione è ottimale
        is_optimal = (best_model is not None and 
                    lower_bound > upper_bound and 
                    not timeout_reached)        
        return {
            'satisfiable': True,
            'solution': best_solution,  # Sempre popolato (almeno con initial_solution)
            'time': total_time,
            'obj': best_obj,
            'optimal': is_optimal,
            'model': best_model  # Può essere None se non abbiamo migliorato la soluzione iniziale
        }
    
    def _add_objective_constraint(self, max_value: int):
        """Add constraint that maximum imbalance <= max_value."""
        for t in self.Teams:
            home_count = Sum([
                If(self.game[t][w][p][0], 1, 0)
                for w in self.Weeks for p in self.Periods
            ])
            away_count = Sum([
                If(self.game[t][w][p][1], 1, 0)
                for w in self.Weeks for p in self.Periods
            ])
            
            # Create absolute difference
            abs_diff = If(home_count >= away_count,
                         home_count - away_count,
                         away_count - home_count)
            # Constrain each team's abs diff to be <= max_value
            self.solver.add(abs_diff <= max_value)

    def _calculate_objective(self, model: Any) -> int:
        """Calculate objective value: maximum home/away imbalance across all teams."""
        max_abs_diff = 0

        for t in self.Teams:
            home_count = 0
            away_count = 0

            for w in self.Weeks:
                for p in self.Periods:
                    if is_true(model.eval(self.game[t][w][p][0])):
                        home_count += 1
                    elif is_true(model.eval(self.game[t][w][p][1])):
                        away_count += 1

            abs_diff = abs(home_count - away_count)
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff

        return max_abs_diff

    def _extract_solution(self, model: Any) -> List[List[List[int]]]:
        """Extract solution from Z3 model in week x period format."""
        schedule = [[[] for _ in self.Periods] for _ in self.Weeks]
        
        for w in self.Weeks:
            for p in self.Periods:
                home_team = None
                away_team = None
                
                for t in self.Teams:
                    if is_true(model.eval(self.game[t][w][p][0])):
                        home_team = t + 1  # Convert to 1-indexed
                    if is_true(model.eval(self.game[t][w][p][1])):
                        away_team = t + 1  # Convert to 1-indexed
                
                if home_team and away_team:
                    schedule[w][p] = [home_team, away_team]
        
        return schedule
    
    def get_solution_stats(self, model: Any) -> Dict[str, Any]:
        """Get statistics about the solution."""
        stats = {
            'teams': {},
            'max_deviation': 0,
            'sum_abs_diff': 0,
            'is_balanced': True
        }

        for t in self.Teams:
            home_count = 0
            away_count = 0

            for w in self.Weeks:
                for p in self.Periods:
                    if is_true(model.eval(self.game[t][w][p][0])):
                        home_count += 1
                    elif is_true(model.eval(self.game[t][w][p][1])):
                        away_count += 1

            abs_diff = abs(home_count - away_count)
            stats['teams'][t + 1] = {
                'home': home_count,
                'away': away_count,
                'abs_diff': abs_diff
            }
            stats['max_deviation'] = max(stats['max_deviation'], abs_diff)
            stats['sum_abs_diff'] += abs_diff

        # Define balanced as max absolute difference <= 1
        stats['is_balanced'] = stats['max_deviation'] <= 1

        return stats