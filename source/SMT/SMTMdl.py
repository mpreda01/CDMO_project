#!/usr/bin/env python3
import json
import time
import os
from z3 import *

class STSModel:
    def __init__(self, n_teams,
                 sb_weeks=False, sb_periods=False, sb_teams=False,
                 ic_matches_per_team=False, ic_period_count=False):
        self.n = n_teams
        self.n_weeks = n_teams - 1
        self.n_periods = n_teams // 2
        self.sb_weeks = sb_weeks
        self.sb_periods = sb_periods
        self.sb_teams = sb_teams
        self.ic_matches_per_team = ic_matches_per_team
        self.ic_period_count = ic_period_count

        # boolean var x[w][p][h][a]
        self.x = {}
        for w in range(self.n_weeks):
            self.x[w] = {}
            for p in range(self.n_periods):
                self.x[w][p] = {}
                for h in range(self.n):
                    self.x[w][p][h] = {}
                    for a in range(self.n):
                        self.x[w][p][h][a] = Bool(f'x_{w}_{p}_{h}_{a}')

    def create_constraints(self, solver):
        # Base constraints 
        for w in range(self.n_weeks):
            for p in range(self.n_periods):
                for t in range(self.n):
                    solver.add(Not(self.x[w][p][t][t]))

        for w in range(self.n_weeks):
            for p in range(self.n_periods):
                game_pairs = [self.x[w][p][h][a] for h in range(self.n) for a in range(self.n) if h != a]
                solver.add(PbEq([(pair, 1) for pair in game_pairs], 1))

        for w in range(self.n_weeks):
            for t in range(self.n):
                plays_in_week = []
                for p in range(self.n_periods):
                    for a in range(self.n):
                        if a != t:
                            plays_in_week.append(self.x[w][p][t][a])
                    for h in range(self.n):
                        if h != t:
                            plays_in_week.append(self.x[w][p][h][t])
                solver.add(PbEq([(game, 1) for game in plays_in_week], 1))

        for t1 in range(self.n):
            for t2 in range(t1 + 1, self.n):
                games_between = []
                for w in range(self.n_weeks):
                    for p in range(self.n_periods):
                        games_between.append(self.x[w][p][t1][t2])
                        games_between.append(self.x[w][p][t2][t1])
                solver.add(PbEq([(game, 1) for game in games_between], 1))

        for t in range(self.n):
            for p in range(self.n_periods):
                period_games = []
                for w in range(self.n_weeks):
                    for a in range(self.n):
                        if a != t:
                            period_games.append(self.x[w][p][t][a])
                    for h in range(self.n):
                        if h != t:
                            period_games.append(self.x[w][p][h][t])
                solver.add(PbLe([(game, 1) for game in period_games], 2))

        
        if self.ic_matches_per_team:
            self.add_ic_matches_per_team(solver)
        if self.ic_period_count:
            self.add_ic_period_count(solver)
        if self.sb_teams or self.sb_weeks or self.sb_periods:
            self.add_symmetry_breaking(solver)

    def add_ic_matches_per_team(self, solver):
        for t in range(self.n):
            total_games = []
            for w in range(self.n_weeks):
                for p in range(self.n_periods):
                    for a in range(self.n):
                        if a != t:
                            total_games.append(self.x[w][p][t][a])
                            total_games.append(self.x[w][p][a][t])
            solver.add(PbEq([(g, 1) for g in total_games], self.n - 1))

    def add_ic_period_count(self, solver):
        for p in range(self.n_periods):
            total_games = []
            for w in range(self.n_weeks):
                for h in range(self.n):
                    for a in range(self.n):
                        if h != a:
                            total_games.append(self.x[w][p][h][a])
            solver.add(PbEq([(g, 1) for g in total_games], self.n_weeks))

    def add_symmetry_breaking(self, solver):
        if self.sb_teams and self.n >= 2:
            solver.add(self.x[0][0][0][1])

    def optimize_schedule(self, schedule):
        """Reduce max imbalance between home and away games."""
        home_count = [0] * self.n
        away_count = [0] * self.n

        for period in schedule:
            for match in period:
                h, a = match
                if h > 0 and a > 0:
                    home_count[h-1] += 1
                    away_count[a-1] += 1

        improved = True
        while improved:
            improved = False
            for period in schedule:
                for i, match in enumerate(period):
                    h, a = match
                    if h == 0 or a == 0:
                        continue
                    diff_h_before = abs(home_count[h-1] - away_count[h-1])
                    diff_a_before = abs(home_count[a-1] - away_count[a-1])

                    # temp invertion
                    home_count[h-1] -= 1
                    away_count[a-1] -= 1
                    home_count[a-1] += 1
                    away_count[h-1] += 1

                    diff_h_after = abs(home_count[h-1] - away_count[h-1])
                    diff_a_after = abs(home_count[a-1] - away_count[a-1])

                    if (max(diff_h_after, diff_a_after) < max(diff_h_before, diff_a_before)):
                        period[i] = [a, h]
                        improved = True
                    else:
                        # revert
                        home_count[h-1] += 1
                        away_count[a-1] += 1
                        home_count[a-1] -= 1
                        away_count[h-1] -= 1

        max_diff = max(abs(home_count[t] - away_count[t]) for t in range(self.n))
        return schedule, max_diff

    def solve(self, time_limit=300, optimize=False):
        solver = Solver()
        solver.set("timeout", time_limit * 1000)
        self.create_constraints(solver)
        start = time.time()
        result = solver.check()
        runtime = int(time.time() - start)

        if result == sat:
            model = solver.model()
            schedule = self.extract_solution(model)
            obj_val = None
            if optimize:
                schedule, obj_val = self.optimize_schedule(schedule)
            return {
                "time": runtime,
                "optimal": (optimize and obj_val == 1) if optimize else True,
                "obj": obj_val if optimize else None,
                "sol": schedule
            }

        return {"time": runtime, "optimal": False, "obj": None, "sol": None}

    def extract_solution(self, model):
        schedule = []
        for p in range(self.n_periods):
            period_schedule = []
            for w in range(self.n_weeks):
                found = False
                for h in range(self.n):
                    for a in range(self.n):
                        if h != a and is_true(model.eval(self.x[w][p][h][a])):
                            period_schedule.append([h+1, a+1])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    period_schedule.append([0, 0])
            schedule.append(period_schedule)
        return schedule
