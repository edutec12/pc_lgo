"""Linear EV charging benchmark with soft physics penalties."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ..core import Array, BenchmarkProblem, Context, Guidance, SolverResult
from ..physics import ev_schedule_matrix, ev_shortfall_and_grid_residuals, ev_soft_penalty


class EVChargingBenchmark(BenchmarkProblem):
    """Small EV charging problem with SOC and grid-limit constraints."""

    name = "ev_charging"

    def __init__(self, *, n_evs: int = 4, horizon: int = 8, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.n_evs = int(n_evs)
        self.horizon = int(horizon)

    def sample_context(self) -> Context:
        n_evs = self.n_evs
        horizon = self.horizon
        departure = self.rng.integers(max(2, horizon // 2), horizon + 1, size=n_evs)
        battery_capacity = self.rng.uniform(38.0, 72.0, size=n_evs)
        initial_soc = self.rng.uniform(0.2, 0.5, size=n_evs)
        max_rate = self.rng.uniform(5.0, 10.0, size=n_evs)
        feasible_energy = 0.48 * max_rate * departure
        requested_energy = self.rng.uniform(0.2, 0.42, size=n_evs) * battery_capacity
        energy_need = np.minimum(requested_energy, feasible_energy)
        target_soc = np.minimum(0.95, initial_soc + energy_need / battery_capacity)
        grid_limit = np.full(horizon, max(np.max(max_rate), 0.68 * float(np.sum(max_rate))), dtype=float)
        price = self.rng.uniform(0.08, 0.32, size=horizon)
        return {
            "n_evs": n_evs,
            "horizon": horizon,
            "dt": 1.0,
            "departure_time": departure,
            "battery_capacity": battery_capacity,
            "initial_soc": initial_soc,
            "target_soc": target_soc,
            "max_rate": max_rate,
            "grid_limit": grid_limit,
            "price": price,
        }

    def objective(self, x: Array, theta: Context) -> float:
        schedule = ev_schedule_matrix(x, theta)
        aggregate = np.sum(schedule, axis=0)
        price = np.asarray(theta["price"], dtype=float)
        dt = float(theta.get("dt", 1.0))
        return float(np.dot(price, aggregate) * dt)

    def violation(self, x: Array, theta: Context) -> Array:
        schedule = ev_schedule_matrix(x, theta)
        max_rate = np.asarray(theta["max_rate"], dtype=float)[:, None]
        lower = -schedule.ravel()
        upper = (schedule - max_rate).ravel()
        shortfall, grid_residual = ev_shortfall_and_grid_residuals(x, theta)
        departure = np.asarray(theta["departure_time"], dtype=int)
        after_departure = []
        for ev_idx, dep in enumerate(departure):
            after_departure.extend(schedule[ev_idx, dep:].tolist())
        return np.concatenate(
            [
                lower,
                upper,
                shortfall,
                grid_residual,
                np.asarray(after_departure, dtype=float),
            ]
        )

    def expert_solve(self, theta: Context) -> SolverResult:
        start = perf_counter()
        schedule = self._greedy_cost_schedule(theta)
        x = schedule.ravel()
        guidance = Guidance(warm_start=x, strategy_name="expert_greedy_ev")
        runtime = perf_counter() - start
        return SolverResult(
            x=x,
            objective=self.objective(x, theta),
            violations=self.violation(x, theta),
            runtime=runtime,
            guidance=guidance,
            info={"heuristic": "earliest-departure-cheapest-slot"},
        )

    def budgeted_solve(
        self, theta: Context, guidance: Guidance | None, budget: int
    ) -> SolverResult:
        start = perf_counter()
        guidance = guidance or Guidance(strategy_name="unguided")
        if guidance.warm_start is not None:
            schedule = ev_schedule_matrix(guidance.warm_start, theta).copy()
        else:
            schedule = np.zeros((self.n_evs, self.horizon), dtype=float)

        schedule = self._clip_operational(schedule, theta)
        best_score = self._penalized_score(schedule.ravel(), theta)
        for _ in range(max(0, int(budget))):
            candidate = schedule.copy()
            shortfall, grid_residual = ev_shortfall_and_grid_residuals(candidate.ravel(), theta)
            if np.max(shortfall) > 1e-8:
                self._add_charge_for_shortfall(candidate, theta, shortfall)
            else:
                self._try_price_shift(candidate, theta)
            candidate = self._clip_operational(candidate, theta)
            score = self._penalized_score(candidate.ravel(), theta)
            if score <= best_score + 1e-12:
                schedule = candidate
                best_score = score
            elif np.max(grid_residual) > 0:
                schedule = self._reduce_grid_violation(candidate, theta)
                best_score = self._penalized_score(schedule.ravel(), theta)

        x = schedule.ravel()
        runtime = perf_counter() - start
        return SolverResult(
            x=x,
            objective=self.objective(x, theta),
            violations=self.violation(x, theta),
            runtime=runtime,
            guidance=guidance,
            info={"budget": int(budget)},
        )

    def feature_vector(self, theta: Context) -> Array:
        return np.concatenate(
            [
                np.asarray(theta["departure_time"], dtype=float) / float(self.horizon),
                np.asarray(theta["battery_capacity"], dtype=float) / 100.0,
                np.asarray(theta["initial_soc"], dtype=float),
                np.asarray(theta["target_soc"], dtype=float),
                np.asarray(theta["max_rate"], dtype=float) / 10.0,
                np.asarray(theta["grid_limit"], dtype=float) / 30.0,
                np.asarray(theta["price"], dtype=float),
            ]
        )

    def _greedy_cost_schedule(self, theta: Context) -> Array:
        n_evs = int(theta["n_evs"])
        horizon = int(theta["horizon"])
        schedule = np.zeros((n_evs, horizon), dtype=float)
        grid_remaining = np.asarray(theta["grid_limit"], dtype=float).copy()
        departure = np.asarray(theta["departure_time"], dtype=int)
        price = np.asarray(theta["price"], dtype=float)
        max_rate = np.asarray(theta["max_rate"], dtype=float)
        capacity = np.asarray(theta["battery_capacity"], dtype=float)
        need = (np.asarray(theta["target_soc"], dtype=float) - np.asarray(theta["initial_soc"], dtype=float)) * capacity
        dt = float(theta.get("dt", 1.0))

        for ev_idx in np.argsort(departure):
            remaining = max(0.0, float(need[ev_idx]))
            slots = sorted(range(int(departure[ev_idx])), key=lambda idx: (price[idx], idx))
            for slot in slots:
                if remaining <= 1e-10:
                    break
                add_power = min(max_rate[ev_idx], grid_remaining[slot], remaining / dt)
                if add_power <= 0.0:
                    continue
                schedule[ev_idx, slot] += add_power
                grid_remaining[slot] -= add_power
                remaining -= add_power * dt
        return schedule

    def _penalized_score(self, x: Array, theta: Context) -> float:
        schedule = ev_schedule_matrix(x, theta)
        max_rate = np.asarray(theta["max_rate"], dtype=float)[:, None]
        bounds = np.concatenate([(-schedule).ravel(), (schedule - max_rate).ravel()])
        return self.objective(x, theta) + ev_soft_penalty(x, theta) + 1_000.0 * float(
            np.dot(np.maximum(bounds, 0.0), np.maximum(bounds, 0.0))
        )

    def _clip_operational(self, schedule: Array, theta: Context) -> Array:
        max_rate = np.asarray(theta["max_rate"], dtype=float)
        departure = np.asarray(theta["departure_time"], dtype=int)
        clipped = np.maximum(schedule, 0.0)
        clipped = np.minimum(clipped, max_rate[:, None])
        for ev_idx, dep in enumerate(departure):
            clipped[ev_idx, dep:] = 0.0
        return clipped

    def _add_charge_for_shortfall(self, schedule: Array, theta: Context, shortfall: Array) -> None:
        ev_idx = int(np.argmax(shortfall))
        capacity = np.asarray(theta["battery_capacity"], dtype=float)
        max_rate = np.asarray(theta["max_rate"], dtype=float)
        departure = np.asarray(theta["departure_time"], dtype=int)
        price = np.asarray(theta["price"], dtype=float)
        grid_limit = np.asarray(theta["grid_limit"], dtype=float)
        dt = float(theta.get("dt", 1.0))
        needed_energy = max(0.0, float(shortfall[ev_idx] * capacity[ev_idx]))
        slots = sorted(range(int(departure[ev_idx])), key=lambda idx: (price[idx], idx))
        aggregate = np.sum(schedule, axis=0)
        for slot in slots:
            if needed_energy <= 1e-10:
                break
            room_rate = min(max_rate[ev_idx] - schedule[ev_idx, slot], grid_limit[slot] - aggregate[slot])
            add_power = min(max(0.0, room_rate), needed_energy / dt)
            schedule[ev_idx, slot] += add_power
            aggregate[slot] += add_power
            needed_energy -= add_power * dt

    def _try_price_shift(self, schedule: Array, theta: Context) -> None:
        price = np.asarray(theta["price"], dtype=float)
        max_rate = np.asarray(theta["max_rate"], dtype=float)
        departure = np.asarray(theta["departure_time"], dtype=int)
        grid_limit = np.asarray(theta["grid_limit"], dtype=float)
        aggregate = np.sum(schedule, axis=0)
        ev_idx = int(self.rng.integers(0, schedule.shape[0]))
        available = np.arange(int(departure[ev_idx]))
        if available.size < 2 or np.sum(schedule[ev_idx, available]) <= 1e-10:
            return
        expensive = int(available[np.argmax(price[available] + (schedule[ev_idx, available] <= 0.0) * 10.0)])
        cheap = int(available[np.argmin(price[available] + (aggregate[available] >= grid_limit[available]) * 10.0)])
        if price[cheap] >= price[expensive] or schedule[ev_idx, expensive] <= 0.0:
            return
        delta = min(schedule[ev_idx, expensive], max_rate[ev_idx] - schedule[ev_idx, cheap], grid_limit[cheap] - aggregate[cheap])
        delta = max(0.0, float(delta))
        schedule[ev_idx, expensive] -= delta
        schedule[ev_idx, cheap] += delta

    def _reduce_grid_violation(self, schedule: Array, theta: Context) -> Array:
        price = np.asarray(theta["price"], dtype=float)
        grid_limit = np.asarray(theta["grid_limit"], dtype=float)
        reduced = schedule.copy()
        aggregate = np.sum(reduced, axis=0)
        for slot in np.flatnonzero(aggregate > grid_limit + 1e-10):
            excess = aggregate[slot] - grid_limit[slot]
            ev_order = np.argsort(-reduced[:, slot])
            for ev_idx in ev_order:
                remove = min(excess, reduced[ev_idx, slot])
                reduced[ev_idx, slot] -= remove
                excess -= remove
                if excess <= 1e-10:
                    break
        del price
        return self._clip_operational(reduced, theta)


def make_default(**kwargs: Any) -> EVChargingBenchmark:
    return EVChargingBenchmark(**kwargs)
