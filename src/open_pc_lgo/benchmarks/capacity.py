"""Binary capacity expansion toy benchmark."""

from __future__ import annotations

from itertools import product
from time import perf_counter
from typing import Any

import numpy as np

from ..core import Array, BenchmarkProblem, Context, Guidance, SolverResult
from ..optim import binary_local_search
from ..physics import reserve_soft_penalty


class CapacityExpansionToyBenchmark(BenchmarkProblem):
    """Small binary investment problem with reserve adequacy."""

    name = "capacity_expansion_toy"

    def __init__(self, *, n_assets: int = 8, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.n_assets = int(n_assets)

    def sample_context(self) -> Context:
        capacities = self.rng.integers(8, 45, size=self.n_assets).astype(float)
        investment_cost = capacities * self.rng.uniform(0.75, 1.65, size=self.n_assets)
        reserve_margin = float(self.rng.uniform(0.10, 0.25))
        peak_demand = float(self.rng.uniform(0.35, 0.62) * np.sum(capacities))
        reserve_requirement = peak_demand * (1.0 + reserve_margin)
        return {
            "capacities": capacities,
            "investment_cost": investment_cost,
            "peak_demand": peak_demand,
            "reserve_margin": reserve_margin,
            "reserve_requirement": reserve_requirement,
        }

    def objective(self, x: Array, theta: Context) -> float:
        plan = np.asarray(x, dtype=float)
        return float(np.dot(np.asarray(theta["investment_cost"], dtype=float), plan))

    def violation(self, x: Array, theta: Context) -> Array:
        plan = np.asarray(x, dtype=float)
        capacities = np.asarray(theta["capacities"], dtype=float)
        reserve_shortfall = float(theta["reserve_requirement"]) - float(np.dot(capacities, plan))
        lower = -plan
        upper = plan - 1.0
        integrality = np.abs(plan - np.rint(plan))
        return np.concatenate([[reserve_shortfall], lower, upper, integrality])

    def expert_solve(self, theta: Context) -> SolverResult:
        start = perf_counter()
        best_plan: Array | None = None
        best_obj = float("inf")
        for bits in product([0.0, 1.0], repeat=self.n_assets):
            plan = np.asarray(bits, dtype=float)
            if np.max(np.maximum(self.violation(plan, theta), 0.0)) <= 1e-9:
                value = self.objective(plan, theta)
                if value < best_obj:
                    best_obj = value
                    best_plan = plan
        if best_plan is None:
            best_plan = np.ones(self.n_assets, dtype=float)
            best_obj = self.objective(best_plan, theta)

        scores = self._asset_scores(theta)
        guidance = Guidance(
            warm_start=best_plan,
            candidate_mask=best_plan.astype(bool),
            active_constraint_scores=scores,
            strategy_name="expert_exhaustive_capacity",
        )
        runtime = perf_counter() - start
        return SolverResult(
            x=best_plan,
            objective=best_obj,
            violations=self.violation(best_plan, theta),
            runtime=runtime,
            guidance=guidance,
            info={"enumerated_plans": 2**self.n_assets},
        )

    def budgeted_solve(
        self, theta: Context, guidance: Guidance | None, budget: int
    ) -> SolverResult:
        start = perf_counter()
        guidance = guidance or Guidance(strategy_name="unguided")
        if guidance.warm_start is not None:
            x0 = np.asarray(guidance.warm_start, dtype=float)
        elif guidance.active_constraint_scores is not None:
            order = np.argsort(-np.asarray(guidance.active_constraint_scores, dtype=float))
            x0 = np.zeros(self.n_assets, dtype=float)
            for idx in order:
                x0[idx] = 1.0
                if self.violation(x0, theta)[0] <= 0.0:
                    break
        else:
            x0 = np.zeros(self.n_assets, dtype=float)

        candidate_mask = guidance.candidate_mask
        x0 = self._optimizer_fill_shortfall(x0, theta, candidate_mask)

        def score(plan: Array) -> float:
            integrality = np.abs(plan - np.rint(plan))
            return self.objective(plan, theta) + reserve_soft_penalty(plan, theta, weight=1_000.0) + 1_000.0 * float(
                np.dot(integrality, integrality)
            )

        x_best, _ = binary_local_search(
            score,
            x0,
            budget=int(budget),
            rng=self.rng,
            candidate_mask=candidate_mask,
        )
        x_best = self._optimizer_fill_shortfall(x_best, theta, candidate_mask)
        runtime = perf_counter() - start
        return SolverResult(
            x=x_best,
            objective=self.objective(x_best, theta),
            violations=self.violation(x_best, theta),
            runtime=runtime,
            guidance=guidance,
            info={"budget": int(budget)},
        )

    def feature_vector(self, theta: Context) -> Array:
        return np.concatenate(
            [
                np.asarray(theta["capacities"], dtype=float) / 50.0,
                np.asarray(theta["investment_cost"], dtype=float) / 80.0,
                np.asarray([theta["peak_demand"], theta["reserve_margin"], theta["reserve_requirement"]], dtype=float) / 100.0,
            ]
        )

    def _optimizer_fill_shortfall(
        self,
        plan: Array,
        theta: Context,
        candidate_mask: np.ndarray | None,
    ) -> Array:
        filled = np.rint(np.asarray(plan, dtype=float)).clip(0.0, 1.0)
        capacities = np.asarray(theta["capacities"], dtype=float)
        costs = np.asarray(theta["investment_cost"], dtype=float)
        allowed = np.ones(self.n_assets, dtype=bool) if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
        if not np.any(allowed):
            allowed = np.ones(self.n_assets, dtype=bool)
        ratio = costs / np.maximum(capacities, 1e-9)
        for idx in np.argsort(ratio):
            if self.violation(filled, theta)[0] <= 0.0:
                break
            if allowed[idx]:
                filled[idx] = 1.0
        if self.violation(filled, theta)[0] > 0.0:
            for idx in np.argsort(ratio):
                if self.violation(filled, theta)[0] <= 0.0:
                    break
                filled[idx] = 1.0
        return filled

    def _asset_scores(self, theta: Context) -> Array:
        capacities = np.asarray(theta["capacities"], dtype=float)
        costs = np.asarray(theta["investment_cost"], dtype=float)
        scores = capacities / np.maximum(costs, 1e-9)
        return scores / max(float(np.max(scores)), 1e-9)


def make_default(**kwargs: Any) -> CapacityExpansionToyBenchmark:
    return CapacityExpansionToyBenchmark(**kwargs)
