"""Synthetic second-order-cone active-set benchmark."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ..core import Array, BenchmarkProblem, Context, Guidance, SolverResult, max_positive_violation
from ..optim import random_pattern_search
from ..physics import conic_constraint_values, conic_soft_penalty


class ConicActiveSetBenchmark(BenchmarkProblem):
    """SOC benchmark where learning proposes active-constraint scores."""

    name = "conic_active_set"

    def __init__(
        self,
        *,
        dimension: int = 3,
        cone_dim: int = 2,
        n_constraints: int = 8,
        top_k: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.dimension = int(dimension)
        self.cone_dim = int(cone_dim)
        self.n_constraints = int(n_constraints)
        self.top_k = int(top_k)

    def sample_context(self) -> Context:
        anchor = self.rng.normal(scale=0.25, size=self.dimension)
        target = anchor + self.rng.normal(scale=1.6, size=self.dimension)
        matrices = self.rng.normal(scale=0.55, size=(self.n_constraints, self.cone_dim, self.dimension))
        centers = anchor + self.rng.normal(scale=0.8, size=(self.n_constraints, self.dimension))
        offsets = -np.einsum("mcd,md->mc", matrices, centers)
        anchor_lhs = np.linalg.norm(np.einsum("mcd,d->mc", matrices, anchor) + offsets, axis=1)
        radii = anchor_lhs + self.rng.uniform(0.35, 0.95, size=self.n_constraints)
        return {
            "anchor": anchor,
            "target": target,
            "soc_matrices": matrices,
            "soc_offsets": offsets,
            "soc_radii": radii,
        }

    def objective(self, x: Array, theta: Context) -> float:
        target = np.asarray(theta["target"], dtype=float)
        residual = np.asarray(x, dtype=float) - target
        return float(0.5 * np.dot(residual, residual))

    def violation(self, x: Array, theta: Context) -> Array:
        return conic_constraint_values(x, theta)

    def expert_solve(self, theta: Context) -> SolverResult:
        start = perf_counter()
        selected = np.arange(self.n_constraints)
        x = self._solve_with_constraints(theta, selected, budget=1_200, warm_start=np.asarray(theta["anchor"], dtype=float))
        violations = self.violation(x, theta)
        if max_positive_violation(violations) > 1e-5:
            x = np.asarray(theta["anchor"], dtype=float).copy()
            violations = self.violation(x, theta)
        active_scores = self._active_scores(violations)
        guidance = Guidance(
            warm_start=x,
            candidate_mask=self._top_k_mask(active_scores, self.top_k),
            active_constraint_scores=active_scores,
            strategy_name="expert_active_set",
        )
        runtime = perf_counter() - start
        return SolverResult(
            x=x,
            objective=self.objective(x, theta),
            violations=violations,
            runtime=runtime,
            guidance=guidance,
            info={"selected_constraints": selected.tolist()},
        )

    def budgeted_solve(
        self, theta: Context, guidance: Guidance | None, budget: int
    ) -> SolverResult:
        start = perf_counter()
        guidance = guidance or Guidance(strategy_name="unguided")
        selected = self._selected_constraints_from_guidance(guidance)
        warm_start = (
            np.asarray(guidance.warm_start, dtype=float)
            if guidance.warm_start is not None
            else np.asarray(theta["anchor"], dtype=float)
        )
        x = self._solve_with_constraints(theta, selected, budget=budget, warm_start=warm_start)
        runtime = perf_counter() - start
        return SolverResult(
            x=x,
            objective=self.objective(x, theta),
            violations=self.violation(x, theta),
            runtime=runtime,
            guidance=guidance,
            info={"budget": int(budget), "selected_constraints": selected.tolist()},
        )

    def feature_vector(self, theta: Context) -> Array:
        return np.concatenate(
            [
                np.asarray(theta["target"], dtype=float).ravel(),
                np.asarray(theta["anchor"], dtype=float).ravel(),
                np.asarray(theta["soc_radii"], dtype=float).ravel(),
                np.asarray(theta["soc_matrices"], dtype=float).ravel(),
            ]
        )

    def _solve_with_constraints(
        self,
        theta: Context,
        selected: Array,
        *,
        budget: int,
        warm_start: Array,
    ) -> Array:
        lower = np.full(self.dimension, -4.0, dtype=float)
        upper = np.full(self.dimension, 4.0, dtype=float)

        def score(x: Array) -> float:
            return self.objective(x, theta) + conic_soft_penalty(
                x,
                theta,
                selected_constraints=selected.astype(int),
                weight=100_000.0,
            )

        x_best, _ = random_pattern_search(
            score,
            warm_start,
            budget=int(budget),
            rng=self.rng,
            step_scale=0.45,
            bounds=(lower, upper),
        )
        return x_best

    def _selected_constraints_from_guidance(self, guidance: Guidance) -> Array:
        if guidance.candidate_mask is not None:
            selected = np.flatnonzero(np.asarray(guidance.candidate_mask, dtype=bool))
            if selected.size:
                return selected.astype(float)
        if guidance.active_constraint_scores is not None:
            return np.flatnonzero(self._top_k_mask(np.asarray(guidance.active_constraint_scores, dtype=float), self.top_k)).astype(float)
        return np.arange(min(self.top_k, self.n_constraints), dtype=float)

    @staticmethod
    def _active_scores(violations: Array) -> Array:
        closeness = np.exp(-np.abs(np.asarray(violations, dtype=float)) / 0.15)
        total = float(np.sum(closeness))
        if total <= 0.0:
            return np.ones_like(closeness) / max(1, closeness.size)
        return closeness / total

    @staticmethod
    def _top_k_mask(scores: Array, k: int) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=float)
        mask = np.zeros(scores_arr.size, dtype=bool)
        if scores_arr.size == 0:
            return mask
        k_eff = min(max(1, int(k)), scores_arr.size)
        mask[np.argsort(scores_arr)[-k_eff:]] = True
        return mask


def make_default(**kwargs: Any) -> ConicActiveSetBenchmark:
    return ConicActiveSetBenchmark(**kwargs)
