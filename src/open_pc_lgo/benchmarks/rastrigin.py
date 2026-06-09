"""Continuous nonconvex warm-start benchmark based on Rastrigin."""

from __future__ import annotations

from itertools import product
from time import perf_counter
from typing import Any

import numpy as np

from ..core import Array, BenchmarkProblem, Context, Guidance, SolverResult
from ..optim import random_pattern_search


class RastriginWarmStartBenchmark(BenchmarkProblem):
    """Rastrigin benchmark where learning proposes a warm start."""

    name = "rastrigin_warm_start"

    def __init__(
        self,
        *,
        dimension: int = 2,
        grid_points: int = 9,
        bounds: tuple[float, float] = (-5.12, 5.12),
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.dimension = int(dimension)
        self.grid_points = int(grid_points)
        self.bounds_scalar = bounds

    def sample_context(self) -> Context:
        shift = self.rng.uniform(-1.5, 1.5, size=self.dimension)
        amplitude = float(self.rng.uniform(8.0, 12.0))
        return {"shift": shift, "amplitude": amplitude, "dimension": self.dimension}

    def objective(self, x: Array, theta: Context) -> float:
        z = np.asarray(x, dtype=float) - np.asarray(theta["shift"], dtype=float)
        amplitude = float(theta["amplitude"])
        return float(amplitude * z.size + np.sum(z * z - amplitude * np.cos(2.0 * np.pi * z)))

    def violation(self, x: Array, theta: Context) -> Array:
        del theta
        lower, upper = self.bounds_scalar
        x_arr = np.asarray(x, dtype=float)
        return np.concatenate([np.full_like(x_arr, lower) - x_arr, x_arr - np.full_like(x_arr, upper)])

    def expert_solve(self, theta: Context) -> SolverResult:
        start = perf_counter()
        warm_start = self._best_grid_point(theta)
        guidance = Guidance(warm_start=warm_start, strategy_name="expert_grid_warm_start")
        result = self.budgeted_solve(theta, guidance, budget=600)
        result.runtime += perf_counter() - start
        result.guidance = guidance
        result.info["expert_warm_start"] = warm_start
        return result

    def budgeted_solve(
        self, theta: Context, guidance: Guidance | None, budget: int
    ) -> SolverResult:
        start = perf_counter()
        guidance = guidance or Guidance(strategy_name="unguided")
        if guidance.warm_start is not None:
            x0 = np.asarray(guidance.warm_start, dtype=float)
        else:
            x0 = np.zeros(self.dimension, dtype=float)

        lower, upper = self.bounds_scalar
        bounds = (
            np.full(self.dimension, lower, dtype=float),
            np.full(self.dimension, upper, dtype=float),
        )

        def score(x: Array) -> float:
            residuals = self.violation(x, theta)
            box_penalty = 1_000.0 * float(np.dot(np.maximum(residuals, 0.0), np.maximum(residuals, 0.0)))
            return self.objective(x, theta) + box_penalty

        x_best, _ = random_pattern_search(
            score,
            x0,
            budget=int(budget),
            rng=self.rng,
            step_scale=0.75,
            bounds=bounds,
        )
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
                np.asarray(theta["shift"], dtype=float),
                np.asarray([float(theta["amplitude"]), float(self.dimension)], dtype=float),
            ]
        )

    def _best_grid_point(self, theta: Context) -> Array:
        lower, upper = self.bounds_scalar
        axes = [np.linspace(lower, upper, self.grid_points) for _ in range(self.dimension)]
        best_x: Array | None = None
        best_value = float("inf")
        for point in product(*axes):
            x = np.asarray(point, dtype=float)
            value = self.objective(x, theta)
            if value < best_value:
                best_x = x
                best_value = value
        if best_x is None:
            raise RuntimeError("empty Rastrigin grid")
        return best_x


def make_default(**kwargs: Any) -> RastriginWarmStartBenchmark:
    return RastriginWarmStartBenchmark(**kwargs)
