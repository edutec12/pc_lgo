"""Small deterministic optimizers used by benchmark baselines.

These routines are deliberately lightweight and solver-free. They are not
intended to replace domain optimizers; they provide reproducible budgeted
search behavior for benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .core import Array


def clip_to_bounds(x: Array, bounds: tuple[Array, Array] | None) -> Array:
    if bounds is None:
        return np.asarray(x, dtype=float)
    lower, upper = bounds
    return np.minimum(np.maximum(np.asarray(x, dtype=float), lower), upper)


def random_pattern_search(
    score_fn: Callable[[Array], float],
    x0: Array,
    *,
    budget: int,
    rng: np.random.Generator,
    step_scale: float = 1.0,
    bounds: tuple[Array, Array] | None = None,
) -> tuple[Array, float]:
    """Simple random pattern search with an explicit evaluation budget."""

    x_best = clip_to_bounds(np.asarray(x0, dtype=float).copy(), bounds)
    best_score = float(score_fn(x_best))
    if budget <= 0:
        return x_best, best_score

    step = float(step_scale)
    dim = x_best.size
    for iteration in range(int(budget)):
        if iteration % max(1, dim) == 0:
            direction = np.zeros(dim)
            direction[iteration % dim] = rng.choice([-1.0, 1.0])
        else:
            direction = rng.normal(size=dim)
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                continue
            direction = direction / norm

        candidate = clip_to_bounds(x_best + step * direction, bounds)
        candidate_score = float(score_fn(candidate))
        if candidate_score < best_score:
            x_best = candidate
            best_score = candidate_score
            step *= 1.04
        else:
            step *= 0.96
        step = max(step, 1e-6)
    return x_best, best_score


def binary_local_search(
    score_fn: Callable[[Array], float],
    x0: Array,
    *,
    budget: int,
    rng: np.random.Generator,
    candidate_mask: np.ndarray | None = None,
) -> tuple[Array, float]:
    """Bit-flip local search for small binary planning benchmarks."""

    x_best = np.rint(np.asarray(x0, dtype=float)).clip(0.0, 1.0)
    best_score = float(score_fn(x_best))
    n = x_best.size
    allowed = np.ones(n, dtype=bool) if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
    allowed_indices = np.flatnonzero(allowed)
    if allowed_indices.size == 0:
        allowed_indices = np.arange(n)

    for _ in range(max(0, int(budget))):
        idx = int(rng.choice(allowed_indices))
        candidate = x_best.copy()
        candidate[idx] = 1.0 - candidate[idx]
        candidate_score = float(score_fn(candidate))
        if candidate_score < best_score:
            x_best = candidate
            best_score = candidate_score
    return x_best, best_score
