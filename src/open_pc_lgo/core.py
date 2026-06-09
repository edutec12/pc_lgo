"""Core abstractions for Projection-Clean Learning-Guided Optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Context = Mapping[str, Any]


@dataclass(frozen=True)
class Guidance:
    """Learning-produced hints consumed by a budgeted optimizer.

    Guidance is intentionally not a solution certificate. A benchmark solver may
    use any subset of these hints, but final feasibility must be determined by a
    full audit.
    """

    warm_start: Array | None = None
    candidate_mask: BoolArray | None = None
    active_constraint_scores: Array | None = None
    strategy_name: str = "unguided"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Raw solver output before projection-clean audit."""

    x: Array
    objective: float
    violations: Array
    runtime: float
    guidance: Guidance = field(default_factory=Guidance)
    accepted: bool | None = None
    info: dict[str, Any] = field(default_factory=dict)


class BenchmarkProblem(ABC):
    """Interface implemented by all OPEN-pc_lgo benchmark problems."""

    name: str = "benchmark"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample_context(self) -> Context:
        """Sample a reproducible problem context theta."""

    @abstractmethod
    def objective(self, x: Array, theta: Context) -> float:
        """Return the objective value for decision x in context theta."""

    @abstractmethod
    def violation(self, x: Array, theta: Context) -> Array:
        """Return constraint residuals where positive entries are violations."""

    @abstractmethod
    def expert_solve(self, theta: Context) -> SolverResult:
        """Return an oracle or high-budget reference solve for theta."""

    @abstractmethod
    def budgeted_solve(
        self, theta: Context, guidance: Guidance | None, budget: int
    ) -> SolverResult:
        """Return a raw budgeted solution using optional learning guidance."""

    @abstractmethod
    def feature_vector(self, theta: Context) -> Array:
        """Return a numeric context representation for learning models."""


def as_float_array(values: Any) -> Array:
    """Convert input to a one-dimensional float array when possible."""

    return np.asarray(values, dtype=float)


def positive_part(values: Array) -> Array:
    """Return elementwise positive residuals."""

    return np.maximum(as_float_array(values), 0.0)


def max_positive_violation(values: Array) -> float:
    """Maximum positive violation with empty-vector safety."""

    positives = positive_part(values)
    return float(np.max(positives)) if positives.size else 0.0


def mean_positive_violation(values: Array) -> float:
    """Mean positive violation with empty-vector safety."""

    positives = positive_part(values)
    return float(np.mean(positives)) if positives.size else 0.0


def default_guidance() -> Guidance:
    """Return an explicit unguided strategy marker."""

    return Guidance(strategy_name="unguided")
