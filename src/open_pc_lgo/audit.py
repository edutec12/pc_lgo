"""Projection-clean audits and benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .core import (
    Array,
    BenchmarkProblem,
    SolverResult,
    max_positive_violation,
    mean_positive_violation,
)


@dataclass
class AuditResult:
    objective: float
    gap_to_oracle: float
    max_violation: float
    mean_violation: float
    runtime: float
    accepted: bool
    violations: Array
    correction_norm: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, float | bool]:
        return {
            "objective": self.objective,
            "gap_to_oracle": self.gap_to_oracle,
            "max_violation": self.max_violation,
            "mean_violation": self.mean_violation,
            "runtime": self.runtime,
            "accepted": self.accepted,
        }


def full_constraint_audit(
    problem: BenchmarkProblem,
    x: Array,
    theta: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
    oracle_objective: float | None = None,
    runtime: float = 0.0,
) -> AuditResult:
    """Audit x against every constraint in the benchmark problem."""

    start = perf_counter()
    objective = float(problem.objective(x, theta))
    violations = problem.violation(x, theta)
    max_violation = max_positive_violation(violations)
    mean_violation = mean_positive_violation(violations)
    audit_runtime = perf_counter() - start
    gap = float("nan") if oracle_objective is None else objective - float(oracle_objective)
    return AuditResult(
        objective=objective,
        gap_to_oracle=gap,
        max_violation=max_violation,
        mean_violation=mean_violation,
        runtime=float(runtime + audit_runtime),
        accepted=bool(max_violation <= tolerance),
        violations=np.asarray(violations, dtype=float),
    )


def projection_clean_acceptance(
    problem: BenchmarkProblem,
    x: Array,
    theta: Mapping[str, Any],
    *,
    raw_x: Array | None = None,
    tolerance: float = 1e-6,
    correction_tolerance: float = 1e-9,
    oracle_objective: float | None = None,
    runtime: float = 0.0,
) -> AuditResult:
    """Accept or reject without counting hidden projection as learning success.

    If ``raw_x`` is supplied, the candidate ``x`` must be within
    ``correction_tolerance`` of that raw optimizer output. Any larger correction
    is reported and rejected even when the corrected point is feasible.
    """

    audit = full_constraint_audit(
        problem,
        x,
        theta,
        tolerance=tolerance,
        oracle_objective=oracle_objective,
        runtime=runtime,
    )
    correction_norm = 0.0
    if raw_x is not None:
        correction_norm = float(np.linalg.norm(np.asarray(x, dtype=float) - np.asarray(raw_x, dtype=float)))
        if correction_norm > correction_tolerance:
            audit.accepted = False
            audit.info["rejection_reason"] = "hidden_projection_correction"
    elif not audit.accepted:
        audit.info["rejection_reason"] = "constraint_violation"
    audit.correction_norm = correction_norm
    return audit


def metrics_from_solution(
    problem: BenchmarkProblem,
    theta: Mapping[str, Any],
    result: SolverResult,
    *,
    oracle: SolverResult | None = None,
    tolerance: float = 1e-6,
) -> dict[str, float | bool | str]:
    """Return CSV-friendly metrics for a raw solver result."""

    oracle_objective = None if oracle is None else oracle.objective
    audit = projection_clean_acceptance(
        problem,
        result.x,
        theta,
        raw_x=result.x,
        tolerance=tolerance,
        oracle_objective=oracle_objective,
        runtime=result.runtime,
    )
    result.accepted = audit.accepted
    metrics: dict[str, float | bool | str] = audit.to_metrics()
    metrics["strategy_name"] = result.guidance.strategy_name
    return metrics
