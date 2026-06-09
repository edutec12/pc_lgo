"""Physics-inspired soft penalties for benchmark optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .core import Array, positive_part


@dataclass
class BasePhysicsPenalty(ABC):
    """Base class for differentiable or derivative-free soft penalties."""

    weight: float = 1.0

    @abstractmethod
    def penalty(self, x: Array, theta: Mapping[str, Any]) -> float:
        """Return a nonnegative penalty value."""

    def __call__(self, x: Array, theta: Mapping[str, Any]) -> float:
        return self.penalty(x, theta)


def conic_constraint_values(
    x: Array,
    theta: Mapping[str, Any],
    selected_constraints: Sequence[int] | np.ndarray | None = None,
) -> Array:
    """Evaluate synthetic SOC residuals.

    The benchmark uses constraints of the form
    ``||M_i x + q_i||_2 <= r_i``. Positive residuals are infeasible.
    """

    matrices = np.asarray(theta["soc_matrices"], dtype=float)
    offsets = np.asarray(theta["soc_offsets"], dtype=float)
    radii = np.asarray(theta["soc_radii"], dtype=float)
    if selected_constraints is not None:
        indices = np.asarray(selected_constraints, dtype=int)
        matrices = matrices[indices]
        offsets = offsets[indices]
        radii = radii[indices]
    transformed = np.einsum("mcd,d->mc", matrices, np.asarray(x, dtype=float)) + offsets
    return np.linalg.norm(transformed, axis=1) - radii


def conic_soft_penalty(
    x: Array,
    theta: Mapping[str, Any],
    *,
    selected_constraints: Sequence[int] | np.ndarray | None = None,
    weight: float = 1_000.0,
) -> float:
    """Quadratic soft penalty for SOC residuals."""

    residuals = conic_constraint_values(x, theta, selected_constraints)
    positives = positive_part(residuals)
    return float(weight * np.dot(positives, positives))


def reserve_soft_penalty(
    plan: Array,
    theta: Mapping[str, Any],
    *,
    weight: float = 1_000.0,
) -> float:
    """Quadratic reserve-adequacy penalty for binary expansion plans."""

    capacities = np.asarray(theta["capacities"], dtype=float)
    requirement = float(theta["reserve_requirement"])
    shortfall = max(0.0, requirement - float(np.dot(capacities, plan)))
    return float(weight * shortfall * shortfall)


def ev_schedule_matrix(x: Array, theta: Mapping[str, Any]) -> Array:
    n_evs = int(theta["n_evs"])
    horizon = int(theta["horizon"])
    return np.asarray(x, dtype=float).reshape(n_evs, horizon)


def ev_soc_at_departure(x: Array, theta: Mapping[str, Any]) -> Array:
    """Compute per-EV state of charge at departure from charging rates."""

    schedule = ev_schedule_matrix(x, theta)
    initial_soc = np.asarray(theta["initial_soc"], dtype=float)
    capacity = np.asarray(theta["battery_capacity"], dtype=float)
    departure = np.asarray(theta["departure_time"], dtype=int)
    dt = float(theta.get("dt", 1.0))
    delivered = np.zeros(schedule.shape[0], dtype=float)
    for ev_idx, dep in enumerate(departure):
        delivered[ev_idx] = float(np.sum(schedule[ev_idx, :dep]) * dt)
    return initial_soc + delivered / capacity


def ev_shortfall_and_grid_residuals(x: Array, theta: Mapping[str, Any]) -> tuple[Array, Array]:
    """Return EV departure shortfalls and aggregate grid residuals."""

    schedule = ev_schedule_matrix(x, theta)
    target_soc = np.asarray(theta["target_soc"], dtype=float)
    shortfall = target_soc - ev_soc_at_departure(x, theta)
    grid_limit = np.asarray(theta["grid_limit"], dtype=float)
    aggregate = np.sum(schedule, axis=0)
    return shortfall, aggregate - grid_limit


def ev_soft_penalty(
    x: Array,
    theta: Mapping[str, Any],
    *,
    shortfall_weight: float = 1_000.0,
    grid_weight: float = 1_000.0,
) -> float:
    """Penalty for EV shortfall and grid-limit violations."""

    shortfall, grid_residual = ev_shortfall_and_grid_residuals(x, theta)
    sf = positive_part(shortfall)
    gv = positive_part(grid_residual)
    return float(shortfall_weight * np.dot(sf, sf) + grid_weight * np.dot(gv, gv))


@dataclass
class ConicPhysicsPenalty(BasePhysicsPenalty):
    selected_constraints: Sequence[int] | np.ndarray | None = None

    def penalty(self, x: Array, theta: Mapping[str, Any]) -> float:
        return conic_soft_penalty(
            x,
            theta,
            selected_constraints=self.selected_constraints,
            weight=self.weight,
        )


@dataclass
class ReservePhysicsPenalty(BasePhysicsPenalty):
    def penalty(self, x: Array, theta: Mapping[str, Any]) -> float:
        return reserve_soft_penalty(x, theta, weight=self.weight)


@dataclass
class EVPhysicsPenalty(BasePhysicsPenalty):
    grid_weight: float = 1_000.0

    def penalty(self, x: Array, theta: Mapping[str, Any]) -> float:
        return ev_soft_penalty(
            x,
            theta,
            shortfall_weight=self.weight,
            grid_weight=self.grid_weight,
        )
