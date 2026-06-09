"""Learning components that propose guidance for budgeted optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .core import Array, Guidance

try:  # PyTorch is a package dependency, but this keeps imports diagnosable.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch.
    torch = None
    nn = None


TaskName = Literal["regression", "binary"]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("BehavioralCloningPrior requires PyTorch; install OPEN-pc_lgo with torch support.")


class BehavioralCloningPrior:
    """Small PyTorch MLP trained from expert demonstrations.

    The model predicts guidance vectors. Downstream code decides whether those
    outputs are interpreted as warm starts, active-constraint scores, candidate
    masks, or mixed guidance.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        task: TaskName = "regression",
        seed: int = 0,
    ) -> None:
        _require_torch()
        if task not in {"regression", "binary"}:
            raise ValueError("task must be 'regression' or 'binary'")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.task: TaskName = task
        self.seed = int(seed)
        torch.manual_seed(self.seed)

        layers: list[nn.Module] = []
        previous = self.input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(previous, int(width)))
            layers.append(nn.ReLU())
            previous = int(width)
        layers.append(nn.Linear(previous, self.output_dim))
        self.model = nn.Sequential(*layers)

    def fit(
        self,
        features: Array,
        targets: Array,
        *,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> list[float]:
        """Train on full-batch demonstrations and return loss history."""

        _require_torch()
        x_train = torch.as_tensor(np.asarray(features, dtype=np.float32))
        y_train = torch.as_tensor(np.asarray(targets, dtype=np.float32))
        if x_train.ndim == 1:
            x_train = x_train[None, :]
        if y_train.ndim == 1:
            y_train = y_train[None, :]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn: nn.Module
        if self.task == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.MSELoss()

        history: list[float] = []
        self.model.train()
        for _ in range(max(0, int(epochs))):
            optimizer.zero_grad()
            prediction = self.model(x_train)
            loss = loss_fn(prediction, y_train)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu().item()))
        return history

    def predict(self, features: Array) -> Array:
        """Return model predictions as a NumPy array."""

        _require_torch()
        x_eval = torch.as_tensor(np.asarray(features, dtype=np.float32))
        squeeze = x_eval.ndim == 1
        if squeeze:
            x_eval = x_eval[None, :]
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_eval)
            if self.task == "binary":
                output = torch.sigmoid(output)
        prediction = output.detach().cpu().numpy().astype(float)
        return prediction[0] if squeeze else prediction

    def guidance(
        self,
        features: Array,
        *,
        warm_start_dim: int = 0,
        active_scores_dim: int = 0,
        candidate_mask_dim: int = 0,
        mask_threshold: float = 0.5,
        strategy_name: str = "behavioral_cloning",
    ) -> Guidance:
        """Interpret one prediction vector as a Guidance object."""

        output = np.asarray(self.predict(features), dtype=float).ravel()
        cursor = 0
        warm_start = None
        active_scores = None
        candidate_mask = None
        if warm_start_dim:
            warm_start = output[cursor : cursor + warm_start_dim]
            cursor += warm_start_dim
        if active_scores_dim:
            active_scores = output[cursor : cursor + active_scores_dim]
            cursor += active_scores_dim
        if candidate_mask_dim:
            mask_values = output[cursor : cursor + candidate_mask_dim]
            candidate_mask = mask_values >= mask_threshold
        return Guidance(
            warm_start=warm_start,
            active_constraint_scores=active_scores,
            candidate_mask=candidate_mask,
            strategy_name=strategy_name,
        )


@dataclass
class BanditDecision:
    strategy_name: str
    action_index: int
    exploration: bool
    scores: Array


class ContextualBanditSelector:
    """Epsilon-greedy linear contextual bandit over guidance strategies."""

    def __init__(
        self,
        strategies: Sequence[str],
        *,
        feature_dim: int,
        epsilon: float = 0.1,
        learning_rate: float = 0.05,
        seed: int = 0,
    ) -> None:
        if not strategies:
            raise ValueError("at least one strategy is required")
        self.strategies = list(strategies)
        self.feature_dim = int(feature_dim)
        self.epsilon = float(epsilon)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((len(self.strategies), self.feature_dim), dtype=float)
        self.counts = np.zeros(len(self.strategies), dtype=int)
        self.reward_baseline = 0.0
        self.updates = 0

    def select_with_info(self, features: Array) -> BanditDecision:
        context = self._context(features)
        scores = self.weights @ context
        exploration = bool(self.rng.random() < self.epsilon)
        if exploration:
            action_index = int(self.rng.integers(0, len(self.strategies)))
        else:
            action_index = int(np.argmax(scores))
        return BanditDecision(
            strategy_name=self.strategies[action_index],
            action_index=action_index,
            exploration=exploration,
            scores=scores.copy(),
        )

    def select(self, features: Array) -> str:
        return self.select_with_info(features).strategy_name

    def update(self, features: Array, strategy_name: str, reward: float) -> None:
        """Apply an epsilon-greedy linear value update for one observed reward."""

        if strategy_name not in self.strategies:
            raise KeyError(f"unknown strategy {strategy_name!r}")
        action_index = self.strategies.index(strategy_name)
        context = self._context(features)
        prediction = float(self.weights[action_index] @ context)
        error = float(reward) - prediction
        self.weights[action_index] += self.learning_rate * error * context
        self.counts[action_index] += 1
        self.updates += 1
        rate = 1.0 / float(self.updates)
        self.reward_baseline = (1.0 - rate) * self.reward_baseline + rate * float(reward)

    def update_from_metrics(self, features: Array, strategy_name: str, metrics: dict[str, float | bool]) -> None:
        """Use audit metrics to define a reward and update the selector."""

        accepted_bonus = 1.0 if bool(metrics.get("accepted", False)) else -1.0
        gap = float(metrics.get("gap_to_oracle", 0.0))
        max_violation = float(metrics.get("max_violation", 0.0))
        reward = accepted_bonus - gap - 10.0 * max_violation
        self.update(features, strategy_name, reward)

    def _context(self, features: Array) -> Array:
        context = np.asarray(features, dtype=float).ravel()
        if context.size != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {context.size}")
        norm = float(np.linalg.norm(context))
        return context if norm == 0.0 else context / norm
