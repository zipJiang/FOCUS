"""Rank eligible checkpoints using critic predictions and collected evidence."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from focus.allocation.calibration import BetaBernoulli, CalibrationModel
from focus.allocation.refinement import retraining_gain
from focus.harness.turn import TurnTag
from focus.scheduler.core.allocation import Observations
from focus.scheduler.core.entropy import _edge_entropy
from focus.scheduler.core.tree import Node, SchedulerView
from focus.statistics import RunningStats

if TYPE_CHECKING:
    from focus.harness.rollout import AnyRollout
    from focus.values import RewardConfig


@dataclass(frozen=True)
class NodeInfo:
    """What a priority may read about one candidate checkpoint."""

    node: Node[AnyRollout]
    tree: SchedulerView[AnyRollout]
    reward: RewardConfig
    observations: Observations
    #: Position in the order checkpoints became eligible during this phase; lower is
    #: earlier. A checkpoint that leaves and re-enters the frontier arrives again.
    arrival: int

    @property
    def id(self) -> int:
        return self.node.id

    @property
    def depth(self) -> int:
        return self.node.depth

    @property
    def prior(self) -> float | None:
        """The critic prediction :math:`V_\\phi`, if this checkpoint was scored."""
        return self.node.critic

    @property
    def outcomes(self) -> RunningStats:
        """Completed continuation outcomes observed through this checkpoint."""
        return self.observations.returns

    @property
    def pending(self) -> int:
        """Continuations launched from this checkpoint and still running."""
        return self.observations.m_pending

    @property
    def children(self) -> list[Node[AnyRollout]]:
        return self.tree.children(self.node)


class Priority(ABC):
    """Rank candidate checkpoints; the search continues the highest one.

    Return ``None`` (or a non-finite or NaN value other than ``+inf``) to leave a
    checkpoint unselected. Ties go to the lower node id.
    """

    #: Shown in search logs. Defaults to the class name.
    name: str = ""
    #: Whether the priority reads critic predictions; a search using it needs a critic.
    requires_critic: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("name"):
            cls.name = cls.__name__

    @abstractmethod
    def __call__(self, node: NodeInfo) -> float | None: ...


class Fifo(Priority):
    """First in, first out: continue checkpoints in the order they became eligible."""

    name = "fifo"

    def __call__(self, node: NodeInfo) -> float:
        return -float(node.arrival)


class ErrorReduction(Priority):
    """Expected reduction of critic error from one more continuation (Eq. gain).

    :math:`g = (2\\kappa^2 \\widehat{e^2} + (N - \\kappa)\\widehat{\\sigma^2}) /
    (\\kappa + N)^3`, with pending continuations counted in :math:`N`. Checkpoints
    with non-positive gain are not continued. ``calibration`` supplies the posterior
    error and predictive variance; the default Beta-Bernoulli model expects outcomes in
    :math:`[0, 1]`.
    """

    name = "error_reduction"
    requires_critic = True

    def __init__(self, calibration: CalibrationModel | None = None) -> None:
        self.calibration = calibration if calibration is not None else BetaBernoulli()

    def __call__(self, node: NodeInfo) -> float | None:
        kappa = node.reward.value_prior_strength
        prior = 0.0 if node.prior is None else node.prior
        e2, variance = self.calibration.posterior(prior, node.outcomes, kappa)
        gain = retraining_gain(e2, variance, node.observations.n_eff, kappa)
        return gain if math.isfinite(gain) and gain > 0.0 else None


class TokenEntropy(Priority):
    """Continue where the policy's own children were generated least certainly.

    Per child edge: the highest sliding-window mean of ``p * -log p`` over generated
    tokens. Per checkpoint: the mean over children. A checkpoint with no scored
    children ranks first (``+inf``). Needs per-token logprobs from the backend.
    """

    name = "token_entropy"

    def __init__(
        self, window: int = 8, tags: Sequence[TurnTag] | None = ("task",)
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.tags = None if tags is None else frozenset(tags)

    def __call__(self, node: NodeInfo) -> float:
        scores = []
        for child in node.children:
            edge = child.payload.generated_since(node.node.payload, tags=self.tags)
            entropy = _edge_entropy(edge, self.window)
            if entropy is not None:
                scores.append(entropy)
        return sum(scores) / len(scores) if scores else math.inf


__all__ = ["ErrorReduction", "Fifo", "NodeInfo", "Priority", "TokenEntropy"]
