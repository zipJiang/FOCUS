"""Where to spend the rollout budget: search phases ranked by a priority function.

A :class:`Search` runs its :class:`Phase` s in order over one tree. Each phase spends a
budget of task transitions; whenever a worker frees up it continues the eligible
checkpoint with the highest :class:`Priority`. Branching limits per checkpoint are part
of the phase.

:class:`Selective` is the paper's recipe: a first-in-first-out warm-up that collects
initial evidence, then selective allocation by expected critic-error reduction
(:class:`ErrorReduction`). Write a new allocation rule as a :class:`Priority`: a
function from one checkpoint's :class:`NodeInfo` to a number, or ``None`` when the
checkpoint should not be continued.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from focus.allocation.calibration import BetaBernoulli, CalibrationModel
from focus.allocation.refinement import retraining_gain
from focus.harness.turn import TurnTag
from focus.scheduler.core.allocation import Observations, RankedSession
from focus.scheduler.core.entropy import _edge_entropy
from focus.scheduler.core.gating import Gate, GatingPolicy
from focus.scheduler.core.tree import Node, SchedulerView
from focus.statistics import RunningStats

if TYPE_CHECKING:
    from focus.allocation.evidence import RolloutLedger
    from focus.generation import SamplingParams
    from focus.harness import Agent, AnyRollout
    from focus.scheduler.core.critic import NodeScorer
    from focus.scheduler.core.policies import Budget
    from focus.scheduler.core.tree import SchedulerState
    from focus.values import RewardConfig

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class Phase:
    """One stage of a search: a priority, a budget, and per-checkpoint branching caps.

    ``budget`` counts new task transitions; a failed continuation costs one. In-flight
    continuations finish after the budget is spent, so a phase can overshoot slightly
    and the next phase still receives its full budget. ``max_children`` caps a
    checkpoint's total branches (children plus running); ``max_pending`` caps only its
    running continuations. ``sampling`` overrides the agent's sampling parameters.
    """

    priority: Priority
    budget: int
    max_children: int | None = None
    max_pending: int | None = None
    sampling: SamplingParams | None = None

    def __post_init__(self) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, int):
            raise TypeError("Phase.budget must be an integer")
        if self.budget < 0:
            raise ValueError("Phase.budget must be nonnegative")
        for cap in (self.max_children, self.max_pending):
            if cap is not None and cap < 1:
                raise ValueError("branching caps must be positive")


class Search:
    """Phases run in order over one tree, with at most ``max_concurrency`` workers."""

    def __init__(self, phases: Sequence[Phase], *, max_concurrency: int = 4) -> None:
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self.phases = tuple(phases)
        self.max_concurrency = max_concurrency

    @property
    def requires_critic(self) -> bool:
        return any(phase.priority.requires_critic for phase in self.phases)

    async def run(
        self,
        tree: SchedulerState[AnyRollout],
        agent: Agent[Any, Any],
        *,
        ledger: RolloutLedger,
        scorer: NodeScorer[AnyRollout] | None = None,
    ) -> None:
        """Run every phase over ``tree``, continuing checkpoints with ``agent``.

        Each phase regates the tree with its caps and spends its budget on top of
        what earlier phases already spent. A phase whose priority needs critic
        predictions is skipped when every prediction failed.
        """
        from focus.scheduler.core.execution import PASS_KEY, RolloutExpander
        from focus.scheduler.core.scheduler import Scheduler

        view = SchedulerView(tree)
        for index, phase in enumerate(self.phases):
            name = phase.priority.name
            if phase.budget == 0:
                logger.info("search skipped phase=%d priority=%s budget=0", index, name)
                continue
            if phase.priority.requires_critic and not any(
                node.critic is not None for node in view.nodes()
            ):
                logger.warning(
                    "search skipped phase=%d priority=%s: every critic prediction "
                    + "failed",
                    index,
                    name,
                )
                continue
            async with tree.lock:
                tree.regate(_PhaseGating(phase))
            spent_before = dict(tree.stats)
            await Scheduler(
                expander=RolloutExpander(
                    agent, sampling_params=phase.sampling, annotation={PASS_KEY: index}
                ),
                allocation=_PrioritySession(phase.priority, ledger),
                termination=_budget(tree, phase.budget),
                scorer=scorer,
                max_concurrency=self.max_concurrency,
            ).run_on(tree)
            logger.info(
                "search finished phase=%d priority=%s budget=%d spent=%s",
                index,
                name,
                phase.budget,
                {k: v - spent_before.get(k, 0.0) for k, v in tree.stats.items()},
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(phases={self.phases!r}, "
            f"max_concurrency={self.max_concurrency})"
        )


class Selective(Search):
    """The paper's search: a FIFO warm-up, then allocation by error reduction.

    The warm-up spends ``warmup`` transitions with at most ``warmup_width`` branches per
    checkpoint. Selective allocation then spends ``budget`` transitions; total width is
    uncapped and each checkpoint runs at most ``max_pending_per_node`` continuations at
    once.
    """

    def __init__(
        self,
        budget: int,
        *,
        warmup: int,
        warmup_width: int = 2,
        max_pending_per_node: int = 2,
        max_concurrency: int = 4,
        calibration: CalibrationModel | None = None,
    ) -> None:
        super().__init__(
            (
                Phase(Fifo(), warmup, max_children=warmup_width),
                Phase(
                    ErrorReduction(calibration),
                    budget,
                    max_pending=max_pending_per_node,
                ),
            ),
            max_concurrency=max_concurrency,
        )


def _budget(state: SchedulerState[AnyRollout], phase_turns: int) -> Budget[AnyRollout]:
    """A cap for one phase: what the tree has already spent, plus this phase's share.

    "Spent" counts realized turns *plus* failures, since a dead expansion is charged a
    turn so a broken backend depletes a budget instead of being retried forever. A
    phase ends by draining its in-flight expansions, so it can land a few turns past
    its own cap; because the next cap is read off the tree's realized spend, that
    overshoot is already in the baseline and the next phase still gets its full share.
    """
    from focus.scheduler.core.policies import Budget

    spent = Budget.turns_spent(SchedulerView(state))
    return Budget(max_turns=int(spent) + phase_turns)


class _PhaseGating(GatingPolicy):
    def __init__(self, phase: Phase) -> None:
        self._max_children = phase.max_children
        self._max_pending = phase.max_pending

    def gate(self, node: Node[Any]) -> Gate:
        return Gate(max_children=self._max_children, max_pending=self._max_pending)


class _PrioritySession(RankedSession["AnyRollout"]):
    """Adapt a priority to the scheduler's allocation lifecycle."""

    def __init__(self, priority: Priority, ledger: RolloutLedger) -> None:
        super().__init__(ledger)
        self.priority = priority
        self._reward = ledger.reward_config

    def score(self, view: SchedulerView[AnyRollout], node: Node[AnyRollout]) -> float:
        # Read without inserting: an unvisited node has no outcomes yet, and the
        # ledger is stored on the tree, so scanning must not grow it.
        observed = self.ledger.observations.get(node.id)
        info = NodeInfo(
            node,
            view,
            self._reward,
            observed if observed is not None else Observations(),
            view.arrival(node),
        )
        value = self.priority(info)
        if value is None or math.isnan(value) or value == -math.inf:
            return math.nan
        return float(value)


__all__ = [
    "ErrorReduction",
    "Fifo",
    "NodeInfo",
    "Phase",
    "Priority",
    "Search",
    "Selective",
    "TokenEntropy",
]
