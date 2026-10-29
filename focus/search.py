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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from focus.priority import ErrorReduction, Fifo, NodeInfo, Priority
from focus.scheduler.core.allocation import Observations, RankedSession
from focus.scheduler.core.gating import Gate, GatingPolicy
from focus.scheduler.core.tree import Node, SchedulerView

if TYPE_CHECKING:
    from focus.allocation.calibration import CalibrationModel
    from focus.allocation.evidence import RolloutLedger
    from focus.generation import SamplingParams
    from focus.harness import Agent, AnyRollout
    from focus.scheduler.core.critic import NodeScorer
    from focus.scheduler.core.policies import Budget
    from focus.scheduler.core.tree import SchedulerState


logger = logging.getLogger(__name__)


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


__all__ = ["Phase", "Search", "Selective"]
