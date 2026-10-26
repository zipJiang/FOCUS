"""Run explicitly configured allocation passes and return the settled search tree."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from focus.allocation import RolloutLedger
from focus.config import RolloutConfig, SearchPass
from focus.critic import Critic
from focus.execution import Usage
from focus.harness import AnyRollout, RolloutState, Runner
from focus.harness.workspace import Workspace
from focus.preparation import Estimator, PreparedBatch, prepare_samples
from focus.scheduler.core.critic import NodeScorer
from focus.scheduler.core.execution import PASS_KEY, RolloutExpander
from focus.scheduler.core.gating import ConcurrencyGating
from focus.scheduler.core.policies import Budget
from focus.scheduler.core.scheduler import Scheduler
from focus.scheduler.core.tree import (
    STAT_FAILURES,
    STAT_SCORE_FAILURES,
    STAT_TURNS,
    SchedulerState,
    SchedulerView,
)

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Connect a runner, an optional critic, and a search configuration.

    Construction preserves each collaborator's configuration and service ownership.
    Each search creates its own tree and workspace.
    """

    runner: Runner[Any, Any]
    #: Predicts remaining return at checkpoints.
    critic: Critic | None = None
    config: RolloutConfig = field(default_factory=RolloutConfig)


def _expander(
    runtime: Runtime, pass_index: int, search_pass: SearchPass
) -> RolloutExpander[Any]:
    """The pass's expander: the shared runner, this pass's sampling and index."""
    return RolloutExpander(
        runtime.runner,
        sampling_params=search_pass.sampling_params,
        annotation={PASS_KEY: pass_index},
    )


def _value_head(runtime: Runtime) -> NodeScorer[AnyRollout] | None:
    """Build this runtime's value head, sharing the runner's execution services."""
    if runtime.critic is None:
        return None
    return NodeScorer(runtime.critic, services=runtime.runner.services)


def _budget(state: SchedulerState[AnyRollout], pass_turns: int) -> Budget[AnyRollout]:
    """A cap for one pass: what the tree has already spent, plus this pass's share.

    "Spent" is read the way :class:`Budget` reads it -- realized turns *plus* failures,
    since a dead expansion is charged a turn so a broken backend depletes a budget
    instead of being retried forever. Counting only the turns here made every earlier
    pass's failures come out of the next pass's share: a pass that lost 20
    expansions to a flaky server handed the next pass an already-exceeded cap.
    Later passes silently bought nothing at exactly the moment the tree most needed
    more evidence.

    The same arithmetic is what makes the drain-at-budget overshoot harmless. A pass
    ends by draining its in-flight expansions, so it can land a few turns past its own
    cap; because the next cap is read off the tree's *realized* spend rather than off a
    running plan, that overshoot is already in the baseline and the next pass still gets
    ``pass_turns`` of its own. Each pass is promised its share beyond whatever has been
    spent -- overshoot included -- not a slice of a fixed total.
    """
    spent = state.stats.get(STAT_TURNS, 0.0) + state.stats.get(STAT_FAILURES, 0.0)
    return Budget(max_turns=int(spent) + pass_turns)


async def _score_root(
    critic: NodeScorer[AnyRollout] | None, state: SchedulerState[AnyRollout]
) -> None:
    """Score the root prior before sampling its first continuation.

    Scoring failures preserve search data and are counted here. Refined TD later
    refuses actor preparation if a required nonterminal prediction is missing.
    """
    if critic is None:
        return
    node = state.nodes[state.root_id]
    try:
        scores = await critic.score([node.payload])
    except Exception as exc:
        state.stats[STAT_SCORE_FAILURES] += 1.0
        node.metadata["score_error"] = repr(exc)
        logger.warning("root scoring failed node=%d error=%r", node.id, exc)
        return
    critic.write([node], scores)


def _spent(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    """What one pass added to each running stat -- its cost, not the tree's total.

    The stats are cumulative over the whole tree, so the raw dict after a pass
    says nothing about that pass. The difference does, and it is the number a budget is
    read against.
    """
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


async def run_search(
    prompt: str | AnyRollout,
    runtime: Runtime,
    *,
    workspace: Workspace | None = None,
) -> SchedulerState[AnyRollout]:
    """Initialize a root and execute the authored passes."""
    if not isinstance(prompt, (str, RolloutState)):
        raise TypeError(
            "prompt must be a str or a RolloutState, not " + type(prompt).__name__
        )
    root = (
        prompt
        if isinstance(prompt, RolloutState)
        else await runtime.runner.start(prompt, workspace=workspace)
    )
    before_usage = asdict(runtime.runner.services.usage.get(root.episode_id, Usage()))
    state: SchedulerState[AnyRollout] = SchedulerState.root(
        root, gating=ConcurrencyGating(1)
    )
    critic = _value_head(runtime)
    await _score_root(critic, state)
    ledger = RolloutLedger(runtime.config.reward_config)
    view = SchedulerView(state)
    ledger.bind(view)
    for index, search_pass in enumerate(runtime.config.passes):
        pass_id = f"pass-{index}"
        allocator = search_pass.allocator
        if search_pass.budget == 0:
            logger.info(
                "search skipped pass=%s allocator=%s budget=0", pass_id, allocator.name
            )
            continue
        async with state.lock:
            state.regate(search_pass.gating)
        allocation = allocator.build(view, ledger)
        if allocation is None:
            logger.info("search skipped pass=%s allocator=%s", pass_id, allocator.name)
            continue
        before = dict(state.stats)
        await Scheduler(
            expander=_expander(runtime, index, search_pass),
            allocation=allocation,
            termination=_budget(state, search_pass.budget),
            scorer=critic,
            max_concurrency=runtime.config.max_concurrency,
        ).run_on(state)
        logger.info(
            "search finished pass=%s allocator=%s turn_budget=%d spent=%s",
            pass_id,
            allocator.name,
            search_pass.budget,
            _spent(before, state.stats),
        )
    _resource_stats(state, runtime, root.episode_id, before_usage)
    return state


async def collect_samples(
    prompt: str | AnyRollout,
    runtime: Runtime,
    *,
    estimator: Estimator | None,
    workspace: Workspace | None = None,
) -> PreparedBatch:
    """Collect a tree and prepare actor and critic samples for a training batch.

    Use the runtime's reward configuration for both collection
    and preparation. ``estimator`` remains explicit: choose a training recipe, or
    pass ``None`` to prepare only critic targets. Inspect ``batch.stats`` and
    ``batch.failures`` before training; successful samples from a partial collection
    are retained. Export with ``to_samples`` or consume the typed records directly.

    The caller owns execution services and model clients. Keep those services open
    around a batch or several concurrent collections, then close them when finished.
    This function returns the prepared batch without retaining the rollout tree.
    Use ``run_search`` followed by ``prepare_samples`` when you need the tree for
    inspection or want to compare several estimators on the same evidence.
    """
    tree = await run_search(prompt, runtime, workspace=workspace)
    return prepare_samples(
        tree,
        estimator=estimator,
        reward_config=runtime.config.reward_config,
    )


def _resource_stats(
    state: SchedulerState[AnyRollout],
    runtime: Runtime,
    episode: str,
    before: dict[str, Any],
) -> None:
    current = asdict(runtime.runner.services.usage.get(episode, Usage()))
    for name, value in current.items():
        state.stats["resource_" + name] = float(value - before[name])


__all__ = ["Runtime", "collect_samples", "run_search"]
