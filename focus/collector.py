"""Collect a rollout tree and prepare its training batch in one object.

A :class:`Collector` binds an agent, an optional critic, a search, and an actor recipe.
:meth:`Collector.collect` searches from a prompt and returns the prepared
:class:`~focus.preparation.records.Batch`; :meth:`Collector.search` and
:meth:`Collector.prepare` expose the two stages for inspecting a tree or comparing
recipes on the same evidence. Used as an async context manager, a collector closes its
agent's execution services on exit; policies and critics belong to the caller, who
often shares them across collectors.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from types import EllipsisType
from typing import Any

from focus.allocation.evidence import RolloutLedger
from focus.critic import Critic
from focus.estimation import DirectBranchTd, Estimator
from focus.execution import Usage
from focus.harness import Agent, AnyRollout, RolloutState
from focus.harness.workspace import Workspace
from focus.preparation import Batch, prepare_samples
from focus.scheduler.core.critic import NodeScorer
from focus.scheduler.core.gating import ConcurrencyGating
from focus.scheduler.core.tree import (
    STAT_SCORE_FAILURES,
    SchedulerState,
    SchedulerView,
)
from focus.search import Search
from focus.values import RewardConfig

logger = logging.getLogger(__name__)

#: A settled search tree: every checkpoint, its critic prediction, and the edges.
Tree = SchedulerState


class Collector:
    """Search from prompts with one agent, critic, and search; prepare training data.

    ``estimator`` defaults to the paper's :class:`~focus.preparation.DirectBranchTd`;
    pass ``None`` to prepare only critic targets. ``reward`` weights outcome and
    step rewards for both search statistics and preparation.

    ``own=True`` (the default) makes ``async with collector`` close the agent's
    execution services on exit. Pass ``own=False`` when an enclosing context shares
    those services across collectors. Close policy clients and critics yourself.
    """

    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        search: Search,
        critic: Critic | None = None,
        estimator: Estimator | None = DirectBranchTd(),  # noqa: B008 (stateless)
        reward: RewardConfig | None = None,
        own: bool = True,
    ) -> None:
        if search.requires_critic and critic is None:
            names = [
                p.priority.name for p in search.phases if p.priority.requires_critic
            ]
            raise ValueError(
                f"search phases {names} rank checkpoints by critic predictions; "
                + "pass critic=..."
            )
        self.agent = agent
        self.search_config = search
        self.critic = critic
        self.estimator = estimator
        self.reward = reward if reward is not None else RewardConfig()
        self.own = own

    async def __aenter__(self) -> Collector:
        await self.agent.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.own:
            await self.aclose()

    async def aclose(self) -> None:
        """Close the agent's execution services."""
        await self.agent.aclose()

    async def collect(
        self, prompt: str | AnyRollout, *, workspace: Workspace | None = None
    ) -> Batch:
        """Search from ``prompt`` and prepare the training batch.

        Inspect ``batch.stats`` and ``batch.failures`` before training: successful
        samples from a partial collection are retained.
        """
        return self.prepare(await self.search(prompt, workspace=workspace))

    def prepare(
        self,
        tree: Tree[AnyRollout],
        *,
        estimator: Estimator | EllipsisType | None = ...,
    ) -> Batch:
        """Prepare actor and critic samples from a settled tree.

        ``estimator`` overrides the collector's recipe for this call; ``None``
        prepares critic targets only.
        """
        chosen = self.estimator if estimator is ... else estimator
        return prepare_samples(tree, estimator=chosen, reward_config=self.reward)

    async def search(
        self, prompt: str | AnyRollout, *, workspace: Workspace | None = None
    ) -> Tree[AnyRollout]:
        """Start an episode from ``prompt`` (or continue a checkpoint) and search."""
        if not isinstance(prompt, (str, RolloutState)):
            raise TypeError(
                "prompt must be a str or a RolloutState, not " + type(prompt).__name__
            )
        agent = self.agent
        root = (
            prompt
            if isinstance(prompt, RolloutState)
            else await agent.start(prompt, workspace=workspace)
        )
        before = asdict(agent.services.usage.get(root.episode_id, Usage()))
        state: Tree[AnyRollout] = SchedulerState.root(root, gating=ConcurrencyGating(1))
        scorer: NodeScorer[AnyRollout] | None = (
            None
            if self.critic is None
            else NodeScorer(self.critic, services=agent.services)
        )
        await _score_root(scorer, state)
        ledger = RolloutLedger(self.reward)
        ledger.bind(SchedulerView(state))
        await self.search_config.run(state, agent, ledger=ledger, scorer=scorer)
        current = asdict(agent.services.usage.get(root.episode_id, Usage()))
        for key, value in current.items():
            state.stats["resource_" + key] = float(value - before[key])
        return state


async def _score_root(
    scorer: NodeScorer[AnyRollout] | None, state: SchedulerState[AnyRollout]
) -> None:
    """Score the root before sampling its first continuation.

    A scoring failure keeps the search going and is counted here; the actor recipe
    later refuses a tree whose required prediction is missing.
    """
    if scorer is None:
        return
    node = state.nodes[state.root_id]
    try:
        scores = await scorer.score([node.payload])
    except Exception as exc:
        state.stats[STAT_SCORE_FAILURES] += 1.0
        node.metadata["score_error"] = repr(exc)
        logger.warning("root scoring failed node=%d error=%r", node.id, exc)
        return
    scorer.write([node], scores)


__all__ = ["Collector", "Tree"]
