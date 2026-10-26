"""Prepare one training recipe from a settled tree without a live runtime."""

from __future__ import annotations

import logging
from types import MappingProxyType

from focus.allocation.evidence import evidence_of
from focus.harness import AnyRollout
from focus.harness.turn import logprob
from focus.preparation.base import AdvantageEstimator
from focus.preparation.direct_branch import critic_targets
from focus.preparation.records import (
    ActorSample,
    CriticSample,
    PreparedBatch,
    _refuse_non_finite,
    _trainable_edges,
    edge_spans,
)
from focus.reward.config import RewardConfig
from focus.scheduler.core.execution import PASS_KEY
from focus.scheduler.core.returns import finished_through
from focus.scheduler.core.tree import SchedulerState, SchedulerView

logger = logging.getLogger(__name__)


def prepare_samples(
    state: SchedulerState[AnyRollout],
    *,
    estimator: AdvantageEstimator | None,
    reward_config: RewardConfig,
) -> PreparedBatch:
    """Compute one actor recipe and the critic's branch-mean targets.

    Critic targets do not depend on the actor recipe: every expanded checkpoint the
    critic scored regresses on the mean of its direct-branch backups. ``estimator=None``
    prepares only those targets.
    """
    view = SchedulerView(state)
    evidence_of(view, reward_config)  # refuses trees searched under other rewards
    actor: list[ActorSample] = []
    critic: list[CriticSample] = []
    for node, parent in _trainable_edges(view) if estimator is not None else ():
        if (
            estimator is not None
            and estimator.requires_completed_paths
            and not finished_through(view, node)
        ):
            continue
        spans = edge_spans(node, parent)
        if not spans:
            continue
        if any(not span.exact for span in spans):
            raise ValueError(
                f"edge into node {node.id} is not trainable: inexact generated tokens"
            )
        logprob(node.payload.edge(parent.payload))  # behavior logprobs are required
        if estimator is not None:
            actor.append(
                ActorSample(
                    node_id=node.id,
                    spans=spans,
                    group_id=parent.id,
                    estimator=estimator.name,
                    search_pass=node.metadata.get(PASS_KEY),
                )
            )
    critic.extend(critic_targets(view, reward_config))
    if estimator is not None:
        estimator.assign(tuple(actor), view=view, reward_config=reward_config)
    for record in actor:
        record.diagnostics = MappingProxyType(dict(record.diagnostics))
        _refuse_non_finite(record)
    for target in critic:
        _refuse_non_finite(target)
    result = PreparedBatch(
        actor=tuple(actor),
        critic=tuple(critic),
        reward_config=reward_config,
        stats=MappingProxyType(dict(view.stats)),
        failures=view.failures,
    )
    logger.log(
        logging.INFO if actor or critic else logging.WARNING,
        "samples prepared actor=%d critic=%d estimator=%s stats=%s failures=%d",
        len(actor),
        len(critic),
        estimator.name if estimator is not None else None,
        dict(result.stats),
        len(result.failures),
    )
    return result
