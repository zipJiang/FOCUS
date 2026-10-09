"""Prepare one training recipe from a settled tree without a live runtime."""

from __future__ import annotations

import logging
from types import MappingProxyType

from focus.allocation.evidence import evidence_of
from focus.harness import AnyRollout
from focus.harness.turn import logprob
from focus.preparation.base import AdvantageEstimator
from focus.preparation.records import (
    ActorSample,
    CriticSample,
    PreparedBatch,
    _refuse_non_finite,
    _trainable_edges,
    edge_spans,
)
from focus.preparation.refined_td import checkpoint_value
from focus.reward.config import RewardConfig
from focus.scheduler.core.critic import VALUE_CONTEXTS_KEY
from focus.scheduler.core.execution import provenance_of
from focus.scheduler.core.returns import finished_through, importance
from focus.scheduler.core.tree import SchedulerState, SchedulerView
from focus.vine import VINE_READY

logger = logging.getLogger(__name__)


def prepare_samples(
    state: SchedulerState[AnyRollout],
    *,
    estimator: AdvantageEstimator | None,
    reward_config: RewardConfig,
    behavior_version: str,
) -> PreparedBatch:
    """Compute one actor recipe and critic targets before flattening token spans."""
    view = SchedulerView(state)
    if VINE_READY in view.root.metadata and (
        estimator is None or estimator.name != "vine_ppo"
    ):
        raise ValueError("VinePPO collections require the vine_ppo estimator")
    evidence = evidence_of(view, reward_config)
    if evidence.behavior_version not in (None, behavior_version):
        raise ValueError(
            f"rollout evidence behavior version {evidence.behavior_version!r} "
            f"differs from behavior_version={behavior_version!r}"
        )
    stamped = {
        p.behavior_version for n in view.nodes() if (p := provenance_of(n)) is not None
    }
    if stamped and stamped != {behavior_version}:
        raise ValueError(
            f"tree behavior versions {sorted(stamped)!r} differ from "
            f"behavior_version={behavior_version!r}"
        )
    if estimator is not None and not estimator.name:
        raise ValueError(
            f"{type(estimator).__name__} has no `name`; "
            "use @register_estimator or set the name class attribute"
        )
    actor: list[ActorSample] = []
    critic: list[CriticSample] = []
    for node, parent, stamp in _trainable_edges(view) if estimator is not None else ():
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
        turns = node.payload.edge(parent.payload)
        if any(
            t.actor and t.policy_id and t.policy_id != behavior_version for t in turns
        ):
            raise ValueError("one trainable actor version is supported per preparation")
        logprob(turns, behavior_version)  # strict even on on-policy edges
        if estimator is not None:
            actor.append(
                ActorSample(
                    node_id=node.id,
                    spans=spans,
                    group_id=parent.id,
                    reward_config_id=reward_config.config_id,
                    estimator=estimator.name,
                    importance=importance(
                        node,
                        parent,
                        reward_config,
                        behavior_version,
                        reward_config.clip,
                    ),
                    provenance=stamp,
                )
            )
    critic_override = (
        estimator.critic_targets(view=view, reward_config=reward_config)
        if estimator is not None
        else None
    )
    for node in view.nodes() if critic_override is None else ():
        counts = evidence.observations.get(node.id)
        if counts is None or counts.returns.n == 0:
            continue
        context = node.metadata.get(VALUE_CONTEXTS_KEY, {}).get(
            reward_config.value_version
        )
        if context is None:
            # No value model was run here: GRPO and custom actor-only recipes
            # remain usable without inventing a critic input or network prior.
            continue
        value = checkpoint_value(node, evidence, reward_config)
        critic.append(
            CriticSample(
                node_id=node.id,
                context=context,
                target=value.value,
                reward_config_id=reward_config.config_id,
                value_version=reward_config.value_version,
                diagnostics=MappingProxyType(
                    {
                        "prior": value.prior,
                        "observations": float(value.observations),
                        "mean_return": counts.returns.mean,
                    }
                ),
            )
        )
    if critic_override is not None:
        critic.extend(critic_override)
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
        behavior_version=behavior_version,
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
