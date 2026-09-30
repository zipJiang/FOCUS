"""Local TD advantages using directly observed checkpoint-value refinements."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from focus.allocation.evidence import RolloutEvidence, evidence_of
from focus.harness import AnyRollout
from focus.preparation.base import (
    AdvantageEstimator,
    _resolved,
    register_estimator,
)
from focus.preparation.records import ActorSample
from focus.reward.config import RewardConfig
from focus.scheduler.core.returns import edge_return
from focus.scheduler.core.tree import Node, SchedulerView
from focus.statistics import RunningStats, refined_mean


@dataclass(frozen=True)
class RefinedValue:
    prior: float
    value: float
    observations: int


def checkpoint_value(
    node: Node[AnyRollout], evidence: RolloutEvidence, config: RewardConfig
) -> RefinedValue:
    counts = evidence.observations.get(node.id)
    stats = counts.returns if counts is not None else RunningStats()
    if node.payload.done:
        return RefinedValue(0.0, 0.0, stats.n)
    prior = node.critic.get(config.value_version)
    if prior is None or not math.isfinite(prior):
        raise ValueError(
            f"checkpoint {node.id} requires a finite network value "
            f"under {config.value_version!r}"
        )
    value = refined_mean(prior, stats, config.value_prior_strength)
    if not math.isfinite(value):
        raise ValueError(f"checkpoint {node.id} has non-finite refined value")
    return RefinedValue(prior, value, stats.n)


@register_estimator("refined_td")
class RefinedTdEstimator(AdvantageEstimator):
    def assign(
        self,
        records: Sequence[ActorSample],
        *,
        view: SchedulerView[AnyRollout],
        reward_config: RewardConfig,
    ) -> None:
        evidence = evidence_of(view, reward_config)
        for record, node in _resolved(records, view):
            parent = view.node(record.group_id)
            if parent is None:
                raise ValueError(f"edge {node.id} has no parent")
            before = checkpoint_value(parent, evidence, reward_config)
            after = checkpoint_value(node, evidence, reward_config)
            duration = sum(
                t.transition is not None for t in node.payload.edge(parent.payload)
            )
            reward = edge_return(node, parent, reward_config)
            record.weight = (
                reward + reward_config.gamma**duration * after.value - before.value
            )
            record.diagnostics = {
                "advantage": record.weight,
                "edge_reward": reward,
                "duration": float(duration),
                "parent_prior": before.prior,
                "successor_prior": after.prior,
                "parent_value": before.value,
                "successor_value": after.value,
                "parent_observations": float(before.observations),
                "successor_observations": float(after.observations),
            }
