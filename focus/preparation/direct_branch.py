"""Recursive direct-branch values with unshrunk critic supervision."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from focus.harness import AnyRollout
from focus.preparation.base import (
    AdvantageEstimator,
    _resolved,
    register_estimator,
)
from focus.preparation.records import ActorSample, CriticSample
from focus.reward.config import RewardConfig
from focus.scheduler.core.critic import VALUE_CONTEXTS_KEY
from focus.scheduler.core.returns import edge_return
from focus.scheduler.core.tree import SchedulerView


@dataclass(frozen=True)
class DirectBranchValue:
    prior: float
    value: float
    mean: float | None
    branches: int


def direct_branch_values(
    view: SchedulerView[AnyRollout], config: RewardConfig
) -> dict[int, DirectBranchValue]:
    """Back up each sampled edge once, irrespective of descendant multiplicity.

    Successor values include their own prior blend. A terminal's remaining value
    is zero; its reward is on the incoming edge. An unexplored nonterminal uses
    its frozen network prior and has no critic target. No tree state is mutated.
    """
    values: dict[int, DirectBranchValue] = {}
    for node in sorted(view.nodes(), key=lambda n: n.depth, reverse=True):
        if node.payload.done:
            if node.children:
                raise ValueError(f"terminal checkpoint {node.id} has children")
            values[node.id] = DirectBranchValue(0.0, 0.0, None, 0)
            continue
        prior = node.critic.get(config.value_version)
        if prior is None or not math.isfinite(prior):
            raise ValueError(f"checkpoint {node.id} requires a finite network value")
        contributions = []
        for child_id in node.children:
            child = view.node(child_id)
            if child is None or child.parent_id != node.id:
                raise ValueError(f"invalid direct branch from checkpoint {node.id}")
            duration = sum(
                t.transition is not None for t in child.payload.edge(node.payload)
            )
            contributions.append(
                edge_return(child, node, config)
                + config.gamma**duration * values[child.id].value
            )
        count = len(contributions)
        total = math.fsum(contributions)
        mean = total / count if count else None
        value = (
            (config.value_prior_strength * prior + total)
            / (config.value_prior_strength + count)
            if count
            else prior
        )
        if not math.isfinite(value) or (mean is not None and not math.isfinite(mean)):
            raise ValueError(f"checkpoint {node.id} has non-finite branch values")
        values[node.id] = DirectBranchValue(prior, value, mean, count)
    return values


@register_estimator("direct_branch_td")
class DirectBranchTdEstimator(AdvantageEstimator):
    """Refined direct-branch TD for actors; detached branch means for critics."""

    def assign(
        self,
        records: Sequence[ActorSample],
        *,
        view: SchedulerView[AnyRollout],
        reward_config: RewardConfig,
    ) -> None:
        values = direct_branch_values(view, reward_config)
        for record, node in _resolved(records, view):
            parent = view.node(record.group_id)
            if parent is None or node.parent_id != parent.id:
                raise ValueError(f"edge {node.id} has no matching parent")
            before, after = values[parent.id], values[node.id]
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
                "parent_direct_branches": float(before.branches),
                "successor_direct_branches": float(after.branches),
            }

    def critic_targets(
        self,
        *,
        view: SchedulerView[AnyRollout],
        reward_config: RewardConfig,
    ) -> Sequence[CriticSample]:
        values = direct_branch_values(view, reward_config)
        result = []
        for node in view.nodes():
            value = values[node.id]
            if value.mean is None:
                continue
            context = node.metadata.get(VALUE_CONTEXTS_KEY, {}).get(
                reward_config.value_version
            )
            if context is None:
                raise ValueError(
                    f"checkpoint {node.id} has no serialized critic context"
                )
            result.append(
                CriticSample(
                    node_id=node.id,
                    context=context,
                    target=value.mean,
                    reward_config_id=reward_config.config_id,
                    value_version=reward_config.value_version,
                    diagnostics=MappingProxyType(
                        {
                            "prior": value.prior,
                            "observations": float(value.branches),
                            "direct_branches": float(value.branches),
                            "mean_return": value.mean,
                            "refined_value": value.value,
                        }
                    ),
                )
            )
        return result
