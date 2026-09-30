"""Training coefficients calculated from the finished tree."""

from __future__ import annotations

import math
from collections.abc import Sequence

from focus.harness import AnyRollout
from focus.preparation.base import (
    AdvantageEstimator,
    _resolved,
    register_estimator,
)
from focus.preparation.records import ActorSample
from focus.reward.config import RewardConfig
from focus.scheduler.core.returns import finished_below
from focus.scheduler.core.tree import SchedulerView
from focus.statistics import RunningStats

_GRPO_STD_EPSILON = 1e-6


@register_estimator("grpo")
class GrpoEstimator(AdvantageEstimator):
    requires_completed_paths = True

    def assign(
        self,
        records: Sequence[ActorSample],
        *,
        view: SchedulerView[AnyRollout],
        reward_config: RewardConfig,
    ) -> None:
        if any(
            node.id != view.root.id and len(node.children) > 1 for node in view.nodes()
        ):
            raise ValueError(
                "GRPO requires independent root rollouts; "
                "branching below the root is unsupported"
            )
        returns = {
            node.id: reward_config.configured_return(
                node.payload.turns, len(view.root.payload.turns)
            )
            for node in view.terminals()
        }
        if not returns:
            return
        group = RunningStats.of(list(returns.values()))
        baseline, spread = group.mean, math.sqrt(group.variance)
        for record, node in _resolved(records, view):
            finished = finished_below(view, node) or [node]
            values = [returns[leaf.id] for leaf in finished if leaf.id in returns]
            if not values:
                continue
            weight = sum(values) / len(values) - baseline
            if spread > 0.0:
                weight /= spread + _GRPO_STD_EPSILON
            record.weight = weight
            # the spread rides along because it is the whole group's, not this edge's:
            # a batch whose weights all look enormous is usually a group that barely
            # varied, and this is the number that says so
            record.diagnostics = {"advantage": weight, "std": spread}
