"""Group-relative advantages over independent root rollouts."""

from __future__ import annotations

import math
from collections.abc import Sequence

from focus.preparation.base import Advantage, Edge, Estimator
from focus.scheduler.core.returns import finished_below
from focus.statistics import RunningStats
from focus.values import TreeValues

_GRPO_STD_EPSILON = 1e-6


class Grpo(Estimator):
    """Each edge's mean finished return, centered and scaled within the prompt group.

    Requires independent rollouts from the root; branching below it is refused.
    """

    name = "grpo"
    requires_completed_paths = True

    def advantages(
        self, edges: Sequence[Edge], values: TreeValues
    ) -> Sequence[Advantage]:
        view, reward = values.tree, values.reward
        if any(
            node.id != view.root.id and len(node.children) > 1 for node in view.nodes()
        ):
            raise ValueError(
                "GRPO requires independent root rollouts; "
                "branching below the root is unsupported"
            )
        returns = {
            node.id: reward.configured_return(
                node.payload.turns, len(view.root.payload.turns)
            )
            for node in view.terminals()
        }
        if not returns:
            return [Advantage(0.0) for _ in edges]
        group = RunningStats.of(list(returns.values()))
        baseline, spread = group.mean, math.sqrt(group.variance)
        result = []
        for edge in edges:
            finished = finished_below(view, edge.node) or [edge.node]
            outcomes = [returns[leaf.id] for leaf in finished if leaf.id in returns]
            if not outcomes:
                result.append(Advantage(0.0))
                continue
            weight = sum(outcomes) / len(outcomes) - baseline
            if spread > 0.0:
                weight /= spread + _GRPO_STD_EPSILON
            # the spread is the whole group's: a batch whose weights all look enormous
            # is usually a group that barely varied, and this is the number that says so
            result.append(Advantage(weight, {"advantage": weight, "std": spread}))
        return result


__all__ = ["Grpo"]
