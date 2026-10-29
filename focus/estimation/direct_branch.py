"""The paper's actor recipe: segment TD residuals over refined direct-branch values."""

from __future__ import annotations

from collections.abc import Sequence

from focus.estimation.base import Advantage, Edge, Estimator
from focus.values import TreeValues


class DirectBranchTd(Estimator):
    """Segment TD advantages :math:`R + \\gamma^d \\tilde V(c') - \\tilde V(c)`.

    Every nonterminal checkpoint on an actor edge needs a critic prediction; a tree
    with a missing one is refused rather than trained against a partial backup.
    """

    name = "direct_branch_td"

    def advantages(
        self, edges: Sequence[Edge], values: TreeValues
    ) -> Sequence[Advantage]:
        gamma = values.reward.gamma
        result = []
        for edge in edges:
            before, after = values[edge.parent.id], values[edge.node.id]
            for node, value in ((edge.parent, before), (edge.node, after)):
                if value.refined is None:
                    raise ValueError(
                        f"checkpoint {node.id} requires a finite network value"
                    )
            assert before.refined is not None
            assert after.refined is not None
            weight = edge.reward + gamma**edge.duration * after.refined - before.refined
            result.append(
                Advantage(
                    weight,
                    {
                        "advantage": weight,
                        "edge_reward": edge.reward,
                        "duration": float(edge.duration),
                        "parent_prior": before.prior or 0.0,
                        "successor_prior": after.prior or 0.0,
                        "parent_value": before.refined,
                        "successor_value": after.refined,
                        "parent_direct_branches": float(before.branches),
                        "successor_direct_branches": float(after.branches),
                    },
                )
            )
        return result


__all__ = ["DirectBranchTd"]
