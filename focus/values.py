"""Per-checkpoint values of a settled search tree, computed once per preparation.

Every quantity the paper names at a checkpoint lives here: the critic prediction
:math:`V_\\phi`, the direct-branch backup mean :math:`\\bar G` over its :math:`n` direct
children, the refined value :math:`\\tilde V`, and the observed continuation outcomes
(:math:`N` and their mean) that selective allocation reads. Estimators and critic
targets read these values instead of recomputing them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from focus.allocation.evidence import EVIDENCE_KEY, RolloutEvidence
from focus.harness import AnyRollout
from focus.reward.config import RewardConfig
from focus.scheduler.core.returns import edge_return
from focus.scheduler.core.tree import SchedulerView


@dataclass(frozen=True)
class NodeValues:
    """One checkpoint's values.

    ``refined`` and ``backup`` are ``None`` when they depend on a checkpoint the critic
    failed to score. A terminal checkpoint has ``refined == 0``: its reward is on the
    incoming edge.
    """

    #: The critic prediction :math:`V_\\phi`, or ``None`` if the node was not scored.
    prior: float | None
    #: :math:`n`, the number of direct children.
    branches: int
    #: :math:`\\bar G`, the mean direct-branch backup; ``None`` without children.
    backup: float | None
    #: :math:`\\tilde V`, the prior blended with the backup at strength :math:`\\kappa`.
    refined: float | None
    #: :math:`N`, completed continuation outcomes observed through this checkpoint.
    observations: int
    #: The mean of those outcomes; ``None`` when there are none.
    observed_mean: float | None


@dataclass(frozen=True)
class TreeValues(Mapping[int, NodeValues]):
    """Values for every node of one tree, keyed by node id."""

    tree: SchedulerView[AnyRollout]
    reward: RewardConfig
    _values: Mapping[int, NodeValues]

    def __getitem__(self, node_id: int) -> NodeValues:
        return self._values[node_id]

    def __iter__(self) -> Iterator[int]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def tree_values(view: SchedulerView[AnyRollout], reward: RewardConfig) -> TreeValues:
    """Back up every checkpoint once, deepest first.

    Each direct child contributes its edge return plus its own discounted refined
    value, once, irrespective of how many descendants it has (Eq. backup). The refined
    value blends the critic prior with that mean at pseudo-count :math:`\\kappa`
    (Eq. refined); an unexpanded checkpoint keeps its prior. No tree state is mutated.
    """
    stored = view.root.metadata.get(EVIDENCE_KEY)
    observed = stored.observations if isinstance(stored, RolloutEvidence) else {}
    kappa = reward.value_prior_strength
    values: dict[int, NodeValues] = {}
    for node in sorted(view.nodes(), key=lambda n: n.depth, reverse=True):
        counts = observed.get(node.id)
        n_obs = counts.returns.n if counts is not None else 0
        mean_obs = counts.returns.mean if counts is not None and n_obs else None
        if node.payload.done:
            if node.children:
                raise ValueError(f"terminal checkpoint {node.id} has children")
            values[node.id] = NodeValues(None, 0, None, 0.0, n_obs, mean_obs)
            continue
        prior = node.critic
        if prior is not None and not math.isfinite(prior):
            prior = None
        contributions: list[float] | None = []
        for child_id in node.children:
            child = view.node(child_id)
            if child is None or child.parent_id != node.id:
                raise ValueError(f"invalid direct branch from checkpoint {node.id}")
            successor = values[child.id].refined
            if successor is None or contributions is None:
                contributions = None
                continue
            duration = sum(
                t.transition is not None for t in child.payload.edge(node.payload)
            )
            contributions.append(
                edge_return(child, node, reward) + reward.gamma**duration * successor
            )
        count = len(node.children)
        backup: float | None = None
        refined: float | None = prior
        if contributions is None:
            refined = None
        elif count:
            total = math.fsum(contributions)
            backup = total / count
            refined = (
                None if prior is None else (kappa * prior + total) / (kappa + count)
            )
        if (refined is not None and not math.isfinite(refined)) or (
            backup is not None and not math.isfinite(backup)
        ):
            raise ValueError(f"checkpoint {node.id} has non-finite branch values")
        values[node.id] = NodeValues(prior, count, backup, refined, n_obs, mean_obs)
    return TreeValues(view, reward, values)


__all__ = ["NodeValues", "TreeValues", "tree_values"]
