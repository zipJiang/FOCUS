"""Per-checkpoint values of a settled search tree, computed once per preparation.

Every quantity the paper names at a checkpoint lives here: the critic prediction
:math:`V_\\phi`, the direct-branch backup mean :math:`\\bar G` over its :math:`n` direct
children, the refined value :math:`\\tilde V`, and the observed continuation outcomes
(:math:`N` and their mean) that selective allocation reads. Estimators and critic
targets read these values instead of recomputing them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# A leaf at import time: search, allocation, and preparation all import RewardConfig
# from here, so the tree machinery is imported inside ``tree_values``.
if TYPE_CHECKING:
    from focus.harness import AnyRollout
    from focus.harness.env import StepResult
    from focus.harness.turn import Turn
    from focus.scheduler.core.tree import SchedulerView


@dataclass(frozen=True, kw_only=True)
class RewardConfig:
    """How raw rewards become one training number -- frozen for a whole batch.

    The envs store ``reward_outcome`` and ``reward_step`` unmixed, so every downstream
    statistic has to say which weighting it used. This record *is* that statement, and
    preparation refuses a tree whose evidence was collected under another one. Frozen
    per batch, which is what makes an estimate collected before a decision and one
    collected after it commensurable.

    Keyword-only: the fields are a bag of
    independent knobs in no meaningful order, so a positional caller is writing down an
    order that is not part of the design -- and reordering two floats would then change
    what an existing call *means* rather than failing it.
    """

    #: Per-transition discount applied along a turn chain.
    gamma: float = 1.0
    #: Pseudo-observations assigned to the network checkpoint value.
    value_prior_strength: float = 1.0
    #: What one unit of shaping reward is worth against one unit of outcome reward.
    beta_step: float = 1.0
    component_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.value_prior_strength)
            or self.value_prior_strength <= 0
        ):
            raise ValueError("value_prior_strength must be positive and finite")

    def train_reward(self, transition: StepResult[object]) -> float:
        """One transition's reward under this weighting."""
        return transition.reward_outcome + self.beta_step * transition.reward_step

    def shaping(self, turn: Turn[object]) -> float:
        """Protocol reward components do not advance task discount time."""
        return self.beta_step * turn.compaction_penalty + sum(
            self.component_weights.get(name, 1.0) * value
            for name, value in turn.reward_components.items()
        )

    def configured_return(self, turns: Sequence[Turn[object]], start: int = 0) -> float:
        """Discounted return over a turn chain, or a suffix of one from ``start``.

        Discounting is per *env transition*, and a turn carries at most one -- so the
        discount no longer depends on how the rollout happened to be cut into regions,
        which is what made this arithmetic fragile when a region was the unit.

        A fold turn (``transition is None``) contributes its explicit compaction
        penalty and **no discount step**. Folding rewrites the working context; it
        does not act on the world, so
        it is not a time step and must not shrink the weight of the task work that
        follows it. A rollout that folded twice discounts exactly like one that never
        folded.
        """
        total, discount = 0.0, 1.0
        for turn in turns[start:]:
            total += discount * self.shaping(turn)
            if turn.transition is None:
                continue
            total += discount * self.train_reward(turn.transition)
            discount *= self.gamma
        return total


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
    from focus.allocation.evidence import EVIDENCE_KEY, RolloutEvidence
    from focus.scheduler.core.returns import edge_return

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


__all__ = ["NodeValues", "RewardConfig", "TreeValues", "tree_values"]
