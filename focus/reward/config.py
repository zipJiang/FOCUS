"""Shared reward semantics used by search and training preparation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from focus.harness.env import StepResult
    from focus.harness.turn import Turn


@dataclass(frozen=True, kw_only=True)
class RewardConfig:
    """How raw rewards become one training number -- frozen for a whole batch.

    The envs store ``reward_outcome`` and ``reward_step`` unmixed and the critic writes
    its estimates under a version key, so every downstream statistic has to say which
    weighting and which value channel it used. This record *is* that statement: it is
    stamped into the provenance of everything derived under it, so two numbers computed
    under different coefficients can never be silently compared. Frozen per allocation
    batch, which is what makes an estimate collected before a decision and one collected
    after it commensurable.

    ``value_version`` names the channel the critic's estimates are stored under.

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
    value_version: str = "v-base"
    #: Identifies this configuration in every record derived under it.
    config_id: str = "rc-0"
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


__all__ = ["RewardConfig"]
