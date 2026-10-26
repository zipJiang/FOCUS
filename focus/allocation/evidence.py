"""Rollout suffix evidence and the scoring extension of the allocation lifecycle."""

from __future__ import annotations

import math
from dataclasses import dataclass

from focus.harness import AnyRollout
from focus.scheduler.core.allocation import (
    AllocationLedger,
    Observations,
)
from focus.scheduler.core.tree import Node, SchedulerView
from focus.values import RewardConfig

EVIDENCE_KEY = "rollout_evidence"


@dataclass(frozen=True)
class RolloutEvidence:
    """Persisted suffix observations shared by allocation and preparation."""

    reward_config: RewardConfig
    observations: dict[int, Observations]


def evidence_of(
    view: SchedulerView[AnyRollout], config: RewardConfig
) -> RolloutEvidence:
    evidence = view.root.metadata.get(EVIDENCE_KEY)
    if not isinstance(evidence, RolloutEvidence):
        raise ValueError(
            "tree has no rollout evidence; historical trees are analysis-only"
        )
    if evidence.reward_config != config:
        raise ValueError(
            "rollout evidence reward configuration differs from preparation"
        )
    return evidence


class RolloutLedger(AllocationLedger[AnyRollout]):
    def __init__(
        self,
        reward_config: RewardConfig,
        observations: dict[int, Observations] | None = None,
    ) -> None:
        self.reward_config = reward_config
        super().__init__(observations=observations)

    def bind(self, view: SchedulerView[AnyRollout]) -> None:
        super().bind(view)
        stored = view.root.metadata.get(EVIDENCE_KEY)
        if stored is None:
            view.root.metadata[EVIDENCE_KEY] = RolloutEvidence(
                self.reward_config, self.observations
            )
        else:
            evidence = evidence_of(view, self.reward_config)
            if self.observations and self.observations is not evidence.observations:
                raise ValueError("tree already owns a different rollout ledger")
            self.observations = evidence.observations

    def record(self, parent: Node[AnyRollout], chain: list[Node[AnyRollout]]) -> None:
        if not chain or not chain[-1].payload.done:
            return
        turns = chain[-1].payload.turns
        cursor, total = len(turns), 0.0
        values = []
        # All checkpoint suffixes in one backward pass. Folds may shape reward
        # without discounting; repeated checkpoint lengths share the same suffix.
        for node in reversed((parent, *chain)):
            start = len(node.payload.turns)
            if not 0 <= start <= cursor:
                raise ValueError(
                    "rollout checkpoints must have nondecreasing turn counts"
                )
            while cursor > start:
                cursor -= 1
                transition = turns[cursor].transition
                if transition is not None:
                    total = (
                        self.reward_config.train_reward(transition)
                        + self.reward_config.gamma * total
                    )
                total += self.reward_config.shaping(turns[cursor])
            if not math.isfinite(total):
                raise ValueError("non-finite rollout suffix return")
            values.append((node, total))
        # Calculate first, then commit so malformed boundaries leave no partial
        # evidence.
        for node, value in reversed(values):
            self.counts(node).returns.add(value)


__all__ = [
    "EVIDENCE_KEY",
    "Observations",
    "RolloutEvidence",
    "RolloutLedger",
    "evidence_of",
]
