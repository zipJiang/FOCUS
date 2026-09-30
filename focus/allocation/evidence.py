"""Rollout suffix evidence and the scoring extension of the allocation lifecycle."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from focus.harness import AnyRollout
from focus.reward.config import RewardConfig
from focus.scheduler.core.allocation import (
    AllocationLedger,
    Observations,
    RankedSession,
)
from focus.scheduler.core.execution import drawn_from_anchor, provenance_of
from focus.scheduler.core.tree import Node, SchedulerView

EVIDENCE_KEY = "rollout_evidence"


@dataclass(frozen=True)
class RolloutEvidence:
    """Persisted suffix observations shared by allocation and preparation."""

    reward_config: RewardConfig
    observations: dict[int, Observations]
    behavior_version: str | None = None


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
        # A proposal samples the entire continuation. Its suffix is anchor-value
        # evidence only when that continuation was drawn from the anchor.
        if not all(
            drawn_from_anchor(n, self.reward_config.anchor_version) for n in chain
        ):
            return
        stamps = [provenance_of(n) for n in chain]
        versions = {p.behavior_version for p in stamps if p is not None}
        configs = {p.reward_config_id for p in stamps if p is not None}
        if len(versions) != 1 or configs != {self.reward_config.config_id}:
            raise ValueError("inconsistent rollout evidence provenance")
        if self._root is not None:
            stored = self._root.metadata[EVIDENCE_KEY]
            assert isinstance(stored, RolloutEvidence)
            version = next(iter(versions))
            if stored.behavior_version not in (None, version):
                raise ValueError("rollout evidence mixes behavior versions")
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
                total += self.reward_config.beta_step * turns[cursor].compaction_penalty
            if not math.isfinite(total):
                raise ValueError("non-finite rollout suffix return")
            values.append((node, total))
        if self._root is not None:
            self._root.metadata[EVIDENCE_KEY] = RolloutEvidence(
                self.reward_config, self.observations, next(iter(versions))
            )
        # Calculate first, then commit so malformed boundaries leave no partial
        # evidence.
        for node, value in reversed(values):
            self.counts(node).returns.add(value)


class NodeWorth(ABC):
    """Expected benefit of another purchase. Nonpositive gain declines the purchase."""

    def __init__(self, reward_config: RewardConfig) -> None:
        self.reward_config = reward_config

    @abstractmethod
    def worth(
        self,
        view: SchedulerView[AnyRollout],
        node: Node[AnyRollout],
        counts: Observations,
    ) -> float: ...


class ScoredSession(RankedSession[AnyRollout]):
    def __init__(self, worth: NodeWorth, ledger: RolloutLedger) -> None:
        if worth.reward_config != ledger.reward_config:
            raise ValueError(
                "worth and evidence must use the same reward configuration"
            )
        super().__init__(ledger)
        self.worth = worth

    def score(self, view: SchedulerView[AnyRollout], node: Node[AnyRollout]) -> float:
        if not node.payload.forkable or node.payload.done:
            return -math.inf
        return self.worth.worth(view, node, self.ledger.counts(node))

    def eligible(self, score: float) -> bool:
        return math.isfinite(score) and score > 0.0


__all__ = ["NodeWorth", "Observations", "RolloutLedger", "ScoredSession"]
