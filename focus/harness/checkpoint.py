"""Checkpoint selection is independent of generation purpose and token packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from focus.harness.rollout import AnyRollout


@dataclass(frozen=True)
class Effect:
    """A committed operation, linked to the generation that requested it."""

    kind: str
    turn_index: int
    context_before: int = 0
    context_after: int = 0
    detail: str = ""
    invocation_id: str = "root"
    #: Exclusive end of the committed operation in the rollout turn log.
    turn_end: int = -1


@runtime_checkable
class CheckpointPolicy(Protocol):
    def eligible(self, state: AnyRollout) -> bool:
        """Whether the completed state is a candidate for another continuation."""
        ...


class SparseCheckpoints:
    def eligible(self, state: AnyRollout) -> bool:
        if not state.turns:
            return True
        for effect in reversed(state.effects):
            if effect.turn_end < len(state.turns):
                break
            if (
                effect.turn_end == len(state.turns)
                and effect.invocation_id == state.invocation_id
                and effect.kind in {"compaction", "context"}
            ):
                return True
        return False


class EveryDecision:
    def eligible(self, state: AnyRollout) -> bool:
        return True
