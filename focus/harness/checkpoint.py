"""Checkpoint selection is independent of generation purpose and token packing."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from focus.registry import register


@dataclass(frozen=True)
class Effect:
    """A committed operation, linked to the generation that requested it."""

    kind: str
    turn_index: int
    context_before: int = 0
    context_after: int = 0
    detail: str = ""
    invocation_id: str = "root"


@runtime_checkable
class CheckpointPolicy(Protocol):
    def selects(self, effects: tuple[Effect, ...]) -> bool: ...


@register(CheckpointPolicy, "sparse")
class SparseCheckpoints:
    def selects(self, effects: tuple[Effect, ...]) -> bool:
        return any(
            effect.kind in {"compaction", "context", "memory", "invoke"}
            for effect in effects
        )


@register(CheckpointPolicy, "every_decision")
class EveryDecision:
    def selects(self, effects: tuple[Effect, ...]) -> bool:
        return True
