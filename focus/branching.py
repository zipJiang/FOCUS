"""Eligibility rules for choosing resumable branch points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from focus.harness.rollout import AnyRollout


@runtime_checkable
class BranchPoints(Protocol):
    """Which completed states search may continue from.

    Implement :meth:`eligible`; the agent calls it after every decision.
    """

    def eligible(self, state: AnyRollout) -> bool:
        """Whether the completed state is a candidate for another continuation."""
        ...


class SparseCheckpoints:
    """The root, and the states that end a compaction or context edit."""

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
    """Every live state after every decision."""

    def eligible(self, state: AnyRollout) -> bool:
        return True


__all__ = ["BranchPoints", "EveryDecision", "SparseCheckpoints"]
