"""Reusable allocator configuration creates a fresh session for each search pass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from focus.allocation.evidence import RolloutLedger
from focus.harness import AnyRollout
from focus.harness.turn import TurnTag
from focus.registry import register, registrable
from focus.scheduler.core.allocation import (
    AllocationSession,
    FifoSession,
)
from focus.scheduler.core.tree import SchedulerView


@registrable(slot="allocator")
class Allocator(ABC):
    name: str = ""

    @abstractmethod
    def build(
        self, view: SchedulerView[AnyRollout], ledger: RolloutLedger
    ) -> AllocationSession[AnyRollout] | None: ...


def register_allocator[A: Allocator](name: str) -> Callable[[type[A]], type[A]]:
    def decorator(cls: type[A]) -> type[A]:
        cls.name = name
        return register(Allocator, name, cls)

    return decorator


@register_allocator("fifo")
class FifoAllocator(Allocator):
    def build(
        self, view: SchedulerView[AnyRollout], ledger: RolloutLedger
    ) -> AllocationSession[AnyRollout]:
        return FifoSession(ledger)


@register_allocator("windowed_token_entropy")
class TokenEntropyAllocator(Allocator):
    def __init__(
        self,
        window: int = 8,
        version: str = "policy",
        tags: tuple[TurnTag, ...] | None = ("task",),
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window, self.version = window, version
        self.tags = tuple(tags) if tags is not None else None

    def build(
        self, view: SchedulerView[AnyRollout], ledger: RolloutLedger
    ) -> AllocationSession[AnyRollout]:
        from focus.scheduler.core.entropy import TokenEntropySession

        return TokenEntropySession(self.window, self.version, self.tags, ledger=ledger)
