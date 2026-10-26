"""Reusable allocator configuration creates a fresh session for each search pass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from focus.allocation.evidence import RolloutLedger
from focus.harness import AnyRollout
from focus.harness.turn import TurnTag
from focus.scheduler.core.allocation import (
    AllocationSession,
    FifoSession,
)
from focus.scheduler.core.tree import SchedulerView


class Allocator(ABC):
    #: Shown in search logs. Defaults to the class name.
    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("name"):
            cls.name = cls.__name__

    @abstractmethod
    def build(
        self, view: SchedulerView[AnyRollout], ledger: RolloutLedger
    ) -> AllocationSession[AnyRollout] | None: ...


class FifoAllocator(Allocator):
    name = "fifo"

    def build(
        self, view: SchedulerView[AnyRollout], ledger: RolloutLedger
    ) -> AllocationSession[AnyRollout]:
        return FifoSession(ledger)


class TokenEntropyAllocator(Allocator):
    name = "windowed_token_entropy"

    def __init__(
        self,
        window: int = 8,
        tags: tuple[TurnTag, ...] | None = ("task",),
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.tags = tuple(tags) if tags is not None else None

    def build(
        self, view: SchedulerView[AnyRollout], ledger: RolloutLedger
    ) -> AllocationSession[AnyRollout]:
        from focus.scheduler.core.entropy import TokenEntropySession

        return TokenEntropySession(self.window, self.tags, ledger=ledger)
