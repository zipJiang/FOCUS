"""Per-node branching control: how many workers may branch from a node right now.

A :class:`Gate` is attached to each node and answers "is a branch slot free?" plus
reserve/release around an expansion -- a non-blocking, semaphore-style counter. A
:class:`GatingPolicy` mints the gate for each new node, so the cap can vary per node
(by depth, payload, etc.); :class:`WidthGating` is the flat one cap for every node.

The policy is the seam; the gate is not. Both shipped policies mint the *same* counter
with different numbers, so ``Gate`` is that counter rather than an interface with one
implementation hidden behind it -- a policy wanting different bookkeeping subclasses it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from focus.scheduler.core.tree import Node


class Gate:
    """Reserve/release branch slots for one node: a non-blocking semaphore over it.

    Two optional caps and an in-flight count: ``max_children`` bounds a node's total
    branches (children plus in-flight), ``max_pending`` only its in-flight ones.

    ``Node[Any]``: gating reads a node's ``children`` and nothing else, so the payload
    type is irrelevant -- and ``Node`` is invariant in it (``payload`` is a mutable
    field), so ``Any`` rather than ``object`` is what a concrete tree's node satisfies.
    """

    def __init__(
        self, *, max_children: int | None = None, max_pending: int | None = None
    ) -> None:
        self._max_children = max_children
        self._max_pending = max_pending
        self._in_use = 0

    @property
    def in_use(self) -> int:
        return self._in_use

    def available(self, node: Node[Any]) -> bool:
        """Whether a branch may launch from ``node`` right now."""
        if self._max_pending is not None and self._in_use >= self._max_pending:
            return False
        return (
            self._max_children is None
            or self._in_use + len(node.children) < self._max_children
        )

    def reserve(self) -> None:
        """Claim a slot (on launch)."""
        self._in_use += 1

    def release(self) -> None:
        """Free a slot (on completion)."""
        self._in_use -= 1


class GatingPolicy(ABC):
    """Mint a :class:`Gate` for each new node -- branch policy, decided per node."""

    @abstractmethod
    def gate(self, node: Node[Any]) -> Gate: ...


class WidthGating(GatingPolicy):
    """Cap *total* branches per node (children + in-flight) -- the classic width."""

    def __init__(self, width: int = 1) -> None:
        self._width = max(1, width)

    def gate(self, node: Node[Any]) -> Gate:
        return Gate(max_children=self._width)


class ConcurrencyGating(GatingPolicy):
    """Cap only *concurrent* in-flight branches per node; total children unbounded."""

    def __init__(self, limit: int = 1) -> None:
        self._limit = max(1, limit)

    def gate(self, node: Node[Any]) -> Gate:
        return Gate(max_pending=self._limit)


__all__ = ["ConcurrencyGating", "Gate", "GatingPolicy", "WidthGating"]
