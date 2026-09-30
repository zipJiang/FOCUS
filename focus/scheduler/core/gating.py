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

from focus.registry import register, registrable

if TYPE_CHECKING:
    from focus.scheduler.core.tree import Node


class Gate:
    """Reserve/release branch slots for one node: a non-blocking semaphore over it.

    A capacity and an in-flight count, and that is the whole of it -- which is why it is
    a class rather than an interface with one implementation behind it. What *varies*
    between policies is the two numbers a :class:`GatingPolicy` mints it with: the
    capacity, and whether the children a node already has count against it.

    ``Node[Any]``: gating reads a node's ``children`` and nothing else, so the payload
    type is irrelevant -- and ``Node`` is invariant in it (``payload`` is a mutable
    field), so ``Any`` rather than ``object`` is what a concrete tree's node satisfies.
    """

    def __init__(self, capacity: int, *, count_children: bool) -> None:
        self._capacity = capacity
        self._count_children = count_children
        self._in_use = 0

    @property
    def in_use(self) -> int:
        return self._in_use

    def available(self, node: Node[Any]) -> bool:
        """Whether a branch may launch from ``node`` right now."""
        used = self._in_use + (len(node.children) if self._count_children else 0)
        return used < self._capacity

    def reserve(self) -> None:
        """Claim a slot (on launch)."""
        self._in_use += 1

    def release(self) -> None:
        """Free a slot (on completion)."""
        self._in_use -= 1


@registrable(slot="gating")
class GatingPolicy(ABC):
    """Mint a :class:`Gate` for each new node -- branch policy, decided per node."""

    @abstractmethod
    def gate(self, node: Node[Any]) -> Gate: ...


@register(GatingPolicy, "width")
class WidthGating(GatingPolicy):
    """Cap *total* branches per node (children + in-flight) -- the classic width."""

    def __init__(self, width: int = 1) -> None:
        self._width = max(1, width)

    def gate(self, node: Node[Any]) -> Gate:
        return Gate(self._width, count_children=True)


@register(GatingPolicy, "concurrency")
class ConcurrencyGating(GatingPolicy):
    """Cap only *concurrent* in-flight branches per node; total children unbounded."""

    def __init__(self, limit: int = 1) -> None:
        self._limit = max(1, limit)

    def gate(self, node: Node[Any]) -> Gate:
        return Gate(self._limit, count_children=False)


__all__ = ["ConcurrencyGating", "Gate", "GatingPolicy", "WidthGating"]
