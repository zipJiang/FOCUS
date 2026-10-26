"""One allocation lifecycle, with search-owned evidence and per-pass reservations.

All lifecycle methods run synchronously under the scheduler's tree lock. Configuration
never owns a session: each search/pass constructs its own, sharing only that search's
ledger. Generic payloads count purchases; rollout ledgers also record suffix returns.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import final

from focus.scheduler.core.tree import (
    Node,
    Payload,
    SchedulerState,
    SchedulerView,
)
from focus.statistics import RunningStats


@dataclass
class Observations:
    """Direct purchases and completed suffix evidence, never pooled from descendants."""

    m_done: int = 0
    m_pending: int = 0
    returns: RunningStats = field(default_factory=RunningStats)

    @property
    def m_alloc(self) -> int:
        return self.m_done + self.m_pending

    @property
    def n_eff(self) -> int:
        return self.returns.n + self.m_pending


class AllocationLedger[P: Payload]:
    """Evidence shared by all allocation sessions belonging to one search."""

    def __init__(self, *, observations: dict[int, Observations] | None = None) -> None:
        self.observations = observations if observations is not None else {}
        self._root: Node[P] | None = None

    def bind(self, view: SchedulerView[P]) -> None:
        if self._root is not None and self._root is not view.root:
            raise ValueError("an allocation ledger belongs to one search tree")
        self._root = view.root

    def counts(self, node: Node[P]) -> Observations:
        if (counts := self.observations.get(node.id)) is None:
            counts = self.observations[node.id] = Observations()
        return counts

    def record(self, parent: Node[P], chain: list[Node[P]]) -> None:
        """Rollout ledgers record suffix evidence; opaque payloads only count
        purchases."""
        return None


@dataclass(frozen=True, eq=False)
class Reservation[P: Payload]:
    """An opaque purchase identity. Two purchases of one node remain distinct."""

    node: Node[P]


class AllocationSession[P: Payload](ABC):
    """Choose nodes and settle their purchases through one owned lifecycle.

    Override ``choose`` (or ``score`` on RankedSession), not the accounting methods.
    ``completed`` is an optional synchronous extension for additional statistics or
    study diagnostics, called after common evidence has been recorded.
    """

    requires_scorer = False

    def __init__(self, ledger: AllocationLedger[P] | None = None) -> None:
        self.ledger = ledger if ledger is not None else AllocationLedger()
        self._active: set[Reservation[P]] = set()
        self._running = False

    def start(self, view: SchedulerView[P]) -> None:
        if self._running or self._active:
            raise RuntimeError(
                "an allocation session is already running "
                "or has outstanding reservations"
            )
        self.ledger.bind(view)
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._active:
            raise RuntimeError(
                "allocation session stopped with outstanding reservations"
            )

    @abstractmethod
    def choose(self, view: SchedulerView[P]) -> Node[P] | None: ...

    @final
    def select(self, view: SchedulerView[P]) -> Reservation[P] | None:
        self.ledger.bind(view)
        node = self.choose(view)
        if node is None:
            return None
        if not view.selectable(node):
            raise ValueError(
                f"{type(self).__name__} returned node {node.id}, "
                "which is not expandable "
                f"(forkable={node.payload.forkable}, done={node.payload.done}, "
                f"gate_free={node.gate.available(node)}); select from view.frontier()"
            )
        reservation = Reservation(node)
        self.ledger.counts(node).m_pending += 1
        self._active.add(reservation)
        try:
            self.selected(reservation, view)
        except BaseException:
            self._settle(reservation)
            raise
        return reservation

    def active(self, reservation: Reservation[P]) -> bool:
        return reservation in self._active

    def _settle(self, reservation: Reservation[P]) -> None:
        if reservation not in self._active:
            raise ValueError("reservation is foreign or already settled")
        counts = self.ledger.counts(reservation.node)
        if counts.m_pending <= 0:
            raise RuntimeError("pending purchase accounting is inconsistent")
        self._active.remove(reservation)
        counts.m_pending -= 1

    @final
    def complete(
        self,
        reservation: Reservation[P],
        state: SchedulerState[P],
        chain: list[Node[P]],
    ) -> None:
        if not self.active(reservation):
            raise ValueError("reservation is foreign or already settled")
        if not chain:
            raise ValueError("an empty expansion must fail instead of complete")
        self.ledger.bind(SchedulerView(state))
        parent = reservation.node
        for node in chain:
            if state.nodes.get(node.id) is not node or node.parent_id != parent.id:
                raise ValueError(
                    "completion must contain the attached continuation chain"
                )
            parent = node
        self._settle(reservation)
        self.ledger.counts(reservation.node).m_done += 1
        self.ledger.record(reservation.node, chain)
        self.completed(reservation, state, chain)

    @final
    def fail(self, reservation: Reservation[P], error: BaseException) -> None:
        self._settle(reservation)
        self.released(reservation, error)

    @final
    def cancel(self, reservation: Reservation[P]) -> None:
        self._settle(reservation)
        self.released(reservation, None)

    def selected(self, reservation: Reservation[P], view: SchedulerView[P]) -> None:
        return None

    def completed(
        self,
        reservation: Reservation[P],
        state: SchedulerState[P],
        chain: list[Node[P]],
    ) -> None:
        return None

    def released(
        self, reservation: Reservation[P], error: BaseException | None
    ) -> None:
        return None


class FifoSession[P: Payload](AllocationSession[P]):
    def choose(self, view: SchedulerView[P]) -> Node[P] | None:
        return next(view.iter_frontier(), None)


class RankedSession[P: Payload](AllocationSession[P], ABC):
    """One streaming frontier scan, with deterministic lower-ID tie-breaking."""

    @abstractmethod
    def score(self, view: SchedulerView[P], node: Node[P]) -> float: ...

    def eligible(self, score: float) -> bool:
        return not math.isnan(score)

    def choose(self, view: SchedulerView[P]) -> Node[P] | None:
        best: Node[P] | None = None
        best_score = float("-inf")
        for node in view.iter_frontier():
            score = self.score(view, node)
            if self.eligible(score) and (
                best is None
                or score > best_score
                or (score == best_score and node.id < best.id)
            ):
                best, best_score = node, score
        return best
