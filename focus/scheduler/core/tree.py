"""The scheduler's owned state: a tree of node payloads plus a selectable frontier.

The tree is generic over an opaque payload ``P`` -- the scheduler is pure over the
tree/state and never touches how a payload is made. All it needs from a payload is the
small :class:`Payload` protocol (``forkable`` / ``done`` / ``reward``), which
:class:`~focus.harness.RolloutState` satisfies structurally, so a rollout
checkpoint drops in as the payload and a later caller can use a different one.

Every payload the scheduler produces is stored as a :class:`Node` in an explicit tree
(parent -> children); the subset that may be expanded -- ``forkable``, not terminal, and
with a free slot in its gate -- is tracked in :attr:`SchedulerState.frontier`, which a
:class:`AllocationSession` reads to pick what to expand next.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from focus.scheduler.core.gating import Gate, GatingPolicy

# The keys of `SchedulerState.stats`, named rather than spelled out at each site:
# they are a contract *between* modules -- the scheduler writes them, a `Termination`
# and the search loop read them -- and a misspelt read of a plain dict is a silent zero
# rather than an error.

#: Expansions launched, completed or failed -- what a rollout budget counts.
STAT_ROLLOUTS = "rollouts"
#: Nodes created. One expansion contributes a whole chain of folds, so this is not the
#: same as ``rollouts``.
STAT_NODES = "nodes"
#: Env transitions realized -- the unit compute is spent in, and the one that means the
#: same thing however far a single expansion rolled.
STAT_TURNS = "turns"
#: Expansions that raised (or produced nothing at all).
STAT_FAILURES = "failures"
#: Expansions whose *scoring* raised, the chain having been kept.
STAT_SCORE_FAILURES = "score_failures"


@runtime_checkable
class Payload(Protocol):
    """What the tree needs from a node's contents (``RolloutState`` satisfies it)."""

    @property
    def forkable(self) -> bool:
        """May the scheduler branch (expand) at this node."""

    @property
    def done(self) -> bool:
        """Whether this payload is terminal."""

    @property
    def reward(self) -> float:
        """The default ranking score (used by :meth:`SchedulerState.best`)."""


@dataclass
class Node[P: Payload]:
    """One payload in the search tree, with scheduler bookkeeping."""

    id: int
    parent_id: int | None
    depth: int
    payload: P
    children: list[int] = field(default_factory=list)
    #: Branching control -- minted by the ``GatingPolicy`` the moment the node is added,
    #: so it is always present once the node is in the tree (hence excluded from init).
    gate: Gate = field(init=False)
    #: Ancestor return bookkeeping (a :class:`AllocationSession` maintains these; unused
    #: by default).
    visits: int = 0
    #: Running mean of the *realized* return over the rollouts that finished through
    #: this node -- a Monte-Carlo statistic, not a value function. Prefix-inclusive
    #: (every observation is a whole-episode return, so nodes at different depths stay
    #: comparable) and raw: no ``gamma``, no ``beta_step``.
    mean_return: float = 0.0
    #: The critic's estimate of the return still *to come* (a suffix), or ``None``
    #: until the node is scored -- a different quantity from :attr:`mean_return`.
    critic: float | None = None
    #: The serialized context the critic scored; the input of its training target.
    value_context: str | None = None
    #: Scheduler-related, extensible per-node metadata.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerState[P: Payload]:
    """The tree of every payload plus the incrementally-maintained selectable frontier.

    ``frontier`` is an insertion-ordered set of selectable node ids (``dict`` used as an
    ordered set) so a FIFO policy is deterministic. ``gating`` mints each node's
    :class:`~focus.scheduler.core.gating.Gate` -- how many branches it may
    spawn. Mutated only by the scheduler; policies see it read-only via
    :class:`SchedulerView`.

    ``lock`` guards concurrent access: the scheduler holds it around every critical
    section, and external code inspecting or mutating the tree during a run must too
    (``async with state.lock:``). A single-shot read (one atomic method call) needs no
    lock; a compound read-modify-write does.
    """

    nodes: dict[int, Node[P]]
    root_id: int
    gating: GatingPolicy
    #: Selectable node ids, each mapped to the order it became selectable.
    frontier: dict[int, int] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)
    failures: list[tuple[int, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _next_id: int = 1
    _arrivals: int = 0

    @classmethod
    def root(cls, payload: P, *, gating: GatingPolicy) -> SchedulerState[P]:
        """A fresh state seeded with the initial payload as the tree root."""
        root = Node(id=0, parent_id=None, depth=0, payload=payload)
        state = cls(nodes={0: root}, root_id=0, gating=gating)
        root.gate = gating.gate(root)
        state.stats = {
            STAT_ROLLOUTS: 0.0,
            STAT_NODES: 0.0,
            STAT_TURNS: 0.0,
            STAT_FAILURES: 0.0,
            STAT_SCORE_FAILURES: 0.0,
        }
        state.refresh(0)
        return state

    def selectable(self, node: Node[P]) -> bool:
        """Whether ``node`` may be expanded: forkable, live, and its gate has a slot."""
        return (
            node.payload.forkable
            and not node.payload.done
            and node.gate.available(node)
        )

    def refresh(self, node_id: int) -> None:
        """Recompute one node's frontier membership (idempotent, order-preserving)."""
        if self.selectable(self.nodes[node_id]):
            if node_id not in self.frontier:
                self.frontier[node_id] = self._arrivals
                self._arrivals += 1
        else:
            self.frontier.pop(node_id, None)

    def add_child(self, parent_id: int, payload: P) -> Node[P]:
        """Attach one payload as a child of ``parent_id`` and index it."""
        parent = self.nodes[parent_id]
        node = Node(
            id=self._next_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            payload=payload,
        )
        self._next_id += 1
        self.nodes[node.id] = node
        node.gate = self.gating.gate(node)  # mint before refresh reads availability
        parent.children.append(node.id)
        self.refresh(node.id)
        self.refresh(parent_id)
        return node

    def attach_chain(self, parent_id: int, chain: list[P]) -> list[Node[P]]:
        """Link a rollout's payloads as a descendant line under ``parent_id``.

        The first payload becomes a child of ``parent_id`` (a new branch); each later
        payload is a child of the previous (the step line). Returns the new nodes.
        """
        created: list[Node[P]] = []
        cursor = parent_id
        for payload in chain:
            node = self.add_child(cursor, payload)
            created.append(node)
            cursor = node.id
        return created

    def regate(self, gating: GatingPolicy) -> None:
        """Re-mint every gate under a new policy and rebuild the frontier.

        Gates are minted once, when a node is added, so a node that filled its
        discovery width has no slot left -- correct while discovery runs, and exactly
        wrong for a later pass that must launch from those same nodes. This is the
        whole pass-boundary mechanism the tree needs: one explicit transition, called
        between passes under ``lock``, rather than per-pass gate bookkeeping smeared
        through the loop.
        """
        if any(node.gate.in_use for node in self.nodes.values()):
            raise RuntimeError("cannot regate while expansions hold gate reservations")
        self.gating = gating
        for node in self.nodes.values():
            node.gate = gating.gate(node)
        self.frontier.clear()
        for node_id in self.nodes:
            self.refresh(node_id)

    def record_failure(self, node_id: int, exc: BaseException) -> None:
        self.failures.append((node_id, repr(exc)))
        self.stats[STAT_FAILURES] += 1.0

    def terminals(self) -> list[Node[P]]:
        """Every node whose payload ended the episode naturally."""
        return [n for n in self.nodes.values() if n.payload.done]

    def best(self) -> Node[P] | None:
        """The highest-reward terminal node, or ``None`` if nothing finished.

        Ranked by the payload's own reward -- the one score every payload has. A caller
        ranking by anything else (a critic channel, a custom key) is choosing among
        :meth:`terminals` and can say so in its own terms.
        """
        return max(self.terminals(), key=lambda n: n.payload.reward, default=None)

    def describe(self, batch: Any = None) -> str:
        """The tree, one node per line, with any actor weights from ``batch`` joined.

        See :func:`~focus.export.format_tree` for the line format.
        """
        from focus.export import format_tree

        return format_tree(SchedulerView(self), batch)  # type: ignore[arg-type]

    def __getstate__(self) -> dict[str, Any]:
        """Pickle everything but the lock, which is minted fresh on the way back in.

        An untouched ``asyncio.Lock`` happens to pickle and a contended one does not,
        which made "can I save this tree?" depend on whether anything had ever waited
        on it. The same shape as :meth:`~focus.export.Batch.
        __getstate__`: a field that is a property of the *live* object is dropped from
        the wire format and rebuilt on arrival.

        This is for a **settled** tree -- one whose search has returned, written to
        disk to be re-scored later or handed to another process for analysis. It is not
        a way to share a running search: the restored lock is a different lock, so two
        processes holding halves of one tree would guard nothing, and the in-flight
        expansions that lock exists for do not cross the boundary either. Which is why
        :meth:`~focus.collector.Collector.search` hands back a tree that is already done
        searching.
        """
        return {key: value for key, value in self.__dict__.items() if key != "lock"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = asyncio.Lock()


@dataclass(frozen=True)
class SchedulerView[P: Payload]:
    """A read-only projection of :class:`SchedulerState` handed to policies.

    Policies observe the tree, the frontier, and the running stats but never mutate --
    all state changes go through the scheduler.
    """

    _state: SchedulerState[P]

    @property
    def stats(self) -> Mapping[str, float]:
        return MappingProxyType(self._state.stats)

    @property
    def failures(self) -> tuple[tuple[int, str], ...]:
        """Every dead expansion so far, as ``(node_id, repr(exc))``.

        Beside ``stats`` because the two are one account of the run and are read
        together -- a policy deciding whether the tree is worth more budget, an export
        handing a trainer what the search cost. A tuple, so what a reader takes away
        cannot change under it when the next expansion dies.
        """
        return tuple(self._state.failures)

    def nodes(self) -> list[Node[P]]:
        """Every node in the tree."""
        return list(self._state.nodes.values())

    def selectable(self, node: Node[P]) -> bool:
        return self._state.nodes.get(node.id) is node and self._state.selectable(node)

    @property
    def frontier_size(self) -> int:
        return len(self._state.frontier)

    def iter_frontier(self) -> Iterator[Node[P]]:
        """Iterate without allocating a frontier copy; hold the tree lock while
        using."""
        return (self._state.nodes[i] for i in self._state.frontier)

    def frontier(self) -> list[Node[P]]:
        """The selectable (forkable, live, free gate slot) nodes, in insertion order."""
        return [self._state.nodes[i] for i in self._state.frontier]

    def arrival(self, node: Node[P]) -> int:
        """When ``node`` last became selectable, as a position in arrival order."""
        return self._state.frontier[node.id]

    def node(self, node_id: int) -> Node[P] | None:
        """One node by id, or ``None`` -- how a policy walks toward the root."""
        return self._state.nodes.get(node_id)

    @property
    def root(self) -> Node[P]:
        """The prompt's own node -- the prefix every edge in this tree descends from.

        Always present (a state is created from one), so this is not an ``Optional`` a
        caller has to defend against. It is what an estimator reading a
        whole-trajectory return needs: a return measured from the start of the episode
        is only comparable to a baseline read at the start of the episode.
        """
        return self._state.nodes[self._state.root_id]

    def children(self, node: Node[P]) -> list[Node[P]]:
        """``node``'s direct children, in the order they were attached.

        :attr:`Node.children` holds ids; resolving them needs the node table, which a
        policy only reaches through this view.
        """
        return [self._state.nodes[i] for i in node.children]

    def terminals(self) -> list[Node[P]]:
        """Every node whose payload ended the episode (the terminal predicate)."""
        return self._state.terminals()

    def path_to(self, node: Node[P]) -> list[Node[P]]:
        """The line from the root down to ``node``, both ends included.

        The counterpart to :meth:`descendants`, and the one upward walk: a return read
        down a realized line, a report that indents by depth, an estimator asking what
        prefix an edge sits behind. A node whose parent chain does not reach the root --
        one from another tree, or one held past a state it was never in -- yields a path
        that does not start at the root, which is how a caller tells the difference.
        """
        line = [node]
        current = node
        while current.parent_id is not None:
            parent = self._state.nodes.get(current.parent_id)
            if parent is None:
                break
            line.append(parent)
            current = parent
        line.reverse()
        return line

    def descendants(self, node: Node[P]) -> Iterator[Node[P]]:
        """Every node below ``node``, depth-first and excluding ``node`` itself.

        The one subtree walk, because every caller was writing it again: a statistic
        over the finished trajectories under a prefix, a check that *some* path through
        one ended. Lazy, so the caller that only needs to know whether such a node
        exists stops at the first instead of materializing the subtree.
        """
        stack = list(self.children(node))
        while stack:
            current = stack.pop()
            yield current
            stack.extend(self.children(current))

    def __len__(self) -> int:
        return len(self._state.nodes)


__all__ = [
    "STAT_FAILURES",
    "STAT_NODES",
    "STAT_ROLLOUTS",
    "STAT_SCORE_FAILURES",
    "STAT_TURNS",
    "Node",
    "Payload",
    "SchedulerState",
    "SchedulerView",
]
