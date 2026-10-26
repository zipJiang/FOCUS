"""Scheduler: drive many rollouts as a parallel tree search, pure over the tree.

A layer above the harness. The harness runs *one* rollout; the :class:`Scheduler`
orchestrates *many*: select a forkable node, ask an injected :class:`Expander` for its
children, attach them to a tree, and select again -- in bounded parallel, until a budget
or goal is met.

One package, :mod:`~focus.scheduler.core`: the live loop (tree, gating,
policies, expansion, orchestration, optional entropy/critic scoring) together with the
tree statistics a return is read off (:mod:`~focus.scheduler.core.returns` --
advantage, its baseline, and which paths ended).

The scheduler imports neither allocation nor preparation contracts. Allocation
builds scheduler policies; preparation reads settled tree views. Shared return
mathematics lives in core, while search objectives live in focus.allocation.
"""

from focus.scheduler.core import (
    PASS_KEY,
    Budget,
    ConcurrencyGating,
    Expander,
    Gate,
    GatingPolicy,
    Node,
    NodeScorer,
    Payload,
    RolloutExpander,
    Scheduler,
    SchedulerState,
    SchedulerView,
    Termination,
    WidthGating,
    edge_return,
    finished_below,
    finished_through,
    value_baseline,
)
from focus.scheduler.core.allocation import (
    AllocationLedger,
    AllocationSession,
    FifoSession,
    RankedSession,
    Reservation,
)
from focus.scheduler.core.entropy import TokenEntropySession

__all__ = [
    "PASS_KEY",
    "AllocationSession",
    "AllocationLedger",
    "Reservation",
    "FifoSession",
    "RankedSession",
    "TokenEntropySession",
    "Budget",
    "ConcurrencyGating",
    "Expander",
    "Gate",
    "GatingPolicy",
    "Node",
    "NodeScorer",
    "Payload",
    "RolloutExpander",
    "Scheduler",
    "SchedulerState",
    "SchedulerView",
    "Termination",
    "WidthGating",
    "edge_return",
    "finished_below",
    "finished_through",
    "value_baseline",
]
