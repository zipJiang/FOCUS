"""Scheduler: drive many rollouts as a parallel tree search, pure over the tree.

A layer above the harness. The harness runs *one* rollout; the :class:`Scheduler`
orchestrates *many*: select a forkable node, ask an injected :class:`Expander` for its
children, attach them to a tree, and select again -- in bounded parallel, until a budget
or goal is met.

One package, :mod:`~focus.scheduler.core`: the live loop (tree, gating,
policies, expansion, orchestration, optional entropy/critic scoring) together with the
tree statistics a return is read off (:mod:`~focus.scheduler.core.returns` --
advantage, its baseline, its importance correction, and which paths ended).

The scheduler imports neither allocation nor preparation contracts. Allocation
builds scheduler policies; preparation reads settled tree views. Shared return
mathematics lives in core, while search objectives live in focus.allocation.
"""

from focus.scheduler.core import (
    PROVENANCE_KEY,
    AllNodes,
    AnchorProposal,
    Budget,
    ConcurrencyGating,
    CriticScheduler,
    CriticSelection,
    EdgeProvenance,
    Expander,
    Gate,
    GatingPolicy,
    MixtureProposal,
    Node,
    NodeScorer,
    Payload,
    ProposalPolicy,
    RolloutExpander,
    Scheduler,
    SchedulerState,
    SchedulerView,
    Terminals,
    Termination,
    WidthGating,
    edge_return,
    finished_below,
    finished_through,
    importance,
    provenance_of,
    rollout_text,
    stamp_provenance,
    value_baseline,
)
from focus.scheduler.core.allocation import (
    AllocationLedger,
    AllocationSession,
    BestFirstSession,
    FifoSession,
    RankedSession,
    Reservation,
)
from focus.scheduler.core.entropy import TokenEntropySession

__all__ = [
    "AllocationSession",
    "AllocationLedger",
    "Reservation",
    "FifoSession",
    "BestFirstSession",
    "RankedSession",
    "TokenEntropySession",
    "PROVENANCE_KEY",
    "AllNodes",
    "AnchorProposal",
    "Budget",
    "ConcurrencyGating",
    "EdgeProvenance",
    "CriticScheduler",
    "CriticSelection",
    "Expander",
    "Gate",
    "GatingPolicy",
    "MixtureProposal",
    "Node",
    "NodeScorer",
    "Payload",
    "ProposalPolicy",
    "RolloutExpander",
    "Scheduler",
    "SchedulerState",
    "SchedulerView",
    "Terminals",
    "Termination",
    "WidthGating",
    "edge_return",
    "finished_below",
    "finished_through",
    "importance",
    "provenance_of",
    "rollout_text",
    "stamp_provenance",
    "value_baseline",
]
