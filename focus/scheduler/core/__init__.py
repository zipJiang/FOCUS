"""Live scheduling loop: tree state, policies, expansion, and orchestration.

Owns everything a :class:`Scheduler` run needs -- every search pass -- plus the
statistics a return is read off the tree with (:mod:`.returns`). Never imports the
layers above it: an estimator pulls core, not the reverse, and where another
rollout is worth buying is an allocator's own question, asked in
:mod:`focus.allocation.evidence`.

The runner owns the completion loop. Proposal policies choose the sampling law;
expanders return its fold checkpoints and endpoint with edge provenance.
"""

from focus.scheduler.core.allocation import (
    AllocationLedger,
    AllocationSession,
    BestFirstSession,
    FifoSession,
    RankedSession,
    Reservation,
)
from focus.scheduler.core.critic import (
    AllNodes,
    CriticScheduler,
    CriticSelection,
    NodeScorer,
    Terminals,
    rollout_text,
)
from focus.scheduler.core.entropy import TokenEntropySession, token_entropies
from focus.scheduler.core.execution import (
    PROVENANCE_KEY,
    EdgeProvenance,
    Expander,
    RolloutExpander,
    drawn_from_anchor,
    provenance_of,
    stamp_provenance,
)
from focus.scheduler.core.gating import (
    ConcurrencyGating,
    Gate,
    GatingPolicy,
    WidthGating,
)
from focus.scheduler.core.policies import (
    Budget,
    Termination,
)
from focus.scheduler.core.proposal import (
    AnchorProposal,
    MixtureProposal,
    ProposalPolicy,
)
from focus.scheduler.core.returns import (
    edge_return,
    finished_below,
    finished_through,
    importance,
    value_baseline,
)
from focus.scheduler.core.scheduler import Scheduler
from focus.scheduler.core.tree import (
    Node,
    Payload,
    SchedulerState,
    SchedulerView,
)

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
    "CriticScheduler",
    "CriticSelection",
    "EdgeProvenance",
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
    "drawn_from_anchor",
    "edge_return",
    "finished_below",
    "finished_through",
    "importance",
    "provenance_of",
    "rollout_text",
    "stamp_provenance",
    "token_entropies",
    "value_baseline",
]
