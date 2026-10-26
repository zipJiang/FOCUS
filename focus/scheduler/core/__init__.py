"""Live scheduling loop: tree state, policies, expansion, and orchestration.

Owns everything a :class:`Scheduler` run needs -- every search pass -- plus the
statistics a return is read off the tree with (:mod:`.returns`). Never imports the
layers above it: an estimator pulls core, not the reverse, and where another
rollout is worth buying is an allocator's own question, asked in
:mod:`focus.allocation.evidence`.

The agent owns the completion loop; expanders return its fold checkpoints and
endpoint with the search pass that produced them.
"""

from focus.scheduler.core.allocation import (
    AllocationLedger,
    AllocationSession,
    FifoSession,
    RankedSession,
    Reservation,
)
from focus.scheduler.core.critic import (
    NodeScorer,
)
from focus.scheduler.core.entropy import token_entropies
from focus.scheduler.core.execution import (
    PASS_KEY,
    Expander,
    RolloutExpander,
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
from focus.scheduler.core.returns import (
    edge_duration,
    edge_return,
    finished_below,
    finished_through,
)
from focus.scheduler.core.scheduler import Scheduler
from focus.scheduler.core.tree import (
    Node,
    Payload,
    SchedulerState,
    SchedulerView,
)

__all__ = [
    "PASS_KEY",
    "AllocationSession",
    "AllocationLedger",
    "Reservation",
    "FifoSession",
    "RankedSession",
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
    "edge_duration",
    "edge_return",
    "finished_below",
    "finished_through",
    "token_entropies",
]
