"""Context compaction: fold a grown conversation down, offloading to the workspace.

Two independent questions, deliberately kept apart:

* **When** -- a :class:`Trigger` (``prompt_tokens``, ``state_counter``, ``any``),
checked
  against the rendered prompt and the task state before each turn. Triggers compose, so
  "fold on length or on a spent per-round budget" is ``AnyOf(...)`` rather than a
  subclass.
* **How** -- a :class:`Compactor`, given a :class:`Compaction`: it saves what matters
  into the workspace and returns the messages the rollout continues from (the seed for a
  plain reset, or the seed plus what it chose to keep), along with any turns it
  generated getting there.

Concrete strategies live as sibling modules (``agentic``, …).

:class:`AgenticCompactor` is the general, LLM-driven strategy, and it is *the same
machinery as the rollout*: a :class:`~focus.harness.agent.Agent` over a
:class:`FoldEnv`, whose tools are the workspace's own. Its turns come back tagged
``"fold"`` and join the rollout's log, so a fold's tool-call turns are first-class
trainable data and an env author never has to know compaction exists.

The fold's ending convention -- reply with no tool call, and that reply is the context
to keep -- is the compactor's own, and no longer the rollout's: the rollout ends on an
explicit ``submit`` call (see :mod:`focus.harness.tools.submission`). The two
are
deliberately different, because the acts are. A rollout produces an *answer*, which is
worth typing and worth confirming; a fold produces the text the agent keeps working
from, so the reply simply is the payload, and asking for it through a tool would buy a
schema nobody reads at the cost of a compaction-only tool this design has always
avoided.
"""

from focus.harness.compaction.agentic import (
    DEFAULT_INSTRUCTION,
    AgenticCompactor,
)
from focus.harness.compaction.base import (
    Compaction,
    CompactionResult,
    Compactor,
)
from focus.harness.compaction.budget import build_budget
from focus.harness.compaction.fold_env import STORE_FIRST, FoldEnv, FoldState
from focus.harness.compaction.triggers import (
    AnyOf,
    PromptTokens,
    StateCounter,
    Trigger,
)

__all__ = [
    "AgenticCompactor",
    "AnyOf",
    "Compaction",
    "CompactionResult",
    "Compactor",
    "DEFAULT_INSTRUCTION",
    "FoldEnv",
    "FoldState",
    "PromptTokens",
    "STORE_FIRST",
    "StateCounter",
    "Trigger",
    "build_budget",
]
