"""Model-driven context compaction: fold a grown conversation into notes.

:class:`Summarize` is a :class:`~focus.harness.context.ContextManager` whose fold is a
bounded rollout of the agent's own policy over a :class:`FoldEnv`, whose tools are the
workspace's own. Its turns join the rollout's log tagged ``"fold"``, so a fold's
generations are trainable data. *When* to fold is a composable
:class:`~focus.harness.triggers.Trigger`.
"""

from focus.harness.compaction.agentic import DEFAULT_INSTRUCTION, Summarize
from focus.harness.compaction.budget import build_budget
from focus.harness.compaction.fold_env import STORE_FIRST, FoldEnv, FoldState

__all__ = [
    "DEFAULT_INSTRUCTION",
    "FoldEnv",
    "FoldState",
    "STORE_FIRST",
    "Summarize",
    "build_budget",
]
