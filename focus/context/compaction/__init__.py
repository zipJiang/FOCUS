"""Model-driven context compaction: fold a grown conversation into notes.

:class:`Summarize` is a :class:`~focus.context.base.ContextManager` whose fold is a
bounded rollout of the agent's own policy over a :class:`FoldEnv`, whose tools are the
workspace's own. Its turns join the rollout's log tagged ``"fold"``, so a fold's
generations are trainable data. *When* to fold is a composable
:class:`~focus.context.triggers.Trigger`.
"""

from focus.context.compaction.agentic import DEFAULT_INSTRUCTION, Summarize
from focus.context.compaction.budget import build_budget
from focus.context.compaction.fold_env import STORE_FIRST, FoldEnv, FoldState

__all__ = [
    "DEFAULT_INSTRUCTION",
    "FoldEnv",
    "FoldState",
    "STORE_FIRST",
    "Summarize",
    "build_budget",
]
