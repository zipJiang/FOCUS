"""Interfaces and building blocks for extending FOCUS.

Subclass these to add a search priority (:class:`Priority`), an actor recipe
(:class:`Estimator`), context management (:class:`ContextManager`, :class:`Trigger`),
tools (:class:`BaseTool`), workspaces, branch-point rules, or persistence.
"""

from focus.harness.agent import DEFAULT_FINISH_PROMPT
from focus.harness.checkpoint import BranchPoints, EveryDecision, SparseCheckpoints
from focus.harness.context import (
    Compaction,
    CompactionResult,
    ContextError,
    ContextManager,
    ContextTools,
    IncompleteCompactionError,
)
from focus.harness.packing import PackedSequence
from focus.harness.persistence import (
    Checkpointer,
    MemoryCheckpointer,
    SQLiteCheckpointer,
)
from focus.harness.rollout import RolloutState
from focus.harness.tools import BaseTool, FunctionTool, Tool
from focus.harness.tools.budget import RoundBudget
from focus.harness.tools.submission import SUBMIT, SubmitFields, SubmitTool
from focus.harness.triggers import AnyOf, PromptTokens, StateCounter, Trigger
from focus.harness.turn import Turn
from focus.harness.workspace import (
    InMemoryWorkspace,
    NullWorkspace,
    StorageTool,
    Workspace,
)
from focus.preparation import Advantage, Edge, Estimator
from focus.search import NodeInfo, Priority
from focus.values import NodeValues, TreeValues, tree_values

__all__ = [
    # Search and preparation
    "Priority",
    "NodeInfo",
    "Estimator",
    "Edge",
    "Advantage",
    "TreeValues",
    "NodeValues",
    "tree_values",
    # Context management
    "ContextManager",
    "ContextTools",
    "ContextError",
    "Compaction",
    "CompactionResult",
    "IncompleteCompactionError",
    "Trigger",
    "PromptTokens",
    "StateCounter",
    "AnyOf",
    # Tools, task state, and workspaces
    "BaseTool",
    "FunctionTool",
    "Tool",
    "SubmitTool",
    "SUBMIT",
    "SubmitFields",
    "RoundBudget",
    "Workspace",
    "InMemoryWorkspace",
    "NullWorkspace",
    "StorageTool",
    # Rollouts, branch points, and persistence
    "RolloutState",
    "Turn",
    "PackedSequence",
    "BranchPoints",
    "EveryDecision",
    "SparseCheckpoints",
    "Checkpointer",
    "MemoryCheckpointer",
    "SQLiteCheckpointer",
    "DEFAULT_FINISH_PROMPT",
]
