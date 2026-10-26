"""Harness: the agent-facing runtime around the generation backends."""

from focus.harness.agent import DEFAULT_FINISH_PROMPT, Agent, Execution
from focus.harness.compaction import FoldEnv, FoldState, Summarize, build_budget
from focus.harness.context import (
    AppendOnly,
    Compaction,
    CompactionResult,
    ContextError,
    ContextManager,
    ContextTools,
    IncompleteCompactionError,
    Segments,
)
from focus.harness.env import Env, StepResult
from focus.harness.packing import PackedSequence, regions
from focus.harness.persistence import (
    Checkpointer,
    CheckpointInfo,
    CorruptCheckpoint,
    MemoryCheckpointer,
    SQLiteCheckpointer,
)
from focus.harness.rollout import TASK, AnyRollout, RolloutState
from focus.harness.tools import (
    BaseTool,
    Tool,
    ToolError,
    dispatch,
    tool_schema,
)
from focus.harness.tools.budget import RoundBudget
from focus.harness.tools.environment import ToolEnv
from focus.harness.tools.submission import SUBMIT, SubmitFields, SubmitTool
from focus.harness.transcript import (
    flatten_tool_calls,
    format_transcript,
)
from focus.harness.triggers import AnyOf, PromptTokens, StateCounter, Trigger
from focus.harness.turn import Turn, TurnTag, logprob, trainable_logprobs
from focus.harness.workspace import (
    InMemoryWorkspace,
    NullWorkspace,
    StorageTool,
    Workspace,
    WorkspaceError,
)

__all__ = [
    "AnyOf",
    "AppendOnly",
    "ContextError",
    "ContextManager",
    "ContextTools",
    "IncompleteCompactionError",
    "Segments",
    "Summarize",
    "AnyRollout",
    "BaseTool",
    "Compaction",
    "CompactionResult",
    "Checkpointer",
    "CheckpointInfo",
    "CorruptCheckpoint",
    "MemoryCheckpointer",
    "SQLiteCheckpointer",
    "DEFAULT_FINISH_PROMPT",
    "Env",
    "FoldEnv",
    "FoldState",
    "InMemoryWorkspace",
    "NullWorkspace",
    "PackedSequence",
    "PromptTokens",
    "RolloutState",
    "RoundBudget",
    "Agent",
    "Execution",
    "SUBMIT",
    "StateCounter",
    "StepResult",
    "StorageTool",
    "SubmitFields",
    "SubmitTool",
    "TASK",
    "Tool",
    "ToolEnv",
    "ToolError",
    "Trigger",
    "Turn",
    "TurnTag",
    "Workspace",
    "WorkspaceError",
    "build_budget",
    "dispatch",
    "flatten_tool_calls",
    "format_transcript",
    "logprob",
    "regions",
    "tool_schema",
    "trainable_logprobs",
]
