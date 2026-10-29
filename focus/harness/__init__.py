"""Harness: the agent-facing runtime around the generation backends."""

from focus.harness.agent import DEFAULT_FINISH_PROMPT, Agent, Execution
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
    FunctionTool,
    Tool,
    ToolError,
    dispatch,
    tool,
    tool_schema,
)
from focus.harness.tools.budget import RoundBudget
from focus.harness.tools.environment import ToolEnv
from focus.harness.tools.submission import SUBMIT, SubmitFields, SubmitTool
from focus.harness.transcript import (
    flatten_tool_calls,
    format_transcript,
)
from focus.harness.turn import Turn, TurnTag, logprob, trainable_logprobs
from focus.harness.workspace import (
    InMemoryWorkspace,
    NullWorkspace,
    StorageTool,
    Workspace,
    WorkspaceError,
)

__all__ = [
    "AnyRollout",
    "BaseTool",
    "Checkpointer",
    "CheckpointInfo",
    "CorruptCheckpoint",
    "MemoryCheckpointer",
    "SQLiteCheckpointer",
    "DEFAULT_FINISH_PROMPT",
    "Env",
    "InMemoryWorkspace",
    "NullWorkspace",
    "PackedSequence",
    "RolloutState",
    "RoundBudget",
    "Agent",
    "Execution",
    "SUBMIT",
    "StepResult",
    "StorageTool",
    "SubmitFields",
    "SubmitTool",
    "TASK",
    "Tool",
    "ToolEnv",
    "ToolError",
    "Turn",
    "TurnTag",
    "Workspace",
    "WorkspaceError",
    "dispatch",
    "tool",
    "FunctionTool",
    "flatten_tool_calls",
    "format_transcript",
    "logprob",
    "regions",
    "tool_schema",
    "trainable_logprobs",
]
