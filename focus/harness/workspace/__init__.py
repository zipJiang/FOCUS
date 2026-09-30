"""Agent workspace: a small read/write scratch space for LLM inference."""

from focus.harness.workspace.base import Workspace, WorkspaceError
from focus.harness.workspace.memory import InMemoryWorkspace
from focus.harness.workspace.null import NullWorkspace
from focus.harness.workspace.storage import StorageTool

__all__ = [
    "InMemoryWorkspace",
    "NullWorkspace",
    "StorageTool",
    "Workspace",
    "WorkspaceError",
]
