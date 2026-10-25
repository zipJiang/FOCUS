"""A qwen-agent ``storage``-format tool over :class:`InMemoryWorkspace`.

qwen-agent ships a single ``storage`` tool that multiplexes ``put``/``get``/``delete``/
``scan`` over path-like keys and returns plain strings. :class:`StorageTool` mirrors it
exactly -- so the model sees the tool it was trained to use -- and executes those ops
against an :class:`InMemoryWorkspace` (passed per-call as ``workspace=`` so forks stay
isolated). Other workspace backends export their own tools; they need not speak KV.
"""

from __future__ import annotations

from typing import Any

from focus.harness.tools import BaseTool, ToolError
from focus.harness.workspace.base import Workspace, WorkspaceError
from focus.harness.workspace.memory import InMemoryWorkspace


class StorageTool(BaseTool):
    """qwen-agent's ``storage`` tool over an :class:`~...memory.InMemoryWorkspace`.

    Execution is here: ``put``/``get``/… map onto the workspace's KV methods. The
    rollout passes the per-branch workspace into :meth:`call` so forks stay isolated.
    """

    name = "storage"
    # The one place the memory discipline is stated. Tool schemas are rendered into
    # every prompt (``codec.render(messages, tools=...)``), so the rollout, the
    # compactor, and any future caller inherit it here instead of each re-writing it as
    # prose -- which is how three prompts came to say it three different ways, while a
    # fourth never mentioned memory at all.
    description = (
        "Store and retrieve text across the conversation -- durable scratch space, and "
        + "the only memory that survives context compaction. Save each finding under a "
        + "descriptive key. Read memory back ('scan' lists every entry, 'get' returns "
        + "one) rather than deriving a fact you may already hold. Put to an EXISTING "
        + "key to correct or extend it; create a key only for something genuinely "
        + "new, never "
        + "a second spelling of an entry you already have."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operate": {
                "type": "string",
                "enum": ["put", "get", "delete", "scan"],
                "description": (
                    "put (save), get (read one entry), delete, or scan (list every "
                    + "entry with its contents)."
                ),
            },
            "key": {
                "type": "string",
                "description": "The entry name, e.g. 'notes' or 'plan/step1'.",
            },
            "value": {
                "type": "string",
                "description": "The text to store (required for put).",
            },
        },
        "required": ["operate"],
    }

    def __init__(self, workspace: InMemoryWorkspace | None = None) -> None:
        self.workspace = workspace if workspace is not None else InMemoryWorkspace()

    def validate_workspace(self, workspace: Workspace) -> None:
        """Require the key-value operations used by the storage tool."""
        required = ("read", "write", "delete", "paths")
        if not all(callable(getattr(workspace, name, None)) for name in required):
            raise ValueError(
                "storage requires a workspace with read, write, delete, and paths; "
                f"received {type(workspace).__name__}. Pass workspace= with a "
                "compatible store when starting the rollout."
            )

    def call(
        self,
        params: str | dict[str, Any],
        *,
        workspace: InMemoryWorkspace | None = None,
        **_: object,
    ) -> str:
        # The rollout and compactor pass the workspace per call (dispatch forwards it);
        # fall back to the constructor default only for standalone / direct use.
        ws = workspace if workspace is not None else self.workspace
        data = self.verify_args(params)
        operate = data["operate"]
        key = str(data.get("key", "")).lstrip("/")
        if operate == "put":
            if "value" not in data:
                raise ToolError("put requires a 'value'")
            ws.write(key, str(data["value"]))
            return f"Successfully saved {key}."
        if operate == "get":
            try:
                return ws.read(key)
            except WorkspaceError:
                return f"Get Failed: {key} does not exist."
        if operate == "delete":
            try:
                ws.delete(key)
                return f"Successfully deleted {key}."
            except WorkspaceError:
                return f"Delete Failed: {key} does not exist."
        if operate == "scan":
            paths = ws.paths()
            return (
                "\n".join(f"{p}: {ws.read(p)}" for p in paths)
                if paths
                else "(workspace is empty)"
            )
        raise ToolError(f"unknown operate {operate!r}; use put/get/delete/scan")


__all__ = ["StorageTool"]
