"""In-memory KV workspace + the storage tool that exposes it.

This backend *is* a path→text store. :class:`~...storage.StorageTool` executes model
calls by reading and writing that store (passed per-call as ``workspace=`` so forks
stay isolated). Other backends need not look like this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from focus.harness.workspace.base import Workspace, WorkspaceError
from focus.registry import register

if TYPE_CHECKING:
    from focus.harness.tools import Tool

_GUIDANCE = """\
You have a workspace -- durable memory, reached with the storage tool. It is the only \
thing that survives context compaction, and it costs you no context until you read a \
key back, so it is where the long, structured, precise material belongs: figures, \
dates, names, quotes, per-item records -- anything you would later have to be exact \
about. Keep the context itself for the short version: what is established, what is \
still open, and which keys hold the detail. Save each finding as you make it, and read \
memory back rather than re-deriving something you may already hold.\
"""

_COMPACTION_GUIDANCE = """\
Everything long, structured, or precise belongs in durable memory: figures, dates, \
names, quotes, per-item records -- anything that would have to stay exact. Put it \
there by actually calling the storage tool, once per finding, under a descriptive key; \
describing a save in your reply stores nothing. Memory survives this fold and every \
later one, and costs the agent no context until it reads a key back. It already holds \
what earlier rounds saved and is deliberately not in this prompt, so read it back \
before you store: correct an existing entry rather than save a second spelling of it.

Your final reply is the other half of the job, and it is short -- the orientation the \
agent re-reads every turn: what is established, what is still unknown, what to do \
next, and the names of the keys that hold the detail. Name keys; do not copy their \
contents back into it. Anything you neither store nor keep is gone.\
"""


@register(Workspace, "memory")
class InMemoryWorkspace(Workspace):
    """A path→text dict, exported to the model via :class:`~...storage.StorageTool`."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}

    def tools(self) -> Sequence[Tool]:
        from focus.harness.workspace.storage import StorageTool

        return (StorageTool(),)

    def guidance(self) -> str:
        return _GUIDANCE

    def compaction_guidance(self) -> str:
        return _COMPACTION_GUIDANCE

    def fork(self) -> InMemoryWorkspace:
        return type(self).deserialize(self.serialize())

    # -- KV store (used by StorageTool and by callers that inspect memory) -----------

    def read(self, path: str) -> str:
        try:
            return self._entries[path]
        except KeyError:
            raise WorkspaceError(f"no such path: {path!r}") from None

    def write(self, path: str, content: str) -> None:
        self._entries[path] = content

    def delete(self, path: str) -> None:
        try:
            del self._entries[path]
        except KeyError:
            raise WorkspaceError(f"no such path: {path!r}") from None

    def paths(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def exists(self, path: str) -> bool:
        return path in self._entries

    def clear(self) -> None:
        self._entries.clear()

    def update(self, entries: Mapping[str, str]) -> None:
        for path, content in entries.items():
            self.write(path, content)

    def init(self, entries: Mapping[str, str] | None = None) -> None:
        self.clear()
        if entries:
            self.update(entries)

    def snapshot(self) -> Mapping[str, str]:
        return self.serialize()

    def serialize(self) -> dict[str, str]:
        return dict(self._entries)

    @classmethod
    def deserialize(cls, state: Mapping[str, str]) -> InMemoryWorkspace:
        workspace = cls()
        workspace._entries = dict(state)
        return workspace

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and self.exists(path)

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["InMemoryWorkspace"]
