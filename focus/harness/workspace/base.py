"""The workspace: per-rollout scratch the model reaches through tools.

A workspace owns three things:

* **tools** -- :meth:`tools` / :meth:`compaction_tools`: schemas plus the ``call``
  implementations the model may invoke. How a call is executed is the tool's job
  (often closing over, or receiving, this workspace via
  ``dispatch(..., workspace=self)``).
* **guidance** -- :meth:`guidance` / :meth:`folded_guidance` /
  :meth:`compaction_guidance`: prose for the agent, for the agent once it has folded,
  and for the fold itself. Empty means "say nothing."
* **fork** -- an independent copy for branch isolation.

There is no uniform store API on the base. A KV workspace
(:class:`~...memory.InMemoryWorkspace`) exposes ``read``/``write`` for its
:class:`~...storage.StorageTool`; a custom workspace can expose whatever its tools need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from focus.harness.tools import Tool


class WorkspaceError(Exception):
    """Raised by a workspace or its tools when an operation cannot complete."""


class Workspace(ABC):
    """Per-rollout scratch: tools, guidance, and fork -- nothing else mandated."""

    def tools(self) -> Sequence[Tool]:
        """Agent-facing tools. Base returns none; a backend with tools overrides."""
        return ()

    def compaction_tools(self) -> Sequence[Tool]:
        """Tools offered to a fold. Default: the same as :meth:`tools`."""
        return self.tools()

    def guidance(self) -> str:
        """How the agent should use this workspace during the rollout.

        Goes in the system prompt, and -- via :meth:`folded_guidance`, which defaults to
        it -- in the post-fold continue text. Empty → omitted.
        """
        return ""

    def folded_guidance(self) -> str:
        """What the *post-fold* context says about memory. Empty → omitted.

        Defaults to :meth:`guidance`, because a workspace's standing advice usually
        still holds once the transcript is gone. Override where the fold changes what is
        true: under :class:`~...workspace.null.NullWorkspace` there is nothing durable
        behind the agent, so the fold's own notes became the only surviving record --
        which is worth saying after a fold and pointless before one.
        """
        return self.guidance()

    def compaction_guidance(self) -> str:
        """Advice for a fold. Empty → omitted from the compactor instruction."""
        return ""

    def snapshot(self) -> Mapping[str, str]:
        """Inspectable contents, if this backend has any. Default: empty.

        For metrics and debugging only -- the model reaches the workspace through
        :meth:`tools`, not through this map. KV backends override; others leave it
        empty.
        """
        return {}

    @abstractmethod
    def fork(self) -> Workspace:
        """An independent copy that starts equal and then diverges."""


__all__ = ["Workspace", "WorkspaceError"]
