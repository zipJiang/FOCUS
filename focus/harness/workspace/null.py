"""The empty workspace: no store, no tools, fold summary is the only memory."""

from __future__ import annotations

from focus.harness.workspace.base import Workspace
from focus.registry import register

_COMPACTION_GUIDANCE = """\
There is no durable memory here: your reply is the only thing that survives this fold, \
so anything you leave out is gone for good. Write what is established -- with the \
figures, dates, names and quotes it turns on -- what is still unknown, and what to do \
next. Prefer precise over short; do not copy the transcript back wholesale.\
"""


#: Said after a fold and not before it: with no store behind the agent, the rollout's
#: system prompt has nothing to advertise, but once the transcript is gone the notes'
#: standing changed -- they are the whole record, and their gaps are gaps in the record
#: rather than things the agent may fill in from what it recalls.
_FOLDED_GUIDANCE = """\
There is no store behind you in this run -- these notes are the only record of your \
earlier work, and everything not in them is gone. Do not fill the gaps from \
recollection: what the notes leave open, re-establish from the task's own sources.\
"""


@register(Workspace, "null")
class NullWorkspace(Workspace):
    """No tools, no store -- only fold advice that the reply is the memory."""

    def fork(self) -> NullWorkspace:
        return NullWorkspace()

    def folded_guidance(self) -> str:
        return _FOLDED_GUIDANCE

    def compaction_guidance(self) -> str:
        return _COMPACTION_GUIDANCE


__all__ = ["NullWorkspace"]
