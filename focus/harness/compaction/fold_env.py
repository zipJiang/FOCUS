"""The environment a fold runs in -- so a compaction *is* a rollout.

A fold is an agent doing a bounded piece of tool work and then saying what it kept,
which is exactly what an :class:`Env` describes -- so the ordinary
:class:`~focus.harness.agent.Agent` drives it, and the compactor is left
owning only its *strategy* (the instruction, the budget, the trigger). The alternative
is a second copy of the agent loop inside the compactor -- generate, parse, dispatch,
observe, finish, with its own nudge and its own budget handling -- drifting from the one
every rollout uses.

The protocol is the older, tool-free one: **a reply with no tool call ends the fold, and
its text is the context to keep.** That is deliberately not the rollout's ``submit``
convention (see the package docstring): a rollout produces an answer worth typing and
worth confirming, while a fold produces the text the agent keeps working from, so the
reply simply *is* the payload.
:class:`~focus.harness.tools.environment.ToolEnv` already spells that
convention ``submit=None``, so this env is one line of configuration plus three hooks.

One nudge, not a loop: the convention that ends a fold ("no tool call = done") is also
the cheapest turn to produce, so a model asked for a summary writes the whole thing in
one go -- keys and all -- having stored none of it. The first such reply is sent back
once; a fold that called *some* tool (even just reading memory) is taken at its word.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from focus.generation.parsing import ToolTurn
from focus.harness.env import StepResult
from focus.harness.tools import Tool
from focus.harness.tools.environment import ToolEnv
from focus.harness.workspace import Workspace

#: Sent once, when a fold's first reply ends it without ever having called a tool.
STORE_FIRST = (
    "You replied without storing anything. Everything not in durable memory is lost "
    + "when this transcript is discarded, and a key you only named in your reply does "
    + "not exist. Save what matters by calling the storage tool now; then reply with "
    + "the "
    + "context to keep."
)


@dataclass(frozen=True)
class FoldState:
    """What a fold accumulates: the kept text, and the two facts the nudge needs."""

    #: The context to keep -- filled from the ending turn's visible text.
    kept: str = ""
    #: Whether any tool ran. A fold that touched memory is not nudged.
    used_tools: bool = False
    #: Whether the store-first nudge was already sent. At most one, ever.
    nudged: bool = False


class FoldEnv(ToolEnv[FoldState]):
    """A bounded tool-use episode that ends on a no-tool reply, keeping its text.

    ``tools`` come from the workspace (:meth:`Workspace.compaction_tools`), so a KV
    store offers ``storage`` and a :class:`~...workspace.null.NullWorkspace` offers
    nothing -- in which case there is no nudge either, because there would be nothing to
    nudge toward and the fold summary *is* the only memory.
    """

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        super().__init__(tools, submit=None)

    async def initial_state(self) -> FoldState:
        return FoldState()

    async def step(
        self,
        state: FoldState,
        turn: ToolTurn,
        *,
        workspace: Workspace | None = None,
    ) -> StepResult[FoldState]:
        """Dispatch tools, or end -- unless this is the one reply worth questioning."""
        if (
            not turn.tool_calls
            and self._by_name  # nothing to store into: nothing to ask for
            and not state.used_tools
            and not state.nudged
        ):
            return StepResult(
                replace(state, nudged=True),
                messages=({"role": "user", "content": STORE_FIRST},),
            )
        return await super().step(state, turn, workspace=workspace)

    def on_tool_calls(self, state: FoldState, turn: ToolTurn) -> FoldState:
        """Record that memory was touched."""
        del turn
        return replace(state, used_tools=True)

    def on_submit(
        self, state: FoldState, turn: ToolTurn, payload: dict[str, Any]
    ) -> FoldState:
        """The fold's ending: its visible text is the context to keep.

        Reached from both endings -- a voluntary no-tool reply, and the forced one
        :meth:`~focus.harness.agent.Agent.finish` drives when the
        interaction budget is spent -- so there is one place the kept text is written.
        """
        del payload
        return replace(state, kept=turn.text)


__all__ = ["STORE_FIRST", "FoldEnv", "FoldState"]
