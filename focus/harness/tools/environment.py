"""Model-free tool dispatch with a single, explicit submission."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import is_dataclass, replace
from typing import Any, TypeVar, final

from focus.generation.parsing import ToolCall, ToolTurn
from focus.harness.env import Env, StepResult
from focus.harness.tools.base import Tool, ToolError, tool_schema
from focus.harness.tools.dispatch import dispatch, tool_reply
from focus.harness.tools.submission import (
    SUBMIT,
    SubmitTool,
    has_submit_fields,
    submit_replies,
    submitted_call,
)
from focus.harness.workspace import Workspace

S = TypeVar("S")


class ToolEnv(Env[S, ToolTurn], ABC):
    """Dispatch tools, or end on the first valid submit.

    No-tool replies receive feedback until the runner's budget expires. With
    ``submit=None``, visible text instead ends the task (used by fold environments).
    Task states may optionally carry an ``answer`` and a ``calls_remaining`` budget.
    No submission or reminder counters are required.
    """

    def __init__(
        self, tools: Sequence[Tool], *, submit: SubmitTool | None = SUBMIT
    ) -> None:
        self._submit = submit
        self._tools = [*tools, submit] if submit is not None else list(tools)
        self._by_name = {t.name: t for t in tools}

    def validate_workspace(self, workspace: Workspace) -> None:
        """Run each tool's workspace check, when it provides one.

        Tools only need the structural Tool interface. Third-party tools without
        this hook remain usable; their environment may add its own checks.
        """
        for tool in self._tools:
            validate = getattr(tool, "validate_workspace", None)
            if callable(validate):
                validate(workspace)

    @final
    async def reset(self) -> S:
        """Validate writable task fields before generation starts."""
        state = await self.initial_state()
        if hasattr(state, "calls_remaining") and not is_dataclass(state):
            raise TypeError(
                f"{type(state).__name__} carries calls_remaining but is not a "
                "dataclass; use @dataclass(frozen=True) and RoundBudget."
            )
        return state

    @abstractmethod
    async def initial_state(self) -> S:
        """Build fresh task state."""
        ...

    @property
    def tools(self) -> tuple[Tool, ...]:
        """Registered tools, including the intercepted submit tool."""
        return tuple(self._tools)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas for the policy's template or native API."""
        return [tool_schema(t) for t in self._tools]

    def _ended(
        self, state: S, turn: ToolTurn, payload: dict[str, Any]
    ) -> StepResult[S]:
        return StepResult(
            self.on_submit(state, turn, payload),
            reward_outcome=self.reward(state, turn),
            done=True,
        )

    async def step(
        self, state: S, turn: ToolTurn, *, workspace: Workspace | None = None
    ) -> StepResult[S]:
        submitted = submitted_call(turn, self._submit)
        if submitted is not None:
            return self._submit_step(state, turn, submitted)
        if not turn.tool_calls:
            if self._submit is None:
                return self._ended(state, turn, {})
            # Logical feedback, not a new query: reasoning-preserving templates
            # render this as a tool response; native policies adapt it for their API.
            return StepResult(
                state,
                messages=(
                    {
                        "role": "tool",
                        "content": "No answer was recorded. "
                        "Call a tool to continue working, "
                        f"or call {self._submit.name} with your final answer.",
                    },
                ),
            )
        results = await asyncio.gather(
            *(
                dispatch(self._by_name, call, workspace=workspace)
                for call in turn.tool_calls
            )
        )
        messages = tuple(
            tool_reply(call, result)
            for call, result in zip(turn.tool_calls, results, strict=True)
        )
        return StepResult(self.on_tool_calls(state, turn), messages=messages)

    def _submit_step(self, state: S, turn: ToolTurn, call: ToolCall) -> StepResult[S]:
        """Submit takes precedence; sibling calls are never dispatched."""
        assert self._submit is not None
        try:
            payload = self._submit.verify_args(call.arguments)
        except ToolError as exc:
            return StepResult(
                state, messages=submit_replies(turn, call, f"error: {exc}")
            )
        return self._ended(state, turn, payload)

    async def finish(self, state: S, turn: ToolTurn) -> StepResult[S]:
        """Read the final submission without executing tools or retrying errors."""
        if self._submit is None:
            return self._ended(state, turn, {})
        call = submitted_call(turn, self._submit)
        if call is not None:
            try:
                payload = self._submit.verify_args(call.arguments)
            except ToolError:
                pass
            else:
                return self._ended(state, turn, payload)
        # An unsubmitted final turn is unanswered, regardless of visible prose.
        if has_submit_fields(state):
            state = replace(state, answer=None)  # type: ignore[type-var]
        return StepResult(state, done=True)

    def reward(self, state: S, turn: ToolTurn) -> float:
        """Score a valid submission (or a text ending when submit is disabled)."""
        return 0.0

    @property
    def answer_field(self) -> str:
        """The convenience answer key from the submit schema."""
        return self._submit.field if self._submit is not None else "answer"

    def on_submit(self, state: S, turn: ToolTurn, payload: dict[str, Any]) -> S:
        """Record a valid answer. Text is used only with submit disabled."""
        if not has_submit_fields(state):
            return state
        answer = payload.get(self.answer_field, "") if self._submit else turn.text
        return replace(state, answer=str(answer))  # type: ignore[type-var]

    def on_tool_calls(self, state: S, turn: ToolTurn) -> S:
        """Charge optional per-round budget once per dispatched tool call."""
        remaining = getattr(state, "calls_remaining", None)
        if remaining is None:
            return state
        return replace(  # type: ignore[type-var]
            state, calls_remaining=remaining - len(turn.tool_calls)
        )


__all__ = ["ToolEnv"]
