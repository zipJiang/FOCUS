"""Context construction and effects, independent of task dynamics and allocation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from focus.codec import Message
from focus.generation.parsing.base import ToolTurn
from focus.harness.checkpoint import Effect
from focus.harness.env import StepResult
from focus.harness.rollout import AnyRollout
from focus.registry import register

if TYPE_CHECKING:
    from focus.harness.runner import Runner


@dataclass(frozen=True)
class Applied:
    rollout: AnyRollout
    transition: StepResult[Any] | None = None


class ContextProtocol:
    """Stateless protocol; branch-owned data belongs in rollout.execution.

    apply receives the already-recorded generation. Its effects reference that index,
    even when apply appends child generations. Only task operations return transitions.
    """

    manages_context = False

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return ()

    async def initialize(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        return rs

    def needs_prepare(self, rs: AnyRollout, runner: Runner[Any, Any]) -> bool:
        return False

    async def before(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        return rs

    async def compact(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        raise ValueError("Runner has no compactor configured")

    async def apply(
        self,
        rs: AnyRollout,
        action: Any,
        runner: Runner[Any, Any],
        *,
        ending: bool,
    ) -> Applied:
        transition = (
            await runner.env.finish(rs.state, action)
            if ending
            else await runner.env.step(rs.state, action, workspace=rs.workspace)
        )
        messages = () if ending or transition.done else transition.messages
        return Applied(
            replace(
                rs,
                state=transition.next_state,
                messages=(*rs.messages, *messages),
                history=(*rs.history, *messages),
            ),
            transition,
        )


@register(ContextProtocol, "append_only")
class AppendOnly(ContextProtocol):
    """Keep the task transcript as its live context."""


class ToolProtocol(ContextProtocol):
    """Serial operations; mixed task/protocol batches are rejected atomically."""

    manages_context = True

    async def operation(
        self,
        rs: AnyRollout,
        name: str,
        arguments: dict[str, Any],
        runner: Runner[Any, Any],
    ) -> tuple[AnyRollout, str, str]:
        raise NotImplementedError

    async def apply(
        self,
        rs: AnyRollout,
        action: Any,
        runner: Runner[Any, Any],
        *,
        ending: bool,
    ) -> Applied:
        names = {s["function"]["name"] for s in self.schemas()}
        if (
            ending
            or not isinstance(action, ToolTurn)
            or not any(c.name in names for c in action.tool_calls)
        ):
            return await super().apply(rs, action, runner, ending=ending)
        origin = len(rs.turns) - 1
        mixed = any(c.name not in names for c in action.tool_calls)
        for call in action.tool_calls:
            kind = ""
            before = rs.context_version
            try:
                if mixed:
                    raise ValueError(
                        "issue context operations separately from task tools"
                    )
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                rs, text, kind = await self.operation(rs, call.name, arguments, runner)
            except (ValueError, KeyError, TypeError) as exc:
                text = f"Invalid {call.name}: {exc}. Previous valid context preserved."
            message: Message = {"role": "tool", "name": call.name, "content": text}
            if call.call_id:
                message["tool_call_id"] = call.call_id
            rs = replace(
                rs, messages=(*rs.messages, message), history=(*rs.history, message)
            )
            if kind:
                rs = replace(
                    rs,
                    effects=(
                        *rs.effects,
                        Effect(
                            kind,
                            origin,
                            before,
                            rs.context_version,
                            call.name,
                        ),
                    ),
                )
        return Applied(rs)


def schema(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
