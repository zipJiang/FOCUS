"""Context construction and effects, independent of task dynamics and allocation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from focus.codec import Message
from focus.execution import Usage
from focus.generation import GenerateResult, PreparedPrompt
from focus.generation.parsing.base import ToolTurn
from focus.harness.checkpoint import Effect
from focus.harness.env import StepResult
from focus.harness.rollout import AnyRollout
from focus.harness.tools.dispatch import tool_reply
from focus.registry import register

if TYPE_CHECKING:
    from focus.harness.runner import Runner


class ProtocolError(ValueError):
    """Invalid model action; safe to return as corrective feedback."""


@dataclass(frozen=True)
class Applied:
    rollout: AnyRollout
    transition: StepResult[Any] | None = None


class ContextProtocol:
    """Shared protocol; branch-owned data belongs in the rollout state.

    apply receives the already-recorded generation. Its effects reference that index,
    and only task operations return transitions.
    """

    manages_context = False

    def prepare(self, rs: AnyRollout, runner: Runner[Any, Any]) -> PreparedPrompt:
        """Prepare the model input from this branch's current context."""
        return runner.policy.prepare(rs.messages, tools=runner.tools)

    def record_reply(
        self,
        result: GenerateResult,
        messages: list[Message],
        runner: Runner[Any, Any],
    ) -> None:
        """Append the generated reply to the branch's live conversation."""
        runner.policy.record_reply(result, messages)

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
        if not ending and isinstance(action, ToolTurn):
            runner.services.usage.setdefault(rs.episode_id, Usage()).tool_calls += len(
                action.tool_calls
            )
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
                context_refs=(
                    *rs.context_refs,
                    *range(len(rs.history), len(rs.history) + len(messages)),
                ),
            ),
            transition,
        )


@register(ContextProtocol, "append_only")
class AppendOnly(ContextProtocol):
    """Retain context as chat turns or exact segments of one assistant response.

    ``continuation=True`` preserves original generated token IDs between requests.
    It requires an exact token-input policy and uninterrupted assistant output.
    """

    def __init__(self, *, continuation: bool = False) -> None:
        self.continuation = continuation

    def _check_policy(self, runner: Runner[Any, Any]) -> None:
        if self.continuation and (
            not runner.policy.provides_exact_tokens
            or runner.policy.provides_native_tool_calls
        ):
            raise ValueError("Token continuation requires an exact token-input policy")

    async def initialize(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        self._check_policy(runner)
        return rs

    def _boundary(self, messages: Sequence[Message], runner: Runner[Any, Any]) -> int:
        boundary = next(
            (
                i
                for i, message in enumerate(messages)
                if "focus_continuation_tokens" in message
            ),
            len(messages),
        )
        for message in messages[boundary:]:
            if (
                message.get("role") != "assistant"
                or "focus_continuation_tokens" not in message
            ):
                raise ValueError(
                    "Token continuation requires uninterrupted assistant output"
                )
            tokens = message["focus_continuation_tokens"]
            if not isinstance(tokens, (tuple, list)) or any(
                type(token) is not int for token in tokens
            ):
                raise ValueError("Continuation metadata must contain integer token IDs")
            if message.get("content") != runner.policy.decode(
                tokens, skip_special_tokens=True
            ):
                raise ValueError(
                    "Continuation text was edited; its token IDs are unchanged"
                )
        return boundary

    def prepare(self, rs: AnyRollout, runner: Runner[Any, Any]) -> PreparedPrompt:
        """Render chat context, or append exact segments to the initial prompt."""
        if not self.continuation:
            return super().prepare(rs, runner)
        self._check_policy(runner)
        boundary = self._boundary(rs.messages, runner)
        initial = runner.policy.prepare(rs.messages[:boundary], tools=runner.tools)
        tokens = initial.tokens + tuple(
            token
            for message in rs.messages[boundary:]
            for token in message["focus_continuation_tokens"]
        )
        return replace(
            initial,
            tokens=tokens,
            _messages=deepcopy(rs.messages),
            exact_continuation=True,
        )

    def record_reply(
        self,
        result: GenerateResult,
        messages: list[Message],
        runner: Runner[Any, Any],
    ) -> None:
        """Keep exact token IDs with each assistant segment in continuation mode."""
        if self.continuation and (
            not result.exact_generation
            or result.native_tool_calls
            or result.native_output
        ):
            raise ValueError("Token continuation requires an exact text completion")
        super().record_reply(result, messages, runner)
        if self.continuation:
            messages[-1]["focus_continuation_tokens"] = tuple(result.tokens)


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
        contracts = {
            s["function"]["name"]: s["function"]["parameters"] for s in self.schemas()
        }
        if (
            ending
            or not isinstance(action, ToolTurn)
            or not any(c.name in contracts for c in action.tool_calls)
        ):
            return await super().apply(rs, action, runner, ending=ending)
        # Native adapters require the active call and all of its replies to remain
        # contiguous. Edit only settled context, then restore this exchange intact.
        # History still records every original message at its stable index.
        exchange: tuple[Message, ...] = ()
        exchange_refs: tuple[int | None, ...] = ()
        if runner.policy.provides_native_tool_calls and rs.messages[-1].get(
            "tool_calls"
        ):
            exchange = rs.messages[-1:]
            exchange_refs = rs.context_refs[-1:]
            rs = replace(
                rs, messages=rs.messages[:-1], context_refs=rs.context_refs[:-1]
            )
        origin = len(rs.turns) - 1
        mixed = any(c.name not in contracts for c in action.tool_calls)
        for index, call in enumerate(action.tool_calls):
            kind = ""
            before = rs.context_version
            try:
                if rs.truncated:
                    raise ProtocolError(
                        "rollout is truncated; no further operations can run"
                    )
                if mixed:
                    raise ProtocolError(
                        "issue context operations separately from task tools"
                    )
                try:
                    arguments = json.loads(call.arguments)
                except ValueError as exc:
                    raise ProtocolError(str(exc)) from exc
                if not isinstance(arguments, dict):
                    raise ProtocolError("tool arguments must be an object")
                contract = contracts[call.name]
                missing = set(contract.get("required", ())) - arguments.keys()
                if missing:
                    raise ProtocolError(f"missing arguments: {sorted(missing)}")
                candidate, text, effect_kind = await self.operation(
                    rs, call.name, arguments, runner
                )
                # Complete pending responses provisionally for validation only.
                # Commit neither context nor execution state if the edit is invalid.
                context = candidate.messages
                if exchange:
                    context = (
                        *context,
                        *exchange,
                        *(tool_reply(c, "") for c in action.tool_calls[index:]),
                    )
                try:
                    runner.policy.validate_context(context)
                except ValueError as exc:
                    raise ProtocolError(str(exc)) from exc
                rs, kind = candidate, effect_kind
            except ProtocolError as exc:
                text = f"Invalid {call.name}: {exc}. Previous valid context preserved."
            message: Message = {**tool_reply(call, text), "name": call.name}
            if exchange:
                exchange = (*exchange, message)
                exchange_refs = (*exchange_refs, len(rs.history))
                rs = replace(rs, history=(*rs.history, message))
            else:
                rs = replace(
                    rs,
                    messages=(*rs.messages, message),
                    history=(*rs.history, message),
                    context_refs=(*rs.context_refs, len(rs.history)),
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
                            rs.invocation_id,
                            turn_end=len(rs.turns),
                        ),
                    ),
                )
        return Applied(
            replace(
                rs,
                messages=(*rs.messages, *exchange),
                context_refs=(*rs.context_refs, *exchange_refs),
            )
        )


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
