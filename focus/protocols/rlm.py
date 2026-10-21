"""Recursive LM invocations over external input in managed Python sessions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NoReturn
from uuid import uuid4

from pydantic_monty import ExternalSettledResult

from focus.execution.monty import MontySnapshot, Query
from focus.generation import Policy, SamplingParams
from focus.generation.parsing.base import ToolTurn
from focus.harness.env import Env, StepResult
from focus.harness.rollout import AnyRollout
from focus.harness.tools.dispatch import tool_reply
from focus.harness.workspace import Workspace
from focus.protocols.base import (
    AppendOnly,
    Applied,
    ContextProtocol,
    OperationSuspended,
    ProtocolError,
    ToolProtocol,
    schema,
)
from focus.registry import register

if TYPE_CHECKING:
    from focus.harness.runner import Runner


class InvocationEnv(Env[str, Any]):
    """A child produces text; it never scores or terminates the enclosing task."""

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            schema(
                "rlm_return",
                "Return the answer to your caller.",
                {"answer": {"type": "string"}},
                ["answer"],
            )
        ]

    async def reset(self) -> str:
        return ""

    async def step(
        self, state: str, action: Any, *, workspace: Workspace | None = None
    ) -> StepResult[str]:
        if not isinstance(action, ToolTurn):
            return StepResult(str(action), done=True)
        for call in action.tool_calls:
            if call.name == "rlm_return":
                try:
                    answer = json.loads(call.arguments)["answer"]
                    if not isinstance(answer, str):
                        raise ValueError("answer must be text")
                except (ValueError, KeyError, TypeError) as exc:
                    return StepResult(
                        state,
                        messages=tuple(
                            tool_reply(
                                sibling,
                                str(exc)
                                if sibling is call
                                else "Not executed: rlm_return took precedence.",
                            )
                            for sibling in action.tool_calls
                        ),
                    )
                return StepResult(answer, done=True)
        if not action.tool_calls:
            return StepResult(action.text, done=True)
        return StepResult(
            state,
            messages=tuple(
                tool_reply(call, "Unknown child tool.") for call in action.tool_calls
            ),
        )

    async def finish(self, state: str, action: Any) -> StepResult[str]:
        result = await self.step(state, action)
        return replace(result, done=True)


@dataclass(frozen=True)
class PendingCell:
    """Children of a cell suspended in `rlm_session`, and its stdout so far."""

    query: Query
    children: tuple[AnyRollout, ...]
    output: str


@register(ContextProtocol, "rlm")
class RLM(ToolProtocol):
    def __init__(
        self,
        context: ContextProtocol | None = None,
        *,
        helpers: dict[str, Policy[Any]] | None = None,
        max_depth: int = 4,
        max_fanout: int = 8,
        child_steps: int = 16,
        timeout: float = 120,
        externalize: bool = True,
    ):
        if max_depth < 0 or max_fanout < 1 or child_steps < 1 or timeout <= 0:
            raise ValueError("invalid recursion, fanout, step or timeout limits")
        self.context = context or AppendOnly()
        self.helpers = dict(helpers or {})
        self.max_depth, self.max_fanout = max_depth, max_fanout
        self.child_steps, self.timeout = child_steps, timeout
        self.externalize = externalize

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return (
            *self.context.schemas(),
            schema(
                "rlm_python",
                "Execute Python with external input in `context`. Variables persist. "
                "Uses Monty's Python subset; host filesystem and third-party "
                "packages are unavailable. "
                "Call llm_query(prompt), llm_query_batched(prompts), "
                "rlm_query(prompt), "
                "or rlm_query_batched(prompts). Optional model selects a fixed helper. "
                "Calls return text in input order. Use task tools to finish.",
                {"code": {"type": "string"}},
                ["code"],
            ),
        )

    async def initialize(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        rs = await self.context.initialize(rs, runner)
        content = str(rs.messages[-1].get("content", "")) if rs.messages else ""
        messages = rs.messages
        if self.externalize and messages:
            messages = (
                *messages[:-1],
                {
                    "role": "user",
                    "content": (
                        f"Input is in context ({len(content)} chars). "
                        "Inspect it with rlm_python and solve the task it describes."
                    ),
                },
            )
        return replace(
            rs,
            messages=messages,
            seed=messages,
            context_refs=(*rs.context_refs[:-1], None)
            if self.externalize
            else rs.context_refs,
            execution={
                **rs.execution,
                "rlm_session": MontySnapshot(context=content),
                "rlm_depth": 0,
            },
        )

    def needs_prepare(self, rs: AnyRollout, runner: Runner[Any, Any]) -> bool:
        return self.context.needs_prepare(rs, runner)

    async def before(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        _session(rs)
        return await self.context.before(rs, runner)

    async def compact(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        return await self.context.compact(rs, runner)

    async def task_apply(
        self,
        rs: AnyRollout,
        action: Any,
        runner: Runner[Any, Any],
        *,
        ending: bool,
    ) -> Applied:
        return await self.context.apply(rs, action, runner, ending=ending)

    def _child_runner(
        self, kind: str, model: str | None, runner: Runner[Any, Any]
    ) -> Runner[Any, Any]:
        from focus.harness.runner import Runner

        protocol = (
            RLM(
                self.context,
                helpers=self.helpers,
                max_depth=self.max_depth,
                max_fanout=self.max_fanout,
                child_steps=self.child_steps,
                timeout=self.timeout,
            )
            if kind == "rlm"
            else AppendOnly()
        )
        return Runner(
            policy=runner.policy if model is None else self.helpers[model],
            env=InvocationEnv(),
            system_prompt="Solve the input task. Return an answer.",
            protocol=protocol,
            services=runner.services,
            actor=runner.actor and model is None,
            max_steps=self.child_steps,
            generate_timeout=runner.generate_timeout,
        )

    def _validate_query(
        self, query: Query, depth: int, runner: Runner[Any, Any]
    ) -> None:
        if query.kind not in ("rlm", "llm"):
            raise ValueError("unknown invocation kind")
        if not isinstance(query.prompts, list) or any(
            not isinstance(p, str) for p in query.prompts
        ):
            raise ValueError("child prompts must be a list of strings")
        if len(query.prompts) > self.max_fanout:
            raise ValueError("child fanout limit exceeded")
        if depth >= self.max_depth:
            raise ValueError("recursion depth limit exceeded")
        if query.model is not None:
            if not isinstance(query.model, str) or query.model not in self.helpers:
                raise ValueError(f"unknown fixed helper {query.model!r}")
            if self.helpers[query.model].version == runner.policy.version:
                raise ValueError("fixed helpers require distinct policy versions")

    async def resume_operation(
        self,
        rs: AnyRollout,
        name: str,
        arguments: dict[str, Any],
        runner: Runner[Any, Any],
        sampling_params: SamplingParams | None,
    ) -> tuple[AnyRollout, str, str]:
        if name != "rlm_python":
            if not isinstance(self.context, ToolProtocol):
                raise ValueError("context protocol has no operation handler")
            return await self.context.operation(rs, name, arguments, runner)
        code = arguments["code"]
        if not isinstance(code, str):
            raise ProtocolError("code must be a string")
        snapshot = _session(rs)
        pending: PendingCell | None = rs.execution.get("rlm_pending")
        owner = rs.owner
        output = "" if pending is None else pending.output
        reply: ExternalSettledResult | None = None
        if pending is not None:
            child_runner = self._child_runner(
                pending.query.kind, pending.query.model, runner
            )

            async def advance(child: AnyRollout) -> AnyRollout:
                # Rebind every descendant lazily as it becomes active. The root's
                # remaining budget has already attached the new fork scope.
                child = replace(
                    child,
                    episode_id=rs.episode_id,
                    branch_id=rs.branch_id,
                    budget=runner.services.remaining(owner),
                )
                if child.done or child.truncated:
                    return child
                params = sampling_params if pending.query.model is None else None
                if child.steps_remaining == 0 and not child_runner.protocol.has_pending(
                    child
                ):
                    return await child_runner.finish(child, sampling_params=params)
                return await child_runner.advance(child, sampling_params=params)

            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(advance(c)) for c in pending.children]
            children = tuple(task.result() for task in tasks)
            # Attach only new generations, in batch order, at every decision.
            turns = list(rs.turns)
            effects = list(rs.effects)
            for old, child in zip(pending.children, children, strict=True):
                turns.extend(replace(turn, transition=None) for turn in child.edge(old))
                new_effects = child.effects[len(old.effects) :]
                if not new_effects:
                    continue
                # Siblings interleave across advances; a constant offset is wrong
                # when an effect completes a call recorded in an earlier advance.
                indices = [
                    i
                    for i, turn in enumerate(turns)
                    if turn.invocation_id == child.invocation_id
                    or turn.invocation_id.startswith(child.invocation_id + "/")
                ]
                effects.extend(
                    replace(effect, turn_index=indices[effect.turn_index])
                    for effect in new_effects
                )
            rs = replace(rs, turns=tuple(turns), effects=tuple(effects))
            if any(c.truncated and not c.done for c in children):
                # A failed child has no answer with which to resume its ancestors.
                return (
                    replace(rs, truncated=True, snapshot_available=False),
                    (output + "\nChild invocation stopped without an answer."),
                    "invoke",
                )
            if not all(c.done for c in children):
                _suspend(rs, snapshot, replace(pending, children=children))
            rs = replace(
                rs,
                execution={
                    **rs.execution,
                    "invocations": (*rs.execution.get("invocations", ()), *children),
                },
            )
            reply = {"return_value": [str(c.state) for c in children]}

        while True:
            result = await runner.services.execute_monty(
                code, snapshot, reply=reply, owner=owner, timeout=self.timeout
            )
            output += result.output
            if result.query is None:
                execution = dict(rs.execution)
                execution.pop("rlm_pending", None)
                if result.snapshot is not None:
                    execution["rlm_session"] = result.snapshot
                return (
                    replace(
                        rs,
                        execution=execution,
                        snapshot_available=result.snapshot is not None,
                        truncated=rs.truncated or result.snapshot is None,
                    ),
                    output + ("\n" + result.error if result.error else ""),
                    "invoke",
                )
            assert result.snapshot is not None
            snapshot = result.snapshot
            query = result.query
            try:
                self._validate_query(query, rs.execution["rlm_depth"], runner)
            except ValueError as exc:
                reply = {"exc_type": "ValueError", "message": str(exc)}
                continue
            if not query.prompts:
                reply = {"return_value": []}
                continue
            child_runner = self._child_runner(query.kind, query.model, runner)
            children = tuple(
                [
                    replace(
                        child,
                        episode_id=rs.episode_id,
                        branch_id=rs.branch_id,
                        invocation_id=rs.invocation_id + "/" + uuid4().hex,
                        budget=runner.services.remaining(owner),
                        execution={
                            **child.execution,
                            "rlm_depth": rs.execution["rlm_depth"] + 1,
                        },
                    )
                    for child in [await child_runner.start(p) for p in query.prompts]
                ]
            )
            _suspend(rs, snapshot, PendingCell(query, children, output))


def _session(rs: AnyRollout) -> MontySnapshot:
    snapshot = rs.execution.get("rlm_session")
    if not isinstance(snapshot, MontySnapshot):
        raise ValueError(
            "CPython RLM checkpoints are analysis-only; start a fresh Monty rollout"
        )
    return snapshot


def _suspend(rs: AnyRollout, snapshot: MontySnapshot, pending: PendingCell) -> NoReturn:
    raise OperationSuspended(
        replace(
            rs,
            execution={
                **rs.execution,
                "rlm_session": snapshot,
                "rlm_pending": pending,
            },
        )
    )
