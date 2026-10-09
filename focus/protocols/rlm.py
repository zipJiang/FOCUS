"""Recursive LM invocations over external input in managed Python sessions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from focus.execution import Owner
from focus.execution.process import ProcessSnapshot
from focus.generation import Policy
from focus.generation.parsing.base import ToolTurn
from focus.harness.env import Env, StepResult
from focus.harness.rollout import AnyRollout
from focus.harness.workspace import Workspace
from focus.protocols.base import AppendOnly, ContextProtocol, ToolProtocol, schema
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
                        state, messages=({"role": "tool", "content": str(exc)},)
                    )
                return StepResult(answer, done=True)
        if not action.tool_calls:
            return StepResult(action.text, done=True)
        return StepResult(
            state, messages=({"role": "tool", "content": "Unknown child tool."},)
        )

    async def finish(self, state: str, action: Any) -> StepResult[str]:
        result = await self.step(state, action)
        return replace(result, done=True)


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
                "rlm_session": ProcessSnapshot.variables(context=content),
                "rlm_depth": 0,
            },
        )

    def needs_prepare(self, rs: AnyRollout, runner: Runner[Any, Any]) -> bool:
        return self.context.needs_prepare(rs, runner)

    async def before(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        return await self.context.before(rs, runner)

    async def compact(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        return await self.context.compact(rs, runner)

    async def operation(
        self,
        rs: AnyRollout,
        name: str,
        arguments: dict[str, Any],
        runner: Runner[Any, Any],
    ) -> tuple[AnyRollout, str, str]:
        if name != "rlm_python":
            if not isinstance(self.context, ToolProtocol):
                raise ValueError("context protocol has no operation handler")
            return await self.context.operation(rs, name, arguments, runner)
        code = arguments["code"]
        if not isinstance(code, str):
            raise ValueError("code must be a string")
        completed: list[AnyRollout] = []
        depth = rs.execution["rlm_depth"]
        owner = Owner(rs.episode_id, rs.branch_id, rs.invocation_id)

        async def invoke(kind: str, prompts: list[str], model: str | None) -> list[str]:
            if not isinstance(prompts, list) or any(
                not isinstance(p, str) for p in prompts
            ):
                raise ValueError("child prompts must be a list of strings")
            if len(prompts) > self.max_fanout:
                raise ValueError("child fanout limit exceeded")
            if depth >= self.max_depth:
                raise ValueError("recursion depth limit exceeded")
            if model is not None and model not in self.helpers:
                raise ValueError(f"unknown fixed helper {model!r}")
            policy = runner.policy if model is None else self.helpers[model]
            if model is not None and policy.version == runner.policy.version:
                raise ValueError("fixed helpers require distinct policy versions")

            async def child(prompt: str) -> AnyRollout:
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
                child_runner = Runner(
                    policy=policy,
                    env=InvocationEnv(),
                    system_prompt="Solve the input task. Return an answer.",
                    protocol=protocol,
                    services=runner.services,
                    actor=runner.actor and model is None,
                    max_steps=self.child_steps,
                    generate_timeout=runner.generate_timeout,
                )
                child_rs = await child_runner.start(prompt)
                child_rs = replace(
                    child_rs,
                    episode_id=rs.episode_id,
                    branch_id=rs.branch_id,
                    invocation_id=rs.invocation_id + "/" + uuid4().hex,
                    budget=runner.services.remaining(owner),
                    execution={**child_rs.execution, "rlm_depth": depth + 1},
                )
                return await child_runner.run(child_rs)

            tasks: list[asyncio.Task[AnyRollout]] = []
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(child(prompt)) for prompt in prompts]
            results = [task.result() for task in tasks]
            completed.extend(results)
            return [str(result.state) for result in results]

        result = await runner.services.execute(
            code,
            rs.execution.get("rlm_session"),
            handler=invoke,
            owner=owner,
            timeout=self.timeout,
        )
        # Record batches in prompt order, independent of completion order.
        turns = tuple(
            replace(turn, transition=None)
            for child in completed
            for turn in child.turns
        )
        effects = list(rs.effects)
        offset = len(rs.turns)
        for child in completed:
            effects.extend(
                replace(effect, turn_index=effect.turn_index + offset)
                for effect in child.effects
            )
            offset += len(child.turns)
        execution = {
            **rs.execution,
            "invocations": (*rs.execution.get("invocations", ()), *completed),
        }
        if result.snapshot is not None:
            execution["rlm_session"] = result.snapshot
        updated = replace(
            rs,
            turns=(*rs.turns, *turns),
            effects=tuple(effects),
            execution=execution,
            snapshot_available=result.snapshot is not None,
            truncated=rs.truncated or result.snapshot is None,
        )
        return (
            updated,
            result.output + ("\n" + result.error if result.error else ""),
            "invoke",
        )
