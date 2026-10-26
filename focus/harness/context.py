"""What the policy sees: context managers render prompts and compact long contexts.

A :class:`ContextManager` decides two things the agent asks before every generation:
how to render the branch's context as a prompt (:meth:`ContextManager.prepare`), and
whether to compact it first (:meth:`ContextManager.should_compact`, then
:meth:`ContextManager.compact`). The agent records every compaction -- its generated
turns, the context version, a branch-point effect -- so a manager only produces the
new context.

:class:`AppendOnly` keeps the whole transcript; :class:`Segments` continues one
assistant response in exact token segments;
:class:`~focus.harness.compaction.Summarize` folds the transcript into notes when a
:class:`~focus.harness.triggers.Trigger` fires. :class:`ContextTools` lets the model
edit its own context through tools.
"""

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
from focus.harness.triggers import Trigger
from focus.harness.turn import Turn
from focus.harness.workspace import Workspace

if TYPE_CHECKING:
    from focus.harness.agent import Agent


class ContextError(ValueError):
    """Invalid context operation; safe to return to the model as corrective feedback."""


class IncompleteCompactionError(ValueError):
    """Rejected model output, including the exact generations that produced it.

    Callers may treat this as a terminal task failure. Transport and implementation
    errors remain ordinary exceptions and must not become training outcomes.
    """

    def __init__(
        self, *, turns: tuple[Turn[Any], ...], reasons: tuple[str, ...]
    ) -> None:
        super().__init__(
            "Incomplete compaction reply; refusing to replace task context"
            + (f" ({', '.join(reasons)})" if reasons else "")
        )
        self.turns = turns
        self.reasons = reasons


@dataclass(frozen=True)
class Compaction[S]:
    """Everything a compaction is handed -- the conversation to fold, and the agent.

    A compactor reads ``messages`` (the grown conversation), persists what matters into
    ``workspace``, and returns the messages to continue from -- ``seed`` (the system +
    first user message) for a plain reset, or ``seed`` with what it kept folded in; how
    the fold is shaped is the compactor's own policy.

    Its one collaborator is the ``agent`` being folded, not a bag of that agent's
    parts: a model-driven fold is a rollout under the *same* policy on a different
    world, so it asks for :meth:`~focus.harness.agent.Agent.derive` instead
    of reassembling a agent out of a generator, a codec, a parser and a version tag
    smuggled across this seam -- and cannot end up guessing a tool dialect the rollout
    never spoke. A compactor that generates nothing simply ignores it.
    """

    messages: tuple[Message, ...]
    seed: tuple[Message, ...]
    state: S
    workspace: Workspace
    #: The agent whose rollout is being folded: the model, the codec, the tool dialect
    #: and the policy-version tag in one piece. Loose in the action parameter because a
    #: fold's env speaks the parser's actions, which need not be the task env's.
    agent: Agent[S, Any]

    checkpoint: Any = None


@dataclass(frozen=True)
class CompactionResult[S]:
    """What one compaction produced: the messages to continue from, and the new state.

    Compaction is a state transition like ``env.step`` -- it reads ``state`` (via
    :class:`Compaction`) and returns a new one, so a compactor can reset the counters
    its :attr:`~Compactor.trigger` watched (e.g. a per-compaction search budget), which
    is what :meth:`~...compaction.triggers.Trigger.relieve` is for.

    ``turns`` are the turns a model-driven compactor generated getting here --
    transition-free, because a fold rewrites the working context rather than acting on
    the world. The agent appends them to its log like any others (stamping them
    ``"fold"`` on the way in, so the tag is its own statement rather than a plug-in's),
    so a fold's tool-call turns are first-class trainable data rather than a special
    kind of record. A plain compactor returns none, and the agent leaves one zero-token
    ``"fold"`` turn in their place -- the marker that records that anything happened.
    """

    messages: tuple[Message, ...]
    state: S
    #: ``Turn[Any]``, not ``Turn[S]``: a generative compactor's turns are
    #: generated in the *fold's* own world (``FoldEnv``), so their state
    #: parameter is not the task's -- and it is vacuous besides, since these
    #: turns are retagged transition-free before they get here.
    turns: tuple[Turn[Any], ...] = ()


@dataclass(frozen=True)
class Applied:
    rollout: AnyRollout
    transition: StepResult[Any] | None = None


class ContextManager[S]:
    """Render a branch's context, and compact it when :attr:`trigger` fires.

    A manager is shared by every branch, so branch-owned data belongs in the rollout
    state. The two methods most managers implement are :meth:`should_compact` (by
    default, whether :attr:`trigger` fires on the rendered prompt) and :meth:`compact`.
    :meth:`prepare`, :meth:`record_reply`, :meth:`apply`, and :meth:`schemas` let a
    manager change rendering, replies, or add context operations of its own.
    """

    #: When to compact; ``None`` never compacts.
    trigger: Trigger[S] | None = None

    @property
    def manages_context(self) -> bool:
        """Whether this manager may rewrite the context during a rollout."""
        return self.trigger is not None

    def should_compact(self, rs: AnyRollout, prompt: Sequence[int]) -> bool:
        """Whether to compact before generating from ``prompt``."""
        return self.trigger is not None and self.trigger.fires(prompt, rs.state)

    async def compact(self, compaction: Compaction[S]) -> CompactionResult[S]:
        """Return the context and task state to continue from.

        Called by the agent when :meth:`should_compact` is true or when the caller
        forces a fold; the agent records the result. A manager that compacts must
        also relieve its trigger in the returned state.
        """
        raise ValueError(f"{type(self).__name__} does not compact context")

    def prepare(
        self,
        rs: AnyRollout,
        agent: Agent[Any, Any],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> PreparedPrompt:
        """Render this branch's context; ``tools`` overrides the agent's schemas."""
        return agent.policy.prepare(
            rs.messages, tools=agent.tools if tools is None else tools
        )

    def record_reply(
        self,
        result: GenerateResult,
        messages: list[Message],
        agent: Agent[Any, Any],
    ) -> None:
        """Append the generated reply to the branch's live conversation."""
        agent.policy.record_reply(result, messages)

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return ()

    async def initialize(self, rs: AnyRollout, agent: Agent[Any, Any]) -> AnyRollout:
        """Check the root state; a trigger refuses state fields it cannot read."""
        if self.trigger is not None:
            self.trigger.check(rs.state)
        return rs

    async def apply(
        self,
        rs: AnyRollout,
        action: Any,
        agent: Agent[Any, Any],
        *,
        ending: bool,
    ) -> Applied:
        if not ending and isinstance(action, ToolTurn):
            agent.services.usage.setdefault(rs.episode_id, Usage()).tool_calls += len(
                action.tool_calls
            )
        transition = (
            await agent.env.finish(rs.state, action)
            if ending
            else await agent.env.step(rs.state, action, workspace=rs.workspace)
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


class AppendOnly(ContextManager[Any]):
    """Retain context as chat turns or exact segments of one assistant response.

    ``continuation=True`` preserves original generated token IDs between requests.
    It requires an exact token-input policy and uninterrupted assistant output.
    """

    def __init__(self, *, continuation: bool = False) -> None:
        self.continuation = continuation

    def _check_policy(self, agent: Agent[Any, Any]) -> None:
        if self.continuation and (
            not agent.policy.provides_exact_tokens
            or agent.policy.provides_native_tool_calls
        ):
            raise ValueError("Token continuation requires an exact token-input policy")

    async def initialize(self, rs: AnyRollout, agent: Agent[Any, Any]) -> AnyRollout:
        self._check_policy(agent)
        return rs

    def _boundary(self, messages: Sequence[Message], agent: Agent[Any, Any]) -> int:
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
            if message.get("content") != agent.policy.decode(
                tokens, skip_special_tokens=True
            ):
                raise ValueError(
                    "Continuation text was edited; its token IDs are unchanged"
                )
        return boundary

    def prepare(
        self,
        rs: AnyRollout,
        agent: Agent[Any, Any],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> PreparedPrompt:
        """Render chat context, or append exact segments to the initial prompt."""
        if not self.continuation:
            return super().prepare(rs, agent, tools=tools)
        self._check_policy(agent)
        boundary = self._boundary(rs.messages, agent)
        initial = agent.policy.prepare(
            rs.messages[:boundary], tools=agent.tools if tools is None else tools
        )
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
        agent: Agent[Any, Any],
    ) -> None:
        """Keep exact token IDs with each assistant segment in continuation mode."""
        if self.continuation and (
            not result.exact_generation
            or result.native_tool_calls
            or result.native_output
        ):
            raise ValueError("Token continuation requires an exact text completion")
        super().record_reply(result, messages, agent)
        if self.continuation:
            messages[-1]["focus_continuation_tokens"] = tuple(result.tokens)


class Segments(AppendOnly):
    """Continue one assistant response in exact segments of at most ``tokens`` tokens.

    Each decision generates one segment, appended to the prompt by its original token
    IDs. The agent defaults to branch points after every segment and to ending without
    a final-answer prompt when its step budget runs out. The policy must accept token
    input and parse segments as :class:`~focus.generation.TokenChunk` actions.
    """

    def __init__(self, tokens: int) -> None:
        if type(tokens) is not int or tokens < 1:
            raise ValueError("Segments.tokens must be a positive integer")
        super().__init__(continuation=True)
        self.tokens = tokens


class ContextTools(ContextManager[Any]):
    """Context operations the model calls as tools; mixed batches are refused.

    Implement :meth:`schemas` and :meth:`operation`. Each successful operation
    records a ``"context"`` effect, which branch-point policies may select.
    """

    @property
    def manages_context(self) -> bool:
        return True

    async def operation(
        self,
        rs: AnyRollout,
        name: str,
        arguments: dict[str, Any],
        agent: Agent[Any, Any],
    ) -> tuple[AnyRollout, str, str]:
        raise NotImplementedError

    async def apply(
        self,
        rs: AnyRollout,
        action: Any,
        agent: Agent[Any, Any],
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
            return await super().apply(rs, action, agent, ending=ending)
        # Native adapters require the active call and all of its replies to remain
        # contiguous. Edit only settled context, then restore this exchange intact.
        # History still records every original message at its stable index.
        exchange: tuple[Message, ...] = ()
        exchange_refs: tuple[int | None, ...] = ()
        if agent.policy.provides_native_tool_calls and rs.messages[-1].get(
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
                    raise ContextError(
                        "rollout is truncated; no further operations can run"
                    )
                if mixed:
                    raise ContextError(
                        "issue context operations separately from task tools"
                    )
                try:
                    arguments = json.loads(call.arguments)
                except ValueError as exc:
                    raise ContextError(str(exc)) from exc
                if not isinstance(arguments, dict):
                    raise ContextError("tool arguments must be an object")
                contract = contracts[call.name]
                missing = set(contract.get("required", ())) - arguments.keys()
                if missing:
                    raise ContextError(f"missing arguments: {sorted(missing)}")
                candidate, text, effect_kind = await self.operation(
                    rs, call.name, arguments, agent
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
                    agent.policy.validate_context(context)
                except ValueError as exc:
                    raise ContextError(str(exc)) from exc
                rs, kind = candidate, effect_kind
            except ContextError as exc:
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
