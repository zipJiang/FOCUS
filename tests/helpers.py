"""Shared fakes and constructors for the harness/scheduler tests.

One copy of the chat tokenizer, the scripted generator, the turn/rollout builders, the
whole-episode driver (:func:`complete`) and its reader (:func:`trainable_text`), the
submit-convention env every tool test wants, and the backends that fail -- down,
stalled, one result short -- which the loop and the scheduler drive the same failure
policy with. So a change to the log's shape, or to what "the standard toy rollout" is,
is a change in one place. The tokenizer fakes matter more than they look: packing is
decided by whether one turn's prefix extends the previous pack, and that is a property
of the *template*, so a fake that renders differently from a real one silently tests a
different packing regime.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from focus.codec import ChatCodec, Message
from focus.context import Trigger
from focus.critic import Critic
from focus.generation import (
    GenerateResult,
    Policy,
    PolicyFormat,
    SamplingParams,
    TokenId,
)
from focus.generation.parsing import ActionParser, TextActionParser, ToolTurn
from focus.harness import (
    Agent,
    Env,
    InMemoryWorkspace,
    NullWorkspace,
    RolloutState,
    StepResult,
    SubmitFields,
    ToolEnv,
    Turn,
    TurnTag,
    Workspace,
    regions,
)

VERSION = "policy"


class DemoValue(Critic):
    """A deterministic zero-valued critic for core integration tests."""

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        return [0.0] * len(contexts)


def test_format() -> PolicyFormat[str]:
    return PolicyFormat("test", ChatCodec(FakeChatTokenizer()), TextActionParser())


test_format.__dict__["__test__"] = False


def native_format(codec: Any) -> PolicyFormat[Any]:
    from focus.generation.parsing import NativeToolCallParser

    class EstimateTokenizer(FakeChatTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
            from focus.codec import _encode_text

            return list(_encode_text(codec, text))

        def decode(
            self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
        ) -> str:
            from focus.codec import _decode_text

            return _decode_text(codec, token_ids)

        def encode_single_token(self, raw: bytes) -> int:
            if not hasattr(codec, "encode_single_token"):
                raise KeyError(raw)
            return int(codec.encode_single_token(raw))

    return PolicyFormat(
        "native-test", ChatCodec(EstimateTokenizer()), NativeToolCallParser()
    )


class FakeChatTokenizer:
    """One id per role-initial + content char, plus a generation-prompt marker.

    ``add_generation_prompt=True`` appends ``"a"`` -- the assistant turn's opener --
    which is what a real template does and what makes consecutive turns *chain*: the
    prefix the runner sent for turn ``n+1`` starts with everything rendered for turn
    ``n``, including that marker, so the two pack into one prefix-stable sequence.
    Without it every turn would be its own pack and these tests would quietly stop
    covering the merge.
    """

    chat_template: str | None = "fake"

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return "".join(chr(t) for t in token_ids)

    def apply_chat_template(
        self,
        conversation: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        return_tensors: str | None,
    ) -> list[int]:
        ids: list[int] = []
        for m in conversation:
            ids.append(ord(m["role"][0]))
            ids.extend(ord(c) for c in m["content"])
        if add_generation_prompt:
            ids.append(ord("a"))
        return ids


class InexactBridge:
    """Mixed onto a scripted backend: the hosted chat bridge's claim, both halves.

    The real bridge makes them from one field -- the capability a search reads
    *before* it spends (`provides_exact_tokens`), and the stamp on every result
    (`exact_generation`). One definition here, so two test files cannot model the
    same claim with subtly different fakes.
    """

    provides_exact_tokens = False
    provides_native_tool_calls = True

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        result: GenerateResult = await super().agenerate_tokens(  # type: ignore[misc]
            prefix_tokens, sampling_params
        )
        result.exact_generation = False
        return result


class ScriptedGenerator(Policy[Any]):
    """Returns scripted completions; records the prefix ids and params it was given."""

    def __init__(
        self, completions: Sequence[str], *, logprobs: float | None = None
    ) -> None:
        self._completions = list(completions)
        self._i = 0
        self._logprobs = logprobs
        self.prefixes: list[tuple[int, ...]] = []
        #: The sampling params of each call -- which law that turn was drawn under.
        self.params: list[SamplingParams | None] = []

    @property
    def calls(self) -> int:
        return len(self.prefixes)

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        self.params.append(sampling_params)
        self.prefixes.append(tuple(prefix_tokens))
        text = self._completions[min(self._i, len(self._completions) - 1)]
        self._i += 1
        # tokens decode back to `text` under FakeChatTokenizer (ord/chr round trip)
        tokens = tuple(ord(c) for c in text)
        lps = [] if self._logprobs is None else [self._logprobs] * len(tokens)
        return GenerateResult(text=text, tokens=tokens, logprobs=lps)


class ConstantScorer:
    """A :class:`PolicyScorer` returning a constant logprob on trainable positions.

    The constant is what makes an importance ratio predictable in a test: the anchor
    channel it fills is a known number per generated token, so the expected
    ``exp(anchor - behavior)`` is arithmetic over the behavior logprobs the generator
    recorded, not a second model.
    """

    def __init__(self, value: float) -> None:
        self._value = value
        self.calls = 0

    async def score(
        self, tokens: tuple[int, ...], trainable: tuple[bool, ...]
    ) -> tuple[float, ...]:
        self.calls += 1
        return tuple(self._value if keep else 0.0 for keep in trainable)


class WhenTrigger(Trigger[Any]):
    """A :class:`Trigger` from a plain predicate over ``(prompt, state)``.

    Test compactors vary *how* they fold, not *when*; this is so one of them can say
    when in a lambda rather than growing a Trigger subclass per test. ``relieve`` is
    the identity, like the base's -- a test that needs a counter reset says so by
    passing one.
    """

    def __init__(
        self,
        fires: Callable[[Sequence[TokenId], Any], bool],
        relieve: Callable[[Any], Any] | None = None,
    ) -> None:
        self._fires = fires
        self._relieve = relieve

    def fires(self, prompt: Sequence[TokenId], state: Any) -> bool:
        return self._fires(prompt, state)

    def relieve(self, state: Any) -> Any:
        return state if self._relieve is None else self._relieve(state)


def turn(
    logprobs: Sequence[float] = (),
    *,
    masked: int = 0,
    tag: TurnTag = "task",
    done: bool = False,
    reward: float = 0.0,
    transition: bool | None = None,
    prefix: tuple[int, ...] | None = None,
) -> Turn[None]:
    """One turn: ``masked`` conditioning tokens, then one generated token per logprob.

    ``transition`` forces whether an env transition is recorded; by default a task turn
    has one and a fold turn has none, which is the invariant the runner maintains.
    """
    if transition is None:
        transition = tag != "fold"
    generated = tuple(range(masked, masked + len(logprobs)))
    return Turn(
        prefix=tuple(range(masked)) if prefix is None else prefix,
        tokens=generated,
        logprobs=tuple(logprobs) if logprobs else None,
        transition=(
            StepResult(next_state=None, reward_outcome=reward, done=done)
            if transition
            else None
        ),
        tag=tag,
    )


def rollout(
    *turns: Turn[None],
    truncated: bool = False,
) -> RolloutState[None]:
    """A checkpoint over ``turns`` with a fresh workspace.

    No ``folds`` argument: how many times the context was rewritten is derived from the
    fold tags in ``turns``, so a checkpoint that folded is one built with a fold-tagged
    turn in it -- ``rollout(turn(...), fold_marker(), turn(...))``.
    """
    return RolloutState(
        messages=(),
        seed=(),
        state=None,
        workspace=InMemoryWorkspace(),
        turns=turns,
        checkpoint_at=len(turns) if not turns or turns[-1].tag == "fold" else -1,
        truncated=truncated,
    )


@dataclass(frozen=True)
class ToyState:
    n: int = 0


class ToyEnv(Env[ToyState, str]):
    """'stop' ends the episode; every other action advances the count by one.

    ``done_at`` caps the episode at that many steps, which is how a test asks for a
    rollout of a known length without scripting a matching completion for every turn.
    """

    def __init__(self, done_at: int | None = None) -> None:
        self._done_at = done_at

    async def reset(self) -> ToyState:
        return ToyState()

    async def step(
        self, state: ToyState, action: str, *, workspace: Workspace | None = None
    ) -> StepResult[ToyState]:
        del workspace
        nxt = ToyState(state.n + 1)
        cap = self._done_at is not None and nxt.n >= self._done_at
        done = action == "stop" or cap
        return StepResult(
            next_state=nxt,
            messages=({"role": "user", "content": f"obs{nxt.n}"},),
            reward_outcome=1.0 if done else 0.0,
            done=done,
        )


def toy_policy(
    gen: ScriptedGenerator,
    *,
    parser: ActionParser[Any] | None = None,
) -> Policy[Any]:
    """The fakes as one model: the fake tokenizer's codec, a text parser by default."""
    return Policy(
        format=PolicyFormat(
            "custom",
            ChatCodec(FakeChatTokenizer()),
            parser if parser is not None else TextActionParser(),
        ),
        complete=(gen).agenerate_tokens,
        exact=(gen).provides_exact_tokens,
        native=(gen).provides_native_tool_calls,
    )


def toy_runner(
    gen: ScriptedGenerator,
    *,
    env: Env[Any, Any] | None = None,
    parser: ActionParser[Any] | None = None,
    **kwargs: Any,
) -> Agent[Any, Any]:
    """A runner over the fakes, wired the way every harness test wires one.

    One place decides what "the standard toy rollout" is -- the fake tokenizer's codec,
    a text parser, a one-line system prompt -- so a test that cares about compaction or
    scheduling names only the piece it is actually varying.
    """
    from focus.harness.agent import Execution

    if "compactor" in kwargs:
        kwargs["context"] = kwargs.pop("compactor")
    execution = {
        key: kwargs.pop(key)
        for key in ("services", "checkpointer", "resource_budget", "generate_timeout")
        if key in kwargs
    }
    if execution:
        kwargs["execution"] = Execution(**execution)
    # A test that names a parser means it: the agent uses it as given.
    return Agent(
        policy=toy_policy(gen, parser=parser),
        env=env if env is not None else ToyEnv(),
        system_prompt="sys",
        parser=parser,
        **kwargs,
    )


async def complete(
    runner: Agent[Any, Any],
    prompt: str,
    *,
    workspace: Workspace | None = None,
    **run_kwargs: Any,
) -> RolloutState[Any]:
    """Fresh episode: ``start`` owns the prompt/workspace, ``run`` continues from it.

    Defaults to :class:`NullWorkspace` so toy tests keep a bare system prompt; pass a
    real workspace when the test exercises memory.
    """
    ws = NullWorkspace() if workspace is None else workspace
    return await runner.run(await runner.start(prompt, workspace=ws), **run_kwargs)


def trainable_text(
    rs: RolloutState[Any], tags: tuple[TurnTag, ...] | None = None
) -> str:
    """Everything the model generated on ``rs``, decoded, in generation order.

    Reads every region rather than the first: a template rewrite splits a run of turns
    into several packs, and a reading that stopped at pack 0 would silently drop turns.
    """
    codec = ChatCodec(FakeChatTokenizer())
    return "".join(
        codec.decode(
            [t for t, keep in zip(seq.tokens, seq.trainable, strict=True) if keep]
        )
        for seq in regions(rs.tagged(tags))
    )


def tool_call(name: str, **arguments: Any) -> str:
    """One Hermes ``<tool_call>`` block -- the dialect these tests generate in."""
    obj = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{obj}\n</tool_call>"


@dataclass(frozen=True)
class SubmitState(SubmitFields):
    """A task state under the submit convention, counting the calls that ran."""

    calls: int = 0


class SubmitEnv(ToolEnv[SubmitState]):
    """A submit-aware env that scores 1.0 and counts the tool calls it dispatched.

    The state stores the answer and counts dispatched tool calls.
    """

    async def initial_state(self) -> SubmitState:
        return SubmitState()

    def reward(self, state: SubmitState, turn: ToolTurn) -> float:
        return 1.0

    def on_tool_calls(self, state: SubmitState, turn: ToolTurn) -> SubmitState:
        state = super().on_tool_calls(state, turn)
        return replace(state, calls=state.calls + len(turn.tool_calls))


# -- backends that fail: the three shapes a run has to survive ------------------------


class DeadReward(Critic):
    """A critic server that is down: every call raises.

    One of the two per-expansion faults the scheduler degrades rather than raises on --
    the generation is the expensive half, and a missing estimate is a coarser advantage,
    not a dead rollout.
    """

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        raise RuntimeError("reward server unreachable")


class StalledReward(Critic):
    """A critic server that took the request and never answered.

    Counts the cancellations rather than only the calls: "the timeout fired" and "the
    request was actually let go" are two claims, and only the second one says a stalled
    socket is not still held when the pass moves on.
    """

    def __init__(self, *, timeout: float | None = None) -> None:
        self.cancelled = 0
        self.timeout = timeout

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        raise AssertionError("unreachable")  # pragma: no cover


class ShortBatchReward(Critic):
    """A critic server that answers a batch with one result too few.

    Not a hypothetical shape: the vLLM backend returns one result per entry of the
    pooling response's ``data``, so a truncated or partial response is exactly this.
    """

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        return [1.0] * (len(contexts) - 1)


class StalledGenerator(Policy[Any]):
    """A generation backend that swallowed the request: it never answers.

    The counters are the assertion: ``calls`` says how many requests the bound let
    through, and ``cancelled`` that each one was let go rather than left running behind
    a caller that had already given up on it.
    """

    def __init__(self) -> None:
        super().__init__(format=test_format())
        self.calls = 0
        self.cancelled = 0

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        self.calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "VERSION",
    "ConstantScorer",
    "DeadReward",
    "FakeChatTokenizer",
    "ScriptedGenerator",
    "ShortBatchReward",
    "StalledGenerator",
    "StalledReward",
    "SubmitEnv",
    "SubmitState",
    "ToyEnv",
    "ToyState",
    "WhenTrigger",
    "complete",
    "rollout",
    "tool_call",
    "toy_runner",
    "trainable_text",
    "turn",
]


async def prepared(prompt: Any, collector: Any, *, estimator: Any = None) -> Any:
    """Run a test search and explicitly prepare its chosen training recipe."""
    from focus import DirectBranchTd

    tree = await collector.search(prompt)
    return collector.prepare(
        tree, estimator=estimator if estimator is not None else DirectBranchTd()
    )


def budget_context(
    *,
    remaining: str = "calls_remaining",
    maximum: str = "calls_max",
    max_prompt_tokens: int = 8192,
    **kwargs: Any,
) -> Any:
    """Compose the token and task-counter triggers used by budget tests."""
    from focus.context import AnyOf, PromptTokens, StateCounter, Summarize

    return Summarize(
        trigger=AnyOf(
            PromptTokens(max_prompt_tokens), StateCounter(remaining, maximum)
        ),
        **kwargs,
    )
