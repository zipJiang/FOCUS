"""The Policy record: one model, and whether a rollout from it can train that model.

The capability flags are declared in three different layers -- a generator says whether
its ids are the model's own, a parser says whether the actions are in those ids, and
:class:`Policy` is the one place the two are read together -- so the truth table is
checked here rather than beside any one of them. What the *loop* then does with the
answer is `tests/test_loop.py`; what a turn records under it is below.
"""

from __future__ import annotations

from typing import Any

import pytest

from focus.codec import ChatCodec
from focus.generation import (
    GenerateResult,
    NativeToolCall,
    Policy,
    PolicyFormat,
    SamplingParams,
)
from focus.generation.parsing import (
    HermesToolCallParser,
    NativeToolCallParser,
    QwenXMLToolCallParser,
    TextActionParser,
)
from focus.harness import Agent
from tests.helpers import (
    FakeChatTokenizer,
    InexactBridge,
    ScriptedGenerator,
    SubmitEnv,
    ToyEnv,
    complete,
    native_format,
    test_format,
    toy_policy,
    toy_runner,
)


class _InexactGenerator(InexactBridge, ScriptedGenerator):
    """The hosted chat bridge's claim over a script -- see `InexactBridge`."""


# -- the truth table ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exact_ids", "in_band", "trainable"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_a_policy_is_trainable_only_when_both_halves_hold(
    exact_ids: bool, in_band: bool, trainable: bool
) -> None:
    """Both conditions are about one thing: are the recorded tokens what the policy did.

    Exact ids from a parser reading a channel beside them is as untrainable as in-band
    actions inside ids nobody can vouch for.
    """
    # The native-parser rows need a channel-claiming backend, or `Policy` refuses
    # the pairing at construction before `trainable` can even be read.
    generator: Policy[Any] = (
        (ScriptedGenerator(["x"]) if in_band else _ChannelGenerator(["x"]))
        if exact_ids
        else _InexactGenerator(["x"])
    )
    policy = Policy(
        format=PolicyFormat(
            "custom",
            ChatCodec(FakeChatTokenizer()),
            TextActionParser() if in_band else NativeToolCallParser(),
        ),
        complete=(generator).agenerate_tokens,
        exact=(generator).provides_exact_tokens,
        native=(generator).provides_native_tool_calls,
        version="policy",
    )

    assert policy.trainable is trainable


def test_a_native_parser_on_a_channel_less_backend_is_refused_at_binding() -> None:
    """The category error fails where the category is asserted, not on turn one.

    Both claims are class-level facts -- the parser reads a structured channel, the
    backend declares none -- so `Policy` is the first place they meet and the last
    place the mistake is cheap. The parser's own runtime error stays as the defense
    for a hand-built `GenerateResult`, but a bound policy never reaches it.
    """
    with pytest.raises(ValueError, match="structured tool-call channel"):
        Policy(
            format=PolicyFormat(
                "custom", ChatCodec(FakeChatTokenizer()), NativeToolCallParser()
            ),
            complete=(ScriptedGenerator(["x"])).agenerate_tokens,
            exact=(ScriptedGenerator(["x"])).provides_exact_tokens,
            native=(ScriptedGenerator(["x"])).provides_native_tool_calls,
            version="policy",
        )


def test_every_in_band_dialect_parser_declares_itself_trainable() -> None:
    assert HermesToolCallParser().trainable
    assert QwenXMLToolCallParser().trainable
    assert TextActionParser().trainable
    assert not NativeToolCallParser().trainable


def test_the_version_is_the_channel_the_runner_files_its_generations_under() -> None:
    """One name, on the policy, read back through the runner -- never authored twice."""
    runner = toy_runner(ScriptedGenerator(["go"]), version="pi-7")

    assert runner.policy.version == "pi-7"


# -- what the runner does with it -----------------------------------------------------


class _ChannelGenerator(ScriptedGenerator):
    """A scripted backend that also opens the structured tool-call channel (empty).

    Exact ids by inheritance -- what this fake is for is the *parser's* half of the
    trainability question, on a backend that answers its own half with "yes".
    """

    provides_native_tool_calls = True

    async def agenerate_tokens(
        self,
        prefix_tokens: Any,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        result = await super().agenerate_tokens(prefix_tokens, sampling_params)
        result.native_tool_calls = ()
        return result


@pytest.mark.asyncio
async def test_a_turn_is_not_exact_when_the_parser_reads_beside_the_tokens() -> None:
    """The backend's claim is necessary, not sufficient: the parser has a veto.

    The scripted backends here report ``exact_generation=True`` (the ids really are the
    ones they made up), and under a native parser the turn is still stamped inexact,
    because the action this turn took is not in those ids. The caveat then rides the
    chain the packer already reads, so nothing downstream must know what a parser is.
    """
    native = await complete(
        toy_runner(_ChannelGenerator(["stop"]), parser=NativeToolCallParser()),
        "q",
        max_steps=1,
    )
    in_band = await complete(toy_runner(ScriptedGenerator(["stop"])), "q", max_steps=1)

    assert native.turns
    assert not any(t.exact for t in native.turns)
    assert in_band.turns
    assert all(t.exact for t in in_band.turns)


def test_derive_hands_the_nested_runner_the_same_policy_object() -> None:
    """A fold is this agent on another world -- not a second one assembled by hand."""
    runner = toy_runner(ScriptedGenerator(["go"]))

    nested = runner.derive(
        env=ToyEnv(), system_prompt="fold", max_steps=2, finish_prompt=None
    )

    assert nested.policy is runner.policy


# -- the schemas the structured channel needs -----------------------------------------


@pytest.mark.asyncio
async def test_the_runner_sends_its_schemas_without_claiming_the_sampling_law() -> None:
    """Tools ride the per-call params; nothing else of the backend's is overwritten.

    Every non-``None`` field of a per-call overlay wins, so attaching schemas through a
    freshly built ``SamplingParams`` would pin temperature and the token budget to this
    library's defaults on *every* turn of *every* tool-using rollout -- a 6144-token
    budget quietly answering in 512. The blanks are what say "not this field".
    """
    gen = ScriptedGenerator(["done"])
    runner = toy_runner(gen, env=SubmitEnv([]), parser=HermesToolCallParser())

    runner.policy.default_params = SamplingParams(max_tokens=6144, temperature=0.3)
    await complete(runner, "q", max_steps=1)

    sent = gen.params[0]
    assert sent is not None
    assert sent.max_tokens == 6144
    assert sent.temperature == 0.3
    prepared = runner.policy.prepare([], tools=SubmitEnv([]).schemas)
    assert prepared.tools[0]["function"]["name"] == "submit"


@pytest.mark.asyncio
async def test_a_runners_own_sampling_params_survive_the_schemas() -> None:
    gen = ScriptedGenerator(["done"])
    runner = toy_runner(
        gen,
        env=SubmitEnv([]),
        parser=HermesToolCallParser(),
        sampling_params=SamplingParams(max_tokens=6144),
    )

    await complete(runner, "q", max_steps=1)

    sent = gen.params[0]
    assert sent is not None
    assert sent.max_tokens == 6144


# -- history fidelity -----------------------------------------------------------------


def test_a_natively_called_turn_is_recorded_with_its_calls() -> None:
    """The call is not in the text, so a message stored as text alone forgets it.

    By the next render the tool results would be answering a question the history no
    longer asks -- which is not an error anywhere, just an agent that appears to have
    been told things it never asked for.
    """
    messages: list[Any] = []
    result = GenerateResult(
        tokens=tuple(ord(c) for c in "ok"),
        text="ok",
        native_tool_calls=(NativeToolCall("grep", '{"pattern": "x"}', "call_1"),),
    )

    Policy(format=native_format(FakeChatTokenizer()), native=True).record_reply(
        result, messages
    )

    assert messages == [
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    # arguments parsed to a mapping: HF templates iterate them
                    # (Qwen's does `arguments|items`), and a JSON string there is
                    # a hard template error
                    "function": {"name": "grep", "arguments": '{"pattern": "x"}'},
                }
            ],
        }
    ]


def test_a_turn_with_no_native_calls_is_the_plain_assistant_message() -> None:
    messages: list[Any] = []

    Policy(format=test_format()).record_reply(
        GenerateResult(tokens=(ord("o"),), text="o"), messages
    )

    assert messages == [{"role": "assistant", "content": "o"}]


def test_a_policy_is_a_value_and_is_shared_rather_than_copied() -> None:
    """Frozen, so the one record a run is configured with cannot drift under it."""
    policy = toy_policy(ScriptedGenerator(["go"]))
    runner: Agent[Any, Any] = Agent(policy=policy, env=ToyEnv(), system_prompt="s")

    assert runner.policy is policy
    with pytest.raises(AttributeError):
        policy.version = "other"  # type: ignore[misc]


def _resolved_policy() -> Any:
    from focus.backends import Policy, PolicyFormat
    from focus.codec import ChatCodec
    from focus.generation.parsing import HermesToolCallParser
    from tests.helpers import FakeChatTokenizer, ScriptedGenerator

    gen = ScriptedGenerator(["go"])
    format = PolicyFormat(
        "m:hermes",
        ChatCodec(FakeChatTokenizer()),
        HermesToolCallParser(),
        resolved=True,
    )
    return Policy(format=format, complete=gen.agenerate_tokens)


def test_an_agent_picks_the_parser_for_a_resolved_format() -> None:
    from focus import Agent, Segments
    from focus.backends import TokenChunkParser
    from focus.generation.parsing import HermesToolCallParser, TextActionParser
    from tests.helpers import SubmitEnv, ToyEnv

    policy = _resolved_policy()
    tool_agent = Agent(policy=policy, env=SubmitEnv([]), system_prompt="sys")
    text_agent = Agent(policy=policy, env=ToyEnv(), system_prompt="sys")
    chunk_agent = Agent(
        policy=policy, env=ToyEnv(), system_prompt="sys", context=Segments(8)
    )

    assert tool_agent.policy is policy  # tools keep the model's dialect
    assert isinstance(text_agent.policy.format.parser, TextActionParser)
    assert isinstance(chunk_agent.policy.format.parser, TokenChunkParser)
    assert text_agent.policy.format.codec is policy.format.codec
    # a fold over a tool world gets the dialect back, whatever the task speaks
    from focus.harness.compaction import FoldEnv

    fold = text_agent.derive(
        env=FoldEnv(()), system_prompt="s", max_steps=1, finish_prompt=None
    )
    assert isinstance(fold.policy.format.parser, HermesToolCallParser)


def test_an_explicit_format_is_left_alone() -> None:
    from focus import Agent
    from focus.generation.parsing import HermesToolCallParser
    from tests.helpers import ScriptedGenerator, ToyEnv, toy_policy

    policy = toy_policy(ScriptedGenerator(["go"]), parser=HermesToolCallParser())
    agent = Agent(policy=policy, env=ToyEnv(), system_prompt="sys")
    assert agent.policy is policy
