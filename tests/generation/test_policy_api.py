"""Policy boundaries and native history across environment transitions and forks."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from focus import (
    Agent,
    NullWorkspace,
    OpenAIChatPolicy,
    OpenAIResponsesPolicy,
    Policy,
    PolicyFormat,
    SamplingParams,
    Summarize,
    TinkerPolicy,
    VLLMPolicy,
)
from focus.generation import GenerateResult, NativeToolCall
from focus.generation.parsing import TextActionParser
from focus.harness import format_transcript
from tests.generation.test_openai_chat import (
    _response as chat_response,
)
from tests.generation.test_openai_chat import (
    _tool_call,
)
from tests.generation.test_openai_chat import (
    _wire as chat_wire,
)
from tests.generation.test_openai_responses import (
    _call_item,
    _text_item,
)
from tests.generation.test_openai_responses import (
    _response as responses_response,
)
from tests.generation.test_openai_responses import (
    _wire as responses_wire,
)
from tests.helpers import FakeChatTokenizer, SubmitEnv, native_format, test_format


@pytest.mark.parametrize(
    ("model", "parser"),
    [
        ("Qwen/Qwen3.5-9B", "QwenXMLToolCallParser"),
        ("Qwen/Qwen2.5-7B-Instruct", "HermesToolCallParser"),
    ],
)
def test_known_formats_bind_the_tokenizer_once(model: str, parser: str) -> None:
    tokenizer = FakeChatTokenizer()
    format = PolicyFormat.resolve(model, tokenizer=tokenizer)
    assert format.codec.tokenizer is tokenizer
    assert type(format.parser).__name__ == parser


def test_custom_format_is_explicit_and_prepared_identity_is_checked() -> None:
    with pytest.raises(ValueError, match="supply a profile"):
        PolicyFormat.resolve("custom-model", tokenizer=FakeChatTokenizer())
    format = test_format()
    policy = Policy(format=format)
    prompt = policy.prepare([{"role": "user", "content": "q"}])
    other = Policy(format=test_format())
    with pytest.raises(ValueError, match="different policy format"):
        other.generate(prompt)
    with pytest.raises(AttributeError):
        policy.format = test_format()  # type: ignore[misc]


def test_prepared_prompt_snapshots_messages_and_tool_schemas() -> None:
    policy = Policy(format=test_format())
    messages = [{"role": "user", "content": "q"}]
    tools = SubmitEnv([]).schemas
    expected = deepcopy(tools)
    prompt = policy.prepare(messages, tools)
    messages[0]["content"] = "changed"
    tools[0]["function"]["name"] = "changed"
    prompt.messages[0]["content"] = "changed again"
    prompt.tools[0]["function"]["name"] = "changed again"
    assert prompt.messages == [{"role": "user", "content": "q"}]
    assert prompt.tools == expected
    assert prompt.tokens == tuple(map(ord, "uqa"))


async def test_vllm_sends_prepared_tokens_and_separate_serving_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    token_ids=[120],
                    text="server text",
                    finish_reason="stop",
                    logprobs=None,
                )
            ]
        )

    monkeypatch.setattr("focus.generation.vllm._import_openai", lambda: MagicMock())
    policy = VLLMPolicy(
        model="artifact",
        served_model="actor-v7",
        format=test_format(),
        base_url="http://gh130:8000/v1",
        api_key="EMPTY",
        default_params=SamplingParams(max_tokens=4096, temperature=0.2),
    )
    policy._aclient = SimpleNamespace(completions=SimpleNamespace(create=create))
    prompt = policy.prepare([{"role": "user", "content": "question"}])
    result = await policy.agenerate(prompt)
    assert requests[0]["prompt"] == list(prompt.tokens)
    assert requests[0]["model"] == "actor-v7"
    assert requests[0]["max_tokens"] == 4096
    assert requests[0]["extra_body"]["return_token_ids"]
    assert result.prefix_tokens == prompt.tokens
    assert result.tokens == (120,)
    assert result.exact_generation
    assert policy.trainable
    history: list[dict[str, Any]] = []
    policy.record_reply(result, history)
    assert result.text == "x"
    assert policy.model == "artifact"
    forbidden: dict[str, Any] = {"model": "different"}
    with pytest.raises(TypeError, match="model"):
        SamplingParams(**forbidden)


def test_tinker_uses_client_tokenizer_and_rejects_an_unrelated_binding() -> None:
    client = MagicMock()
    tokenizer = FakeChatTokenizer()
    client.get_tokenizer.return_value = tokenizer
    policy = TinkerPolicy(client, model="custom", profile="text")
    assert policy.format.codec.tokenizer is tokenizer
    client.get_tokenizer.assert_called_once()
    with pytest.raises(ValueError, match="client's tokenizer"):
        TinkerPolicy(client, model="custom", format=test_format())


@pytest.mark.parametrize("backend", ["chat", "responses"])
async def test_native_tools_errors_nudges_and_forks(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire: Any
    policy: Policy[Any]
    if backend == "chat":
        wire = chat_wire(
            monkeypatch,
            chat_response(
                None,
                tool_calls=[
                    _tool_call("missing", "{}", "c1"),
                    _tool_call("also_missing", "{malformed", "c2"),
                ],
            ),
        )
        policy = OpenAIChatPolicy(
            model="corporate", format=native_format(FakeChatTokenizer())
        )
    else:
        wire = responses_wire(
            monkeypatch,
            responses_response(
                _call_item("missing", "{}", "c1"),
                _call_item("also_missing", "{malformed", "c2"),
            ),
        )
        policy = OpenAIResponsesPolicy(
            model="corporate", format=native_format(FakeChatTokenizer())
        )
    runner = Agent(policy=policy, env=SubmitEnv([]), system_prompt="sys")
    rs = await runner.start("q", workspace=NullWorkspace())
    rs = await runner.advance(rs)
    replies = [m for m in rs.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in replies] == ["c1", "c2"]
    assert all("does not exist" in m["content"] for m in replies)
    assert not rs.turns[-1].exact
    assert "missing {}" in format_transcript(rs.messages)
    saved = deepcopy(rs.messages)
    wire.response = (
        chat_response("thinking")
        if backend == "chat"
        else responses_response(_text_item("thinking"))
    )
    left, right = await asyncio.gather(
        runner.advance(rs.snapshot()), runner.advance(rs.snapshot())
    )
    assert rs.messages == saved
    assert left.messages == right.messages
    key = "messages" if backend == "chat" else "input"
    assert wire.calls[1][key] == wire.calls[2][key]
    assert wire.calls[1][key][0] == {"role": "system", "content": "sys"}
    # The nudge has no corresponding call; it is feedback in the native request.
    await runner.advance(left)
    assert wire.calls[-1][key][-1]["role"] == "user"
    if backend == "responses":
        items = wire.calls[1][key]
        assert [
            i["call_id"] for i in items if i.get("type") == "function_call_output"
        ] == ["c1", "c2"]
        assert all(
            c["store"] is False and "previous_response_id" not in c for c in wire.calls
        )
    with pytest.raises(NotImplementedError, match="token input"):
        await policy.agenerate_tokens([1, 2])


@pytest.mark.parametrize("arguments", ["{bad", "{}"])
async def test_malformed_submit_answers_every_skipped_sibling(
    arguments: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = chat_wire(
        monkeypatch,
        chat_response(
            None,
            tool_calls=[
                _tool_call("submit", arguments, "submit1"),
                _tool_call("missing", "{}", "skipped1"),
            ],
        ),
    )
    policy = OpenAIChatPolicy(
        model="corporate", format=native_format(FakeChatTokenizer())
    )
    runner = Agent(policy=policy, env=SubmitEnv([]), system_prompt="sys")
    rs = await runner.advance(await runner.start("q", workspace=NullWorkspace()))
    replies = [m for m in rs.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in replies] == ["submit1", "skipped1"]
    assert "Not executed" in replies[1]["content"]
    wire.response = chat_response("considering")
    await runner.advance(rs)
    assert wire.calls[-1]["messages"][-1] == replies[-1]


async def test_responses_keeps_encrypted_reasoning_for_stateless_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = SimpleNamespace(
        type="reasoning", id="r1", summary=[], encrypted_content="opaque"
    )
    wire = responses_wire(
        monkeypatch, responses_response(reasoning, _call_item("missing", "{}", "c1"))
    )
    policy = OpenAIResponsesPolicy(
        model="corporate", format=native_format(FakeChatTokenizer())
    )
    runner = Agent(policy=policy, env=SubmitEnv([]), system_prompt="sys")
    rs = await runner.advance(await runner.start("q", workspace=NullWorkspace()))
    wire.response = responses_response(_text_item("done"))
    await runner.advance(rs)
    assert wire.calls[-1]["input"][2] == vars(reasoning)
    assert wire.calls[-1]["include"] == ["reasoning.encrypted_content"]


async def test_native_compaction_uses_same_policy_without_orphan_tool_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = chat_wire(monkeypatch, chat_response("notes"))
    policy = OpenAIChatPolicy(
        model="corporate", format=native_format(FakeChatTokenizer())
    )
    runner = Agent(
        policy=policy,
        env=SubmitEnv([]),
        system_prompt="sys",
        context=Summarize(max_prompt_tokens=10000),
    )
    rs = await runner.start("q", workspace=NullWorkspace())
    folded = await runner.compact(rs)
    assert folded.folds == 1
    assert not folded.turns[-1].exact
    await runner.advance(folded)
    assert len(wire.calls) == 2
    assert "notes" in wire.calls[-1]["messages"][-1]["content"]


def test_native_format_cannot_be_bound_to_token_backend() -> None:
    with pytest.raises(ValueError, match="structured tool-call channel"):
        Policy(format=native_format(FakeChatTokenizer()))
    with pytest.raises(ValueError, match="native format"):
        OpenAIChatPolicy(model="corporate", format=test_format())
    assert isinstance(test_format().parser, TextActionParser)


@pytest.mark.parametrize("call_id", ["wrong", ""])
async def test_native_history_refuses_unmatched_tool_results(call_id: str) -> None:
    policy = OpenAIChatPolicy(
        model="corporate", format=native_format(FakeChatTokenizer())
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
    result = GenerateResult(
        text="visible response",
        tokens=(120,),
        native_tool_calls=(NativeToolCall("missing", "{}", "c1"),),
        exact_generation=False,
    )
    policy.record_reply(result, messages)
    assert messages[-1]["content"] == "visible response"
    messages.append({"role": "tool", "content": "output", "tool_call_id": call_id})
    with pytest.raises(ValueError, match=r"Unmatched|missing tool_call_id"):
        await policy.agenerate(policy.prepare(messages))
    assert policy._aclient is None  # rejected before contacting the provider


async def test_native_async_lifecycle_closes_client_once() -> None:
    policy = OpenAIChatPolicy(
        model="corporate", format=native_format(FakeChatTokenizer())
    )
    close = AsyncMock()
    policy._aclient = SimpleNamespace(close=close)
    await policy.aclose()
    await policy.aclose()
    close.assert_awaited_once()


def test_vllm_sync_prepared_generation_uses_sync_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                token_ids=[120],
                text="x",
                finish_reason="stop",
                logprobs=None,
            )
        ]
    )
    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: lambda **kwargs: client,
    )
    policy = VLLMPolicy(
        model="artifact", format=test_format(), base_url="http://x/v1", api_key="EMPTY"
    )
    prompt = policy.prepare([{"role": "user", "content": "q"}])
    result = policy.generate(prompt)
    assert result.prefix_tokens == prompt.tokens
    assert client.completions.create.call_args.kwargs["prompt"] == list(prompt.tokens)
    assert policy._aclient is None


@pytest.mark.parametrize("backend", ["chat", "responses"])
@pytest.mark.parametrize("valid", [True, False])
async def test_native_reminder_and_final_attempt_are_api_valid(
    backend, valid, monkeypatch
):
    wire: Any
    policy: Policy[Any]
    if backend == "chat":
        wire = chat_wire(monkeypatch, chat_response("draft"))
        policy = OpenAIChatPolicy(
            model="corporate", format=native_format(FakeChatTokenizer())
        )
    else:
        reasoning = SimpleNamespace(
            type="reasoning", id="r1", summary=[], encrypted_content="opaque"
        )
        wire = responses_wire(
            monkeypatch, responses_response(reasoning, _text_item("draft"))
        )
        policy = OpenAIResponsesPolicy(
            model="corporate", format=native_format(FakeChatTokenizer())
        )
    runner = Agent(policy=policy, env=SubmitEnv([]), system_prompt="sys", max_steps=1)
    rs = await runner.advance(await runner.start("q"))
    assert not rs.done
    assert rs.state.answer is None
    assert rs.messages[-1]["role"] == "tool"
    if backend == "chat":
        wire.response = (
            chat_response(
                None, tool_calls=[_tool_call("submit", '{"answer":"42"}', "s1")]
            )
            if valid
            else chat_response("42")
        )
    else:
        wire.response = responses_response(
            _call_item("submit", '{"answer":"42"}', "s1") if valid else _text_item("42")
        )
    result = await runner.finish(rs)
    assert result.done
    assert result.truncated
    assert result.state.answer == ("42" if valid else None)
    assert result.reward_outcome == (1 if valid else 0)
    sent = wire.calls[-1]["messages" if backend == "chat" else "input"]
    assert sent[-1]["role"] == sent[-2]["role"] == "user"
    assert "final-answer tool" in sent[-1]["content"]
    assert not any(m.get("type") == "function_call_output" for m in sent)
    if backend == "responses":
        assert any(m.get("encrypted_content") == "opaque" for m in sent)
