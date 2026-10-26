from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from focus import AppendOnly
from focus.backends import (
    Policy,
    PreparedPrompt,
    SamplingParams,
    TokenChunk,
    TokenChunkParser,
)
from focus.execution import ExecutionServices
from focus.generation import GenerateResult
from tests.helpers import test_format


def stub(**kwargs: Any) -> Any:
    return SimpleNamespace(**kwargs)


def prepare(policy, messages, tools=()):
    return AppendOnly(continuation=True).prepare(
        stub(messages=tuple(messages)),
        stub(policy=policy, tools=tools),
    )


def record(policy, result, messages):
    AppendOnly(continuation=True).record_reply(result, messages, stub(policy=policy))


async def generate(policy, tokens, params=None):
    prompt = PreparedPrompt(
        tuple(tokens), policy.format, (), (), exact_continuation=True
    )
    services = ExecutionServices()
    try:
        return await services.generate(policy, prompt, params)
    finally:
        await services.aclose()


def backend():
    return Policy(
        format=test_format(),
        model="test-model",
        served_model="served-model",
        version="policy-7",
        default_params=SamplingParams(temperature=0.3),
    )


def test_original_ids_survive_lossy_decode_json_and_sibling_branches():
    inner = backend()
    # Distinct IDs have the same display text. Re-tokenizing loses that distinction.
    inner.format.codec.decode = Mock(return_value="same text")
    policy = inner
    history = [{"role": "user", "content": "problem"}]
    initial = prepare(policy, history).tokens
    record(policy, GenerateResult(tokens=(12345, 67890)), history)
    checkpoint = deepcopy(history)
    restored = json.loads(json.dumps(history))
    assert prepare(policy, restored).tokens == (*initial, 12345, 67890)
    siblings = [deepcopy(restored), deepcopy(restored)]
    for messages, token in zip(siblings, [91, 92], strict=True):
        record(policy, GenerateResult(tokens=(token,)), messages)
        assert prepare(policy, messages).tokens == (*initial, 12345, 67890, token)
    assert history == checkpoint
    prepared = prepare(policy, history)
    history[-1]["content"] = "edited"
    assert prepared.messages[-1]["content"] == "same text"
    with pytest.raises(ValueError, match="edited"):
        prepare(policy, history)


def test_initial_codec_render_and_tool_schemas_are_preserved():
    policy = backend()
    tools = [{"type": "function", "function": {"name": "test"}}]
    history = [{"role": "user", "content": "problem"}]
    policy.format.codec.render = Mock(return_value=(99, 98))
    record(policy, GenerateResult(tokens=(42,)), history)
    continued = prepare(policy, history, tools)
    assert continued.tokens == (99, 98, 42)
    assert continued.tools == tools
    policy.format.codec.render.assert_called_once_with(
        tuple(history[:1]), tools=tuple(tools)
    )


@pytest.mark.parametrize("role", ["user", "tool", "assistant"])
def test_intervening_messages_rejected_before_generation(role):
    policy = backend()
    history = [{"role": "user", "content": "problem"}]
    record(policy, GenerateResult(tokens=(42,)), history)
    history.append({"role": role, "content": "Continue"})
    with pytest.raises(ValueError, match="uninterrupted"):
        prepare(policy, history)
    with pytest.raises(ValueError, match="uninterrupted"):
        prepare(policy, history)


async def test_custom_parser_and_model_identity_are_independent_of_continuation():
    policy = backend()
    result = GenerateResult(tokens=(42,), logprobs=[-0.2], stop_reason="stop")
    policy._complete = AsyncMock(return_value=result)
    policy.format.parser.parse = Mock(return_value="custom action")
    assert (policy.model, policy.served_model, policy.version) == (
        "test-model",
        "served-model",
        "policy-7",
    )
    prompt = prepare(policy, [{"role": "user", "content": "problem"}])
    assert (
        await generate(policy, prompt.tokens, SamplingParams(max_tokens=256)) is result
    )
    assert policy.parse(result) == "custom action"
    with pytest.raises(ValueError, match="different policy format"):
        await policy.agenerate(backend().prepare([]))


@pytest.mark.parametrize(("exact", "native"), [(False, False), (True, True)])
def test_incompatible_backend_rejected(exact, native):
    policy = Policy(format=test_format(), exact=exact, native=native)
    with pytest.raises(ValueError, match="exact token-input"):
        prepare(policy, [])


async def test_inexact_per_call_provenance_rejected():
    policy = backend()
    result = GenerateResult(tokens=(42,), exact_generation=False)
    policy._complete = AsyncMock(return_value=result)
    with pytest.raises(ValueError, match="exact generated"):
        await generate(policy, (1,))
    with pytest.raises(ValueError, match="exact text"):
        record(policy, result, [])


@pytest.mark.parametrize("tokens", [None, "123", [True], [1.5]])
def test_invalid_continuation_metadata_rejected(tokens):
    policy = backend()
    messages = [
        {"role": "assistant", "content": "", "focus_continuation_tokens": tokens}
    ]
    with pytest.raises(ValueError, match="integer token IDs"):
        prepare(policy, messages)


def test_default_format_renders_each_chat_turn_without_continuation_metadata():
    policy = Policy(format=test_format())
    messages = [{"role": "user", "content": "problem"}]
    protocol = AppendOnly()
    runner = stub(policy=policy, tools=())
    protocol.record_reply(GenerateResult(tokens=(42,)), messages, runner)
    assert "focus_continuation_tokens" not in messages[-1]
    assert protocol.prepare(
        stub(messages=messages), runner
    ).tokens == policy.format.codec.render(messages)


@pytest.mark.parametrize("size", [1, 256, 512])
def test_chunk_parser_boundary_and_natural_completion(size):
    parser = TokenChunkParser()
    tokens = tuple(range(size))
    assert parser.parse(
        GenerateResult(tokens=tokens, stop_reason="length")
    ) == TokenChunk(tokens, False)
    assert parser.parse(
        GenerateResult(tokens=tokens, stop_reason="stop")
    ) == TokenChunk(tokens, True)
    assert parser.parse(GenerateResult(tokens=(), stop_reason="stop")) == TokenChunk(
        (), True
    )
    with pytest.raises(ValueError, match="finish reason"):
        parser.parse(GenerateResult(tokens=tokens, stop_reason="error"))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
async def test_invalid_request_limits_fail_before_generation(limit):
    inner = backend()
    inner._complete = AsyncMock()
    policy = inner
    with pytest.raises(ValueError, match="positive integer"):
        await generate(policy, (1,), SamplingParams(max_tokens=limit))
    inner._complete.assert_not_awaited()


@pytest.mark.parametrize("override", [None, 4, 16])
@pytest.mark.parametrize("finish", ["length", "stop"])
async def test_generation_validates_the_effective_request_limit(override, finish):
    inner = backend()
    inner.default_params = SamplingParams(max_tokens=8, temperature=0.3)
    inner._complete = AsyncMock()
    policy = inner
    params = None if override is None else SamplingParams(max_tokens=override)
    limit = 8 if override is None else override
    inner._complete.return_value = GenerateResult(
        tokens=(42,) * limit,
        stop_reason=finish,
    )
    result = await generate(policy, (1,), params)
    assert len(result.tokens) == limit
    assert inner._complete.call_args.args[1].max_tokens == limit
    inner._complete.return_value = GenerateResult(
        tokens=(42,) * (limit + 1),
        stop_reason=finish,
    )
    with pytest.raises(ValueError, match="exceeded"):
        await generate(policy, (1,), params)
    inner._complete.return_value = GenerateResult(
        tokens=(42,) * (limit - 1),
        stop_reason=finish,
    )
    if finish == "length":
        with pytest.raises(ValueError, match="before the requested"):
            await generate(policy, (1,), params)
    else:
        assert len((await generate(policy, (1,), params)).tokens) == limit - 1


async def test_partial_override_keeps_the_default_token_limit():
    inner = backend()
    inner.default_params = SamplingParams(max_tokens=3, temperature=0.1)
    inner._complete = AsyncMock(
        return_value=GenerateResult(
            tokens=(1, 2, 3),
            stop_reason="length",
        )
    )
    policy = inner
    await generate(policy, (9,), SamplingParams(max_tokens=None, temperature=0.7))
    effective = inner._complete.call_args.args[1]
    assert effective.max_tokens == 3
    assert effective.temperature == 0.7


@pytest.fixture(params=["vllm", "sglang", "slime", "tinker"])
def continuation_backend(request, monkeypatch):
    from focus.generation import SGLangPolicy, SlimePolicy, TinkerPolicy, VLLMPolicy
    from tests.generation.test_tinker import _install_fake_tinker

    format = test_format()
    defaults = SamplingParams(max_tokens=2)
    output = {
        "text": "ab",
        "meta_info": {
            "finish_reason": {"type": "length"},
            "output_token_logprobs": [[-0.1, 97], [-0.2, 98]],
        },
    }
    if request.param == "vllm":
        response = stub(
            choices=[
                stub(
                    text="ab",
                    token_ids=[97, 98],
                    finish_reason="length",
                    logprobs=None,
                )
            ]
        )
        sync = stub(completions=stub(create=Mock(return_value=response)))
        async_client = stub(completions=stub(create=AsyncMock(return_value=response)))
        monkeypatch.setattr(
            "focus.generation.vllm._import_openai", lambda: lambda **kw: sync
        )
        policy = VLLMPolicy(
            model="test",
            format=format,
            base_url="http://test/v1",
            api_key="test",
            default_params=defaults,
        )
        monkeypatch.setattr(policy, "_async_client", lambda: async_client)
        return policy
    if request.param == "sglang":
        response = stub(raise_for_status=lambda: None, json=lambda: output)
        clients = stub(
            Client=lambda **kw: stub(post=Mock(return_value=response)),
            AsyncClient=lambda **kw: stub(post=AsyncMock(return_value=response)),
        )
        monkeypatch.setattr("focus.generation.sglang._import_httpx", lambda: clients)
        return SGLangPolicy(
            model="test", format=format, base_url="http://test", default_params=defaults
        )
    if request.param == "slime":
        return SlimePolicy(
            AsyncMock(return_value=output),
            "http://test/generate",
            model="test",
            format=format,
            default_params=defaults,
        )
    _install_fake_tinker(monkeypatch)
    client: Any = stub(
        get_tokenizer=lambda: format.codec.tokenizer,
        sample_async=AsyncMock(
            return_value=stub(
                sequences=[
                    stub(tokens=[97, 98], stop_reason="length", logprobs=None),
                ]
            )
        ),
    )
    return TinkerPolicy(client, model="test", format=format, default_params=defaults)


async def test_backend_generation_checks_continuation_limits(continuation_backend):
    policy = continuation_backend

    assert (await generate(policy, (1,))).tokens == (97, 98)
    with pytest.raises(ValueError, match="exceeded"):
        await generate(policy, (1,), SamplingParams(max_tokens=1))
    with pytest.raises(ValueError, match="before the requested"):
        await generate(policy, (1,), SamplingParams(max_tokens=3))
    with pytest.raises(ValueError, match="positive integer"):
        await generate(policy, (1,), SamplingParams(max_tokens=0))


def test_same_policy_supports_independent_context_protocols():
    policy = backend()
    history = [{"role": "user", "content": "problem"}]
    initial = prepare(policy, history).tokens
    record(policy, GenerateResult(tokens=(42,)), history)
    continued = prepare(policy, history)
    ordinary = AppendOnly().prepare(
        stub(messages=history), stub(policy=policy, tools=())
    )
    assert continued.tokens == (*initial, 42)
    assert continued.exact_continuation
    assert not ordinary.exact_continuation
    assert ordinary.tokens == policy.format.codec.render(history)
    assert not hasattr(policy.format, "continuation")


def test_segments_sets_the_continuation_defaults_on_the_agent():
    from focus import Agent, Segments
    from focus.backends import PolicyFormat
    from focus.codec import ChatCodec
    from focus.extend import EveryDecision
    from tests.helpers import FakeChatTokenizer, ScriptedGenerator, ToyEnv, toy_policy

    format = PolicyFormat("chunks", ChatCodec(FakeChatTokenizer()), TokenChunkParser())
    policy = toy_policy(ScriptedGenerator(["go"]))
    policy = replace_format(policy, format)
    agent = Agent(policy=policy, env=ToyEnv(), system_prompt="sys", context=Segments(4))

    assert agent._sampling_params is not None
    assert agent._sampling_params.max_tokens == 4
    assert isinstance(agent._checkpoints, EveryDecision)
    assert agent._finish_prompt is None
    assert isinstance(agent.context, Segments)
    assert agent.context.continuation
    with pytest.raises(ValueError, match="positive"):
        Segments(0)


def replace_format(policy, format):
    """The same backend under another format."""
    from focus.backends import Policy

    return Policy(
        format=format,
        complete=policy._complete,
        exact=policy.provides_exact_tokens,
        native=policy.provides_native_tool_calls,
    )
