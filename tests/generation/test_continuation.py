from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from focus import (
    Policy,
    SamplingParams,
    TokenChunk,
    TokenChunkParser,
    TokenContinuationPolicy,
)
from focus.generation import GenerateResult
from focus.generation.parsing import ActionParser
from focus.registry import build
from tests.helpers import test_format


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
    inner.decode = Mock(return_value="same text")
    policy = TokenContinuationPolicy(inner)
    history = [{"role": "user", "content": "problem"}]
    initial = policy.prepare(history).tokens
    policy.record_reply(GenerateResult(tokens=(12345, 67890)), history)
    checkpoint = deepcopy(history)
    restored = json.loads(json.dumps(history))
    assert policy.prepare(restored).tokens == (*initial, 12345, 67890)
    siblings = [deepcopy(restored), deepcopy(restored)]
    for messages, token in zip(siblings, [91, 92], strict=True):
        policy.record_reply(GenerateResult(tokens=(token,)), messages)
        assert policy.prepare(messages).tokens == (*initial, 12345, 67890, token)
    assert history == checkpoint
    prepared = policy.prepare(history)
    history[-1]["content"] = "edited"
    assert prepared.messages[-1]["content"] == "same text"
    with pytest.raises(ValueError, match="edited"):
        policy.prepare(history)


def test_custom_initial_preparation_and_tool_schemas_are_preserved():
    inner = backend()
    tools = [{"type": "function", "function": {"name": "test"}}]
    history = [{"role": "user", "content": "problem"}]
    prompt = replace(inner.prepare(history, tools), tokens=(99, 98))
    inner.prepare = Mock(return_value=prompt)
    policy = TokenContinuationPolicy(inner)
    policy.record_reply(GenerateResult(tokens=(42,)), history)
    continued = policy.prepare(history, tools)
    assert continued.tokens == (99, 98, 42)
    assert continued.tools == tools
    inner.prepare.assert_called_once_with(history[:1], tools)


@pytest.mark.parametrize("role", ["user", "tool", "assistant"])
def test_intervening_messages_rejected_before_generation(role):
    policy = TokenContinuationPolicy(backend())
    history = [{"role": "user", "content": "problem"}]
    policy.record_reply(GenerateResult(tokens=(42,)), history)
    history.append({"role": role, "content": "Continue"})
    with pytest.raises(ValueError, match="uninterrupted"):
        policy.validate_context(history)
    with pytest.raises(ValueError, match="uninterrupted"):
        policy.prepare(history)


@pytest.mark.asyncio
async def test_backend_identity_custom_hooks_and_lifecycle():
    inner = backend()
    result = GenerateResult(tokens=(42,), logprobs=[-0.2], stop_reason="stop")
    inner.agenerate_tokens = AsyncMock(return_value=result)
    inner.parse = Mock(return_value="custom action")
    inner.is_exact = Mock(return_value=False)
    inner.validate_context = Mock()
    inner.startup_check = Mock()
    inner.close = Mock()
    inner.aclose = AsyncMock()
    policy = TokenContinuationPolicy(inner)
    assert policy.format is inner.format
    assert (policy.model, policy.served_model, policy.version) == (
        "test-model",
        "served-model",
        "policy-7",
    )
    assert policy.default_params == inner.default_params
    params = SamplingParams(max_tokens=256)
    prompt = policy.prepare([{"role": "user", "content": "problem"}])
    assert await policy.agenerate(prompt, params) is result
    inner.agenerate_tokens.assert_awaited_once_with(prompt.tokens, params)
    assert policy.parse(result) == "custom action"
    assert not policy.is_exact(result)
    policy.validate_context(prompt.messages)
    inner.validate_context.assert_called_once_with(prompt.messages)
    policy.startup_check()
    inner.startup_check.assert_called_once()
    policy.close()
    inner.close.assert_called_once()
    await policy.aclose()
    inner.aclose.assert_awaited_once()
    with pytest.raises(ValueError, match="different policy format"):
        await policy.agenerate(backend().prepare([]))


@pytest.mark.asyncio
async def test_inexact_backend_and_per_call_provenance_rejected():
    inner = backend()
    inner.provides_exact_tokens = False
    with pytest.raises(ValueError, match="exact token-input"):
        TokenContinuationPolicy(inner)
    inner.provides_exact_tokens = True
    inner.provides_native_tool_calls = True
    with pytest.raises(ValueError, match="exact token-input"):
        TokenContinuationPolicy(inner)
    inner.provides_native_tool_calls = False
    result = GenerateResult(tokens=(42,), exact_generation=False)
    inner.agenerate_tokens = AsyncMock(return_value=result)
    policy = TokenContinuationPolicy(inner)
    with pytest.raises(ValueError, match="exact generated"):
        await policy.agenerate_tokens((1,))
    with pytest.raises(ValueError, match="exact text"):
        policy.record_reply(result, [])


@pytest.mark.parametrize("size", [1, 256, 512])
def test_chunk_parser_boundary_and_natural_completion(size):
    parser = TokenChunkParser(size)
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
    with pytest.raises(ValueError, match="boundary"):
        parser.parse(GenerateResult(tokens=tokens[:-1], stop_reason="length"))
    with pytest.raises(ValueError, match="exceeded"):
        parser.parse(GenerateResult(tokens=(*tokens, 1), stop_reason="stop"))
    with pytest.raises(ValueError, match="finish reason"):
        parser.parse(GenerateResult(tokens=tokens, stop_reason="error"))


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_chunk_sizes(size):
    with pytest.raises(ValueError, match="positive"):
        TokenChunkParser(size)


def test_configuration_registry_supports_the_same_customization():
    parser = build(ActionParser, {"name": "token_chunk", "chunk_size": 32})
    assert isinstance(parser, TokenChunkParser)
    assert parser.chunk_size == 32
    inner = backend()
    policy = build(Policy, {"name": "token_continuation", "backend": inner})
    assert isinstance(policy, TokenContinuationPolicy)
    assert policy.backend is inner
