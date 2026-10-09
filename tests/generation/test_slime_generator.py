"""SlimePolicy: post-injected, async-native, token/logprob-aligned."""

from __future__ import annotations

import asyncio
from typing import Any

from focus.generation import Policy, SamplingParams, SlimePolicy
from focus.registry import build
from tests.helpers import test_format


def _canned_output() -> dict[str, Any]:
    return {
        "text": " hi",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            # [logprob, token_id, ...], aligned one-to-one with sampled tokens.
            "output_token_logprobs": [[-0.1, 11], [-0.4, 22], [-0.5, 33]],
        },
    }


def test_agenerate_aligns_tokens_and_logprobs() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    async def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen.append((url, payload))
        return _canned_output()

    gen = SlimePolicy(
        model="test-model",
        format=test_format(),
        post=fake_post,
        url="http://router:9000/generate",
        default_params=SamplingParams(max_tokens=16, temperature=1.1, top_p=0.9),
    )
    result = asyncio.run(gen.agenerate_tokens((5, 6, 7), SamplingParams(max_tokens=4)))

    assert result.tokens == (11, 22, 33)
    assert result.logprobs == [-0.1, -0.4, -0.5]
    assert len(result.logprobs) == len(result.tokens)
    assert result.prefix_tokens == (5, 6, 7)
    assert result.text == " hi"
    assert result.stop_reason == "stop"

    url, payload = seen[0]
    assert url == "http://router:9000/generate"
    assert payload["input_ids"] == [5, 6, 7]
    assert payload["return_logprob"] is True
    assert payload["sampling_params"]["max_new_tokens"] == 4  # per-call override
    assert "top_k" not in payload["sampling_params"]  # unset top_k is not sent


def test_top_k_is_sent_only_when_set() -> None:
    seen: list[dict[str, Any]] = []

    async def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return _canned_output()

    gen = SlimePolicy(
        model="test-model",
        format=test_format(),
        post=fake_post,
        url="http://router:9000/generate",
        default_params=SamplingParams(
            max_tokens=512, temperature=1.0, top_p=0.99, top_k=10000
        ),
    )
    asyncio.run(gen.agenerate_tokens((1, 2, 3)))
    sp = seen[0]["sampling_params"]
    # matches verl/MemSearcher train rollout sampling
    assert sp["top_k"] == 10000
    assert sp["top_p"] == 0.99
    assert sp["temperature"] == 1.0
    assert sp["max_new_tokens"] == 512

    # top_k = -1 (disabled) is a real value, still forwarded (verl eval greedy)
    seen.clear()
    asyncio.run(
        gen.agenerate_tokens(
            (1, 2, 3), SamplingParams(temperature=0.0, top_p=1.0, top_k=-1)
        )
    )
    assert seen[0]["sampling_params"]["top_k"] == -1


def test_matched_stop_surfaced() -> None:
    seen: list[dict[str, Any]] = []

    def canned(matched: Any) -> dict[str, Any]:
        out = _canned_output()
        out["meta_info"]["finish_reason"] = {"type": "stop", "matched": matched}
        return out

    responses: list[dict[str, Any]] = []

    async def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return responses[-1]

    gen = SlimePolicy(
        model="test-model",
        format=test_format(),
        post=fake_post,
        url="http://router:9000/generate",
    )
    # eos finish: matched is the eos token id (int)
    responses.append(canned(151645))
    r = asyncio.run(gen.agenerate_tokens((1, 2)))
    assert r.stop_reason == "stop"
    assert r.matched_stop == 151645
    # stop-string finish (verl's illegal <|im_start|> stop): matched is the string
    responses.append(canned("<|im_start|>"))
    r2 = asyncio.run(gen.agenerate_tokens((1, 2)))
    assert r2.stop_reason == "stop"
    assert r2.matched_stop == "<|im_start|>"
    # unreported: None
    responses.append(_canned_output())
    r3 = asyncio.run(gen.agenerate_tokens((1, 2)))
    assert r3.matched_stop is None


def test_repetition_penalty_is_sent_only_when_set() -> None:
    seen: list[dict[str, Any]] = []

    async def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return _canned_output()

    gen = SlimePolicy(
        model="test-model",
        format=test_format(),
        post=fake_post,
        url="http://router:9000/generate",
    )
    # unset (None) -> not forwarded, so SGLang keeps its default (1.0, no penalty)
    asyncio.run(gen.agenerate_tokens((1, 2, 3), SamplingParams(max_tokens=8)))
    assert "repetition_penalty" not in seen[0]["sampling_params"]

    # set -> forwarded verbatim
    seen.clear()
    asyncio.run(
        gen.agenerate_tokens(
            (1, 2, 3), SamplingParams(max_tokens=8, repetition_penalty=1.05)
        )
    )
    assert seen[0]["sampling_params"]["repetition_penalty"] == 1.05


def test_sync_generate_bridges_async_post() -> None:
    async def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _canned_output()

    gen = SlimePolicy(
        model="test-model",
        format=test_format(),
        post=fake_post,
        url="http://router/generate",
    )
    result = gen.generate_tokens((1, 2))
    assert result.tokens == (11, 22, 33)


def test_headers_are_forwarded_when_set() -> None:
    seen: dict[str, Any] = {}

    async def fake_post(
        url: str, payload: dict[str, Any], headers: Any = None
    ) -> dict[str, Any]:
        seen["headers"] = headers
        return _canned_output()

    gen = SlimePolicy(
        model="test-model",
        format=test_format(),
        post=fake_post,
        url="http://router/generate",
        headers={"Authorization": "Bearer x"},
    )
    asyncio.run(gen.agenerate_tokens((1,)))
    assert seen["headers"] == {"Authorization": "Bearer x"}


def test_registered_under_slime_name() -> None:
    async def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _canned_output()

    gen = build(
        Policy,
        {"name": "slime", "model": "m", "format": test_format()},
        post=fake_post,
        url="http://router/generate",
    )
    assert isinstance(gen, SlimePolicy)
