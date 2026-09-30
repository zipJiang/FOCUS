"""Async Tinker sampling generator."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from focus.generation import SamplingParams


def _install_fake_tinker(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Register a minimal fake ``tinker`` module in ``sys.modules``."""

    captured: dict[str, object] = {}

    class FakeModelInput:
        def __init__(self, ints: list[int]):
            self.ints = ints

        @classmethod
        def from_ints(cls, ints: list[int]) -> FakeModelInput:
            return cls(list(ints))

    class FakeSamplingParams:
        def __init__(self, **kwargs: object):
            captured["sampling_params_kwargs"] = kwargs
            self.kwargs = kwargs

    # `Any`: these are stub modules populated by assignment, which is exactly the
    # shape a module object has no static attributes for.
    types_module: Any = types.ModuleType("tinker.types")
    types_module.ModelInput = FakeModelInput

    tinker_module: Any = types.ModuleType("tinker")
    tinker_module.types = types_module
    tinker_module.SamplingParams = FakeSamplingParams

    monkeypatch.setitem(sys.modules, "tinker", tinker_module)
    monkeypatch.setitem(sys.modules, "tinker.types", types_module)
    return SimpleNamespace(captured=captured, FakeModelInput=FakeModelInput)


def test_async_tinker_generator_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = _install_fake_tinker(monkeypatch)

    # Import after patching sys.modules so the lazy imports see the fake.
    from focus.generation.tinker import TinkerPolicy

    sequence = SimpleNamespace(
        tokens=[42, 43, 44],
        stop_reason="stop",
        logprobs=[-0.1, -0.2, -0.3],
    )
    response = SimpleNamespace(
        sequences=[sequence],
        prompt_logprobs=[None, -0.05, -0.06],
        topk_prompt_logprobs=[
            None,
            [(10, -0.05), (11, -0.5)],
            [(12, -0.06), (13, -0.7)],
        ],
    )

    sampling_client = MagicMock(spec=["get_tokenizer", "sample_async", "close"])
    sampling_client.sample_async = AsyncMock(return_value=response)
    sampling_client.get_tokenizer.return_value.decode.return_value = "hello"

    engine = TinkerPolicy(
        model="test-model", profile="text", sampling_client=sampling_client
    )
    sp = SamplingParams(
        temperature=0.3,
        max_tokens=64,
        top_p=0.9,
        stop=["</s>"],
        logprobs=1,
        prompt_logprobs=2,
    )
    out = engine.generate_tokens([10, 11, 12], sp)

    assert out.prefix_tokens == (10, 11, 12)
    assert out.tokens == (42, 43, 44)
    assert out.text == "hello"
    assert out.stop_reason == "stop"
    assert out.logprobs == [-0.1, -0.2, -0.3]
    assert out.prompt_logprobs == [None, -0.05, -0.06]
    assert out.prompt_top_logprobs == [
        None,
        {"10": -0.05, "11": -0.5},
        {"12": -0.06, "13": -0.7},
    ]
    assert out.entropy == 0.0
    sampling_client.get_tokenizer.assert_called_once()

    sampling_client.sample_async.assert_awaited_once()
    call_kwargs = sampling_client.sample_async.await_args.kwargs
    assert isinstance(call_kwargs["prompt"], fakes.FakeModelInput)
    assert call_kwargs["prompt"].ints == [10, 11, 12]
    assert call_kwargs["num_samples"] == 1
    assert call_kwargs["include_prompt_logprobs"] is True
    assert call_kwargs["topk_prompt_logprobs"] == 2

    captured = fakes.captured["sampling_params_kwargs"]
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 64
    assert captured["top_p"] == 0.9
    assert captured["stop"] == ["</s>"]

    engine.close()
    sampling_client.close.assert_called_once()


def test_async_tinker_generator_length_stop_and_no_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tinker(monkeypatch)

    from focus.generation.tinker import TinkerPolicy

    sequence = SimpleNamespace(tokens=[99], stop_reason="length", logprobs=[-0.5])
    response = SimpleNamespace(sequences=[sequence])

    sampling_client = MagicMock(spec=["get_tokenizer", "sample_async", "close"])
    sampling_client.sample_async = AsyncMock(return_value=response)
    sampling_client.get_tokenizer.return_value.decode.return_value = "x"

    with TinkerPolicy(
        model="test-model", profile="text", sampling_client=sampling_client
    ) as engine:
        out = engine.generate_tokens([1], SamplingParams())

    assert out.stop_reason == "length"
    assert out.prefix_tokens == (1,)
    assert out.tokens == (99,)
    assert out.text == "x"
    assert out.logprobs == []
    sampling_client.close.assert_called_once()
