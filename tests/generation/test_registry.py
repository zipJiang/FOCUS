from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

import focus.generation as generation
from focus.generation import (
    GenerateResult,
    Policy,
    SamplingParams,
    TinkerPolicy,
    TokenId,
    VLLMPolicy,
    register_policy,
)
from focus.registry import build, construct_target
from tests.helpers import test_format


class CustomGenerator(Policy[Any]):
    def __init__(self, value: int = 0) -> None:
        self.value = value

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        del sampling_params
        return GenerateResult(tokens=(self.value,), prefix_tokens=tuple(prefix_tokens))


@dataclass(frozen=True)
class CustomGeneratorConfig:
    name: str = "unit_custom"
    value: int = 0


class InjectedGenerator(Policy[Any]):
    def __init__(self, value: int, scale: int) -> None:
        self.value = value * scale

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        del sampling_params
        return GenerateResult(tokens=(self.value,), prefix_tokens=tuple(prefix_tokens))


class WrapperGenerator(Policy[Any]):
    def __init__(self, inner: Policy[Any]) -> None:
        self.inner = inner

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        return await self.inner.agenerate_tokens(prefix_tokens, sampling_params)


class FakeTokenizer:
    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(ch) for ch in text)

    def decode(self, tokens: Sequence[TokenId]) -> str:
        return "".join(chr(token) for token in tokens)


def test_builtins_from_params_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: MagicMock(),
    )
    sampling_client = MagicMock(spec=["sample_async", "get_tokenizer"])

    assert isinstance(
        build(
            Policy,
            {
                "name": "openai_completion",
                "base_url": "http://example.test/v1",
                "api_key": "key",
                "model": "test-model",
                "format": test_format(),
            },
        ),
        VLLMPolicy,
    )
    assert isinstance(
        build(
            Policy,
            {
                "name": "vllm",
                "base_url": "http://example.test/v1",
                "api_key": "key",
                "model": "test-model",
                "format": test_format(),
            },
        ),
        VLLMPolicy,
    )
    assert isinstance(
        build(
            Policy,
            {
                "name": "tinker",
                "sampling_client": sampling_client,
                "model": "m",
                "profile": "text",
            },
        ),
        TinkerPolicy,
    )


def test_third_party_generator_registers_and_constructs() -> None:
    register_policy("unit_custom", CustomGenerator, replace=True)

    built = build(Policy, {"name": "unit_custom", "value": 7})

    assert isinstance(built, CustomGenerator)
    assert built.value == 7


def test_a_bound_registrar_describes_itself_rather_than_functools() -> None:
    """``register_policy`` is where the docs send a backend author, and ``help`` on
    it used to answer with ``functools.partial``'s own docstring -- true of the object
    and useless about the slot it registers into."""
    assert "Policy" in (register_policy.__doc__ or "")


def test_a_config_of_the_wrong_shape_says_what_shape_would_work() -> None:
    """A bare ``"vllm"`` is the natural thing to try in YAML, and the refusal has to
    carry the mapping form or the author is left guessing at punctuation."""
    with pytest.raises(TypeError, match=r'expected a mapping like \{"name": \.\.\.\}'):
        build(Policy, "vllm")


def test_from_params_accepts_instances_dataclasses_and_target_configs() -> None:
    register_policy("unit_custom", CustomGenerator, replace=True)
    instance = CustomGenerator(value=2)

    assert build(Policy, instance) is instance

    from_dataclass = build(Policy, CustomGeneratorConfig(value=3))
    assert isinstance(from_dataclass, CustomGenerator)
    assert from_dataclass.value == 3

    from_target = build(
        Policy,
        {
            "_target_": f"{CustomGenerator.__module__}.CustomGenerator",
            "value": 4,
        },
    )
    assert isinstance(from_target, CustomGenerator)
    assert from_target.value == 4


def test_construct_target_builds_unregistered_config_object() -> None:
    built = construct_target(
        {
            "_target_": f"{CustomGenerator.__module__}.CustomGenerator",
            "value": 6,
        }
    )

    assert isinstance(built, CustomGenerator)
    assert built.value == 6


def test_from_params_uses_dependency_injection_kwargs() -> None:
    register_policy("unit_injected", InjectedGenerator, replace=True)

    injected = build(
        Policy,
        {"name": "unit_injected", "value": 5},
        scale=2,
        ignored_dependency=True,
    )
    overridden = build(
        Policy,
        {"name": "unit_injected", "value": 5, "scale": 3},
        scale=2,
    )

    assert isinstance(injected, InjectedGenerator)
    assert injected.value == 10
    assert isinstance(overridden, InjectedGenerator)
    assert overridden.value == 15


def test_from_params_infers_nested_registrable_from_annotation() -> None:
    register_policy("unit_custom", CustomGenerator, replace=True)
    register_policy("unit_wrapper", WrapperGenerator, replace=True)

    wrapper = build(
        Policy,
        {"name": "unit_wrapper", "inner": {"name": "unit_custom", "value": 6}},
    )

    assert isinstance(wrapper, WrapperGenerator)
    assert isinstance(wrapper.inner, CustomGenerator)
    assert wrapper.inner.value == 6


def test_missing_name_unknown_name_duplicate_and_wrong_type_fail() -> None:
    with pytest.raises(ValueError, match="must include a string name"):
        build(Policy, {"value": 1})

    with pytest.raises(ValueError, match="unknown policy 'missing'"):
        build(Policy, {"name": "missing"})

    register_policy("unit_duplicate", CustomGenerator, replace=True)
    with pytest.raises(ValueError, match="already registered"):
        register_policy("unit_duplicate", CustomGenerator)

    register_policy("unit_wrong_type", lambda: object(), replace=True)
    with pytest.raises(TypeError, match="returned object"):
        build(Policy, {"name": "unit_wrong_type"})


def test_removed_build_apis_are_not_exported() -> None:
    assert not hasattr(Policy, "build")
    assert "build_generator" not in generation.__all__
