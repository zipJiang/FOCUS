from __future__ import annotations

import types
from typing import Any, ClassVar
from unittest.mock import Mock

import pytest

from focus.generation import SamplingParams
from focus.generation.vllm import (
    VLLMPolicy,
    _parse_vllm_token_ids,
)
from tests.helpers import test_format


def encode_text(text: str) -> tuple[int, ...]:
    return tuple(ord(ch) for ch in text)


def _default_params(**kwargs: Any) -> SamplingParams:
    return SamplingParams(**kwargs)


class FakeModels:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.list_calls = 0

    def list(self) -> object:
        self.list_calls += 1
        if self.failure is not None:
            raise self.failure
        return object()


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        choice = types.SimpleNamespace(
            text=" done",
            token_ids=list(encode_text(" done")),
            finish_reason="length",
            logprobs=types.SimpleNamespace(
                token_logprobs=[-0.2, -0.3, -0.4, -0.5, -0.6]
            ),
        )
        return types.SimpleNamespace(choices=[choice])


def test_openai_client_construction_missing_sdk_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: (_ for _ in ()).throw(ImportError("openai missing")),
    )

    with pytest.raises(
        RuntimeError,
        match=r"http://missing\.test/v1.*test-model.*openai missing",
    ):
        VLLMPolicy(
            model="test-model",
            format=test_format(),
            base_url="http://missing.test/v1",
            api_key="key",
            default_params=_default_params(),
        )


def test_a_zero_token_budget_is_refused_not_silently_defaulted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`max_tokens=0` raises vLLM's own range error rather than generating.

    Pinned deliberately: an earlier spelling (`max_tokens or DEFAULT`) treated the
    falsy zero as "unset" and quietly generated 512 tokens. Zero is an authored
    number, and a config that says it should hear that vLLM refuses it -- not
    receive a completion under a budget nobody wrote.
    """
    monkeypatch.setattr(
        "focus.generation.vllm._import_openai", lambda: lambda **kw: object()
    )
    generator = VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://example.test/v1",
        api_key="key",
        default_params=_default_params(),
    )

    with pytest.raises(ValueError, match=r"max_tokens must be >= 1"):
        generator._completion_kwargs(
            "test-model", (1, 2), _default_params(max_tokens=0)
        )


def test_openai_client_construction_requires_model() -> None:
    with pytest.raises(ValueError, match="model is required"):
        VLLMPolicy(
            model="",
            format=test_format(),
            base_url="http://example.test/v1",
            api_key="key",
        )


def test_openai_generator_uses_completions_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[FakeOpenAI] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.models = FakeModels()
            self.completions = FakeCompletions()
            self.close = Mock()
            clients.append(self)

    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: FakeOpenAI,
    )
    prefix_tokens = encode_text("Prompt:")
    gen = VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://example.test/v1",
        api_key="test-key",
        default_params=_default_params(),
    )

    gen.startup_check()
    result = gen.generate_tokens(
        prefix_tokens,
        SamplingParams(
            temperature=0.2,
            max_tokens=7,
            top_p=0.9,
            stop=["END"],
            repetition_penalty=1.05,
            seed=1729,
        ),
    )
    gen.close()

    client = clients[0]
    assert client.kwargs == {
        "base_url": "http://example.test/v1",
        "api_key": "test-key",
    }
    assert client.models.list_calls == 1
    assert client.completions.calls == [
        {
            "model": "test-model",
            "prompt": "ping",
            "temperature": 0.0,
            "max_tokens": 1,
            "top_p": 1.0,
        },
        {
            "model": "test-model",
            "prompt": list(prefix_tokens),
            "temperature": 0.2,
            "max_tokens": 7,
            "top_p": 0.9,
            "stop": ["END"],
            "seed": 1729,
            "extra_body": {
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "repetition_penalty": 1.05,
            },
        },
    ]
    assert result.tokens == encode_text(" done")
    assert result.text == " done"
    assert result.prefix_tokens == prefix_tokens
    assert result.stop_reason == "length"
    assert result.logprobs == [-0.2, -0.3, -0.4, -0.5, -0.6]
    client.close.assert_called_once()


def test_openai_generator_defaults_and_stop_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletionsNoLength:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            choice = types.SimpleNamespace(token_ids=[], text="", finish_reason="stop")
            return types.SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.models = FakeModels()
            self.completions = FakeCompletionsNoLength()

    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: FakeOpenAI,
    )
    gen = VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://example.test/v1",
        api_key="test-key",
        default_params=SamplingParams(max_tokens=3, stop=None),
    )

    result = gen.generate_tokens((65,))

    assert result.stop_reason == "stop"
    assert gen._client.completions.calls == [
        {
            "model": "test-model",
            "prompt": [65],
            "max_tokens": 3,
            "extra_body": {
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            },
        }
    ]


def test_openai_startup_failure_mentions_base_url_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.models = FakeModels(OSError("connection refused"))
            self.completions = FakeCompletions()

    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: FakeOpenAI,
    )
    gen = VLLMPolicy(
        model="bad-model",
        format=test_format(),
        base_url="http://bad.test/v1",
        api_key="key",
        default_params=SamplingParams(),
    )

    with pytest.raises(RuntimeError, match=r"http://bad\.test/v1.*bad-model"):
        gen.startup_check()


def test_openai_generate_failure_includes_request_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenAIError(Exception):
        status_code = 422
        body: ClassVar[dict[str, str]] = {"detail": "max_tokens is too large"}

    class FakeCompletionsFailure:
        def create(self, **kwargs: object) -> object:
            raise FakeOpenAIError("unprocessable content")

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.models = FakeModels()
            self.completions = FakeCompletionsFailure()

    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: FakeOpenAI,
    )
    gen = VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://example.test/v1",
        api_key="test-key",
        default_params=_default_params(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        gen.generate_tokens((65,), SamplingParams(max_tokens=999))

    assert getattr(exc_info.value, "status_code", None) == 422
    message = str(exc_info.value)
    assert "max_tokens is too large" in message
    assert "'prefix_tokens': 1" in message
    assert "'max_tokens': 999" in message


def test_openai_generator_filters_vllm_unsafe_stop_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.models = FakeModels()
            self.completions = FakeCompletions()

    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: FakeOpenAI,
    )
    gen = VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://example.test/v1",
        api_key="test-key",
        default_params=_default_params(),
    )

    gen.generate_tokens(
        (65,),
        SamplingParams(stop=["\n\n", "\nDocument", ""]),
    )

    assert gen._client.completions.calls[0]["stop"] == ["\nDocument"]


def test_openai_generator_extracts_prompt_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix_tokens = (80, 114)

    class FakeCompletionsWithPromptLogprobs:
        def create(self, **kwargs: object) -> object:
            choice = types.SimpleNamespace(
                text=" done",
                token_ids=list(encode_text(" done")),
                finish_reason="length",
                logprobs=types.SimpleNamespace(
                    token_logprobs=[-0.2, -0.4, -0.4, -0.4, -0.4]
                ),
                prompt_logprobs=[
                    {80: {"logprob": -0.1, "rank": 1, "decoded_token": "P"}},
                    {114: {"logprob": -0.3, "rank": 2, "decoded_token": "r"}},
                ],
            )
            return types.SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.models = FakeModels()
            self.completions = FakeCompletionsWithPromptLogprobs()

    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: FakeOpenAI,
    )
    gen = VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://example.test/v1",
        api_key="test-key",
        default_params=_default_params(),
    )

    result = gen.generate_tokens(
        prefix_tokens,
        SamplingParams(prompt_logprobs=1),
    )

    assert result.prompt_logprobs == [-0.1, -0.3]
    assert result.prompt_top_logprobs == [{"P": -0.1}, {"r": -0.3}]


def test_openai_generator_passes_prompt_logprobs_in_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.models = FakeModels()
            self.completions = FakeCompletions()

    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: FakeOpenAI,
    )
    gen = VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://example.test/v1",
        api_key="test-key",
        default_params=_default_params(),
    )

    gen.generate_tokens((65,), SamplingParams(prompt_logprobs=1))

    assert gen._client.completions.calls[0]["extra_body"] == {
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "prompt_logprobs": 1,
    }


def test_openai_generator_merges_vllm_extra_body_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.models = FakeModels()
            self.completions = FakeCompletions()

    monkeypatch.setattr(
        "focus.generation.vllm._import_openai",
        lambda: FakeOpenAI,
    )
    gen = VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://example.test/v1",
        api_key="test-key",
        default_params=_default_params(),
    )

    gen.generate_tokens(
        (65,),
        SamplingParams(repetition_penalty=1.05, prompt_logprobs=2),
    )

    assert gen._client.completions.calls[0]["extra_body"] == {
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "repetition_penalty": 1.05,
        "prompt_logprobs": 2,
    }


def test_parse_vllm_token_ids_accepts_token_id_prefix_format() -> None:
    assert _parse_vllm_token_ids(["token_id:21927", "token_id:382"]) == [
        21927,
        382,
    ]


def test_parse_vllm_token_ids_rejects_null_token() -> None:
    # A null token must fail loud: silently dropping it would shift every
    # following logprob (token_logprobs keeps the position) and desync.
    with pytest.raises(ValueError, match="null token at position 1"):
        _parse_vllm_token_ids(["token_id:21927", None, "token_id:382"])


def test_parse_vllm_token_ids_rejects_decoded_text_tokens() -> None:
    # h12 / current vLLM returns decoded substrings in logprobs.tokens, not token_id:N.
    with pytest.raises(ValueError, match="Could not parse vLLM token id"):
        _parse_vllm_token_ids([" Hello", ".\n\n", "Can"])


def _client_kwargs_of(monkeypatch: pytest.MonkeyPatch, **init: Any) -> dict[str, Any]:
    """Build a generator against a recording stand-in and return the client kwargs."""
    seen: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> object:
        seen.update(kwargs)
        return Mock()

    monkeypatch.setattr("focus.generation.vllm._import_openai", lambda: fake_client)
    VLLMPolicy(
        model="test-model",
        format=test_format(),
        base_url="http://x/v1",
        api_key="EMPTY",
        default_params=_default_params(),
        **init,
    )
    return seen


def test_generator_omits_timeout_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not `timeout=None`: the SDK reads that as "no deadline at all", which is a
    # different thing from "the SDK's own default" and is not what an unset flag means.
    assert "timeout" not in _client_kwargs_of(monkeypatch)


def test_generator_passes_timeout_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _client_kwargs_of(monkeypatch, timeout=3600.0)["timeout"] == 3600.0
