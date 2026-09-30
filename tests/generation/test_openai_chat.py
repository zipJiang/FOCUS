from __future__ import annotations

import asyncio
import types
from collections.abc import Sequence
from typing import Any

import pytest

from focus.generation import (
    NativeToolCall,
    Policy,
    SamplingParams,
)
from focus.generation.openai_chat import OpenAIChatPolicy
from focus.registry import build
from tests.helpers import native_format


def _params(**kwargs: Any) -> SamplingParams:
    return SamplingParams(**kwargs)


class ByteCodec:
    """HF-like stub codec: UTF-8 byte values, tolerant of extra kwargs."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        del kwargs
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int], **kwargs: Any) -> str:
        del kwargs
        return bytes(tokens).decode("utf-8", errors="replace")


class StrictByteCodec(ByteCodec):
    """tiktoken-like stub: encode/decode reject the optional kwargs."""

    def encode(self, text: str) -> list[int]:  # type: ignore[override]
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:  # type: ignore[override]
        return bytes(tokens).decode("utf-8", errors="replace")


class ExactCodec(ByteCodec):
    """Stub exposing ``encode_single_token`` for the exact bytes path."""

    def __init__(self) -> None:
        self.table = {b"\xc3\xa9": 9001, b" hi": 9002}

    def encode_single_token(self, raw: bytes) -> int:
        return self.table[raw]


def _entry(token: str, logprob: float, raw_bytes: list[int] | None = None) -> Any:
    return types.SimpleNamespace(token=token, logprob=logprob, bytes=raw_bytes)


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> Any:
    """One ``message.tool_calls`` entry, in the SDK's own nesting."""
    return types.SimpleNamespace(
        id=call_id,
        type="function",
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


def _response(
    content: str | None,
    *,
    entries: list[Any] | None = None,
    finish_reason: str | None = "stop",
    tool_calls: list[Any] | None = None,
) -> Any:
    logprobs = types.SimpleNamespace(content=entries) if entries is not None else None
    choice = types.SimpleNamespace(
        message=types.SimpleNamespace(content=content, tool_calls=tool_calls),
        finish_reason=finish_reason,
        logprobs=logprobs,
    )
    return types.SimpleNamespace(choices=[choice])


class _Completions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _wire(monkeypatch: pytest.MonkeyPatch, response: Any) -> _Completions:
    """Patch the async-client factory to return a fake exposing ``chat.completions``."""
    completions = _Completions(response)

    def factory(*, base_url: Any = None, api_key: Any = None) -> Any:
        del base_url, api_key
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions),
            close=lambda: None,
        )

    # The client factory lives on the shared bridge both hosted backends are built from
    monkeypatch.setattr(
        "focus.generation._openai_shared._import_async_openai",
        lambda: factory,
    )
    return completions


def test_prefix_decoded_into_single_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = ByteCodec()
    completions = _wire(monkeypatch, _response("ok"))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(codec), default_params=_params()
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert completions.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert "logprobs" not in completions.calls[0]
    assert result.text == "ok"
    assert result.tokens == tuple(codec.encode("ok"))
    assert result.logprobs == []
    assert result.exact_generation is False


def test_system_prompt_and_logprob_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = ByteCodec()
    completions = _wire(monkeypatch, _response("ok", entries=[_entry("ok", -0.1)]))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini",
        format=native_format(codec),
        default_params=_params(logprobs=3),
        system_prompt="sys",
    )

    asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    call = completions.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 3


def test_logprob_entries_align_to_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = ByteCodec()
    # "ab" -> 2 ids, "é" -> 2 bytes/ids: tests multi-id pieces and 0.0 padding.
    entries = [_entry("ab", -0.1), _entry("é", -0.2)]
    _wire(monkeypatch, _response("abé", entries=entries))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini",
        format=native_format(codec),
        default_params=_params(logprobs=0),
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert result.tokens == (97, 98, 0xC3, 0xA9)
    assert result.logprobs == [-0.1, 0.0, -0.2, 0.0]
    assert len(result.logprobs) == len(result.tokens)
    assert codec.decode(list(result.tokens)) == "abé"


def test_exact_bytes_path_uses_encode_single_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = ExactCodec()
    entries = [
        _entry(" hi", -0.4, raw_bytes=[0x20, 0x68, 0x69]),
        _entry("é", -0.5, raw_bytes=[0xC3, 0xA9]),
    ]
    _wire(monkeypatch, _response(" hié", entries=entries))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini",
        format=native_format(codec),
        default_params=_params(logprobs=1),
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert result.tokens == (9002, 9001)
    assert result.logprobs == [-0.4, -0.5]


def test_none_content_is_total(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, _response(None, finish_reason="content_filter"))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(ByteCodec()), default_params=_params()
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert result.tokens == ()
    assert result.logprobs == []
    assert result.text == ""
    assert result.stop_reason == "content_filter"


def test_logprobs_requested_but_absent_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = ByteCodec()
    # logprobs requested, but the server returned no logprobs payload.
    _wire(monkeypatch, _response("done", entries=None))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini",
        format=native_format(codec),
        default_params=_params(logprobs=2),
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert result.tokens == tuple(codec.encode("done"))
    assert result.logprobs == []


def test_finish_reason_none_maps_to_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, _response("x", finish_reason=None))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(ByteCodec()), default_params=_params()
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert result.stop_reason == "stop"


def test_strict_codec_kwarg_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = StrictByteCodec()
    completions = _wire(monkeypatch, _response("ok"))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(codec), default_params=_params()
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert completions.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert result.tokens == tuple(codec.encode("ok"))


def test_sync_generate_matches_async(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = ByteCodec()
    _wire(monkeypatch, _response("ok", entries=[_entry("ok", -0.1)]))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini",
        format=native_format(codec),
        default_params=_params(logprobs=1),
    )

    result = gen.generate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))

    assert result.tokens == tuple(codec.encode("ok"))
    assert result.logprobs == [-0.1, 0.0]


def test_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="model is required"):
        OpenAIChatPolicy(
            model="",
            format=native_format(ByteCodec()),
            default_params=SamplingParams(),
        )


@pytest.mark.parametrize("name", ["openai_chat", "openai"])
def test_registry_builds_via_from_params(name: str) -> None:
    gen = build(
        Policy,
        {
            "name": name,
            "api_key": "k",
            "model": "gpt-4o-mini",
        },
        format=native_format(ByteCodec()),
    )
    assert isinstance(gen, OpenAIChatPolicy)


# -- the native tool-call channel -----------------------------------------------------

_SCHEMA = {
    "type": "function",
    "function": {"name": "grep", "description": "search", "parameters": {}},
}


def test_schemas_are_sent_as_the_apis_own_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hosted model calls tools in its own protocol, not in a rendered dialect."""
    codec = ByteCodec()
    completions = _wire(monkeypatch, _response("ok"))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(codec), default_params=_params()
    )

    asyncio.run(
        gen.agenerate(
            gen.prepare([{"role": "user", "content": "hi"}], tools=(_SCHEMA,))
        )
    )

    assert completions.calls[0]["tools"] == [_SCHEMA]
    # left to the API: the harness offers tools, it never demands one be called
    assert "tool_choice" not in completions.calls[0]


def test_no_schemas_means_no_tools_key_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = ByteCodec()
    completions = _wire(monkeypatch, _response("ok"))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(codec), default_params=_params()
    )

    asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert "tools" not in completions.calls[0]


def test_returned_tool_calls_become_the_results_native_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = ByteCodec()
    _wire(
        monkeypatch,
        _response(
            None,
            tool_calls=[
                _tool_call("grep", '{"pattern": "gross income"}'),
                _tool_call("read", '{"path": "s.md"}', "call_2"),
            ],
        ),
    )
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(codec), default_params=_params()
    )

    result = asyncio.run(
        gen.agenerate(
            gen.prepare([{"role": "user", "content": "hi"}], tools=(_SCHEMA,))
        )
    )

    assert result.native_tool_calls == (
        NativeToolCall("grep", '{"pattern": "gross income"}', "call_1"),
        NativeToolCall("read", '{"path": "s.md"}', "call_2"),
    )
    assert result.text == ""  # a pure tool-call turn says nothing out loud


def test_an_offered_but_unused_channel_comes_back_empty_not_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``()`` -- the model was offered tools and answered in prose instead."""
    codec = ByteCodec()
    _wire(monkeypatch, _response("the answer is 4200"))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(codec), default_params=_params()
    )

    result = asyncio.run(
        gen.agenerate(
            gen.prepare([{"role": "user", "content": "hi"}], tools=(_SCHEMA,))
        )
    )

    assert result.native_tool_calls == ()


def test_a_call_sent_no_schemas_still_reports_an_open_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``()`` even with no schemas sent: the channel is the backend's, not the call's.

    A fold's derived runner inherits the rollout's policy, and under a toolless
    ``FoldEnv`` (a ``NullWorkspace`` fold) its generations send no schemas. Reporting
    ``None`` there had the native parser raising its misconfiguration error *inside
    the fold* -- but a prose reply on a chat backend is a turn that called nothing,
    which is exactly what a fold's final reply is. ``None`` stays reserved for
    backends with no structured channel at all.
    """
    codec = ByteCodec()
    _wire(monkeypatch, _response("the answer is 4200"))
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(codec), default_params=_params()
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert result.native_tool_calls == ()


def test_the_budget_travels_as_max_completion_tokens_and_locking_omits_the_law() -> (
    None
):
    """The current wire spelling of the budget, and the reasoning-model escape hatch.

    The reasoning-line chat models refuse any request naming a non-default
    temperature or top_p; `locked_sampling=True` omits both, leaving the sampling law
    to the server -- which is the honest description of those models. The budget key
    is `max_completion_tokens` for every current model; `max_tokens` is deprecated.
    """
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(ByteCodec()), default_params=_params()
    )
    kwargs = gen._chat_kwargs("m", [], _params(max_tokens=64, temperature=0.5), [])
    assert kwargs["max_completion_tokens"] == 64
    assert "max_tokens" not in kwargs
    assert kwargs["temperature"] == 0.5

    locked = OpenAIChatPolicy(
        model="gpt-4o-mini",
        format=native_format(ByteCodec()),
        default_params=_params(),
        locked_sampling=True,
    )
    kwargs = locked._chat_kwargs("m", [], _params(max_tokens=64, temperature=0.5), [])
    assert kwargs["max_completion_tokens"] == 64
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_the_bridge_does_not_claim_the_models_own_tokens() -> None:
    """Both the per-call stamp and the capability a search reads before it spends."""
    gen = OpenAIChatPolicy(
        model="gpt-4o-mini", format=native_format(ByteCodec()), default_params=_params()
    )
    with pytest.raises(TypeError, match="exact_generation_override"):
        OpenAIChatPolicy(  # type: ignore[call-arg]
            model="gpt-4o-mini",
            format=native_format(ByteCodec()),
            default_params=_params(),
            exact_generation_override=True,
        )

    assert not gen.provides_exact_tokens
