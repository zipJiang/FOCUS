"""The Responses bridge: what it puts on the wire, and what it reads back off it.

Same fake-client pattern as ``test_openai_chat``: the async-client factory is patched to
hand back an object exposing ``responses.create``, which records its kwargs and returns
a canned response. Nothing here reaches the network, and the assertions are about the
two places this backend can silently differ from the chat one -- the endpoint's own
spelling of a request (a flat tool schema, ``max_output_tokens``, ``store=False``), and
an output *list* read by item type rather than a single message.
"""

from __future__ import annotations

import asyncio
import types
from collections.abc import Sequence
from typing import Any

import pytest

from focus.generation import (
    NativeToolCall,
    SamplingParams,
)
from focus.generation.openai_responses import OpenAIResponsesPolicy
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


#: The nested shape `SamplingParams.tools` carries -- what `ToolEnv.schemas` produces.
_SCHEMA = {
    "type": "function",
    "function": {"name": "grep", "description": "search", "parameters": {}},
}
#: The same schema as the Responses API spells it: flat.
_FLAT = {
    "type": "function",
    "name": "grep",
    "description": "search",
    "parameters": {},
}


def _text_item(text: str) -> Any:
    return types.SimpleNamespace(
        type="message",
        content=[types.SimpleNamespace(type="output_text", text=text)],
    )


def _call_item(name: str, arguments: str, call_id: str = "call_1") -> Any:
    return types.SimpleNamespace(
        type="function_call", name=name, arguments=arguments, call_id=call_id
    )


def _reasoning_item() -> Any:
    """An item whose ``summary`` is not the completion, and is not read as one."""
    return types.SimpleNamespace(
        type="reasoning",
        summary=[types.SimpleNamespace(type="summary_text", text="thinking out loud")],
        content=[types.SimpleNamespace(type="output_text", text="not the answer")],
    )


def _response(
    *items: Any, status: str = "completed", incomplete_reason: str | None = None
) -> Any:
    details = (
        types.SimpleNamespace(reason=incomplete_reason)
        if incomplete_reason is not None
        else None
    )
    return types.SimpleNamespace(
        output=list(items), status=status, incomplete_details=details
    )


class _Responses:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _wire(monkeypatch: pytest.MonkeyPatch, response: Any) -> _Responses:
    """Patch the async-client factory to return a fake exposing ``responses``."""
    responses = _Responses(response)

    def factory(*, base_url: Any = None, api_key: Any = None) -> Any:
        del base_url, api_key
        return types.SimpleNamespace(responses=responses, close=lambda: None)

    monkeypatch.setattr(
        "focus.generation._openai_shared._import_async_openai",
        lambda: factory,
    )
    return responses


# -- the request ----------------------------------------------------------------------


def test_the_prefix_travels_as_one_user_item_under_the_endpoints_own_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = ByteCodec()
    responses = _wire(monkeypatch, _response(_text_item("ok")))
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2",
        format=native_format(codec),
        default_params=_params(),
        system_prompt="sys",
    )

    result = asyncio.run(
        gen.agenerate(
            gen.prepare([{"role": "user", "content": "hi"}], tools=()),
            _params(max_tokens=64),
        )
    )

    call = responses.calls[0]
    assert call["input"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert call["max_output_tokens"] == 64
    assert "max_tokens" not in call  # chat completions' spelling, not this one's
    assert "messages" not in call
    assert result.text == "ok"
    assert result.tokens == tuple(codec.encode("ok"))


def test_state_always_stays_on_this_side_of_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store=False``, never ``previous_response_id`` -- on every request there is.

    A server-side thread is an append-only history the server owns. This harness folds
    (rewriting the prefix) and forks (sharing it between siblings), so the only prompt
    that means what the runner says it means is the full input, sent every call.
    """
    codec = ByteCodec()
    responses = _wire(monkeypatch, _response(_text_item("ok")))
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params()
    )

    asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )
    asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert [c["store"] for c in responses.calls] == [False, False]
    assert all("previous_response_id" not in c for c in responses.calls)


def test_schemas_are_converted_to_the_flat_shape_this_api_speaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat completions nests the function; the Responses API hoists it."""
    codec = ByteCodec()
    responses = _wire(monkeypatch, _response(_text_item("ok")))
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params()
    )

    asyncio.run(
        gen.agenerate(
            gen.prepare([{"role": "user", "content": "hi"}], tools=(_SCHEMA,))
        )
    )

    assert responses.calls[0]["tools"] == [_FLAT]
    # left to the API: the harness offers tools, it never demands one be called
    assert "tool_choice" not in responses.calls[0]


def test_no_schemas_means_no_tools_key_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = ByteCodec()
    responses = _wire(monkeypatch, _response(_text_item("ok")))
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params()
    )

    asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert "tools" not in responses.calls[0]


def test_reasoning_is_sent_only_when_an_effort_was_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = ByteCodec()
    responses = _wire(monkeypatch, _response(_text_item("ok")))
    plain = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params()
    )
    thinking = OpenAIResponsesPolicy(
        model="gpt-5.2",
        format=native_format(codec),
        default_params=_params(),
        reasoning_effort="high",
    )

    asyncio.run(
        plain.agenerate(plain.prepare([{"role": "user", "content": "hi"}], tools=()))
    )
    asyncio.run(
        thinking.agenerate(
            thinking.prepare([{"role": "user", "content": "hi"}], tools=())
        )
    )

    assert "reasoning" not in responses.calls[0]
    assert responses.calls[1]["reasoning"] == {"effort": "high"}


def test_locked_sampling_omits_the_law_and_otherwise_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = ByteCodec()
    responses = _wire(monkeypatch, _response(_text_item("ok")))
    free = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params()
    )
    locked = OpenAIResponsesPolicy(
        model="gpt-5.2",
        format=native_format(codec),
        default_params=_params(),
        locked_sampling=True,
    )

    asyncio.run(
        free.agenerate(
            free.prepare([{"role": "user", "content": "hi"}], tools=()),
            _params(temperature=0.5),
        )
    )
    asyncio.run(
        locked.agenerate(
            locked.prepare([{"role": "user", "content": "hi"}], tools=()),
            _params(temperature=0.5),
        )
    )

    assert responses.calls[0]["temperature"] == 0.5
    assert "top_p" in responses.calls[0]
    assert "temperature" not in responses.calls[1]
    assert "top_p" not in responses.calls[1]


def test_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="model is required"):
        OpenAIResponsesPolicy(
            model="",
            format=native_format(ByteCodec()),
            default_params=SamplingParams(),
        )


# -- the response ---------------------------------------------------------------------


def test_the_output_list_is_read_by_item_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calls become structure, message parts become the text, reasoning is dropped."""
    codec = ByteCodec()
    _wire(
        monkeypatch,
        _response(
            _reasoning_item(),
            _text_item("checking the statute."),
            _call_item("grep", '{"pattern": "gross income"}'),
            _call_item("read", '{"path": "s.md"}', "call_2"),
        ),
    )
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params()
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
    assert result.text == "checking the statute."
    assert result.tokens == tuple(codec.encode("checking the statute."))
    assert result.exact_generation is False


def test_an_empty_output_is_a_prose_turn_that_said_nothing_not_a_missing_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``()`` and never ``None``: whether the channel exists is the backend's property.

    ``None`` is reserved for a backend with no structured channel at all -- a native
    parser reading it raises a misconfiguration error, which is the wrong thing to say
    about a turn (a toolless fold's reply, say) that simply called nothing.
    """
    _wire(monkeypatch, _response())
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(ByteCodec()), default_params=_params()
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert result.text == ""
    assert result.tokens == ()
    assert result.native_tool_calls == ()


def test_several_message_items_concatenate(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = ByteCodec()
    _wire(monkeypatch, _response(_text_item("one "), _text_item("two")))
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params()
    )

    assert (
        asyncio.run(
            gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
        ).text
        == "one two"
    )


def test_a_truncated_answer_is_reported_as_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The budget ran out: the same event every other backend calls ``length``."""
    _wire(
        monkeypatch,
        _response(
            _text_item("the deduction is"),
            status="incomplete",
            incomplete_reason="max_output_tokens",
        ),
    )
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(ByteCodec()), default_params=_params()
    )

    assert (
        asyncio.run(
            gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
        ).stop_reason
        == "length"
    )


def test_a_finished_answer_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, _response(_text_item("done"), status="completed"))
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(ByteCodec()), default_params=_params()
    )

    assert (
        asyncio.run(
            gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
        ).stop_reason
        == "stop"
    )


def test_another_incomplete_reason_is_reported_as_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(
        monkeypatch,
        _response(status="incomplete", incomplete_reason="content_filter"),
    )
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(ByteCodec()), default_params=_params()
    )

    assert (
        asyncio.run(
            gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
        ).stop_reason
        == "content_filter"
    )


def test_logprobs_are_asked_for_and_quietly_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No per-token channel on this endpoint -- an empty column, never an exception."""
    codec = ByteCodec()
    responses = _wire(monkeypatch, _response(_text_item("ok")))
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params(logprobs=5)
    )

    result = asyncio.run(
        gen.agenerate(gen.prepare([{"role": "user", "content": "hi"}], tools=()))
    )

    assert result.logprobs == []
    assert "logprobs" not in responses.calls[0]


def test_sync_generate_matches_async(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = ByteCodec()
    _wire(monkeypatch, _response(_text_item("ok")))
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(codec), default_params=_params()
    )

    assert gen.generate(
        gen.prepare([{"role": "user", "content": "hi"}], tools=())
    ).tokens == tuple(codec.encode("ok"))


# -- what the backend claims about itself ---------------------------------------------


def test_the_capabilities_a_policy_is_bound_on() -> None:
    """A native channel always; the model's own tokens never, unless told otherwise."""
    gen = OpenAIResponsesPolicy(
        model="gpt-5.2", format=native_format(ByteCodec()), default_params=_params()
    )
    with pytest.raises(TypeError, match="exact_generation_override"):
        OpenAIResponsesPolicy(  # type: ignore[call-arg]
            model="gpt-5.2",
            format=native_format(ByteCodec()),
            default_params=_params(),
            exact_generation_override=True,
        )

    assert gen.provides_native_tool_calls
    assert not gen.provides_exact_tokens
