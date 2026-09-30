"""Core generation value objects."""

from collections.abc import Generator
from typing import Any

import pytest

from focus.generation import (
    GenerateResult,
    NativeToolCall,
    SamplingParams,
)
from focus.generation.interfaces import (
    merge_sampling_params,
    run_async,
)


def test_generate_result_normalizes_token_sequences() -> None:
    # lists on purpose: `__post_init__` coerces, so the annotation (which describes
    # what is *stored*) is deliberately narrower than what the constructor takes.
    result = GenerateResult(tokens=[1, 2], prefix_tokens=[3])  # type: ignore[arg-type]

    assert result.tokens == (1, 2)
    assert result.prefix_tokens == (3,)


def test_generate_result_rejects_mismatched_completion_logprobs() -> None:
    with pytest.raises(ValueError, match="completion logprobs length"):
        GenerateResult(tokens=(1, 2), logprobs=[-0.1])


def test_generate_result_rejects_mismatched_top_logprobs() -> None:
    with pytest.raises(ValueError, match="completion top_logprobs length"):
        GenerateResult(tokens=(1, 2), top_logprobs=[{"a": -0.1}])


def test_generate_result_rejects_mismatched_prefix_logprobs() -> None:
    with pytest.raises(ValueError, match="prefix logprobs length"):
        GenerateResult(prefix_tokens=(1, 2), prompt_logprobs=[-0.1])


def test_run_async_bridges_a_coroutine_to_a_sync_result() -> None:
    async def scored() -> int:
        return 7

    assert run_async(scored()) == 7


def test_run_async_takes_a_coroutine_not_any_awaitable() -> None:
    """Pins the narrowed contract: ``asyncio.run`` rejects non-coroutine awaitables.

    ``run_async`` is annotated ``Coroutine[Any, Any, _T]`` rather than ``Awaitable[_T]``
    because the wider annotation promised something it never accepted -- a plain
    awaitable reaches ``asyncio.run`` and raises. The type checker now catches the
    miswiring; this pins the runtime half so the annotation cannot silently widen again.
    """

    class NotACoroutine:
        def __await__(self) -> Generator[Any, None, int]:
            yield
            return 7

    with pytest.raises(ValueError, match="coroutine was expected"):
        run_async(NotACoroutine())  # type: ignore[arg-type]


# -- the structured tool-call channel -------------------------------------------------

_SCHEMA = {"type": "function", "function": {"name": "grep", "parameters": {}}}


def test_a_result_reports_no_structured_channel_by_default() -> None:
    """``None``, not ``()``: a token-native backend has no such channel to be empty.

    The distinction is the whole point of the field -- "the model called nothing" and
    "this backend cannot report a call" are what a tool harness must never confuse --
    so the default is the second, which every backend but the chat bridge means.
    """
    assert GenerateResult().native_tool_calls is None


def test_an_open_but_unused_channel_is_an_empty_tuple() -> None:
    assert GenerateResult(native_tool_calls=()).native_tool_calls == ()


def test_native_tool_calls_are_tupleized_like_the_token_sequences() -> None:
    result = GenerateResult(
        native_tool_calls=[NativeToolCall("grep", "{}", "c1")]  # type: ignore[arg-type]
    )

    assert result.native_tool_calls == (NativeToolCall("grep", "{}", "c1"),)


def test_a_merged_result_never_carries_an_unset_knob() -> None:
    """Backends read a number off any merged params, never a "not overridden"."""
    merged = merge_sampling_params(
        SamplingParams(temperature=None, max_tokens=None, top_p=None), None
    )

    assert (merged.temperature, merged.max_tokens, merged.top_p) == (0.7, 512, 1.0)
