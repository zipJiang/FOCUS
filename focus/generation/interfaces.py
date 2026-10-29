"""Shared sampling, synchronous bridges, and transport value helpers."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import fields, replace
from typing import Any

from focus.generation.types import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GenerateResult,
    SamplingParams,
    TokenId,
)

_SAMPLING_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(SamplingParams))


def close_client(client: Any) -> None:
    """Close an underlying SDK client if it exposes a synchronous ``close``.

    Only a sync closer is called here -- an async-only client is left for its own
    GC/atexit.
    """

    close = getattr(client, "close", None)
    if close is not None:
        close()


async def aclose_client(client: Any) -> None:
    """Close a client whose close method may be synchronous or asynchronous."""
    close = getattr(client, "close", None)
    if close is not None:
        result = close()
        if inspect.isawaitable(result):
            await result


def merge_sampling_params(
    default_params: SamplingParams,
    sampling_params: SamplingParams | None,
) -> SamplingParams:
    """Merge request overrides and return the complete resolved settings.

    Each non-None request field overrides the corresponding default. Temperature,
    max_tokens, and top_p receive library defaults if neither input sets them.
    Other unset fields remain None for the backend to handle.
    """
    params = default_params
    if sampling_params is not None:
        updates = {
            name: value
            for name in _SAMPLING_FIELD_NAMES
            if (value := getattr(sampling_params, name)) is not None
        }
        params = replace(default_params, **updates)
    if (
        params.temperature is not None
        and params.max_tokens is not None
        and params.top_p is not None
    ):
        return params
    return replace(
        params,
        temperature=(
            DEFAULT_TEMPERATURE if params.temperature is None else params.temperature
        ),
        max_tokens=(
            DEFAULT_MAX_TOKENS if params.max_tokens is None else params.max_tokens
        ),
        top_p=DEFAULT_TOP_P if params.top_p is None else params.top_p,
    )


def prepare_generation(
    default_params: SamplingParams,
    prefix_tokens: Sequence[TokenId],
    sampling_params: SamplingParams | None,
) -> tuple[tuple[TokenId, ...], SamplingParams]:
    """Normalize a generation call: tuple prefix + merged sampling params."""

    return (
        tuple(prefix_tokens),
        merge_sampling_params(default_params, sampling_params),
    )


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Bridge an async generation call into a sync ``generate`` entrypoint.

    A ``Coroutine``, not any ``Awaitable``: ``asyncio.run`` rejects a non-coroutine
    awaitable outright (``ValueError: a coroutine was expected``), so the wider
    annotation promised something this never accepted.
    """

    return asyncio.run(coro)


def normalize_stop_reason(raw: Any) -> str:
    """Coerce a backend's finish reason to a string (``None`` -> ``"stop"``)."""
    if raw is None:
        return "stop"
    if isinstance(raw, Mapping):
        return str(raw.get("type", raw))
    return str(raw)


def finite_logprobs(
    raw: Sequence[float | None] | None,
    length: int,
) -> list[float]:
    """Validate completion likelihoods; absent optional likelihoods remain absent."""
    if not raw:
        return []
    if len(raw) != length or any(v is None or not math.isfinite(v) for v in raw):
        raise ValueError("completion logprobs must be aligned, present and finite")
    return [float(v) for v in raw if v is not None]


def _continuation_params(
    defaults: SamplingParams,
    params: SamplingParams | None,
) -> SamplingParams:
    resolved = merge_sampling_params(defaults, params)
    limit = resolved.max_tokens
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_tokens must be a positive integer")
    return resolved


def _validate_continuation_result(
    result: GenerateResult, params: SamplingParams
) -> None:
    if not result.exact_generation or result.native_tool_calls or result.native_output:
        raise ValueError(
            "Token continuation requires exact generated token IDs and text output"
        )
    limit = params.max_tokens
    assert limit is not None
    if len(result.tokens) > limit:
        raise ValueError(f"Generation exceeded the requested limit of {limit} tokens")
    if result.stop_reason == "length" and len(result.tokens) != limit:
        raise ValueError("Generation stopped before the requested token limit")
