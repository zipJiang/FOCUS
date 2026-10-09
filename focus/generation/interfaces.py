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
    """Overlay each non-``None`` per-call param onto ``default_params``.

    Shared by every backend generator: a call with no per-call params keeps the
    defaults, otherwise any set field (``stop`` included) overrides. ``None`` is
    therefore how an overlay says *not this field* -- which is the only way to overlay
    one knob (a fold's ``max_tokens``) without also pinning the sampling law to
    :class:`SamplingParams`' own defaults.

    What comes back is settled rather than partial: the three always-set knobs are
    filled from the constants they default to when neither side named one, so a backend
    reads a temperature and a token budget off any merged result and never has to ask
    whether the number it was given means "unset".
    """
    if sampling_params is None:
        return settled(default_params)
    updates = {
        name: value
        for name in _SAMPLING_FIELD_NAMES
        if (value := getattr(sampling_params, name)) is not None
    }
    return settled(replace(default_params, **updates))


def settled(params: SamplingParams) -> SamplingParams:
    """``params`` with the three overlay-blankable knobs resolved to real numbers.

    The one boundary between the two modes a :class:`SamplingParams` can be in --
    *overlay* (``None`` legal, meaning "not this field") and *settled* (every knob a
    number). Everything a backend receives has passed through here via
    :func:`merge_sampling_params`. The early return is for identity, not speed: the
    common case is already settled, and handing back the same object keeps "merge then
    read" from allocating a copy per generate call.
    """
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


def settled_knobs(params: SamplingParams) -> tuple[float, int, float]:
    """``(temperature, max_tokens, top_p)`` as the numbers a backend sends on the wire.

    The typed face of :func:`settled`: mypy cannot see that a merged
    :class:`SamplingParams` carries no ``None``, so a backend reading the fields
    directly must either assert three times or respell the defaults -- the duplication
    that produced an ``or``-fallback quietly treating ``0`` as unset. Read the knobs
    through this instead; the ``| None`` never escapes.
    """
    resolved = settled(params)
    assert resolved.temperature is not None  # settled: narrowing, not runtime
    assert resolved.max_tokens is not None
    assert resolved.top_p is not None
    return resolved.temperature, resolved.max_tokens, resolved.top_p


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
