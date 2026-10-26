"""Tinker sampling generator (thinking-machines-lab/tinker)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from focus.codec import HFTokenizer
from focus.generation.interfaces import (
    close_client,
    finite_logprobs,
    normalize_stop_reason,
    prepare_generation,
)
from focus.generation.policy import Policy, PolicyFormat
from focus.generation.types import (
    GenerateResult,
    SamplingParams,
    TokenId,
)

if TYPE_CHECKING:
    import tinker


def _import_tinker() -> Any:
    try:
        import tinker
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "The 'tinker' package is required to use TinkerPolicy. "
            + "Install it with: pip install focus[tinker]"
        ) from exc
    return tinker


def _tinker_sampling_kwargs(params: SamplingParams) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
    }
    if params.stop:
        kwargs["stop"] = list(params.stop)
    return kwargs


def _aligned[T](raw: Sequence[T | None] | None, prefix_len: int) -> list[T | None]:
    """Align ``raw`` to the prompt length, padding a short response with ``None``.

    Unlike :func:`finite_logprobs` (completion side) a missing prompt position stays
    ``None``: the first prompt token has no logprob, and 0.0 would read as certainty.
    ``raw`` is itself optional *per position* -- the backend leaves the first prompt
    token empty -- which is why ``_T`` is the payload type and the ``None`` is spelled
    out on both sides rather than inferred.
    """
    raw = raw or ()
    return [raw[index] if index < len(raw) else None for index in range(prefix_len)]


def _extract_prompt_logprobs(
    raw: Sequence[float | None] | None,
    prefix_len: int,
) -> list[float | None]:
    return _aligned(raw, prefix_len) if raw else []


def _extract_prompt_top_logprobs(
    raw: Sequence[Sequence[tuple[int, float]] | None] | None,
    prefix_len: int,
) -> list[dict[str, float] | None] | None:
    if raw is None:
        return None
    return [_top_dict(entry) for entry in _aligned(raw, prefix_len)]


def _top_dict(entry: Sequence[tuple[int, float]] | None) -> dict[str, float] | None:
    """One prompt position's ``(token_id, logprob)`` pairs as a ``{"id": logprob}``."""
    return {str(token_id): float(logprob) for token_id, logprob in entry or ()} or None


class TinkerPolicy(Policy[Any]):
    """Token generation over a ``tinker.SamplingClient``.

    :meth:`agenerate` ``await``s ``sample_async``; sync :meth:`generate` is the
    :class:`~focus.generation.Policy` bridge over it.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        model: str,
        format: PolicyFormat[Any] | None = None,
        profile: str | None = None,
        default_params: SamplingParams | None = None,
    ):
        self._client = sampling_client
        tokenizer = cast(HFTokenizer, sampling_client.get_tokenizer())
        if format is not None and profile is not None:
            raise ValueError("Supply either format or profile")
        if format is not None and format.codec.tokenizer is not tokenizer:
            raise ValueError("Tinker format must use the sampling client's tokenizer")
        super().__init__(
            model=model,
            format=format
            or PolicyFormat.resolve(model, tokenizer=tokenizer, profile=profile),
            default_params=default_params,
        )

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        params, prefix_tuple, kwargs = self._sample_request(
            prefix_tokens, sampling_params
        )
        response = await self._client.sample_async(**kwargs)
        return self._parse_response(response, prefix_tuple, params)

    def _sample_request(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None,
    ) -> tuple[SamplingParams, tuple[TokenId, ...], dict[str, Any]]:
        tinker = _import_tinker()
        prefix_tuple, params = prepare_generation(
            self.default_params, prefix_tokens, sampling_params
        )
        model_input = tinker.types.ModelInput.from_ints(list(prefix_tuple))
        topk = params.prompt_logprobs
        kwargs = {
            "prompt": model_input,
            "num_samples": 1,
            "sampling_params": tinker.SamplingParams(**_tinker_sampling_kwargs(params)),
            "include_prompt_logprobs": topk is not None,
            "topk_prompt_logprobs": int(topk or 0),
        }
        return params, prefix_tuple, kwargs

    def _parse_response(
        self,
        response: Any,
        prefix_tuple: tuple[TokenId, ...],
        params: SamplingParams,
    ) -> GenerateResult:
        seq = response.sequences[0]
        tokens = tuple(seq.tokens)
        raw_logprobs = getattr(seq, "logprobs", None)
        logprobs = (
            finite_logprobs(raw_logprobs, len(tokens))
            if params.logprobs is not None
            else []
        )
        return GenerateResult(
            tokens=tokens,
            text=self.decode(tokens),
            prefix_tokens=prefix_tuple,
            stop_reason=normalize_stop_reason(getattr(seq, "stop_reason", None)),
            logprobs=logprobs,
            prompt_logprobs=_extract_prompt_logprobs(
                getattr(response, "prompt_logprobs", None),
                len(prefix_tuple),
            ),
            prompt_top_logprobs=_extract_prompt_top_logprobs(
                getattr(response, "topk_prompt_logprobs", None),
                len(prefix_tuple),
            ),
        )

    def close(self) -> None:
        close_client(self._client)

    def __enter__(self) -> TinkerPolicy:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "TinkerPolicy",
]
