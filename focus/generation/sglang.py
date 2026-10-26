"""SGLang native ``/generate`` policy with exact token provenance."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from focus.generation.interfaces import (
    close_client,
    normalize_stop_reason,
    prepare_generation,
    run_async,
    settled_knobs,
)
from focus.generation.policy import (
    Policy,
    PolicyFormat,
    PreparedPrompt,
)
from focus.generation.types import GenerateResult, SamplingParams, TokenId


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "The SGLang backend requires httpx. Install it with: pip install httpx"
        ) from exc
    return httpx


def _generate_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url:
        raise ValueError("base_url is required")
    return url if url.endswith("/generate") else f"{url}/generate"


def _sglang_sampling_params(params: SamplingParams) -> dict[str, Any]:
    temperature, max_tokens, top_p = settled_knobs(params)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if params.top_k is not None:
        kwargs["top_k"] = params.top_k
    if params.repetition_penalty is not None:
        kwargs["repetition_penalty"] = params.repetition_penalty
    if params.stop:
        kwargs["stop"] = list(params.stop)
    if params.seed is not None:
        kwargs["sampling_seed"] = params.seed
    return kwargs


def _sglang_payload(
    prefix_tokens: tuple[TokenId, ...], params: SamplingParams
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ids": list(prefix_tokens),
        "sampling_params": _sglang_sampling_params(params),
        "return_logprob": True,
    }
    if params.prompt_logprobs is not None:
        # Score the complete supplied prefix. SGLang reports the first token with a
        # null logprob because there is no preceding token in this request.
        payload["logprob_start_len"] = 0
    topk = max(params.logprobs or 0, params.prompt_logprobs or 0)
    if topk:
        # SGLang uses one width for input and output top-logprob payloads.
        payload["top_logprobs_num"] = topk
    return payload


def _aligned_token_logprobs(
    pairs: Sequence[Sequence[Any]] | None,
) -> tuple[tuple[TokenId, ...], list[float]]:
    """Split SGLang ``[logprob, token_id, ...]`` output pairs."""
    if not pairs:
        return (), []
    tokens: list[TokenId] = []
    logprobs: list[float] = []
    for pair in pairs:
        if pair[0] is None or not math.isfinite(float(pair[0])):
            raise ValueError("missing or non-finite completion logprob")
        logprob = float(pair[0])
        tokens.append(int(pair[1]))
        logprobs.append(logprob)
    return tuple(tokens), logprobs


def _prompt_logprobs(
    pairs: Sequence[Sequence[Any]] | None, prefix_length: int
) -> list[float | None]:
    if not pairs:
        return []
    # Some SGLang versions omit the intrinsically unscored first token; normalize both
    # shapes to the Policy contract of one entry per supplied prefix token.
    normalized: list[Sequence[Any] | None] = list(pairs)
    if len(normalized) == prefix_length - 1:
        normalized.insert(0, None)
    if len(normalized) != prefix_length:
        return []
    result: list[float | None] = []
    for pair in normalized:
        if pair is None or not pair or pair[0] is None:
            result.append(None)
            continue
        value = float(pair[0])
        result.append(value if math.isfinite(value) else None)
    return result


def _one_top_logprobs(raw: Any) -> dict[str, float] | None:
    if raw is None or not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    values: dict[str, float] = {}
    for pair in raw:
        if not isinstance(pair, Sequence) or len(pair) < 2 or pair[0] is None:
            continue
        value = float(pair[0])
        if math.isfinite(value):
            values[f"token_id:{int(pair[1])}"] = value
    return values


def _top_logprobs(
    rows: Sequence[Any] | None,
    length: int,
    *,
    allow_missing_first: bool = False,
) -> list[dict[str, float] | None] | None:
    if rows is None:
        return None
    normalized: list[Any] = list(rows)
    if allow_missing_first and len(normalized) == length - 1:
        normalized.insert(0, None)
    if len(normalized) != length:
        return None
    return [_one_top_logprobs(row) for row in normalized]


def _completion_top_logprobs(
    rows: Sequence[Any] | None, length: int
) -> list[dict[str, float]] | None:
    aligned = _top_logprobs(rows, length)
    if aligned is None:
        return None
    # An output position always exists even when SGLang returned no alternatives for
    # it; represent that position as an empty mapping to retain 1:1 token alignment.
    return [row or {} for row in aligned]


def _parse_sglang_output(
    output: Mapping[str, Any],
    prefix_tokens: tuple[TokenId, ...],
    params: SamplingParams | None = None,
) -> GenerateResult:
    meta = output.get("meta_info") or {}
    if not isinstance(meta, Mapping):
        raise ValueError("SGLang response meta_info must be a mapping")
    raw_output = meta.get("output_token_logprobs")
    pairs = raw_output if isinstance(raw_output, Sequence) else None
    tokens, logprobs = _aligned_token_logprobs(pairs)
    finish = meta.get("finish_reason")
    matched = finish.get("matched") if isinstance(finish, Mapping) else None

    prompt_logprobs: list[float | None] = []
    prompt_top: list[dict[str, float] | None] | None = None
    output_top: list[dict[str, float]] | None = None
    if params is not None and params.prompt_logprobs is not None:
        raw_prompt = meta.get("input_token_logprobs")
        prompt_pairs = raw_prompt if isinstance(raw_prompt, Sequence) else None
        prompt_logprobs = _prompt_logprobs(prompt_pairs, len(prefix_tokens))
        raw_prompt_top = meta.get("input_top_logprobs")
        prompt_top = _top_logprobs(
            raw_prompt_top if isinstance(raw_prompt_top, Sequence) else None,
            len(prefix_tokens),
            allow_missing_first=True,
        )
    if params is not None and params.logprobs is not None:
        raw_output_top = meta.get("output_top_logprobs")
        output_top = _completion_top_logprobs(
            raw_output_top if isinstance(raw_output_top, Sequence) else None,
            len(tokens),
        )

    return GenerateResult(
        tokens=tokens,
        prefix_tokens=prefix_tokens,
        text=str(output.get("text", "")),
        stop_reason=normalize_stop_reason(finish),
        matched_stop=matched if isinstance(matched, (int, str)) else None,
        logprobs=logprobs,
        top_logprobs=output_top,
        prompt_logprobs=prompt_logprobs,
        prompt_top_logprobs=prompt_top,
    )


class SGLangPolicy(Policy[Any]):
    """Exact token generation through an independently hosted SGLang server.

    The policy owns sync and async httpx clients and speaks SGLang's native
    ``POST /generate`` route. Use :class:`SlimePolicy` when a training framework owns
    the transport and injects a gated ``post`` callable instead.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        format: PolicyFormat[Any] | None = None,
        profile: str | None = None,
        default_params: SamplingParams | None = None,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        if format is not None and profile is not None:
            raise ValueError("Supply either format or profile")
        super().__init__(
            model=model,
            format=format or PolicyFormat.resolve(model, profile=profile),
            default_params=default_params,
        )
        self.base_url = base_url.rstrip("/")
        self.url = _generate_url(base_url)
        request_headers = dict(headers or {})
        if api_key is not None:
            request_headers.setdefault("Authorization", f"Bearer {api_key}")
        self._headers = request_headers
        # httpx's default five-second timeout is too short for a deliberately batched
        # inference server. ``None`` intentionally means no transport deadline; the
        # harness can impose its own task deadline, or callers can set this explicitly.
        self._timeout = timeout
        try:
            self._httpx = _import_httpx()
        except ImportError as exc:
            raise RuntimeError(
                f"Could not initialize SGLang policy at {self.url!r}: {exc}"
            ) from exc
        self._client = self._httpx.Client(headers=self._headers, timeout=self._timeout)
        self._aclient: Any = None

    def generate(
        self, prompt: PreparedPrompt, sampling_params: SamplingParams | None = None
    ) -> GenerateResult:
        self._check_prompt(prompt)
        return self.generate_tokens(prompt.tokens, sampling_params)

    def generate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        prefix, params = prepare_generation(
            self.default_params, prefix_tokens, sampling_params
        )
        try:
            response = self._client.post(self.url, json=_sglang_payload(prefix, params))
            response.raise_for_status()
            output = response.json()
        except Exception as exc:
            raise self._request_error(exc, prefix, params) from exc
        return _parse_sglang_output(_mapping_response(output), prefix, params)

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        prefix, params = prepare_generation(
            self.default_params, prefix_tokens, sampling_params
        )
        try:
            response = await self._async_client().post(
                self.url, json=_sglang_payload(prefix, params)
            )
            response.raise_for_status()
            output = response.json()
        except Exception as exc:
            raise self._request_error(exc, prefix, params) from exc
        return _parse_sglang_output(_mapping_response(output), prefix, params)

    def startup_check(self) -> None:
        try:
            self.generate_tokens(
                (0,),
                SamplingParams(temperature=0.0, max_tokens=1, top_p=1.0, top_k=-1),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach SGLang /generate at {self.url!r}: {exc}"
            ) from exc

    def _async_client(self) -> Any:
        if self._aclient is None:
            self._aclient = self._httpx.AsyncClient(
                headers=self._headers, timeout=self._timeout
            )
        return self._aclient

    def _request_error(
        self,
        exc: Exception,
        prefix: tuple[TokenId, ...],
        params: SamplingParams,
    ) -> RuntimeError:
        _, max_tokens, _ = settled_knobs(params)
        return RuntimeError(
            "SGLang /generate request failed "
            + f"for {self.url!r}: {type(exc).__name__}: {exc}; "
            + f"prompt_tokens={len(prefix)}, max_tokens={max_tokens}"
        )

    def close(self) -> None:
        close_client(self._client)
        if self._aclient is not None:
            run_async(self._aclient.aclose())
            self._aclient = None

    async def aclose(self) -> None:
        close_client(self._client)
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None


def _mapping_response(output: Any) -> Mapping[str, Any]:
    if not isinstance(output, Mapping):
        raise ValueError(
            f"SGLang /generate returned {type(output).__name__}, expected an object"
        )
    return output


__all__ = ["SGLangPolicy"]
