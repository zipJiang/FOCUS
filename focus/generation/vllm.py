"""vLLM completions generator (OpenAI ``/v1/completions`` wire format, vLLM-only)."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from focus.generation._openai_shared import (
    _import_async_openai,
    _import_openai,
    _request_error,
    _startup_error,
)
from focus.generation.interfaces import (
    aclose_client,
    close_client,
    finite_logprobs,
    prepare_generation,
    run_async,
    settled_knobs,
)
from focus.generation.policy import (
    Policy,
    PolicyFormat,
    PreparedPrompt,
    register_policy,
)
from focus.generation.types import (
    GenerateResult,
    SamplingParams,
    TokenId,
)

if TYPE_CHECKING:
    # Typing-only: the openai SDK is loaded lazily at call time (``_import_openai``),
    # so importing this backend never requires ``openai`` -- it stays an optional extra.
    from openai.types.completion_choice import CompletionChoice


_DEFAULT_SAMPLING = SamplingParams()
_TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")
logger = logging.getLogger(__name__)


@register_policy("vllm", aliases=("openai_completion",))
class VLLMPolicy(Policy[Any]):
    """vLLM completions generator for token rollout prompts.

    Speaks the OpenAI ``/v1/completions`` wire format but is vLLM-specific: it sends a
    token-id ``prompt`` plus vLLM ``extra_body`` (``return_token_ids`` etc.) and parses
    vLLM token ids back. It does **not** work against the hosted OpenAI API; for that
    use :class:`~focus.generation.openai_chat.OpenAIChatPolicy`. The
    ``openai_completion`` registry name is kept as a back-compat alias.
    """

    def __init__(
        self,
        *,
        model: str,
        format: PolicyFormat[Any] | None = None,
        profile: str | None = None,
        served_model: str | None = None,
        version: str = "policy",
        base_url: str,
        api_key: str,
        default_params: SamplingParams | None = None,
        timeout: float | None = None,
    ) -> None:
        """``timeout`` is the per-request deadline in seconds; ``None`` keeps the SDK's.

        Worth a knob rather than the SDK default (600s), because that default is a
        *hosted-API* number and this backend talks to a batch server the caller is
        deliberately saturating. Under a deep batch the slowest generations pass 600s,
        the SDK retries them, and the retry re-queues behind the same backlog -- so load
        converts into duplicated work rather than into a slower answer, and throughput
        collapses while the server still reads as busy. Measured on a 9B at ~700
        concurrent requests: 248 timeouts in 90 minutes, most of the traffic retries.
        """
        if format is not None and profile is not None:
            raise ValueError("Supply either format or profile")
        super().__init__(
            model=model,
            served_model=served_model,
            version=version,
            format=format or PolicyFormat.resolve(model, profile=profile),
            default_params=default_params,
        )
        self.base_url = base_url
        self._timeout = timeout
        try:
            openai_client = _import_openai()
        except ImportError as exc:
            raise RuntimeError(
                "Could not initialize the OpenAI-compatible completions generator "
                + f"at {base_url!r} for model {self.served_model!r}: {exc}"
            ) from exc
        self._client = openai_client(
            base_url=base_url, api_key=api_key, **self._client_kwargs()
        )
        self._api_key = api_key
        self._aclient: Any = None

    def startup_check(self) -> None:
        model = self.served_model
        assert model is not None
        try:
            self._client.models.list()
            self._client.completions.create(
                model=model,
                prompt="ping",
                temperature=0.0,
                max_tokens=1,
                top_p=1.0,
            )
        except Exception as exc:
            raise _startup_error(self.base_url, model, exc) from exc

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
        prefix_tuple, params, kwargs = self._prepare(prefix_tokens, sampling_params)
        try:
            response = self._client.completions.create(**kwargs)
        except Exception as exc:
            raise self._wrap_request_error(exc, prefix_tuple, params) from exc
        return _parse_completion(response, prefix_tuple)

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        prefix_tuple, params, kwargs = self._prepare(prefix_tokens, sampling_params)
        try:
            response = await self._async_client().completions.create(**kwargs)
        except Exception as exc:
            raise self._wrap_request_error(exc, prefix_tuple, params) from exc
        return _parse_completion(response, prefix_tuple)

    def _prepare(
        self,
        prefix_tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None,
    ) -> tuple[tuple[TokenId, ...], SamplingParams, dict[str, Any]]:
        prefix_tuple, params = prepare_generation(
            self.default_params, prefix_tokens, sampling_params
        )
        model = self.served_model
        kwargs = self._completion_kwargs(model, prefix_tuple, params)
        return prefix_tuple, params, kwargs

    def _wrap_request_error(
        self,
        exc: Exception,
        prefix_tuple: tuple[TokenId, ...],
        params: SamplingParams,
    ) -> RuntimeError:
        return _request_error(
            exc,
            base_url=self.base_url,
            model=self.served_model,
            prefix_tokens=prefix_tuple,
            params=params,
        )

    def _client_kwargs(self) -> dict[str, Any]:
        """Client options that are ours to set, omitted entirely when unset.

        Passing ``timeout=None`` would mean *no* timeout to the SDK rather than *its*
        timeout, so an unset knob has to not appear in the call at all.
        """
        return {} if self._timeout is None else {"timeout": self._timeout}

    def _async_client(self) -> Any:
        """Lazily build the ``AsyncOpenAI`` client (only when async is used)."""
        if self._aclient is None:
            self._aclient = _import_async_openai()(
                base_url=self.base_url,
                api_key=self._api_key,
                **self._client_kwargs(),
            )
        return self._aclient

    def _completion_kwargs(
        self,
        model: str,
        prefix_tokens: tuple[TokenId, ...],
        params: SamplingParams,
    ) -> dict[str, Any]:
        # The three blankable knobs, read through the typed face of the settle rule --
        # never off the fields, where `| None` invites re-spelled defaults per backend.
        raw_temperature, raw_max_tokens, raw_top_p = settled_knobs(params)
        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": list(prefix_tokens),
            "max_tokens": _checked("max_tokens", int(raw_max_tokens)),
        }
        temperature = _checked("temperature", float(raw_temperature))
        if temperature != _DEFAULT_SAMPLING.temperature:
            kwargs["temperature"] = temperature
        top_p = _checked("top_p", float(raw_top_p))
        if top_p != _DEFAULT_SAMPLING.top_p:
            kwargs["top_p"] = top_p
        stop = _normalize_stop(params.stop)
        if stop is not None:
            kwargs["stop"] = stop
        if params.logprobs is not None:
            kwargs["logprobs"] = _checked("logprobs", int(params.logprobs))
        if params.seed is not None:
            kwargs["seed"] = params.seed
        kwargs["extra_body"] = _vllm_extra_body(params)
        return kwargs

    def close(self) -> None:
        close_client(self._client)
        if self._aclient is not None:
            run_async(aclose_client(self._aclient))
            self._aclient = None

    async def aclose(self) -> None:
        close_client(self._client)
        if self._aclient is not None:
            await aclose_client(self._aclient)
            self._aclient = None


__all__ = [
    "VLLMPolicy",
]


def _parse_completion(
    response: Any, prefix_tuple: tuple[TokenId, ...]
) -> GenerateResult:
    """Build a :class:`GenerateResult` from a completions response (sync or async)."""
    choice = response.choices[0]
    tokens = _extract_tokens(choice)
    prompt_logprobs, prompt_top_logprobs = _extract_prompt_logprobs(
        response,
        choice,
        prefix_tuple,
    )
    return GenerateResult(
        tokens=tokens,
        text=choice.text,
        prefix_tokens=prefix_tuple,
        stop_reason=choice.finish_reason,
        logprobs=_extract_token_logprobs(choice, len(tokens)),
        top_logprobs=_extract_top_logprobs(choice, len(tokens)),
        prompt_logprobs=prompt_logprobs,
        prompt_top_logprobs=prompt_top_logprobs,
    )


def _extract_tokens(choice: CompletionChoice) -> tuple[TokenId, ...]:
    token_ids = _choice_extra(choice, "token_ids")
    if token_ids is not None:
        return tuple(token_ids)

    raw_tokens = _logprobs_field(choice, "tokens")
    if raw_tokens is not None:
        return tuple(_parse_vllm_token_ids(raw_tokens))

    raise ValueError(
        "Completion response did not include token ids. "
        + "Ensure the backend returns choice.token_ids with return_token_ids enabled."
    )


def _parse_vllm_token_ids(tokens: Sequence[str | None]) -> list[int]:
    token_ids: list[int] = []
    for index, token in enumerate(tokens):
        if token is None:
            raise ValueError(
                f"vLLM returned a null token at position {index} in "
                + "logprobs.tokens; dropping it would desync token ids from "
                + "their logprobs. Enable return_token_ids so token ids come "
                + "from choice.token_ids instead of this fallback."
            )
        match = _TOKEN_ID_RE.match(token)
        if not match:
            raise ValueError(
                f"Could not parse vLLM token id from {token!r}. Ensure the "
                + "backend honors return_tokens_as_token_ids and emits "
                + "'token_id:<int>' strings rather than decoded text."
            )
        token_ids.append(int(match.group(1)))
    return token_ids


def _logprobs_field(choice: Any, name: str) -> Any:
    """Read ``choice.logprobs.<name>``, tolerating a choice with no logprobs payload."""
    return getattr(getattr(choice, "logprobs", None), name, None)


def _extract_token_logprobs(choice: Any, token_count: int) -> list[float]:
    raw = _logprobs_field(choice, "token_logprobs")
    return finite_logprobs(raw, token_count)


def _extract_top_logprobs(
    choice: Any,
    token_count: int,
) -> list[dict[str, float]] | None:
    raw = _logprobs_field(choice, "top_logprobs")
    if token_count == 0 or raw is None or len(raw) != token_count:
        return None
    # `raw` comes off the backend's response object, so it is `Any` by construction:
    # the length check above is all that can be verified about it here.
    return cast("list[dict[str, float]]", raw)


def _extract_prompt_logprobs(
    response: Any,
    choice: Any,
    prefix_tokens: tuple[TokenId, ...],
) -> tuple[list[float | None], list[dict[str, float] | None] | None]:
    raw = _choice_extra(choice, "prompt_logprobs")
    if raw is None:
        raw = getattr(response, "prompt_logprobs", None)
    if not raw:
        return [], None

    per_token: list[float | None] = []
    top: list[dict[str, float] | None] = []
    for index, entry in enumerate(raw):
        token_id = prefix_tokens[index] if index < len(prefix_tokens) else None
        per_token.append(
            _selected_logprob(entry, token_id) if token_id is not None else None
        )
        top.append(_logprob_top_dict(entry) or None)
    return per_token, top


def _logprob_top_dict(entry: Any) -> dict[str, float]:
    if not isinstance(entry, Mapping):
        return {}
    top: dict[str, float] = {}
    for token_id, logprob_obj in entry.items():
        logprob_value, label = _logprob_fields(logprob_obj, token_id)
        if logprob_value is None:
            continue
        top[label] = logprob_value
    return top


def _selected_logprob(entry: Any, token_id: TokenId) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    # vLLM keys prompt_logprobs by int token id; a JSON round-trip makes it a string.
    logprob_obj = entry.get(token_id, entry.get(str(token_id)))
    if logprob_obj is None:
        return None
    return _logprob_fields(logprob_obj, token_id)[0]


def _logprob_fields(
    logprob_obj: Any,
    token_id: Any,
) -> tuple[float | None, str]:
    if isinstance(logprob_obj, Mapping):
        raw_logprob = logprob_obj.get("logprob")
        decoded = logprob_obj.get("decoded_token")
    else:
        raw_logprob = getattr(logprob_obj, "logprob", None)
        decoded = getattr(logprob_obj, "decoded_token", None)
    label = str(decoded or token_id)
    if raw_logprob is None:
        return None, label
    numeric = float(raw_logprob)
    return (numeric if math.isfinite(numeric) else None), label


def _choice_extra(choice: Any, name: str) -> Any:
    value = getattr(choice, name, None)
    if value is not None:
        return value
    extra = getattr(choice, "model_extra", None)
    if isinstance(extra, Mapping):
        return extra.get(name)
    return None


def _vllm_extra_body(params: SamplingParams) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    if params.repetition_penalty is not None:
        extra["repetition_penalty"] = _checked(
            "repetition_penalty", float(params.repetition_penalty)
        )
    if params.prompt_logprobs is not None:
        extra["prompt_logprobs"] = _checked(
            "prompt_logprobs", int(params.prompt_logprobs)
        )
    return extra


#: What vLLM accepts per sampling field: the range predicate and the message a value
#: outside it fails with. Rejecting here keeps a 400 from the server -- with its own
#: wording -- from being the first time a caller learns the bound.
_RANGES: dict[str, tuple[Callable[[float], bool], str]] = {
    "logprobs": (lambda v: v >= 0, "logprobs must be >= 0 for vLLM"),
    "prompt_logprobs": (
        lambda v: v >= 0,
        "prompt_logprobs must be >= 0 for vLLM",
    ),
    "temperature": (
        lambda v: 0.0 <= v <= 2.0,
        "temperature must be between 0 and 2 for vLLM",
    ),
    "max_tokens": (lambda v: v >= 1, "max_tokens must be >= 1 for vLLM"),
    "top_p": (lambda v: 0.0 < v <= 1.0, "top_p must be in (0, 1] for vLLM"),
    "repetition_penalty": (
        lambda v: v > 0.0,
        "repetition_penalty must be > 0 for vLLM",
    ),
}


def _checked[Number: (int, float)](field: str, value: Number) -> Number:
    """Return ``value`` when it is inside vLLM's accepted range for ``field``."""
    predicate, message = _RANGES[field]
    if not predicate(value):
        raise ValueError(message)
    return value


def _normalize_stop(stop: list[str] | None) -> list[str] | None:
    if not stop:
        return None
    normalized = [item for item in stop if item and item.strip()]
    return normalized or None
