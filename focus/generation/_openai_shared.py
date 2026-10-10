"""Shared SDK plumbing and native corporate-policy history validation.

The SDK is optional and loaded when a backend creates its client. vLLM shares the SDK
helpers but retains its exact-token completions protocol.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from focus.codec import Message
from focus.generation.interfaces import aclose_client, run_async
from focus.generation.policy import Policy, PolicyFormat
from focus.generation.types import SamplingParams, TokenId

logger = logging.getLogger(__name__)

_SDK_HINT = (
    "The OpenAI SDK is required for the OpenAI-compatible generators. "
    + "Install it with: pip install focus[vllm]"
)


def _import_openai() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(_SDK_HINT) from exc
    return OpenAI


def _import_async_openai() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(_SDK_HINT) from exc
    return AsyncOpenAI


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _exception_detail(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if body is not None:
        return _compact_json(body)
    response = getattr(exc, "response", None)
    text = getattr(response, "text", None)
    if text:
        return str(text)
    return str(exc)


def _request_error(
    exc: Exception,
    *,
    base_url: str | None,
    model: str,
    prefix_tokens: tuple[TokenId, ...],
    params: SamplingParams,
    label: str = "completions",
) -> RuntimeError:
    # The summary rides along in both the log record and the message: a failed request
    # is otherwise unreproducible from the SDK's own error text alone.
    summary: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "prefix_tokens": len(prefix_tokens),
        "prefix_preview": list(prefix_tokens[:24]),
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "stop": list(params.stop or []),
        "logprobs": params.logprobs,
        "prompt_logprobs": params.prompt_logprobs,
        "repetition_penalty": params.repetition_penalty,
    }
    detail = _exception_detail(exc)
    status_code = _exception_status_code(exc)
    logger.exception(
        "OpenAI-compatible %s request failed",
        label,
        extra={"request_summary": summary, "status_code": status_code},
    )
    message = (
        f"OpenAI-compatible {label} request failed "
        + f"for {base_url!r} model {model!r}: {detail} request={summary}"
    )
    error = RuntimeError(message)
    if status_code is not None:
        error.status_code = status_code  # type: ignore[attr-defined]
    return error


class NativePolicy(Policy[Any]):
    """Corporate API policy. Native messages, estimated tokens, always eval-only."""

    def __init__(
        self,
        *,
        model: str,
        format: PolicyFormat[Any] | None = None,
        version: str = "policy",
        default_params: SamplingParams | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        locked_sampling: bool = False,
    ) -> None:
        resolved = format or PolicyFormat.resolve(model, profile="native")
        if not resolved.parser.requires_native_channel:
            raise ValueError("Corporate API policies require a native format")
        super().__init__(
            model=model,
            format=resolved,
            version=version,
            default_params=default_params,
            exact=False,
            native=True,
        )
        self._codec = resolved.codec
        self.base_url = base_url
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._locked_sampling = locked_sampling
        self._aclient: Any = None

    def validate_context(self, messages: Sequence[Message]) -> None:
        self._to_messages(list(messages))

    def _to_messages(self, history: list[Message]) -> list[Message]:
        messages: list[Message] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        pending: set[str] = set()
        seen: set[str] = set()
        for original in history:
            message = {k: v for k, v in original.items() if k != "native_output"}
            if message["role"] == "tool":
                call_id = message.get("tool_call_id")
                if call_id:
                    if call_id not in pending:
                        raise ValueError(f"Unmatched tool response {call_id!r}")
                    pending.remove(call_id)
                elif pending:
                    raise ValueError("Tool response is missing tool_call_id")
                else:
                    message["role"] = "user"  # control feedback, e.g. stray nudge
            else:
                if pending:
                    raise ValueError("Missing responses for native tool calls")
                for call in message.get("tool_calls", ()):
                    call_id = call.get("id")
                    if not call_id or call_id in seen:
                        raise ValueError("Native tool calls require unique call IDs")
                    pending.add(call_id)
                    seen.add(call_id)
            messages.append(message)
        if pending:
            raise ValueError("Missing responses for native tool calls")
        return messages

    def _async_client(self) -> Any:
        if self._aclient is None:
            self._aclient = _import_async_openai()(
                base_url=self.base_url, api_key=self._api_key
            )
        return self._aclient

    def close(self) -> None:
        run_async(self.aclose())

    async def aclose(self) -> None:
        if self._aclient is not None:
            await aclose_client(self._aclient)
            self._aclient = None


def _startup_error(base_url: str | None, model: str, exc: Exception) -> RuntimeError:
    detail = _exception_detail(exc)
    error = RuntimeError(
        "Could not reach the OpenAI-compatible server at "
        + f"{base_url!r} for model {model!r}. "
        + "Start the server, verify the model name, and confirm the "
        + f"/v1/completions endpoint works. {detail}"
    )
    status_code = _exception_status_code(exc)
    if status_code is not None:
        error.status_code = status_code  # type: ignore[attr-defined]
    return error
