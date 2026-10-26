"""Corporate Chat Completions policy using branch-local native message history.

The provider owns its prompt template, so local IDs are estimates and every result is
inexact. Tool schemas belong to PreparedPrompt; returned calls retain their IDs and
verbatim JSON arguments for subsequent tool responses.
"""

from __future__ import annotations

from typing import Any

from focus.codec import (
    Message,
    encode_text,
)
from focus.generation._openai_shared import (
    NativePolicy,
    _request_error,
)
from focus.generation.interfaces import (
    merge_sampling_params,
    normalize_stop_reason,
)
from focus.generation.policy import (
    PolicyFormat,
    PreparedPrompt,
)
from focus.generation.types import (
    GenerateResult,
    NativeToolCall,
    SamplingParams,
    TokenId,
)


class OpenAIChatPolicy(NativePolicy):
    """Native Chat Completions with estimated local token accounting."""

    #: A chat backend always has `message.tool_calls` to read: the channel is the
    #: backend's, so every result carries a tuple (see `_native_calls`), never `None`.
    provides_native_tool_calls = True

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
        super().__init__(
            model=model,
            format=format,
            version=version,
            default_params=default_params,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            locked_sampling=locked_sampling,
        )
        # tiktoken exposes ``encode_single_token`` for exact id recovery; cache the
        # lookup once rather than probing the codec for every completion token.
        self._encode_single: Any = getattr(
            self.format.codec.tokenizer, "encode_single_token", None
        )

    async def agenerate(
        self,
        prompt: PreparedPrompt,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        self._check_prompt(prompt)
        prefix_tuple = prompt.tokens
        params = merge_sampling_params(self.default_params, sampling_params)
        model = self.served_model
        messages = self._to_messages(prompt.messages)

        try:
            response = await self._async_client().chat.completions.create(
                **self._chat_kwargs(model, messages, params, prompt.tools)
            )
        except Exception as exc:
            raise _request_error(
                exc,
                base_url=self.base_url,
                model=model,
                prefix_tokens=prefix_tuple,
                params=params,
                label="chat completions",
            ) from exc

        return self._parse_choice(response.choices[0], prefix_tuple, params)

    def _chat_kwargs(
        self,
        model: str,
        messages: list[Message],
        params: SamplingParams,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # `max_completion_tokens` is the current spelling of the budget and the only
        # one the reasoning-line models accept; the older `max_tokens` is deprecated
        # and every current chat model takes the new name.
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": params.max_tokens,
        }
        if not self._locked_sampling:
            kwargs["temperature"] = params.temperature
            kwargs["top_p"] = params.top_p
        if params.stop:
            kwargs["stop"] = list(params.stop)
        if params.logprobs is not None:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = int(params.logprobs)
        if tools:
            # The model's own tool protocol (see the module docstring for why).
            # `tool_choice` is left at the API's default -- the harness asks for tools
            # to be available, never for one to be called.
            kwargs["tools"] = list(tools)
        return kwargs

    def _parse_choice(
        self,
        choice: Any,
        prefix_tuple: tuple[TokenId, ...],
        params: SamplingParams,
    ) -> GenerateResult:
        content = getattr(choice.message, "content", None) or ""
        tokens, logprobs = self._tokens_and_logprobs(
            choice, content, want_logprobs=params.logprobs is not None
        )
        return GenerateResult(
            tokens=tokens,
            prefix_tokens=prefix_tuple,
            text=content,
            stop_reason=normalize_stop_reason(getattr(choice, "finish_reason", None)),
            logprobs=logprobs,
            exact_generation=False,
            native_tool_calls=self._native_calls(choice),
        )

    def _native_calls(self, choice: Any) -> tuple[NativeToolCall, ...]:
        """The structured calls this choice carries -- possibly none, never ``None``.

        Whether the channel *exists* is a property of the backend, not of one call:
        this is a chat backend, so ``message.tool_calls`` is always there to be read,
        and a turn that carries none -- schemas not sent, or sent and declined -- is a
        prose turn, reported as ``()``. That distinction is load-bearing for folds: the
        fold's derived runner inherits the rollout's policy, and under a toolless
        ``FoldEnv`` (a ``NullWorkspace`` fold) its generations send no schemas -- a
        native parser must read those as "no call made", not as the misconfiguration
        ``None`` names (a token-native backend that has no structured channel at all).
        """
        raw = getattr(choice.message, "tool_calls", None) or ()
        return tuple(
            NativeToolCall(
                # `function.name`/`arguments` is the OpenAI shape; a call missing either
                # is reported as-is rather than dropped, so `dispatch` tells the model
                # about it the way a malformed in-band call is told about.
                name=str(getattr(call.function, "name", "") or ""),
                arguments=str(getattr(call.function, "arguments", "") or ""),
                call_id=str(getattr(call, "id", "") or ""),
            )
            for call in raw
        )

    def _tokens_and_logprobs(
        self,
        choice: Any,
        content: str,
        *,
        want_logprobs: bool,
    ) -> tuple[tuple[TokenId, ...], list[float]]:
        payload = getattr(choice, "logprobs", None) if want_logprobs else None
        entries = getattr(payload, "content", None)
        if not entries:
            # No per-token logprobs to align to: re-encode the whole completion and
            # leave logprobs empty (an empty list skips the alignment check).
            return tuple(encode_text(self._codec, content)), []

        tokens: list[TokenId] = []
        logprobs: list[float] = []
        for entry in entries:
            ids = self._entry_token_ids(entry)
            if not ids:
                continue
            tokens.extend(ids)
            # Attach the entry's logprob to its first sub-id; pad any extra sub-ids with
            # 0.0 so ``len(tokens) == len(logprobs)`` always holds.
            logprobs.append(float(getattr(entry, "logprob", 0.0)))
            logprobs.extend([0.0] * (len(ids) - 1))
        return tuple(tokens), logprobs

    def _entry_token_ids(self, entry: Any) -> list[TokenId]:
        """Recover the local token ids for one ``logprobs.content`` entry.

        Prefers the exact path -- a codec exposing ``encode_single_token`` (tiktoken)
        maps the entry's raw ``bytes`` straight to OpenAI's own token id, robust to
        byte-fallback tokens that render as ``token``. Falls back to the token string.
        """
        raw_bytes = getattr(entry, "bytes", None)
        if self._encode_single is not None and raw_bytes is not None:
            try:
                return [self._encode_single(bytes(raw_bytes))]
            except (KeyError, ValueError):
                pass
        return encode_text(self._codec, getattr(entry, "token", "") or "")


__all__ = [
    "OpenAIChatPolicy",
]
