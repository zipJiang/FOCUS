"""Continue a single assistant response across exact token boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from focus.codec import Message
from focus.generation.interfaces import merge_sampling_params
from focus.generation.parsing.base import ActionParser
from focus.generation.policy import Policy, PreparedPrompt, register_policy
from focus.generation.types import GenerateResult, SamplingParams, TokenId
from focus.registry import register

_TOKENS = "focus_continuation_tokens"


@dataclass(frozen=True)
class TokenChunk:
    """One generated segment and whether the model finished its response."""

    tokens: tuple[TokenId, ...]
    finished: bool


@register(ActionParser, "token_chunk")
class TokenChunkParser(ActionParser[TokenChunk]):
    """Extract original tokens and distinguish continuation from completion.

    ``length`` leaves the response open; ``stop`` finishes it. Other finish reasons
    are rejected. The generation layer validates the request's token limit.
    """

    def parse(self, result: GenerateResult) -> TokenChunk:
        if result.stop_reason not in {"length", "stop"}:
            raise ValueError(f"Unexpected finish reason: {result.stop_reason}")
        return TokenChunk(tuple(result.tokens), result.stop_reason == "stop")


@register_policy("token_continuation")
class TokenContinuationPolicy[A](Policy[A]):
    """Wrap a token-input policy to continue one response in successive segments.

    The backend prepares the initial prompt. Later prompts append the original
    generated IDs, preserving the exact prefix used to sample each segment. Replies
    carry those IDs in branch-local message metadata, which survives checkpoint
    copies and serialization. Use uninterrupted assistant output with ``AppendOnly``;
    tool observations or edited segment text require a regular chat policy.

    The backend retains its format, parser, model identity, version, and sampling
    defaults. Custom parsers remain supported; ``TokenChunkParser`` is convenient
    for fixed-size segments. Closing this wrapper closes its backend, so close one
    of them once the shared backend is no longer needed.
    """

    def __init__(self, backend: Policy[A]) -> None:
        if not backend.provides_exact_tokens or backend.provides_native_tool_calls:
            raise ValueError("Token continuation requires an exact token-input policy")
        self.backend = backend
        super().__init__(
            format=backend.format,
            model=backend.model,
            served_model=backend.served_model,
            version=backend.version,
            default_params=backend.default_params,
        )

    @property
    def trainable(self) -> bool:
        return self.backend.trainable

    def _boundary(self, messages: Sequence[Message]) -> int:
        boundary = next(
            (i for i, message in enumerate(messages) if _TOKENS in message),
            len(messages),
        )
        for message in messages[boundary:]:
            if message.get("role") != "assistant" or _TOKENS not in message:
                raise ValueError(
                    "Token continuation requires uninterrupted assistant output"
                )
            tokens = message[_TOKENS]
            if not isinstance(tokens, (tuple, list)) or any(
                type(token) is not int for token in tokens
            ):
                raise ValueError("Continuation metadata must contain integer token IDs")
            if message.get("content") != self.decode(tokens, skip_special_tokens=True):
                raise ValueError(
                    "Continuation text was edited; its token IDs are unchanged"
                )
        return boundary

    def validate_context(self, messages: Sequence[Message]) -> None:
        boundary = self._boundary(messages)
        self.backend.validate_context(messages[:boundary])

    def prepare(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
    ) -> PreparedPrompt:
        boundary = self._boundary(messages)
        initial = self.backend.prepare(messages[:boundary], tools)
        tokens = initial.tokens + tuple(
            token for message in messages[boundary:] for token in message[_TOKENS]
        )
        return replace(initial, tokens=tokens, _messages=deepcopy(tuple(messages)))

    async def agenerate_tokens(
        self,
        tokens: Sequence[TokenId],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        params = merge_sampling_params(self.default_params, sampling_params)
        limit = params.max_tokens
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_tokens must be a positive integer")
        result = await self.backend.agenerate_tokens(tokens, params)
        if not result.exact_generation:
            raise ValueError("Token continuation requires exact generated token IDs")
        if len(result.tokens) > limit:
            raise ValueError(
                f"Generation exceeded the requested limit of {limit} tokens"
            )
        if result.stop_reason == "length" and len(result.tokens) != limit:
            raise ValueError("Generation stopped before the requested token limit")
        return result

    def decode(
        self, tokens: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self.backend.decode(tokens, skip_special_tokens=skip_special_tokens)

    def parse(self, result: GenerateResult) -> A:
        return self.backend.parse(result)

    def is_exact(self, result: GenerateResult) -> bool:
        return self.backend.is_exact(result)

    def record_reply(self, result: GenerateResult, messages: list[Message]) -> None:
        if (
            not result.exact_generation
            or result.native_tool_calls
            or result.native_output
        ):
            raise ValueError("Token continuation requires an exact text completion")
        super().record_reply(result, messages)
        messages[-1][_TOKENS] = tuple(result.tokens)

    def startup_check(self) -> None:
        self.backend.startup_check()

    def close(self) -> None:
        self.backend.close()

    async def aclose(self) -> None:
        await self.backend.aclose()
