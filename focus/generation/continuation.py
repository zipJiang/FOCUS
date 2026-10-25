"""Continue a single assistant response across exact token boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from focus.codec import Message
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
    """Read fixed-size segments, allowing the final segment to end earlier.

    Set the policy's ``SamplingParams(max_tokens=chunk_size)`` to the same size.
    A ``length`` finish must reach that boundary; a ``stop`` finish ends the response.
    Other finish reasons fail explicitly so interrupted generations cannot be graded
    as completed answers.
    """

    def __init__(self, chunk_size: int = 256) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def parse(self, result: GenerateResult) -> TokenChunk:
        if result.stop_reason not in {"length", "stop"}:
            raise ValueError(f"Unexpected finish reason: {result.stop_reason}")
        if len(result.tokens) > self.chunk_size:
            raise ValueError(f"A segment exceeded {self.chunk_size} tokens")
        if result.stop_reason == "length" and len(result.tokens) != self.chunk_size:
            raise ValueError("Generation stopped before the requested chunk boundary")
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
        result = await self.backend.agenerate_tokens(tokens, sampling_params)
        if not result.exact_generation:
            raise ValueError("Token continuation requires exact generated token IDs")
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
