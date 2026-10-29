"""Continue a single assistant response across exact token boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from focus.generation.parsing.base import ActionParser
from focus.generation.types import GenerateResult, TokenId

_TOKENS = "focus_continuation_tokens"


@dataclass(frozen=True)
class TokenChunk:
    """One generated segment and whether the model finished its response."""

    tokens: tuple[TokenId, ...]
    finished: bool


class TokenChunkParser(ActionParser[TokenChunk]):
    """Extract original tokens and distinguish continuation from completion.

    ``length`` leaves the response open; ``stop`` finishes it. Other finish reasons
    are rejected. The generation layer validates the request's token limit.
    """

    action = "tokens"

    def parse(self, result: GenerateResult) -> TokenChunk:
        if result.stop_reason not in {"length", "stop"}:
            raise ValueError(f"Unexpected finish reason: {result.stop_reason}")
        return TokenChunk(tuple(result.tokens), result.stop_reason == "stop")


__all__ = ["TokenChunk", "TokenChunkParser"]
