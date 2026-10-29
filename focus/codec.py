"""Codec: the tokenization / conversation-rendering seam, independent of any backend.

A codec turns content into token ids and back. The base :class:`TextCodec` is the flat
text<->token seam used by the generation backends (satisfied by a HuggingFace
``PreTrainedTokenizer`` or a ``tiktoken.Encoding``); :class:`ChatCodec` wraps a
HuggingFace tokenizer to render a conversation -- not just raw text -- into tokens,
when that tokenizer carries a chat template.

Codecs are structural, injected seams: they wrap a live tokenizer and are duck-typed.
This module depends on nothing else in the package, so
both the generation backends and the harness can build on it without an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

#: One chat message: ``{"role": ..., "content": ...}``, and whatever else that role's
#: template reads. ``Any`` rather than ``str`` because an assistant turn that called a
#: tool through a backend's *structured* channel carries the calls themselves --
#: ``{"tool_calls": [{"id", "type", "function": {...}}]}``, the OpenAI shape an HF chat
#: template renders back into the model's own dialect (see
#: :func:`~focus.harness.agent._record_reply`). Every message this library
#: writes still has a string ``content``.
Message = dict[str, Any]


@runtime_checkable
class TextCodec(Protocol):
    """Minimal tokenizer seam: ``encode`` text and ``decode`` ids.

    Satisfied by both a HuggingFace ``PreTrainedTokenizer`` and a ``tiktoken.Encoding``.
    A codec that also exposes ``encode_single_token(bytes) -> int`` (tiktoken) unlocks
    the exact per-token recovery in the OpenAI-chat backend.
    """

    def encode(self, text: str) -> Sequence[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


def _decode_text(
    codec: TextCodec, ids: Sequence[int], *, skip_special_tokens: bool = False
) -> str:
    """Decode ids, passing ``skip_special_tokens`` only to codecs accepting it (HF)."""
    try:
        return codec.decode(  # type: ignore[call-arg]
            list(ids), skip_special_tokens=skip_special_tokens
        )
    except TypeError:
        return codec.decode(list(ids))


def _encode_text(codec: TextCodec, text: str) -> list[int]:
    """Encode text, tolerating special-token strings (``tiktoken`` raises otherwise)."""
    try:
        return list(codec.encode(text, disallowed_special=()))  # type: ignore[call-arg]
    except TypeError:
        return list(codec.encode(text))


@runtime_checkable
class HFTokenizer(Protocol):
    """The HuggingFace ``PreTrainedTokenizer`` surface :class:`ChatCodec` uses.

    Structural, so any HF tokenizer satisfies it -- but naming the surface makes the
    HuggingFace assumption explicit rather than hidden behind ``Any``. ``chat_template``
    is ``None`` when the tokenizer carries no chat template (so ``apply_chat_template``
    would raise); ``apply_chat_template`` returns a plain ``list[int]`` for these
    arguments (``tokenize=True``, ``return_tensors=None``).
    """

    chat_template: str | None

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str: ...

    def apply_chat_template(
        self,
        conversation: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        return_tensors: str | None,
    ) -> list[int]: ...


class ChatCodec:
    """Text and conversation rendering over an injected HuggingFace tokenizer.

    ``encode`` / ``decode`` work with any :class:`HFTokenizer`; ``render``
    additionally needs a chat template (:attr:`supports_chat`) and raises a clear
    error on a tokenizer without one.
    """

    def __init__(self, tokenizer: HFTokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> HFTokenizer:
        """The tokenizer bound to this codec, including optional byte recovery."""
        return self._tokenizer

    @property
    def supports_chat(self) -> bool:
        """Whether the tokenizer carries a chat template (needed to render turns)."""
        return self._tokenizer.chat_template is not None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]:
        """Encode a raw text fragment to token ids.

        ``add_special_tokens`` defaults to ``False`` (unlike HuggingFace's own default):
        a codec is used to *extend* a prefix, and the special tokens belong to the chat
        template that :meth:`render` emits, not to a mid-stream fragment.
        """
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self, tokens: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self._tokenizer.decode(
            list(tokens), skip_special_tokens=skip_special_tokens
        )

    def render(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> tuple[int, ...]:
        """Template the whole ``messages`` list into token ids (a full re-render).

        ``tools`` are OpenAI-style function schemas (``{"type": "function", "function":
        {...}}``); the template renders them into the system turn and teaches the
        ``<tool_call>`` emission format. Raises :class:`ValueError` if the tokenizer has
        no chat template.
        """
        if not self.supports_chat:
            raise ValueError(
                "tokenizer has no chat template; ChatCodec.render "
                + "needs one -- encode raw text instead"
            )
        return tuple(
            self._tokenizer.apply_chat_template(
                list(messages),
                tools=list(tools) if tools else None,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
                return_tensors=None,
            )
        )


__all__ = ["ChatCodec", "HFTokenizer", "Message", "TextCodec"]
