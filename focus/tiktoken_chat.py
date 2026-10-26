"""A chat tokenizer for the hosted models, over ``tiktoken`` and nothing else.

:class:`~focus.codec.ChatCodec` wraps anything satisfying the
:class:`~focus.codec.HFTokenizer` protocol, and until now the only thing that
did was a real HuggingFace tokenizer -- which meant a hosted-model run had to borrow
some open model's chat template to count its tokens and render its transcript. That
pairing works, but it lies twice: the counts a fold triggers on are another vocabulary's
counts, and the transcript is rendered in a dialect the hosted model was never trained
to read.

:class:`TiktokenChatTokenizer` is the honest pairing. It satisfies that same protocol
with the hosted model's *own* encoding (``o200k_base`` for the current line), so token
counts -- the numbers that decide when a context folds and whether a budget is spent --
are the ones the server will also arrive at. Its template is deliberately ours and
deliberately plain: on this path the prompt reaches the model as a *decoded* user turn
(see
:class:`~focus.generation._openai_shared.NativePolicy`) and the tool
schemas reach it as the API's own ``tools``, so the template's only job is to render a
conversation a reader -- and a model reading it as prose -- can follow unambiguously.

``tiktoken`` is included in the standard installation and loaded on first use.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from focus.codec import Message

_TIKTOKEN_HINT = (
    "tiktoken is required for TiktokenChatTokenizer. "
    + "Install it with: pip install tiktoken"
)


def _import_tiktoken() -> Any:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(_TIKTOKEN_HINT) from exc
    return tiktoken


def _arguments_text(arguments: Any) -> str:
    """A call's arguments as JSON text, whether they arrived parsed or raw.

    The harness stores them *parsed* on the assistant message (an HF template has to
    iterate them); the wire carries them as text. Both render the same line here.
    """
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except TypeError:  # pragma: no cover - a mapping the runner never builds
        return str(arguments)


class TiktokenChatTokenizer:
    """The :class:`~focus.codec.HFTokenizer` surface over a tiktoken encoding.

    Structural, like every codec here: it is handed to
    ``ChatCodec(TiktokenChatTokenizer())`` and that is the whole wiring.
    """

    #: The protocol's "is there a template" flag. There is one -- this class *is* it --
    #: so ``ChatCodec.supports_chat`` is true and ``render`` works. The string is a name
    #: rather than Jinja source: nothing reads it but that check.
    chat_template: str | None = "tiktoken-plain"

    def __init__(self, encoding: str = "o200k_base") -> None:
        self.encoding_name = encoding
        self._encoding: Any = None

    def _enc(self) -> Any:
        if self._encoding is None:
            self._encoding = _import_tiktoken().get_encoding(self.encoding_name)
        return self._encoding

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Text to ids. ``add_special_tokens`` is accepted and ignored: this tokenizer
        has no special ids to add -- the template above is plain text, so its role
        markers are ordinary tokens. ``disallowed_special=()`` keeps a transcript that
        happens to quote ``<|endoftext|>`` from raising instead of encoding."""
        del add_special_tokens
        return list(self._enc().encode(text, disallowed_special=()))

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        """Ids back to text. ``skip_special_tokens`` is accepted and ignored for the
        same reason: there are no special ids in this vocabulary to skip, so the flag
        the bridge passes is satisfied by doing nothing rather than by raising."""
        del skip_special_tokens
        return str(self._enc().decode(list(token_ids)))

    def encode_single_token(self, raw: bytes) -> int:
        """One token's bytes to its id -- tiktoken's exact-recovery path.

        Forwarded because the chat bridge probes for it: with it, a completion's
        ``logprobs.content`` entries map straight back to the model's own ids instead of
        being re-encoded from their rendered strings.
        """
        return int(self._enc().encode_single_token(raw))

    def apply_chat_template(
        self,
        conversation: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        return_tensors: str | None,
    ) -> list[int]:
        """Render the conversation to plain text, then to ids.

        ``tools`` is deliberately **ignored**. On this path the schemas reach the model
        as the API's own ``tools`` parameter, so rendering them into the transcript as
        well would state them twice -- in two dialects -- and invite the model to answer
        in the rendered one instead of through the channel the parser reads.
        """
        del tools
        if not tokenize or return_dict or return_tensors is not None:
            raise ValueError(
                "TiktokenChatTokenizer renders straight to ids: it supports only "
                + "tokenize=True, return_dict=False, return_tensors=None"
            )
        return self.encode(self.render_text(conversation, add_generation_prompt))

    def render_text(
        self, conversation: Sequence[Message], add_generation_prompt: bool = True
    ) -> str:
        """The transcript this tokenizer counts: one role-tagged block per message.

        Public because it is the thing to *look at* when a hosted run reads oddly -- the
        prompt the bridge sends is exactly this text, decoded back out of the ids.
        """
        blocks = [_render_message(message) for message in conversation]
        if add_generation_prompt:
            blocks.append("<|assistant|>\n")
        return "".join(blocks)


def _render_message(message: Message) -> str:
    """One message as a role-tagged block, tool calls on their own readable lines."""
    role = str(message.get("role", "user"))
    lines = [f"<|{role}|>\n"]
    content = message.get("content")
    if content:
        lines.append(f"{content}\n")
    for call in message.get("tool_calls") or ():
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name", "")
        arguments = _arguments_text(function.get("arguments", {}))
        call_id = call.get("id", "") if isinstance(call, dict) else ""
        lines.append(f'<|tool_call|> {name} {arguments} (id: "{call_id}")\n')
    return "".join(lines)


__all__ = [
    "TiktokenChatTokenizer",
]
