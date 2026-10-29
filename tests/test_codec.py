from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from focus.codec import (
    ChatCodec,
    Message,
    TextCodec,
    _decode_text,
    _encode_text,
)


class ByteCodec:
    """A TextCodec double that tolerates the optional kwargs (like an HF tokenizer)."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        del kwargs
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int], **kwargs: Any) -> str:
        del kwargs
        return bytes(tokens).decode("utf-8")


class StrictByteCodec(ByteCodec):
    """Reject optional keyword arguments to exercise codec fallbacks."""

    def encode(self, text: str) -> list[int]:  # type: ignore[override]
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:  # type: ignore[override]
        return bytes(tokens).decode("utf-8")


def test_bytecodec_satisfies_the_textcodec_protocol() -> None:
    assert isinstance(ByteCodec(), TextCodec)


def test_encode_text_round_trips_and_tolerates_kwargs() -> None:
    for codec in (ByteCodec(), StrictByteCodec()):
        ids = _encode_text(codec, "hi")
        assert ids == list(b"hi")
        assert _decode_text(codec, ids, skip_special_tokens=True) == "hi"


class FakeChatTokenizer:
    """A stateless chat tokenizer double: one id per role-initial + content char.

    A trailing ``0`` marks ``add_generation_prompt``, so the returned ids alone prove
    the role, content, and generation-prompt flag were all forwarded through.
    """

    chat_template: str | None = "fake-template"

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) for c in text]
        return [1, *ids] if add_special_tokens else ids  # 1 is a fake BOS

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        ids = [t for t in token_ids if not (skip_special_tokens and t == 1)]
        return "".join(chr(t) for t in ids)

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
        assert tokenize is True
        assert return_dict is False
        assert return_tensors is None
        ids: list[int] = []
        if tools:
            ids.append(ord("T"))  # a marker so a tools-render is distinguishable
        for m in conversation:
            ids.append(ord(m["role"][0]))
            ids.extend(ord(c) for c in m["content"])
        if add_generation_prompt:
            ids.append(0)
        return ids


@pytest.fixture
def codec() -> ChatCodec:
    return ChatCodec(FakeChatTokenizer())


def test_supports_chat_reflects_the_chat_template() -> None:
    assert ChatCodec(FakeChatTokenizer()).supports_chat

    tok = FakeChatTokenizer()
    tok.chat_template = None  # a tokenizer without a chat template
    codec = ChatCodec(tok)
    assert not codec.supports_chat
    with pytest.raises(ValueError, match="no chat template"):
        codec.render([{"role": "user", "content": "hi"}])


def test_render_forwards_messages_and_generation_prompt(codec: ChatCodec) -> None:
    out = codec.render([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert out == (ord("u"), ord("h"), ord("i"), 0)  # role + content + gen marker


def test_chatcodec_encode_omits_special_tokens_by_default(codec: ChatCodec) -> None:
    # default: no fake BOS (1) prepended -- fragments append cleanly to a prefix
    assert list(codec.encode("ab")) == [ord("a"), ord("b")]
    # opt in explicitly to get them
    assert list(codec.encode("ab", add_special_tokens=True)) == [1, ord("a"), ord("b")]


def test_chatcodec_decode_can_skip_special_tokens(codec: ChatCodec) -> None:
    assert codec.decode([1, ord("a"), ord("b")]) == "\x01ab"
    assert codec.decode([1, ord("a"), ord("b")], skip_special_tokens=True) == "ab"
