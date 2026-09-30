"""The hosted models' own tokenizer, wrapped by ``ChatCodec`` unchanged.

Two claims are worth a test here. The first is *conformance*: this class exists to
satisfy the ``HFTokenizer`` protocol, and the only proof that it does is a live
``ChatCodec`` driving it -- rendering, encoding, decoding -- through the same calls the
runner makes. The second is what the template says: role markers a reader (and a model
reading prose) can follow, tool calls visible in the assistant turn that made them, and
the schemas deliberately *absent*, because on this path they reach the model as the
API's own ``tools`` parameter.

Skipped where ``tiktoken`` is not installed: it ships in the ``openai-chat`` extra, and
the package must be importable without it.
"""

from __future__ import annotations

import pytest

from focus.codec import ChatCodec, HFTokenizer, Message
from focus.tiktoken_chat import TiktokenChatTokenizer

pytest.importorskip("tiktoken")


_CONVERSATION: tuple[Message, ...] = (
    {"role": "system", "content": "You apply the statute."},
    {"role": "user", "content": "How much tax is owed?"},
    {
        "role": "assistant",
        "content": "Let me look it up.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "grep", "arguments": {"pattern": "gross income"}},
            }
        ],
    },
    {"role": "tool", "content": "s.md:12: gross income means all income"},
)

_SCHEMA = {
    "type": "function",
    "function": {"name": "grep", "description": "search", "parameters": {}},
}


def _codec() -> ChatCodec:
    return ChatCodec(TiktokenChatTokenizer())


def test_it_satisfies_the_protocol_chatcodec_is_written_against() -> None:
    assert isinstance(TiktokenChatTokenizer(), HFTokenizer)


def test_a_chatcodec_over_it_renders_encodes_and_decodes() -> None:
    codec = _codec()

    ids = codec.render(_CONVERSATION)

    assert codec.supports_chat
    assert ids
    assert all(isinstance(i, int) for i in ids)
    assert codec.decode(list(codec.encode("gross income"))) == "gross income"


def test_decoding_a_render_gives_back_the_transcript_it_counted() -> None:
    """The round trip is the point: these ids are what a fold trigger measures."""
    tokenizer = TiktokenChatTokenizer()
    codec = ChatCodec(tokenizer)

    text = codec.decode(list(codec.render(_CONVERSATION)))

    assert text == tokenizer.render_text(_CONVERSATION)


def test_the_transcript_names_every_role_and_the_call_that_was_made() -> None:
    text = TiktokenChatTokenizer().render_text(_CONVERSATION)

    for marker in ("<|system|>", "<|user|>", "<|assistant|>", "<|tool|>"):
        assert marker in text
    assert "s.md:12: gross income means all income" in text
    # the call itself: name, arguments as JSON, and the id a history has to quote back
    assert '<|tool_call|> grep {"pattern": "gross income"} (id: "call_1")' in text


def test_a_generation_prompt_is_the_assistant_opener() -> None:
    tokenizer = TiktokenChatTokenizer()
    conversation: tuple[Message, ...] = ({"role": "user", "content": "hi"},)

    with_prompt = tokenizer.render_text(conversation, add_generation_prompt=True)
    without = tokenizer.render_text(conversation, add_generation_prompt=False)

    assert with_prompt == without + "<|assistant|>\n"


def test_the_schemas_are_not_rendered_into_the_transcript() -> None:
    """They travel as the API's own ``tools``; rendering them would state them twice."""
    codec = _codec()

    plain = codec.render(_CONVERSATION)
    with_tools = codec.render(_CONVERSATION, tools=[_SCHEMA])

    assert plain == with_tools


def test_wire_shaped_arguments_render_as_they_arrived() -> None:
    """Arguments reach this template parsed (from the runner) or raw (from the wire)."""
    tokenizer = TiktokenChatTokenizer()
    message: Message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_9",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "s.md"}'},
            }
        ],
    }

    text = tokenizer.render_text((message,), add_generation_prompt=False)

    assert '<|tool_call|> read {"path": "s.md"} (id: "call_9")' in text


def test_special_token_text_encodes_instead_of_raising() -> None:
    """A transcript may quote anything; tiktoken's default would raise on this one."""
    tokenizer = TiktokenChatTokenizer()

    ids = tokenizer.encode("<|endoftext|> appears in the statute")

    assert tokenizer.decode(ids) == "<|endoftext|> appears in the statute"
    # and the flags the protocol carries are accepted, not honoured -- no special ids
    assert tokenizer.decode(ids, skip_special_tokens=True) == tokenizer.decode(ids)


def test_it_renders_straight_to_ids_and_says_so() -> None:
    with pytest.raises(ValueError, match="straight to ids"):
        TiktokenChatTokenizer().apply_chat_template(
            _CONVERSATION,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
            return_tensors=None,
        )
