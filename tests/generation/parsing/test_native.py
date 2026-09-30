"""The native dialect: decoding a backend's structured ``tool_calls`` into a ToolTurn.

Parsing only -- what the harness *does* with a decoded turn is
`tools/test_environment.py`, and
what puts the calls on the result in the first place is
`tests/generation/test_openai_chat.py`.
"""

from __future__ import annotations

import pytest

from focus.generation import GenerateResult, NativeToolCall
from focus.generation.parsing import NativeToolCallParser, ToolCall
from focus.generation.parsing.base import ActionParser
from focus.registry import build


def test_parse_maps_every_native_call_onto_the_turn() -> None:
    turn = NativeToolCallParser().parse(
        GenerateResult(
            text="  Looking it up.  ",
            native_tool_calls=(
                NativeToolCall("grep", '{"pattern": "gross income"}', "call_1"),
                NativeToolCall("read", '{"path": "statutes/section63.md"}', "call_2"),
            ),
        )
    )

    # the same ToolCall the in-band dialects produce: JSON-text arguments, so
    # `verify_args` types them from the tool's schema either way
    assert turn.tool_calls == (
        ToolCall("grep", '{"pattern": "gross income"}', "call_1"),
        ToolCall("read", '{"path": "statutes/section63.md"}', "call_2"),
    )
    assert turn.text == "Looking it up."
    # a hosted model does not return its reasoning, so none is claimed
    assert turn.reasoning == ""


def test_an_open_channel_with_no_call_is_an_ordinary_prose_turn() -> None:
    """``()`` is the model declining to call anything -- the env's stray/nudge path."""
    turn = NativeToolCallParser().parse(
        GenerateResult(text="The answer is 4200.", native_tool_calls=())
    )

    assert turn.tool_calls == ()
    assert turn.text == "The answer is 4200."


def test_no_channel_at_all_raises_rather_than_reading_as_an_answer() -> None:
    """The silent-mismatch failure gets a loud spelling here, not a third silent one.

    ``None`` is a backend that never had a structured channel -- every token-native one,
    and a chat call sent no schemas. Decoding that as "the model called nothing" is
    exactly the misconfiguration this parser exists to make impossible: every turn would
    read as prose and the run would report a model that cannot use tools.
    """
    with pytest.raises(RuntimeError, match="no native tool-call channel"):
        NativeToolCallParser().parse(GenerateResult(text="<tool_call>{}</tool_call>"))


def test_the_error_names_the_parsers_an_in_band_model_wants() -> None:
    with pytest.raises(RuntimeError, match="hermes` / `qwen_xml"):
        NativeToolCallParser().parse(GenerateResult(text=""))


def test_the_parser_is_registered_and_declares_itself_untrainable() -> None:
    parser = build(ActionParser, {"name": "native"})

    assert isinstance(parser, NativeToolCallParser)
    # the actions were never in the tokens, so the tokens cannot be trained on
    assert parser.trainable is False
