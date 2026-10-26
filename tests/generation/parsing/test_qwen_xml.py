from __future__ import annotations

from focus.generation import GenerateResult
from focus.generation.parsing import (
    HermesToolCallParser,
    QwenXMLToolCallParser,
    ToolCall,
)


def _result(text: str) -> GenerateResult:
    return GenerateResult(text=text)


CALL = (
    "<tool_call>\n<function=search>\n<parameter=query>\n"
    + "who founded it\n</parameter>\n</function>\n</tool_call>"
)


def test_parse_splits_reasoning_and_extracts_the_call() -> None:
    turn = QwenXMLToolCallParser().parse(
        _result(f"reasoning here</think>\n\nLet me look.\n{CALL}")
    )
    assert turn.reasoning == "reasoning here"
    assert turn.text == "Let me look."  # tool_call block stripped from visible text
    assert turn.tool_calls == (ToolCall("search", '{"query": "who founded it"}'),)


def test_every_parameter_arrives_as_a_string() -> None:
    # the dialect carries no types; verify_args/_coerce turn these into schema types
    turn = QwenXMLToolCallParser().parse(
        _result(
            "t</think><tool_call>\n<function=read>\n<parameter=docid>\n40103\n"
            + "</parameter>\n<parameter=offset>\n6000\n</parameter>\n"
            + "</function>\n</tool_call>"
        )
    )
    assert turn.tool_calls == (
        ToolCall("read", '{"docid": "40103", "offset": "6000"}'),
    )


def test_parse_extracts_several_calls_from_one_turn() -> None:
    turn = QwenXMLToolCallParser().parse(_result(f"t</think>{CALL}\n{CALL}"))
    assert [c.name for c in turn.tool_calls] == ["search", "search"]


def test_parse_ignores_a_call_mentioned_inside_reasoning() -> None:
    # a call the model muses about in its <think> region must not fire
    turn = QwenXMLToolCallParser().parse(
        _result(f"I could write {CALL} but wait</think>\n\nActually, done.")
    )
    assert turn.tool_calls == ()
    assert turn.text == "Actually, done."


def test_a_turn_with_no_closing_think_is_all_visible() -> None:
    # Thinking models pre-fill the opener, so a completion cut off mid-reasoning has no
    # </think> at all; treating it as reasoning would silently drop the whole turn.
    turn = QwenXMLToolCallParser().parse(_result("still thinking, no closer"))
    assert turn.reasoning == ""
    assert turn.text == "still thinking, no closer"


def test_the_reasoning_split_is_the_one_the_hermes_dialect_gets() -> None:
    """Both dialects share `parser.tool_turn`, so the trimming around the split is one
    rule, not one per dialect: the reasoning region comes back stripped exactly as the
    visible text does, whichever tags the model wrote its calls in."""
    text = "  reasoning here  </think>  visible  "

    qwen = QwenXMLToolCallParser().parse(_result(text))
    hermes = HermesToolCallParser().parse(_result(text))

    assert qwen.reasoning == hermes.reasoning == "reasoning here"
    assert qwen.text == hermes.text == "visible"


def test_unparseable_block_is_a_call_not_a_final_answer() -> None:
    # a present-but-malformed block must register a (nameless) call, so dispatch reports
    # it back to the model rather than the turn reading as an answer
    turn = QwenXMLToolCallParser().parse(
        _result("t</think><tool_call>garbage</tool_call>")
    )
    assert turn.tool_calls == (ToolCall("", "garbage"),)


def test_parse_final_answer_has_no_calls() -> None:
    turn = QwenXMLToolCallParser().parse(_result("t</think>\n\nThe answer is 42"))
    assert turn.tool_calls == ()
    assert turn.text == "The answer is 42"
