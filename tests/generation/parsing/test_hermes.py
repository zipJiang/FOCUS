"""The Nous/Hermes dialect: decoding an assistant turn into a ToolTurn.

Parsing only -- what the harness *does* with a decoded turn is
`tools/test_environment.py`, and the tools it dispatches to are
`tools/test_dispatch.py`.
"""

from __future__ import annotations

from focus.generation import GenerateResult
from focus.generation.parsing import HermesToolCallParser, ToolCall
from focus.generation.parsing.base import ActionParser
from focus.registry import build


def _result(text: str) -> GenerateResult:
    return GenerateResult(text=text)


TOOL_CALL = (
    '<tool_call>\n{"name": "storage", "arguments": '
    + '{"operate": "put", "key": "k", "value": "v"}}\n</tool_call>'
)


def test_parse_splits_reasoning_and_extracts_the_call() -> None:
    turn = HermesToolCallParser().parse(
        _result(f"reasoning here\n</think>\n\nLet me store it.\n{TOOL_CALL}")
    )
    assert turn.reasoning == "reasoning here"
    assert turn.text == "Let me store it."  # tool_call block stripped from visible text
    assert turn.tool_calls == (
        ToolCall("storage", '{"operate": "put", "key": "k", "value": "v"}'),
    )


def test_parse_ignores_tool_call_mentioned_inside_reasoning() -> None:
    # a <tool_call> the model muses about in its <think> region must not fire
    turn = HermesToolCallParser().parse(
        _result(f"I could write {TOOL_CALL} but wait\n</think>\n\nActually, done.")
    )
    assert turn.tool_calls == ()
    assert turn.text == "Actually, done."


def test_parse_final_answer_has_no_calls() -> None:
    turn = HermesToolCallParser().parse(_result("think\n</think>\n\nThe answer is 42"))
    assert turn.tool_calls == ()
    assert turn.text == "The answer is 42"


def test_parse_multiple_parallel_calls() -> None:
    two = TOOL_CALL + "\n" + TOOL_CALL
    turn = HermesToolCallParser().parse(_result(f"go\n</think>\n\n{two}"))
    assert len(turn.tool_calls) == 2


def test_parse_malformed_block_degrades_to_a_nameless_call() -> None:
    bad = "<tool_call>\n{not json}\n</tool_call>"
    turn = HermesToolCallParser().parse(_result(f"x\n</think>\n\n{bad}"))
    assert turn.tool_calls[0].name == ""  # dispatch will report it back to the model


def test_parse_tolerates_text_around_the_json_inside_the_tags() -> None:
    # leading and trailing prose inside the tags must not drop the call (finding #5)
    messy = '<tool_call>hi {"name": "search", "arguments": {"q": "x"}} ok</tool_call>'
    turn = HermesToolCallParser().parse(_result(f"go\n</think>\n\n{messy}"))
    assert turn.tool_calls == (ToolCall("search", '{"q": "x"}'),)


def test_parse_handles_braces_inside_json_strings() -> None:
    # the brace matcher must ignore ``}``/``{`` inside string values
    block = '<tool_call>{"name": "x", "arguments": {"v": "a}b{c"}}</tool_call>'
    turn = HermesToolCallParser().parse(_result(f"go\n</think>\n\n{block}"))
    assert turn.tool_calls[0].name == "x"
    assert turn.tool_calls[0].arguments == '{"v": "a}b{c"}'


def test_parse_unparseable_block_is_a_call_not_a_final_answer() -> None:
    # a present-but-unparseable <tool_call> must register a (nameless) call, so the env
    # does not mistake the turn for a final answer and end the episode
    turn = HermesToolCallParser().parse(
        _result("go\n</think>\n\n<tool_call>???</tool_call>")
    )
    assert turn.tool_calls
    assert turn.tool_calls[0].name == ""


def test_parser_from_registry() -> None:
    assert isinstance(build(ActionParser, {"name": "hermes"}), HermesToolCallParser)
