import json

import pytest

from focus.generation.parsing.hermes import HermesToolCallParser
from tests.helpers import ScriptedGenerator, tool_call, toy_runner
from tests.protocols.helpers import KeepRecent


async def test_custom_protocol_changes_context_without_task_transition():
    runner = toy_runner(
        ScriptedGenerator([tool_call("keep_recent", n=1)]),
        parser=HermesToolCallParser(),
        protocol=KeepRecent(),
    )
    root = await runner.start("question")
    result = await runner.advance(root)
    assert result.context_version == 1
    assert len(result.turns) == 1
    assert result.transitions == ()
    assert result.effects[0].turn_index == 0
    assert result.forkable
    assert root.messages[0]["role"] == "system"


async def test_mixed_mutations_are_rejected_without_partial_execution():
    text = "".join(
        "<tool_call>" + json.dumps(call) + "</tool_call>"
        for call in [
            {"name": "keep_recent", "arguments": {"n": 1}},
            {"name": "task_operation", "arguments": {}},
        ]
    )
    runner = toy_runner(
        ScriptedGenerator([text]), parser=HermesToolCallParser(), protocol=KeepRecent()
    )
    result = await runner.advance(await runner.start("question"))
    assert result.context_version == 0
    assert result.transitions == ()
    assert "separately" in result.messages[-1]["content"]


async def test_programming_failures_are_not_reported_as_invalid_model_actions():
    from focus.protocols.base import ToolProtocol, schema

    class Broken(ToolProtocol):
        def schemas(self):
            return (schema("break", "break", {}, []),)

        async def operation(self, rs, name, arguments, runner):
            raise KeyError("backend not registered")

    runner = toy_runner(
        ScriptedGenerator([tool_call("break")]),
        parser=HermesToolCallParser(),
        protocol=Broken(),
    )
    with pytest.raises(KeyError, match="backend not registered"):
        await runner.advance(await runner.start("question"))


@pytest.mark.parametrize("arguments", [{}, {"n": True}, {"n": 0}, {"n": "1"}, []])
async def test_invalid_custom_arguments_preserve_context(arguments):
    text = (
        "<tool_call>"
        + json.dumps({"name": "keep_recent", "arguments": arguments})
        + "</tool_call>"
    )
    runner = toy_runner(
        ScriptedGenerator([text]), parser=HermesToolCallParser(), protocol=KeepRecent()
    )
    try:
        root = await runner.start("question")
        result = await runner.advance(root)
        assert result.messages[: len(root.messages)] == root.messages
        assert result.context_version == 0
        assert result.effects == ()
        assert result.transitions == ()
        assert "Invalid keep_recent" in result.messages[-1]["content"]
    finally:
        await runner.services.aclose()


async def test_context_operations_obey_runner_step_limit():
    generator = ScriptedGenerator([tool_call("keep_recent", n=1)])
    runner = toy_runner(
        generator,
        parser=HermesToolCallParser(),
        protocol=KeepRecent(),
        max_steps=2,
        finish_prompt=None,
    )
    try:
        result = await runner.run_from("question")
        assert result.truncated
        assert generator.calls == 2
        assert result.context_version == 2
        assert len(result.turns) == 2
        assert result.steps_remaining == 0
        assert result.transitions == ()
    finally:
        await runner.services.aclose()
