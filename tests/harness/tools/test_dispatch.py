"""The tool contract: dispatching a call, the schema it renders to, StorageTool."""

from __future__ import annotations

import pytest

from focus.generation.parsing import ToolCall
from focus.harness import (
    BaseTool,
    InMemoryWorkspace,
    StorageTool,
    ToolError,
    dispatch,
    tool_schema,
)


class _EchoSchema(BaseTool):
    """The shared schema. ``call`` is left to the two tools below, which differ only
    in whether they answer synchronously -- so neither overrides the other's return."""

    description = "echo the text"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }


class _Echo(_EchoSchema):
    name = "echo"

    def call(self, params, **kwargs) -> str:
        return str(self.verify_args(params)["text"])


class _AsyncEcho(_EchoSchema):
    name = "aecho"

    async def call(self, params, **kwargs) -> str:
        return "async:" + str(self.verify_args(params)["text"])


@pytest.mark.asyncio
async def test_dispatch_runs_sync_and_async_tools() -> None:
    tools = {"echo": _Echo(), "aecho": _AsyncEcho()}
    assert await dispatch(tools, ToolCall("echo", '{"text": "hi"}')) == "hi"
    assert await dispatch(tools, ToolCall("aecho", '{"text": "hi"}')) == "async:hi"


@pytest.mark.asyncio
async def test_dispatch_forwards_context_to_the_tool() -> None:
    # a call-time workspace reaches StorageTool, not the constructor default
    ws = InMemoryWorkspace()
    call = ToolCall("storage", '{"operate": "put", "key": "k", "value": "v"}')
    out = await dispatch({"storage": StorageTool()}, call, workspace=ws)
    assert out == "Successfully saved k."
    assert ws.read("k") == "v"


@pytest.mark.asyncio
async def test_dispatch_reports_failures_as_strings_never_raises() -> None:
    tools = {"echo": _Echo()}
    assert "does not exist" in await dispatch(tools, ToolCall("missing", "{}"))
    missing_arg = await dispatch(tools, ToolCall("echo", "{}"))
    assert missing_arg.startswith("error:")
    assert (await dispatch(tools, ToolCall("echo", "not json"))).startswith("error:")


def test_tool_schema_shape() -> None:
    schema = tool_schema(_Echo())
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["required"] == ["text"]


# -- StorageTool -------------------------------------------------------------------


def test_storage_tool_put_get_delete_scan() -> None:
    ws = InMemoryWorkspace()
    tool = StorageTool(ws)
    assert tool.call('{"operate": "put", "key": "a", "value": "hello"}') == (
        "Successfully saved a."
    )
    assert ws.read("a") == "hello"  # writes through to the backing workspace
    assert tool.call('{"operate": "get", "key": "a"}') == "hello"
    assert tool.call('{"operate": "scan"}') == "a: hello"
    assert tool.call('{"operate": "delete", "key": "a"}') == "Successfully deleted a."
    assert "does not exist" in tool.call('{"operate": "get", "key": "a"}')


def test_storage_tool_put_requires_value() -> None:
    with pytest.raises(ToolError):
        StorageTool().call('{"operate": "put", "key": "a"}')
