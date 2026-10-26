"""Tools written as functions: schema from the signature, validation on the way in."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import pytest

from focus import ToolError, tool
from focus.harness import InMemoryWorkspace, ToolEnv
from focus.harness.tools import tool_schema


@tool
async def search(
    query: str,
    top_k: int = 5,
    mode: Literal["exact", "fuzzy"] = "exact",
    tags: list[str] | None = None,
    *,
    workspace: InMemoryWorkspace,
) -> dict[str, object]:
    """Search the notes.

    The second paragraph is not part of the description.

    Args:
        query: What to look for.
        top_k: How many results
            to return.
    """
    workspace.write("last", query)
    return {"query": query, "top_k": top_k, "mode": mode, "tags": tags}


def test_the_schema_comes_from_the_signature_and_docstring() -> None:
    assert search.name == "search"
    assert search.description == "Search the notes."
    assert search.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "top_k": {
                "type": "integer",
                "description": "How many results to return.",
                "default": 5,
            },
            "mode": {"type": "string", "enum": ["exact", "fuzzy"], "default": "exact"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    assert tool_schema(search)["function"]["name"] == "search"


async def test_calls_coerce_validate_and_receive_the_workspace() -> None:
    workspace = InMemoryWorkspace()
    reply = await search.call('{"query": "x", "top_k": "3"}', workspace=workspace)
    assert json.loads(reply) == {
        "query": "x",
        "top_k": 3,
        "mode": "exact",
        "tags": None,
    }
    assert workspace.read("last") == "x"
    with pytest.raises(ToolError, match="missing required"):
        await search.call("{}", workspace=workspace)
    with pytest.raises(ToolError, match="unknown"):
        await search.call('{"query": "x", "bogus": 1}', workspace=workspace)
    with pytest.raises(ToolError, match="one of"):
        await search.call('{"query": "x", "mode": "loose"}', workspace=workspace)


async def test_names_annotations_and_sync_functions() -> None:
    @tool(name="add", description="Add two numbers.")
    def plus(a: Annotated[float, "first"], b: float) -> str:
        return str(a + b)

    assert plus.name == "add"
    assert plus.parameters["properties"]["a"]["description"] == "first"
    assert await plus.call({"a": 1, "b": 2.5}) == "3.5"


def test_unsupported_parameter_types_are_refused() -> None:
    with pytest.raises(TypeError, match="unsupported type"):

        @tool
        def odd(value: complex) -> str:
            return ""


async def test_function_tools_run_inside_a_tool_env() -> None:
    from focus.generation.parsing import ToolCall, ToolTurn

    class Env(ToolEnv[None]):
        async def initial_state(self) -> None:
            return None

    env = Env([search])
    result = await env.step(
        None,
        ToolTurn(text="", tool_calls=(ToolCall("search", '{"query": "q"}', "c1"),)),
        workspace=InMemoryWorkspace(),
    )
    assert json.loads(result.messages[0]["content"])["query"] == "q"
