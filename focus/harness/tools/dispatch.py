"""Execute decoded tool calls and format their replies."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import isawaitable
from typing import Any

from focus.generation.parsing.base import ToolCall

from .base import Tool, ToolError


async def dispatch(tools: Mapping[str, Tool], call: ToolCall, **context: object) -> str:
    """Run one tool call, returning the string fed back to the model as a tool result.

    Awaits the tool if its ``call`` is a coroutine (async I/O) or returns synchronously.
    Any ``context`` (e.g. ``workspace=...``, the rollout's per-branch scratch store) is
    forwarded as keyword args to the tool -- a tool ignores what it does not use. Never
    raises: an unknown tool, bad arguments, or a tool exception all become a string the
    model can read and react to.
    """
    tool = tools.get(call.name)
    if tool is None:
        available = ", ".join(tools) or "(none)"
        return f"Tool {call.name!r} does not exist. Available tools: {available}."
    try:
        result = tool.call(call.arguments, **context)
        if isawaitable(result):
            result = await result
        return str(result)
    except ToolError as exc:
        return f"error: {exc}"
    except Exception as exc:  # a tool fault is fed back, not fatal, per qwen-agent
        return f"error calling {call.name!r}: {exc}"


__all__ = ["dispatch"]


def tool_reply(call: ToolCall, content: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "tool", "content": content}
    if call.call_id:
        message["tool_call_id"] = call.call_id
    return message
