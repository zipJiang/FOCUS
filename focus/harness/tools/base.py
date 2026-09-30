"""The Nous/Hermes tool contract, aligned with qwen-agent's ``BaseTool``.

A tool is anything the model can invoke by name with JSON arguments. The :class:`Tool`
protocol is structural, so a real ``qwen_agent.tools.BaseTool`` instance satisfies it --
no dependency in either direction: tools written for qwen-agent drop straight in, and
ours run in their framework. :func:`tool_schema` turns a tool into the OpenAI function
schema the Qwen chat template renders into its ``<tools>`` block; :func:`dispatch` runs
a parsed call and turns every failure (unknown tool, bad args, an exception) into text
the model reads back -- a tool fault must never kill a rollout (qwen-agent's rule).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """A callable the model may invoke. Structurally like a qwen-agent tool."""

    name: str
    description: str
    #: OpenAI-style JSON Schema for the args (``{"type","properties","required"}``).
    parameters: dict[str, Any]

    #: ``kwargs`` is the per-call context :func:`dispatch` forwards (``workspace=``,
    #: and whatever else an env threads through). Deliberately ``Any``, not
    #: ``object``: the context is open, and a tool that consumes one of those
    #: keywords declares it with its real type (see ``StorageTool.call``) -- which
    #: against a narrower ``object`` would be an LSP violation.
    def call(
        self, params: str | dict[str, Any], **kwargs: Any
    ) -> str | Awaitable[str]: ...


def tool_schema(tool: Tool) -> dict[str, Any]:
    """The OpenAI function schema the chat template renders into ``<tools>``."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class ToolError(Exception):
    """A bad-argument problem surfaced back to the model as the tool result string."""


class BaseTool(ABC):
    """Author a tool by subclassing: set name/description/parameters, define ``call``.

    Mirrors ``qwen_agent.tools.base.BaseTool``: :attr:`function` self-describes, and
    :meth:`verify_args` parses the model's JSON args and checks required keys.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    @property
    def function(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def verify_args(self, params: str | dict[str, Any]) -> dict[str, Any]:
        """Parse ``params``, check required keys, and coerce to the declared types."""
        data = params if isinstance(params, dict) else _loads(params)
        required = self.parameters.get("required", [])
        missing = [k for k in required if k not in data]
        if missing:
            raise ToolError(f"missing required argument(s): {', '.join(missing)}")
        return _coerce(data, self.parameters.get("properties", {}))

    @abstractmethod
    def call(
        self, params: str | dict[str, Any], **kwargs: Any
    ) -> str | Awaitable[str]: ...


def _loads(params: str) -> dict[str, Any]:
    try:
        data = json.loads(params)
    except (ValueError, TypeError) as exc:
        raise ToolError(f"arguments are not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ToolError("arguments must be a JSON object")
    return data


#: How to read a value declared as a non-string JSON type. ``bool`` first: a model that
#: writes ``"false"`` means false, which ``bool(str)`` would read as true.
_COERCE: dict[str, Callable[[Any], Any]] = {
    "integer": lambda v: bool(v) if isinstance(v, bool) else int(str(v).strip()),
    "number": lambda v: float(v) if isinstance(v, bool) else float(str(v).strip()),
    "boolean": lambda v: (
        v
        if isinstance(v, bool)
        else {"true": True, "false": False}[str(v).strip().lower()]
    ),
}


def _coerce(data: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    """Read each argument as the schema says it is typed, or raise :class:`ToolError`.

    A tool declares ``{"type": "integer"}`` and should be able to trust it. It cannot,
    because whether the value arrives typed depends on the *model's dialect*, not the
    tool: Hermes sends a JSON object (so ``3`` stays an int), while Qwen3.5's XML
    template has nowhere to put a type and sends ``"3"`` for everything. Coercing here,
    against the schema the tool already declared, is the one place that knows both --
    otherwise every tool author writes the same try/except and invents their own error
    message for it.
    """
    out = dict(data)
    for key, value in data.items():
        coerce = _COERCE.get(properties.get(key, {}).get("type", "string"))
        if coerce is None or value is None or value == "":
            continue
        try:
            out[key] = coerce(value)
        except (TypeError, ValueError, KeyError):
            declared = properties[key]["type"]
            raise ToolError(
                f"`{key}` must be {'an' if declared[0] in 'ai' else 'a'} "
                + f"{declared}, got {value!r}"
            ) from None
    return out


__all__ = ["BaseTool", "Tool", "ToolError", "tool_schema"]
