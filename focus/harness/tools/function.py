"""Tools from plain functions: the schema comes from the signature and the docstring.

    @tool
    async def search(query: str, top_k: int = 5, *, workspace: Workspace) -> str:
        '''Search the corpus.

        Args:
            query: What to look for.
            top_k: How many results to return.
        '''

The first docstring paragraph describes the tool; ``Args:`` entries (or
``Annotated[T, "description"]``) describe parameters. ``str``, ``int``, ``float``,
``bool``, ``list[...]``, ``Literal[...]`` and optional (``X | None``) parameters are
supported; a parameter without a default is required. A parameter named
``workspace`` receives the active branch's workspace and is not shown to the model.
Return a string, or any JSON-serializable value; raise :class:`ToolError` to tell
the model what went wrong.
"""

from __future__ import annotations

import inspect
import json
import re
import types
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from focus.harness.tools.base import BaseTool, ToolError

_SCALARS: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class FunctionTool(BaseTool):
    """A :class:`BaseTool` that calls a function with validated arguments."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        summary, documented = _parse_docstring(inspect.getdoc(fn) or "")
        self.description = description or summary
        self.parameters, self._takes_workspace = _schema(fn, documented)

    async def call(
        self, params: str | dict[str, Any], *, workspace: Any = None, **kwargs: Any
    ) -> str:
        args = self.verify_args(params)
        unknown = set(args) - set(self.parameters["properties"])
        if unknown:
            raise ToolError(f"unknown argument(s): {', '.join(sorted(unknown))}")
        for key, spec in self.parameters["properties"].items():
            allowed = spec.get("enum")
            if allowed is not None and key in args and args[key] not in allowed:
                raise ToolError(f"`{key}` must be one of {allowed}, got {args[key]!r}")
        if self._takes_workspace:
            args["workspace"] = workspace
        result = self.fn(**args)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) else json.dumps(result)


@overload
def tool(fn: Callable[..., Any], /) -> FunctionTool: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Turn a function into a tool; use bare or with ``name=`` / ``description=``."""

    def wrap(function: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(function, name=name, description=description)

    return wrap(fn) if fn is not None else wrap


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """The first paragraph, and ``Args:`` entries keyed by parameter name."""
    summary = " ".join(doc.split("\n\n")[0].split())
    documented: dict[str, str] = {}
    section = re.search(r"^Args:\n((?:[ \t]+.*(?:\n|$))+)", doc, flags=re.M)
    if section is None:
        return summary, documented
    lines = section.group(1).splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    current: str | None = None
    for line in lines:
        depth = len(line) - len(line.lstrip())
        entry = re.match(r"\s*(\w+)(?:\s*\([^)]*\))?:\s*(.*)", line)
        if depth == indent and entry:
            current = entry.group(1)
            documented[current] = entry.group(2).strip()
        elif current is not None and line.strip():
            documented[current] += " " + line.strip()
    return summary, documented


def _schema(
    fn: Callable[..., Any], documented: dict[str, str]
) -> tuple[dict[str, Any], bool]:
    hints = get_type_hints(fn, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    takes_workspace = False
    for parameter in inspect.signature(fn).parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.name == "workspace":
            takes_workspace = True
            continue
        annotation = hints.get(parameter.name, str)
        note = documented.get(parameter.name, "")
        if get_origin(annotation) is Annotated:
            annotation, *extras = get_args(annotation)
            note = next((e for e in extras if isinstance(e, str)), note)
        spec, optional = _json_type(annotation, parameter.name)
        if note:
            spec["description"] = note
        if parameter.default is not inspect.Parameter.empty:
            if parameter.default is not None:
                spec["default"] = parameter.default
        elif not optional:
            required.append(parameter.name)
        properties[parameter.name] = spec
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema, takes_workspace


def _json_type(annotation: Any, name: str) -> tuple[dict[str, Any], bool]:
    """A JSON schema for one parameter type, and whether ``None`` is allowed."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) == 1:
            inner, _ = _json_type(members[0], name)
            return inner, True
    if annotation in _SCALARS:
        return {"type": _SCALARS[annotation]}, False
    if origin is Literal:
        values = list(get_args(annotation))
        kinds = {_SCALARS.get(type(v), "string") for v in values}
        return {
            "type": kinds.pop() if len(kinds) == 1 else "string",
            "enum": values,
        }, False
    if origin in (list, tuple) or annotation in (list, tuple):
        items = get_args(annotation)
        spec: dict[str, Any] = {"type": "array"}
        if items and items[0] is not Ellipsis:
            spec["items"], _ = _json_type(items[0], name)
        return spec, False
    if origin is dict or annotation is dict:
        return {"type": "object"}, False
    raise TypeError(f"tool parameter `{name}` has an unsupported type: {annotation!r}")


__all__ = ["FunctionTool", "tool"]
