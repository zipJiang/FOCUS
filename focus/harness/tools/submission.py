"""The terminal action: ending an episode by *calling* a tool, not by staying silent.

The older convention was that an assistant turn with no tool call **is** the answer.
That reads the ending off an absence, and an absence is a weak signal: a model
post-trained on tool use keeps emitting calls, so the ending it produces most naturally
is the one the convention cannot see. The failure is real -- a rollout whose
out-of-budget turn ended on a tool call scored blank, because
:meth:`~...runner.Agent.finish` does not dispatch those and there was no visible text
left to read.

:class:`SubmitTool` makes the ending a positive act. The model calls ``submit``, its
arguments *are* the answer (typed by a schema the env chooses, so a task wanting an
answer plus a citation plus a confidence asks for three fields rather than parsing them
back out of prose), and :class:`~focus.harness.tools.environment.ToolEnv`
intercepts the call by name before
dispatch.

The tool object here carries only a name and a schema; it is never dispatched, because
:meth:`~focus.harness.tools.environment.ToolEnv.step` catches it first.
:meth:`SubmitTool.call` exists so
that an env which forgot to wire it degrades to a readable message instead of "tool does
not exist".
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any

from focus.generation.parsing import ToolCall, ToolTurn
from focus.harness.tools.base import BaseTool, ToolError
from focus.harness.tools.dispatch import tool_reply

DEFAULT_DESCRIPTION = (
    "Submit your final answer. This ends the task: no tool call after it will run, so "
    + "call it only once the answer is settled."
)


@dataclass(frozen=True)
class SubmitFields:
    """Optional answer storage for a task state dataclass."""

    answer: str | None = None


class SubmitTool(BaseTool):
    """The terminal action's name and schema. Intercepted, never dispatched.

    The default schema takes one string. Pass ``parameters`` (and ``field``, naming
    which key holds the answer for the convenience path) to demand more structure -- the
    whole point of a typed ending is that a task can ask for its answer in pieces.
    """

    name = "submit"
    description = DEFAULT_DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "your final answer, in the form the task asked for",
            }
        },
        "required": ["answer"],
    }

    def __init__(
        self,
        *,
        name: str = "submit",
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        field: str = "answer",
    ) -> None:
        self.name = name
        self.description = description or DEFAULT_DESCRIPTION
        if parameters is not None:
            self.parameters = parameters
        #: Which argument holds the answer, for envs that just want the string.
        self.field = field

    def verify_args(self, params: str | dict[str, Any]) -> dict[str, Any]:
        """Validate declared string fields as well as required keys."""
        payload = super().verify_args(params)
        for key, value in payload.items():
            declared = self.parameters.get("properties", {}).get(key, {}).get("type")
            if declared == "string" and not isinstance(value, str):
                raise ToolError(f"`{key}` must be a string")
        return payload

    def call(self, params: str | dict[str, Any], **kwargs: object) -> str:
        # Unreachable through a submit-aware ToolEnv, which catches it by name.
        # Reached only if a submit tool was registered on an env that does not know
        # about it -- better a message the model can act on than a silent no-op.
        del params, kwargs
        return (
            f"{self.name} is not wired up in this environment, so nothing was recorded."
        )


#: The default terminal action, shared by every env that does not configure its own.
SUBMIT = SubmitTool()

__all__ = [
    "DEFAULT_DESCRIPTION",
    "SUBMIT",
    "SubmitFields",
    "SubmitTool",
]


def has_submit_fields(state: object) -> bool:
    """Whether the default submission hook can store an answer on the state."""
    return is_dataclass(state) and hasattr(state, "answer")


def submitted_call(turn: ToolTurn, submit: SubmitTool | None) -> ToolCall | None:
    """Find the first terminal call, which takes precedence over ordinary tools."""
    if submit is None:
        return None
    return next((call for call in turn.tool_calls if call.name == submit.name), None)


def submission_payload(
    call: ToolCall | None, submit: SubmitTool | None
) -> dict[str, Any]:
    """Read submission arguments, returning an empty payload when invalid.

    Normal steps report malformed calls so the model can retry. A forced ending has
    no remaining turn for a retry, so an unreadable call yields an empty payload.
    """
    if call is None or submit is None:
        return {}
    try:
        return submit.verify_args(call.arguments)
    except ToolError:
        return {}


def submit_replies(
    turn: ToolTurn, submitted: ToolCall, content: str
) -> tuple[dict[str, Any], ...]:
    if not submitted.call_id:
        return (tool_reply(submitted, content),)
    return tuple(
        tool_reply(
            call,
            content
            if call is submitted
            else "Not executed: a submit call took precedence in this turn.",
        )
        for call in turn.tool_calls
    )
