"""The third protocol: tool calls a backend hands back as *structure*, not as text.

:mod:`~focus.generation.parsing.hermes` and
:mod:`~focus.generation.parsing.qwen_xml` read
a dialect the model wrote into its completion. A hosted chat model does neither: given
real ``tools``, it returns ``message.tool_calls`` -- a field beside the completion, in
the API's own shape -- and its completion text contains no call at all. Asking such a
model for someone else's in-band dialect is what produced the nameless, malformed calls
the OpenAI-bridge experiments measured; asking it for its own produces named calls with
typed arguments, and this parser is the one that reads them.

The price is exactness, and it is why :attr:`NativeToolCallParser.trainable` is
``False``: the call never was tokens, so a training pass over this turn's ids would be a
pass over the prose around an action it cannot see. A rollout under this parser is
evaluation: preparation refuses actor samples from its inexact generations.
"""

from __future__ import annotations

from focus.generation.parsing.base import ActionParser, ToolCall, ToolTurn
from focus.generation.types import GenerateResult


class NativeToolCallParser(ActionParser[ToolTurn]):
    """Read a backend's structured ``tool_calls`` into the env's :class:`ToolTurn`.

    Downstream of here nothing differs from an in-band dialect: the same
    :class:`~focus.generation.parsing.ToolCall` records, with JSON-text
    arguments
    that ``verify_args``/``_coerce`` type from the tool's own schema, reach the same
    :class:`~focus.harness.tools.environment.ToolEnv`.
    """

    action = "tool"

    #: The actions are not in the tokens -- see the module docstring.
    trainable = False
    requires_native_channel = True

    def parse(self, result: GenerateResult) -> ToolTurn:
        calls = result.native_tool_calls
        if calls is None:
            # The silent-mismatch failure this codebase documents twice does not get a
            # third spelling. `None` is the backend saying it has no structured channel
            # at all, and every call would otherwise decode as an answer -- the exact
            # shape of "the model cannot use tools" that is really a misconfiguration.
            raise RuntimeError(
                "no native tool-call channel on this backend: this parser reads a "
                + "backend's own `message.tool_calls`, which only a tools-aware chat "
                + "backend provides. Run it against `OpenAIChatPolicy`, or use the "
                + "`hermes` / `qwen_xml` parsers for a model that calls tools in band"
            )
        return ToolTurn(
            # The whole completion is the visible text: unlike an in-band dialect there
            # is nothing to strip out of it, and a hosted model's reasoning is not
            # returned at all -- so `reasoning` is empty rather than guessed at.
            text=result.text.strip(),
            tool_calls=tuple(
                ToolCall(name=call.name, arguments=call.arguments, call_id=call.call_id)
                for call in calls
            ),
            reasoning="",
        )


__all__ = ["NativeToolCallParser"]
