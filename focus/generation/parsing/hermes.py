"""The Nous/Hermes tool-call dialect: parse an assistant turn.

This is the model-native tool-use path for Qwen (and any Hermes-format model). An
assistant turn is ``[reasoning] </think> [text] <tool_call>{...}</tool_call>*``, where
each call is a JSON object; :class:`HermesToolCallParser` splits it into the
:class:`~focus.generation.parsing.ToolTurn` the harness runs on.

Only the *JSON body* lives here. The turn-shaped part -- splitting the reasoning off,
finding the ``<tool_call>`` blocks both dialects share, clearing them out of the visible
text -- is :func:`~focus.generation.parsing.base.tool_turn`, and what happens
to a
decoded turn (dispatching its calls, intercepting ``submit``, ending the episode)
belongs to :class:`~focus.harness.tools.environment.ToolEnv`, which never sees
a tag.
So a model that speaks a different dialect (see
:mod:`focus.generation.parsing.qwen_xml`) is one ``decode`` function and
nothing else.
"""

from __future__ import annotations

import json

from focus.generation.parsing.base import (
    ActionParser,
    ToolCall,
    ToolTurn,
    tool_turn,
)
from focus.generation.types import GenerateResult

_DECODER = json.JSONDecoder()


class HermesToolCallParser(ActionParser[ToolTurn]):
    """Parse a Hermes-style assistant turn: split reasoning, extract ``<tool_call>``s.

    The splitting is :func:`~focus.generation.parsing.base.tool_turn`, shared
    with
    every other tool dialect; this parser is only :func:`_parse_call`, the JSON body.
    """

    def parse(self, result: GenerateResult) -> ToolTurn:
        return tool_turn(result.text, _parse_call)


def _parse_call(block: str) -> ToolCall:
    """Parse one ``<tool_call>`` block's JSON object into a :class:`ToolCall`.

    ``block`` is the raw text between the tags; the object is located and decoded in one
    pass with :meth:`json.JSONDecoder.raw_decode` from the first ``{``, so prose the
    model adds around it (and braces inside string values) are handled. A block with no
    decodable object degrades to a nameless call carrying the raw text, so ``dispatch``
    reports it back to the model instead of the turn looking like an answer.
    """
    start = block.find("{")
    if start != -1:
        try:
            obj, _ = _DECODER.raw_decode(block, start)
            name = str(obj["name"])
            args = obj.get("arguments", {})
            arguments = (
                args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            )
            return ToolCall(name=name, arguments=arguments)
        except (ValueError, KeyError, TypeError):
            pass
    return ToolCall(name="", arguments=block.strip())


__all__ = ["HermesToolCallParser"]
