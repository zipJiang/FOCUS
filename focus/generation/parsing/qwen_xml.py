"""Qwen3.5's XML tool-call dialect: the second protocol the harness speaks.

Qwen3.5 is not a Hermes-format model. Where the Nous/Hermes template wraps a JSON object
(``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``, parsed by
:class:`~focus.generation.parsing.hermes.HermesToolCallParser`), Qwen3.5's
chat
template renders a call as nested XML::

    <tool_call>
    <function=search>
    <parameter=query>
    who founded it
    </parameter>
    </function>
    </tool_call>

Same outer ``<tool_call>`` tag, different inside -- so the Hermes parser finds no JSON
object and reads every call as a final answer. Only the inside is this module's: the
turn is split by :func:`~focus.generation.parsing.base.tool_turn` exactly as a
Hermes
turn is. Everything downstream of the parser is
dialect-agnostic (:class:`~focus.harness.tools.environment.ToolEnv` dispatches
on
:class:`~focus.generation.parsing.ToolCall`, the turn log records tokens), so
a
model swap is a parser swap and nothing else.

Parameters arrive as text (the template has nowhere to put a type). They are re-emitted
here as a JSON object so ``verify_args`` / ``_coerce`` can type them from the tool
schema, the same path Hermes JSON calls take.
"""

from __future__ import annotations

import json
import re

from focus.generation.parsing.base import (
    ActionParser,
    ToolCall,
    ToolTurn,
    tool_turn,
)
from focus.generation.types import GenerateResult

# Only what is *inside* the shared <tool_call> tag is this module's business; the tag
# itself, and the reasoning split around it, are `parser.tool_turn`'s.
_FUNCTION = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)\s*>\n?(.*?)\n?</parameter>", re.DOTALL)


class QwenXMLToolCallParser(ActionParser[ToolTurn]):
    """Parse Qwen3.5's XML tool calls into the ``ToolTurn`` the env takes.

    The turn is split by :func:`~focus.generation.parsing.base.tool_turn`
    exactly as a
    Hermes turn is; only :func:`_parse_call` differs.
    """

    action = "tool"

    def parse(self, result: GenerateResult) -> ToolTurn:
        return tool_turn(result.text, _parse_call)


def _parse_call(block: str) -> ToolCall:
    """Parse one block's ``<function=...>`` XML into a :class:`ToolCall`.

    Parameters arrive as text (the template has nowhere to put a type) and are
    re-emitted as a JSON object, so they reach ``verify_args`` / ``_coerce`` the way
    Hermes' own JSON arguments do. A block with no decodable ``<function=...>``
    degrades to a nameless call carrying the raw text, so ``dispatch`` reports it back
    to the model instead of the turn reading as an answer.
    """
    match = _FUNCTION.search(block)
    if match is None:
        return ToolCall(name="", arguments=block.strip())
    args = dict(_PARAMETER.findall(match.group(2)))
    return ToolCall(
        name=match.group(1).strip(), arguments=json.dumps(args, ensure_ascii=False)
    )


__all__ = ["QwenXMLToolCallParser"]
