"""Showing a conversation back to a model: ``role: content``, calls flattened.

The rollout's source of truth is a list of **message strings**, and the loop's own
direction of travel is forward -- render them to token ids and generate. This module is
the *other* direction: :func:`format_transcript` renders messages back into flat text
for another model to read (a summarizing fold, a critic's view of a rollout).

It is deliberately dialect-free -- it matches only the outer ``<tool_call>`` tag, which
every dialect the harness speaks shares -- so nothing that merely wants to *show* a
conversation has to import a parser to do it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from focus.codec import Message
from focus.generation.parsing.base import TOOL_CALL

#: How much of a flattened call to keep: enough to be a reminder that one happened, not
#: a transcript of it.
_CALL_CHARS = 200


def flatten_tool_calls(content: str) -> str:
    """One message's text with its ``<tool_call>`` blocks flattened to inert lines.

    Anywhere a transcript is re-shown to a model -- a summarizing fold, a critic's
    view of a rollout -- a call rendered *verbatim* is read as content rather than as
    a thing that happened: the fold imitates the puts it can see, and a critic asked
    to score an answer is handed raw markup instead. Matched with
    :data:`~focus.generation.parsing.base.TOOL_CALL`, whose outer tag both
    dialects
    share (Hermes' JSON body and Qwen3.5's ``<function=...>`` XML alike), so one pass
    covers both.
    """

    def flatten(match: re.Match[str]) -> str:
        # Truncated mid-word, not on a word boundary (textwrap.shorten): compact JSON
        # arguments can be one long "word", and dropping all of it hides the call.
        body = " ".join(match[1].split())
        clipped = body if len(body) <= _CALL_CHARS else body[:_CALL_CHARS] + "..."
        return f"[tool call: {clipped}]"

    return TOOL_CALL.sub(flatten, content)


def format_transcript(messages: Sequence[Message], *, sep: str = "\n") -> str:
    """``role: content`` join, tool calls flattened -- for folds and reward views."""
    return sep.join(f"{m['role']}: {_message_text(m)}" for m in messages)


def _message_text(message: Message) -> str:
    content = flatten_tool_calls(message["content"] or "")
    for call in message.get("tool_calls", ()):
        function = call["function"]
        arguments = function["arguments"]
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        body = " ".join(f"{function['name']} {arguments}".split())
        clipped = body if len(body) <= _CALL_CHARS else body[:_CALL_CHARS] + "..."
        content += f"\n[tool call: {clipped}]"
    return content


__all__ = [
    "flatten_tool_calls",
    "format_transcript",
]
