"""The action parser: turn a model generation into a decoded action (text -> action).

Because the :class:`Env` is pure world dynamics that consumes an *already-decoded*
action, the text->action step lives outside the env, here. An :class:`ActionParser` is
the task's decoder; the action type ``A`` it produces is the same ``A`` the env's
``step`` consumes -- the one shared contract between the two halves.

It is handed the whole :class:`GenerateResult` (not just the text) so it can consult
``matched_stop`` / ``stop_reason`` -- e.g. treat an illegal stop-string hit as a
malformed action -- not only ``text``.

:func:`tool_turn` is the part every *tool-calling* dialect shares: split the reasoning
off, find the ``<tool_call>`` blocks, strip them out of the visible text. Only what is
inside a block differs between dialects (Hermes' JSON body, Qwen3.5's ``<function=...>``
XML), so a dialect module is its per-block decoder and nothing else -- see
:mod:`focus.generation.parsing.hermes` and
:mod:`focus.generation.parsing.qwen_xml`.
:data:`TOOL_CALL`, the outer tag they share, is therefore defined here rather than
beside a caller.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from focus.generation.types import GenerateResult


@dataclass(frozen=True)
class ToolCall:
    """One parsed ``<tool_call>``: the function name and its JSON argument string."""

    name: str
    arguments: str
    call_id: str = ""


@dataclass(frozen=True)
class ToolTurn:
    """A decoded assistant turn: the reasoning, the visible text, and any tool calls.

    The hand-off between a dialect and the world: every parser
    (:class:`~focus.generation.parsing.hermes.HermesToolCallParser`,
    :class:`~focus.generation.parsing.qwen_xml.QwenXMLToolCallParser`)
    produces one of
    these, and :class:`~focus.harness.tools.environment.ToolEnv` consumes it.
    It lives
    beside :class:`ToolCall` because it is the same contract one level up -- the turn is
    what carries the calls -- and because a leaf module lets the dialects and the env
    share the type without either importing the other.

    The three fields are disjoint slices of one completion, which is why ``text`` is so
    often empty: ``reasoning`` is everything before ``</think>``, the calls are the
    ``<tool_call>`` blocks, and ``text`` is what is left once both are removed. A turn
    that did nothing but call a tool therefore has **no** visible text -- an env reading
    an answer wants the ``submit`` call's arguments, and ``text`` only as the fallback
    for an ending that made no call at all.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    reasoning: str = ""


# A <tool_call> block; capture its whole inner content (``.*?`` non-greedy so adjacent
# blocks don't merge). Public because it is the one shared surface of the two dialects:
# a parser decodes what is inside the tags (Hermes' JSON body, Qwen3.5's
# ``<function=...>`` XML), while anything that only needs to *find* a call in text --
# :mod:`focus.harness.transcript`, flattening one for a fold -- matches with
# this rather than keeping a second copy of the pattern.
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class ActionParser[A](ABC):
    """Decode a model :class:`GenerateResult` into a task action."""

    #: Whether the actions this parser reads are *in band* -- written by the model into
    #: the very tokens the result carries. True for every dialect parser here, and the
    #: default, because a dialect is text; ``False`` for
    #: :class:`~focus.generation.parsing.native.NativeToolCallParser`, which
    #: reads a
    #: channel beside the tokens. Why that forfeits training is argued once, at
    #: :attr:`~focus.generation.policy.Policy.trainable`, which pairs this
    #: with
    #: the generator's own claim about its ids.
    trainable: bool = True

    #: Whether this parser reads the backend's *structured* tool-call channel
    #: (:attr:`~focus.generation.types.GenerateResult.native_tool_calls`)
    #: rather than the completion text. ``False`` for every dialect parser. Checked
    #: against the generator's ``provides_native_tool_calls``
    #: (:class:`~focus.generation.Policy`) when a
    #: :class:`~focus.generation.policy.Policy` is built.
    requires_native_channel: bool = False

    @abstractmethod
    def parse(self, result: GenerateResult) -> A:
        """Return the action ``result`` expresses (or a task-defined malformed one)."""


class TextActionParser(ActionParser[str]):
    """The trivial parser: the action is the completion text, stripped."""

    def parse(self, result: GenerateResult) -> str:
        return result.text.strip()


def tool_turn(text: str, decode: Callable[[str], ToolCall]) -> ToolTurn:
    """A completion read as a tool-calling assistant turn, ``decode`` per call block.

    Reasoning is everything up to the first ``</think>``: thinking models emit no opener
    (the generation prompt pre-fills it), and a completion cut off mid-reasoning carries
    no closer at all, so a turn without one is entirely visible rather than entirely
    thought. ``<tool_call>`` blocks inside the reasoning region are ignored
    (qwen-agent's rule) because only ``body`` is searched, and they are stripped out of
    the visible text so what is left is what the model *said*.
    """
    reasoning, closer, body = text.partition("</think>")
    if not closer:
        reasoning, body = "", text
    return ToolTurn(
        text=TOOL_CALL.sub("", body).strip(),
        tool_calls=tuple(decode(block) for block in TOOL_CALL.findall(body)),
        reasoning=reasoning.strip(),
    )


__all__ = [
    "TOOL_CALL",
    "ActionParser",
    "TextActionParser",
    "ToolCall",
    "ToolTurn",
    "tool_turn",
]
