"""``list`` over a :class:`~...corpus.Corpus`: what is mounted, and what is in it."""

from __future__ import annotations

from typing import Any

from focus.harness.tools import BaseTool

from .corpus import Corpus

#: Task-agnostic default, with ``{section}`` standing in for whatever this env calls a
#: section -- the same substitution :mod:`.read` makes, and for the same reason.
DEFAULT_DESCRIPTION = (
    "List the files you can read, with their length and the {section}s each one is "
    + "divided into. Call it first: it is how you find out what evidence exists before "
    + "guessing at a path."
)


class ListTool(BaseTool):
    """``list``: the corpus manifest, on demand.

    The same text :meth:`Corpus.manifest` puts in the opening prompt, reachable as a
    tool. That is not redundant. A manifest pasted into the first user turn is the first
    thing a compactor drops when the context folds, and an agent that has forgotten what
    is mounted greps blind or answers from memory. A tool call re-reads it in the turn
    it is needed, at the cost of one call rather than a permanent tax on every prompt --
    which is also why an env that inlines its evidence has no use for this tool.
    """

    name = "list"

    def __init__(
        self,
        corpus: Corpus,
        *,
        description: str | None = None,
        section_arg: str = "section",
    ):
        self._corpus = corpus
        #: The word the manifest calls its sections by, kept in step with what ``read``
        #: names the argument -- one vocabulary across the prompt, the schema and the
        #: output.
        self._label = section_arg + "s"
        self.description = description or DEFAULT_DESCRIPTION.format(
            section=section_arg
        )
        #: No arguments: the corpus is small enough to print whole, and a filter would
        #: be a second way to say what ``grep`` already says better.
        self.parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def call(self, params: str | dict[str, Any], **kwargs: Any) -> str:
        self.verify_args(params)
        return self._corpus.manifest(label=self._label) or "no files are mounted"


__all__ = ["ListTool"]
