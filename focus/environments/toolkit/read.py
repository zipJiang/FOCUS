"""``read`` over a :class:`~...corpus.Corpus`: page a file, or one section of it."""

from __future__ import annotations

from typing import Any

from focus.harness.tools import BaseTool

from ._args import _as_int
from .corpus import Corpus

#: Task-agnostic default, with ``{section}`` standing in for whatever this env calls a
#: section (``provision``, ``heading``, ``chapter``): the description is prompt text, so
#: it has to use the same word as the argument the model is being asked to fill in.
DEFAULT_DESCRIPTION = (
    "Read a file by path, with 1-based line numbers you can cite. Pass `{section}` to "
    + "read just one section of it (the file list names them); otherwise you get the "
    + "whole file from `offset`. Read the whole section, not just the line grep "
    + "matched -- the sentence that changes the answer is usually a line or two "
    + "below it."
)

DEFAULT_SECTION_HELP = "one section of that file, e.g. '(c)(1)'"


class ReadTool(BaseTool):
    """``read``: a file, or one labelled section of it, with citable line numbers.

    Corpus only -- what the agent wrote down is reached with the workspace's own tools,
    not by remounting workspace keys as files.

    The section argument is *named* by the env (``section_arg``) rather than fixed here,
    because the schema is prompt text: a model reading a statute is asked for a
    ``provision``, and a schema that insisted on ``section`` would be teaching it a word
    the rest of its prompt never uses.
    """

    name = "read"

    def __init__(
        self,
        corpus: Corpus,
        *,
        description: str | None = None,
        section_arg: str = "section",
        section_help: str = DEFAULT_SECTION_HELP,
        max_line_chars: int = 500,
        max_chars: int = 0,
    ):
        #: Ceiling on one call's whole response; 0 disables. ``max_line_chars`` caps a
        #: *line*, which does not bound the response: 400 lines of statute is ~7k
        #: tokens, and four such reads in one turn put 28,448 tokens into a single
        #: region -- over any sane row cap, so the longest and most tool-heavy
        #: trajectories were the ones silently dropped from training. ``grep`` and
        #: ``python`` already cap at 4000 and 3000; this is the same guard for ``read``.
        self._max_chars = max_chars
        self.description = description or DEFAULT_DESCRIPTION.format(
            section=section_arg
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "a path from the file list you were given",
                },
                section_arg: {"type": "string", "description": section_help},
                "offset": {
                    "type": "integer",
                    "description": "first line to show, 1-based (default 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": "how many lines, max 400 (default 200)",
                },
            },
            "required": ["path"],
        }
        self._corpus = corpus
        self._section_arg = section_arg
        self._max_line_chars = max_line_chars

    def call(
        self,
        params: str | dict[str, Any],
        *,
        workspace: object = None,
        **kwargs: object,
    ) -> str:
        del workspace  # the workspace owns durable notes; read does not remount them
        args = self.verify_args(params)
        wanted = str(args["path"])
        # a model that writes the section into the path is being helpful, not wrong
        asked, _, inline = wanted.partition("#")
        section = str(args.get(self._section_arg) or inline or "").strip()
        source = self._corpus.resolve(asked)
        if source is None:
            listing = "\n".join(f"  {p}" for p in self._corpus.paths())
            return f"No single file matches {asked!r}. Available files:\n{listing}"

        lines = source.lines()
        start, end = 1, len(lines)
        if section:
            found = source.find(section)
            if found is None:
                labels = ", ".join(s.label for s in source.sections) or "(none)"
                return (
                    f"No {self._section_arg} {section!r} in {source.path}. "
                    + f"It has: {labels}"
                )
            start, end = found.start, found.end
        else:
            start = min(_as_int(args, "offset", 1, lo=1, hi=10**6), max(1, len(lines)))
            end = min(start - 1 + _as_int(args, "limit", 200, lo=1, hi=400), len(lines))

        shown = [
            f"{i + 1}: {lines[i][: self._max_line_chars]}"
            for i in range(start - 1, end)
        ]
        tail = (
            f"\n[{len(lines) - end} more lines; read again with offset={end + 1}]"
            if end < len(lines)
            else "\n[end of file]"
        )
        where = f" {section}" if section else ""
        header = f"[{source.path}{where}] lines {start}-{end} of {len(lines)}"
        text = header + "\n" + "\n".join(shown) + tail
        if self._max_chars and len(text) > self._max_chars:
            # Truncated rather than paged down: the tool already tells the model how to
            # continue (it prints the next offset and advertises the section argument),
            # so a clipped read is a paging prompt in the tool's own idiom.
            text = text[: self._max_chars] + (
                "\n[truncated here -- read again from a later offset, or name a "
                + "section to fetch just the part you need]"
            )
        return text


__all__ = ["DEFAULT_DESCRIPTION", "DEFAULT_SECTION_HELP", "ReadTool"]
