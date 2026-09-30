"""``grep`` over a :class:`~...corpus.Corpus`: find the passage by the words it uses."""

from __future__ import annotations

import time
from typing import Any

import regex as regex_module  # type: ignore[import-untyped]

from focus.harness.tools import BaseTool, ToolError

from ._args import _as_int
from .corpus import Corpus

#: A task-agnostic account of what this tool is for. An env with a subject of its own
#: passes its own text -- "the operative words a provision would use" reads very
#: differently from "the function that raises this error" -- because the description is
#: rendered into every prompt and is the only place the search *strategy* can be taught.
DEFAULT_DESCRIPTION = (
    "Search the files with a regular expression. Returns matching lines as "
    + "`path:line: text`, naming the section each hit fell in, which is how you cite "
    + "what you relied on. Search for the distinctive words the passage itself would "
    + "use, not a sentence copied from the question."
)


class GrepTool(BaseTool):
    """``grep``: regex search over the corpus, reporting ``path:line`` and the section.

    Corpus only. Whatever the agent wrote down lives behind the workspace's own tools
    (e.g. ``storage``), not under a synthetic mount here: a corpus tool searches task
    evidence, and mixing the two makes "what I read" indistinguishable from "what I
    concluded".
    """

    name = "grep"
    description = DEFAULT_DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "a regular expression, e.g. 'surviving spouse|head of "
                    + "household'"
                ),
            },
            "path": {
                "type": "string",
                "description": "restrict to one file or a prefix like 'statutes/'",
            },
            "context": {
                "type": "integer",
                "description": "lines of context around each hit, 0-10 (default 2)",
            },
        },
        "required": ["pattern"],
    }

    def __init__(
        self,
        corpus: Corpus,
        *,
        description: str = DEFAULT_DESCRIPTION,
        max_hits: int = 40,
        max_chars: int = 4000,
        search_timeout: float = 1.0,
        max_pattern_chars: int = 512,
    ):
        if search_timeout <= 0:
            raise ValueError("search_timeout must be positive")
        if max_pattern_chars <= 0:
            raise ValueError("max_pattern_chars must be positive")
        self.description = description
        self._corpus = corpus
        self._max_hits = max_hits
        self._max_chars = max_chars
        self._search_timeout = search_timeout
        self._max_pattern_chars = max_pattern_chars

    def call(
        self,
        params: str | dict[str, Any],
        *,
        workspace: object = None,
        **kwargs: object,
    ) -> str:
        del workspace  # the workspace owns durable notes; grep does not peek into it
        args = self.verify_args(params)
        pattern = str(args["pattern"])
        if len(pattern) > self._max_pattern_chars:
            raise ToolError(
                "regular expression is too long: "
                f"{len(pattern)} characters exceeds {self._max_pattern_chars}"
            )
        prefix = str(args.get("path") or "").strip().strip("/")
        context = _as_int(args, "context", 2, lo=0, hi=10)
        try:
            regex = regex_module.compile(pattern, regex_module.IGNORECASE)
        except regex_module.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from None

        everything = list(self._corpus.files)
        files = [f for f in everything if prefix in f.path]
        if not files:
            listing = ", ".join(f.path for f in everything)
            return f"No file matches path {prefix!r}. Available files: {listing}"

        out: list[str] = []
        hits = 0
        deadline = time.monotonic() + self._search_timeout
        for f in files:
            lines = f.lines()
            for i, line in enumerate(lines):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ToolError(
                        "regular expression search exceeded "
                        f"{self._search_timeout:g} seconds; use a simpler pattern"
                    )
                try:
                    matched = regex.search(line, timeout=remaining)
                except TimeoutError:
                    raise ToolError(
                        "regular expression search exceeded "
                        f"{self._search_timeout:g} seconds; use a simpler pattern"
                    ) from None
                if not matched:
                    continue
                hits += 1
                if hits > self._max_hits:
                    continue
                if out:
                    out.append("--")
                where = f.section_at(i + 1)
                if where:
                    out.append(f"[{f.path} -- {where}]")
                for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                    sep = ":" if j == i else "-"
                    out.append(f"{f.path}{sep}{j + 1}{sep}{lines[j]}")
        if not hits:
            listing = ", ".join(f.path for f in files)
            return (
                f"No match for {pattern!r} in {len(files)} file(s). Try different "
                + f"wording. Searched: {listing}"
            )
        text = "\n".join(out)
        if len(text) > self._max_chars:
            text = text[: self._max_chars] + "\n[output truncated]"
        if hits > self._max_hits:
            text += f"\n[{hits - self._max_hits} more hits -- narrow the pattern]"
        return text


__all__ = ["DEFAULT_DESCRIPTION", "GrepTool"]
