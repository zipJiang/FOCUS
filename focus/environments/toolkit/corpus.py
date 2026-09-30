"""A read-only corpus: the files an agent may grep and read, addressed by section.

The unit is a :class:`SourceFile` -- one *whole* file, plus an index of the sections it
was split into. Keeping the file whole is what makes a citation work: line numbers are
absolute within the file, so the ``path:line`` a grep hit reports and the ``path:line``
a read shows are the same numbers. The sections are an index *over* that file, which is
what lets ``read`` fetch one subsection instead of four hundred lines, and lets ``grep``
say which section a hit landed in.

Finding the section heads is the task's business, not the corpus's: a statute opens its
provisions with ``(c)(1)``, a manual with markdown headings, a codebase with ``def``.
So a corpus takes sections already computed -- or a ``sectioner`` callable -- and never
guesses at structure itself.

A :class:`Corpus` is deliberately *not* a
:class:`~focus.harness.workspace.base.Workspace`. The corpus is task evidence,
identical across branches and read-only; the workspace is per-rollout scratch the model
reaches through ``workspace.tools()``. Putting evidence in the workspace would fork-copy
it per branch and confuse "what I was given" with "what I wrote down".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

#: Finds the sections of one file's text. Task-specific by nature (see the module
#: docstring), so it is a parameter rather than anything this module implements.
Sectioner = Callable[[str], Sequence["Section"]]


@dataclass(frozen=True)
class Section:
    """One addressable chunk of a file: its label, and where it sits in the file.

    ``start`` and ``end`` are **1-based and inclusive** -- the same numbers ``grep``
    prints and ``read`` shows, so a section's bounds and a citation are one convention.
    A sectioner is task-written (see the module docstring), so an off-by-one here is the
    likeliest mistake in the whole toolkit, and it used to be silent: ``read`` sliced
    from ``start - 1``, so a 0-based ``start`` wrapped to the *end* of the file and
    handed the model confidently mis-cited lines. It is checked instead, once, where the
    section is built.
    """

    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ValueError(
                f"section {self.label!r} spans lines {self.start}-{self.end}: bounds "
                + "are 1-based and inclusive, so start must be >= 1 and end >= start"
            )


@dataclass(frozen=True)
class SourceFile:
    """One mounted file, plus the sections it was split into."""

    path: str
    text: str
    sections: tuple[Section, ...] = ()
    #: The text split into lines, computed once in ``__post_init__``. A declared
    #: ``init=False`` field rather than a bare ``object.__setattr__`` attribute, so it
    #: is typed; kept out of ``repr``/``eq`` since it is derived from ``text``.
    _lines: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Split once, here, rather than per call: grep re-reads every file on every
        # search, and the text is immutable, so re-splitting a 109-line section on each
        # of ~15 greps a rollout is pure waste.
        object.__setattr__(self, "_lines", tuple(self.text.splitlines()))
        # The upper half of :class:`Section`'s own bounds check, which only the file can
        # make. `read` slices `lines[start - 1:end]` by index, so a section claiming
        # lines the file does not have took the rollout down with an ``IndexError`` from
        # inside the tool -- a sectioner's off-by-one, surfacing mid-rollout as a dead
        # expansion with a traceback naming neither the file nor the section.
        for section in self.sections:
            if section.end > len(self._lines):
                raise ValueError(
                    f"section {section.label!r} of {self.path} ends at line "
                    + f"{section.end}, but the file has {len(self._lines)} lines; "
                    + "bounds are 1-based and inclusive"
                )

    @classmethod
    def sectioned(cls, path: str, text: str, sectioner: Sectioner) -> SourceFile:
        """Build a file, running ``sectioner`` over its text for the section index."""
        return cls(path=path, text=text, sections=tuple(sectioner(text)))

    def lines(self) -> tuple[str, ...]:
        return self._lines

    def section_at(self, line: int) -> str:
        """The most specific section containing a 1-based line.

        Spans nest, so a line inside ``(c)(1)`` is also inside ``(c)``. The narrowest
        one is the citation worth printing: "§63(c)" points at sixty lines, while
        "§63(c)(1)" points at the rule.
        """
        holding = [s for s in self.sections if s.start <= line <= s.end]
        return min(holding, key=lambda s: s.end - s.start).label if holding else ""

    def find(self, label: str) -> Section | None:
        for section in self.sections:
            if section.label == label:
                return section
        return None


@dataclass(frozen=True)
class Corpus:
    """The read-only files one task can grep and read, in mount order."""

    files: tuple[SourceFile, ...] = ()

    def paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)

    def resolve(self, path: str) -> SourceFile | None:
        """A file from what the model typed: an exact path, else a unique suffix.

        The suffix has to start at a path-component boundary -- a whole tail of the
        path, never a tail of one name. Bare ``endswith`` made ``"ax.txt"`` resolve
        ``"a/tax.txt"`` uniquely and confidently, which is worse than not resolving:
        the model gets a file it did not ask for and cites it.

        No basename tier: basename equality implies a boundary-aligned suffix, so it
        could only ever fire where the suffix pass already found nothing.
        """
        path = path.strip().strip("/")
        exact = next((f for f in self.files if f.path == path), None)
        if exact is not None:
            return exact
        match = [f for f in self.files if f.path.endswith("/" + path)]
        return match[0] if len(match) == 1 else None

    def manifest(self, *, label: str = "sections") -> str:
        """The file list that goes in the prompt -- the discovery affordance.

        ``label`` is the word this task calls its sections by ("provisions" for a
        statute, "headings" for a manual): the manifest is prompt text a model reads, so
        it should use the task's own vocabulary rather than this module's.
        """
        out = []
        for f in self.files:
            n = len(f.lines())
            out.append(f"  {f.path}  ({n} lines)  {' '.join(f.text.split()[:10])}...")
            if f.sections:
                labels = ", ".join(s.label for s in f.sections[:12])
                more = "..." if len(f.sections) > 12 else ""
                out.append(f"      {label}: {labels}{more}")
        return "\n".join(out)


__all__ = ["Corpus", "Section", "Sectioner", "SourceFile"]
