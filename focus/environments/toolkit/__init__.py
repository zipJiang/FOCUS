"""Toolkit: reusable, task-agnostic tools an env can hand its agent.

This module supplies the concrete corpus and Python tools used by the maintained
benchmark environments. :mod:`focus.harness.tools` says what a tool *is* (the
protocol, the schema,
``dispatch``). This package is the other half: concrete tools that recur across tasks,
so an env author writes the task and not a fourth ``grep``.

Two pieces, and they compose:

* a :class:`Corpus` of :class:`SourceFile` -- read-only evidence, addressed by path and
  by :class:`Section` -- with :class:`ListTool`, :class:`GrepTool` and :class:`ReadTool`
  over it, which are respectively how an agent discovers, finds and reads it;
* :class:`PythonTool`, a sandboxed subprocess for the arithmetic a model should not do
  in its head.

Everything task-specific is a constructor argument. Finding the section heads is the
env's job (a statute opens provisions with ``(c)(1)``, a manual with ``##``), and so is
the prose: each tool's ``description`` is rendered into every prompt, and it is the only
place the *strategy* for using it can be taught, so an env with a subject of its own
should pass its own.

Building an env's tool list, then, is composition rather than subclassing::

    corpus = Corpus(tuple(SourceFile.sectioned(p, t, my_heads) for p, t in files))
    tools = [
        *workspace.tools(),                       # durable notes, if any
        ListTool(corpus, section_arg="provision"),
        GrepTool(corpus, description=MY_GREP),
        ReadTool(corpus, section_arg="provision"),
        PythonTool(timeout=10.0),
    ]

Tools are constructed directly, because a tool needs the task's own corpus and prose.

The modules' ``DEFAULT_DESCRIPTION`` constants are deliberately not re-exported here --
they share a name; import them from :mod:`~.grep` / :mod:`~.list` / :mod:`~.read` when
extending one.
"""

from .corpus import (
    Corpus,
    Section,
    Sectioner,
    SourceFile,
)
from .grep import GrepTool
from .list import ListTool
from .python import (
    DEFAULT_IMPORTS,
    PythonTool,
    screen_code,
)
from .read import ReadTool

__all__ = [
    "DEFAULT_IMPORTS",
    "Corpus",
    "GrepTool",
    "ListTool",
    "PythonTool",
    "ReadTool",
    "Section",
    "Sectioner",
    "SourceFile",
    "screen_code",
]
