"""Offline tests for the task-agnostic toolkit: corpus, grep, read, the python sandbox.

No model, no network, and deliberately no task: these pin the behaviour every env
inherits by composing :mod:`focus.environments.toolkit`, so they use a toy two-file
corpus rather than any one benchmark's data.
"""

from __future__ import annotations

import time

import pytest

from focus.environments.toolkit import (
    Corpus,
    GrepTool,
    PythonTool,
    ReadTool,
    Section,
    SourceFile,
    screen_code,
)
from focus.generation.parsing import ToolCall
from focus.harness import (
    InMemoryWorkspace,
    NullWorkspace,
    ToolError,
    dispatch,
)

MANUAL = """\
# Baggage
Every bag is weighed at check-in.

## Overweight
A bag over 50 lb is charged $100.

## Oversize
A bag over 62 inches is charged $200.
"""


def _headings(text: str) -> tuple[Section, ...]:
    """A toy sectioner: a markdown heading opens a section, nesting by heading level."""
    heads: list[tuple[int, str, int]] = []
    stack: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("#"):
            continue
        depth = len(line) - len(line.lstrip("#"))
        stack = [*stack[: depth - 1], line.lstrip("# ").strip()]
        heads.append((depth, "/".join(stack), i))
    last = len(text.splitlines())
    out = []
    for k, (depth, label, start) in enumerate(heads):
        end = next((n - 1 for d, _, n in heads[k + 1 :] if d <= depth), last)
        out.append(Section(label, start, end))
    return tuple(out)


def _corpus() -> Corpus:
    return Corpus(
        (
            SourceFile("case/facts.txt", "Alice checked two bags in 2017."),
            SourceFile.sectioned("policy/baggage.md", MANUAL, _headings),
        )
    )


# -- the corpus and its section index ------------------------------------------------


def test_section_bounds_are_checked_where_the_sectioner_wrote_them() -> None:
    # 1-based inclusive, and a 0-based sectioner used to be silent: `read` sliced from
    # `start - 1`, so start=0 wrapped to the last line of the file and the model was
    # handed lines it then cited. The env writes the sectioner, so this is the likeliest
    # mistake in the toolkit -- it fails at construction now.
    with pytest.raises(ValueError, match="1-based"):
        Section("Overweight", 0, 3)
    with pytest.raises(ValueError, match="1-based"):
        Section("Overweight", 5, 4)


def test_a_section_running_past_its_file_is_refused_where_the_file_is_built() -> None:
    # The other half of the same mistake, and the half a `Section` cannot see on its
    # own: `read` slices `lines[start - 1:end]` by index, so a section claiming lines
    # the file does not have took the rollout down with an `IndexError` from inside the
    # tool. The file is where both facts are known, so it is where this is said.
    with pytest.raises(ValueError, match="ends at line 3"):
        SourceFile("a.txt", "one\ntwo\n", sections=(Section("s", 1, 3),))
    SourceFile("a.txt", "one\ntwo\n", sections=(Section("s", 1, 2),))
    assert Section("Overweight", 4, 4).end == 4  # a one-line section is fine


def test_a_section_spans_its_own_subsections() -> None:
    # reading the outer section has to give the inner ones with it: the sentence that
    # changes the answer is usually in a subsection, so a span that stopped short would
    # hand back the half that lies
    sections = {s.label: s for s in _headings(MANUAL)}
    assert set(sections) == {"Baggage", "Baggage/Overweight", "Baggage/Oversize"}
    outer, inner = sections["Baggage"], sections["Baggage/Oversize"]
    assert outer.start < inner.start
    assert outer.end >= inner.end


def test_section_at_names_the_narrowest_section_holding_a_line() -> None:
    # spans nest; the narrowest is the citation worth printing
    source = _corpus().resolve("policy/baggage.md")
    assert source is not None
    line = source.lines().index("A bag over 62 inches is charged $200.") + 1
    assert source.section_at(line) == "Baggage/Oversize"
    assert source.section_at(2) == "Baggage"


def test_resolve_takes_an_exact_path_or_a_unique_suffix() -> None:
    corpus = _corpus()

    def path_of(query: str) -> str:
        resolved = corpus.resolve(query)
        assert resolved is not None
        return resolved.path

    assert path_of("policy/baggage.md") == "policy/baggage.md"
    assert path_of("baggage.md") == "policy/baggage.md"
    assert path_of("/policy/baggage.md ") == "policy/baggage.md"
    assert corpus.resolve("nope.md") is None


def test_a_suffix_only_matches_at_a_path_boundary() -> None:
    """A tail of a *name* is not a tail of a path: matching one resolves confidently to
    a file the model never asked for, which is worse than resolving nothing."""
    corpus = Corpus((SourceFile("a/tax.txt", "one"), SourceFile("case/facts.txt", "b")))

    assert corpus.resolve("ax.txt") is None
    assert corpus.resolve("tax.txt").path == "a/tax.txt"  # type: ignore[union-attr]
    assert corpus.resolve("facts.txt").path == "case/facts.txt"  # type: ignore[union-attr]
    assert corpus.resolve("case/facts.txt").path == "case/facts.txt"  # type: ignore[union-attr]


def test_an_ambiguous_suffix_resolves_to_nothing() -> None:
    # two files could be meant, so the tool has to say so rather than pick one
    corpus = Corpus((SourceFile("a/x.txt", "one"), SourceFile("b/x.txt", "two")))
    assert corpus.resolve("x.txt") is None


def test_the_manifest_names_every_file_once_in_the_task_s_own_word() -> None:
    corpus = _corpus()
    manifest = corpus.manifest(label="provisions")
    for path in corpus.paths():
        assert manifest.count(path) == 1
    assert "provisions: Baggage," in manifest
    assert "sections:" in corpus.manifest()  # the generic default


# -- grep ----------------------------------------------------------------------------


def test_grep_reports_path_line_and_the_section_it_landed_in() -> None:
    out = GrepTool(_corpus()).call({"pattern": "62 inches", "context": "0"})
    assert "policy/baggage.md:8:" in out
    assert "Baggage/Oversize" in out


def test_grep_coerces_string_arguments() -> None:
    # Qwen3.5's dialect has no types: every value arrives as text
    tool = GrepTool(_corpus())
    assert "baggage.md" in tool.call({"pattern": "bag", "context": "1"})
    with pytest.raises(ToolError):
        tool.call({"pattern": "bag", "context": "lots"})


async def test_a_bad_regex_comes_back_as_text_not_an_exception() -> None:
    out = await dispatch(
        {"grep": GrepTool(_corpus())}, ToolCall("grep", '{"pattern": "([unclosed"}')
    )
    assert out.startswith("error:")
    assert "regular expression" in out


def test_grep_with_no_hits_names_what_it_searched() -> None:
    out = GrepTool(_corpus()).call({"pattern": "zzz nothing"})
    assert "No match" in out
    assert "baggage.md" in out


def test_grep_rejects_pathological_regex_without_blocking_the_runner() -> None:
    corpus = Corpus((SourceFile("long.txt", "a" * 4000 + "!"),))
    tool = GrepTool(corpus, search_timeout=0.01)
    started = time.monotonic()
    with pytest.raises(ToolError, match="search exceeded"):
        tool.call({"pattern": "(a+)+$"})
    assert time.monotonic() - started < 0.5


def test_grep_caps_pattern_length_before_compiling() -> None:
    tool = GrepTool(_corpus(), max_pattern_chars=8)
    with pytest.raises(ToolError, match="too long"):
        tool.call({"pattern": "a" * 9})


def test_grep_restricts_to_a_path_prefix_and_says_when_none_matches() -> None:
    tool = GrepTool(_corpus())
    out = tool.call({"pattern": "bag", "path": "policy/"})
    assert "case/facts.txt" not in out
    assert "policy/baggage.md" in out
    missing = tool.call({"pattern": "bag", "path": "elsewhere/"})
    assert "No file matches" in missing
    assert "case/facts.txt" in missing


def test_grep_caps_hits_and_output_and_says_it_did() -> None:
    corpus = Corpus((SourceFile("many.txt", "bag\n" * 50),))
    capped = GrepTool(corpus, max_hits=3).call({"pattern": "bag", "context": "0"})
    assert "47 more hits" in capped
    truncated = GrepTool(corpus, max_chars=40).call({"pattern": "bag", "context": "0"})
    assert "[output truncated]" in truncated


def test_grep_stays_on_the_corpus_even_when_the_workspace_has_notes() -> None:
    # Durable notes live behind workspace.tools() (storage); grep/read never see them.
    ws = InMemoryWorkspace()
    ws.write("notes", "Alice checked a surfboard")
    tool = GrepTool(_corpus())
    assert "No match" in tool.call({"pattern": "surfboard"}, workspace=ws)
    assert "No match" in tool.call({"pattern": "surfboard"}, workspace=NullWorkspace())
    stored = ws.tools()[0].call({"operate": "get", "key": "notes"}, workspace=ws)
    assert isinstance(stored, str)  # the storage tool answers synchronously
    assert "surfboard" in stored


# -- read ----------------------------------------------------------------------------


def test_read_returns_the_whole_file_by_default() -> None:
    out = ReadTool(_corpus()).call({"path": "baggage.md"})
    assert "lines 1-8 of 8" in out
    assert "1: # Baggage" in out
    assert out.endswith("[end of file]")


def test_read_returns_one_section_with_its_subsections() -> None:
    out = ReadTool(_corpus()).call({"path": "baggage.md", "section": "Baggage"})
    assert "Every bag is weighed" in out
    assert "62 inches" in out
    scoped = ReadTool(_corpus()).call(
        {"path": "baggage.md", "section": "Baggage/Overweight"}
    )
    assert "50 lb" in scoped
    assert "62 inches" not in scoped


def test_read_accepts_the_section_written_into_the_path() -> None:
    inline = ReadTool(_corpus()).call({"path": "policy/baggage.md#Baggage/Overweight"})
    explicit = ReadTool(_corpus()).call(
        {"path": "baggage.md", "section": "Baggage/Overweight"}
    )
    assert inline == explicit


def test_read_pages_from_an_offset_and_says_how_to_continue() -> None:
    out = ReadTool(_corpus()).call({"path": "baggage.md", "offset": "4", "limit": "2"})
    assert "lines 4-5 of 8" in out
    assert "4: ## Overweight" in out
    assert "6:" not in out
    assert "read again with offset=6" in out


def test_read_lists_what_exists_when_the_path_or_section_is_wrong() -> None:
    tool = ReadTool(_corpus())
    assert "Available files" in tool.call({"path": "nope"})
    listed = tool.call({"path": "baggage.md", "section": "(zz)"})
    assert "Baggage/Oversize" in listed


def test_the_section_argument_is_named_by_the_env() -> None:
    # the schema is prompt text: an env that says "provision" everywhere else must not
    # be made to ask for a "section" here
    tool = ReadTool(_corpus(), section_arg="provision", section_help="e.g. '(c)(1)'")
    assert "provision" in tool.parameters["properties"]
    assert "section" not in tool.parameters["properties"]
    assert "`provision`" in tool.description
    out = tool.call({"path": "baggage.md", "provision": "Baggage/Oversize"})
    assert "62 inches" in out
    assert "No provision" in tool.call({"path": "baggage.md", "provision": "(zz)"})


# -- the python sandbox --------------------------------------------------------------


def test_the_screen_blocks_what_would_leave_the_process() -> None:
    # in an eval harness the answer keys sit on the same filesystem, so this screen is
    # usually the metric rather than the security posture
    for code in ("import os", "open('x')", "__import__('os')", "(1).__class__"):
        with pytest.raises(ToolError):
            screen_code(code)


def test_the_screen_allows_what_arithmetic_actually_needs() -> None:
    screen_code("import math\nsum(b * 40 for b in [1, 2, 3])")
    screen_code("def fee(w):\n    return 40 if w < 50 else 140\nfee(60)")
    screen_code("from datetime import date\ndate(2017, 1, 1)")


def test_the_allowed_imports_are_the_env_s_to_choose() -> None:
    with pytest.raises(ToolError):
        screen_code("import statistics", allowed_imports={"math"})
    screen_code("import statistics", allowed_imports={"math", "statistics"})


def test_syntax_errors_come_back_as_tool_errors() -> None:
    with pytest.raises(ToolError):
        screen_code("def (:")


async def test_a_trailing_expression_is_echoed() -> None:
    # models forget print() constantly; a tool that silently returns nothing on a bare
    # expression teaches them the tool is broken
    assert "535" in await PythonTool().call({"code": "500 + 35"})


async def test_a_trailing_print_is_not_echoed_on_top_of_itself() -> None:
    # `print(x)` is both a bare trailing expression and the commonest last line there
    # is; echoing it appended print's own None to the result, which reads as the tool
    # malfunctioning -- the exact lesson the echo exists to prevent
    assert await PythonTool().call({"code": "print(500 + 35)"}) == "535"
    # ... and a print that is not last still leaves the real trailing expression echoed
    assert await PythonTool().call({"code": "print(1)\n2 + 3"}) == "1\n5"


async def test_a_trailing_call_that_returned_nothing_echoes_nothing() -> None:
    # the same defect one step out from `print`: what must not be echoed is not the
    # *name* print but a value of None, and agent code reaches one every other way --
    # a helper that prints and returns nothing, a list method that sorts in place.
    assert await PythonTool().call({"code": "def p(x):\n    print(x)\np(535)"}) == "535"
    assert "None" not in await PythonTool().call({"code": "xs = [3, 1, 2]\nxs.sort()"})


async def test_output_is_capped() -> None:
    out = await PythonTool(max_chars=50).call({"code": "print('x' * 5000)"})
    assert out.endswith("[output truncated]")
    assert len(out) < 100


async def test_a_runaway_is_killed_with_an_actionable_message() -> None:
    # a timeout well under the child's own 2s CPU rlimit, so what is measured here is
    # the parent's wall-clock kill -- the guarantee that holds on every platform -- and
    # the test costs a fraction of a second rather than waiting out the rlimit.
    out = await PythonTool(timeout=0.3).call({"code": "while True:\n    pass"})
    assert "killed" in out or "timed out" in out


async def test_the_default_description_names_the_imports_on_offer() -> None:
    # the description is the only place the model learns what it may import
    assert "decimal" in PythonTool().description
    assert PythonTool(allowed_imports={"math"}).description.endswith("math.")
