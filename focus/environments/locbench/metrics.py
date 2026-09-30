"""Rank one submission at file and edited-function granularity.

Acc@k requires *all* gold targets, including when there are more than k. Added
functions and bare paths do not consume function ranks. Constructors retain their
full qualified names; no class/function alias or fuzzy matching is applied.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

MAX_LOCATIONS = 10
FILE_K = (1, 3, 5)
FUNCTION_K = (5, 10)


def relative_path(value: str, *, allow_root: bool = False) -> str:
    """Canonical POSIX repository path, with no traversal or metadata access."""
    if not isinstance(value, str) or any(ord(c) < 32 for c in value):
        raise ValueError("path must be a string without control characters")
    if value.startswith("/") or "\\" in value or ":" in value:
        raise ValueError("use a relative POSIX repository path")
    parts = value.split("/")
    if any(p in {"..", ".git"} for p in parts):
        raise ValueError("parent traversal and .git access are unavailable")
    path = "/".join(p for p in parts if p not in {"", "."})
    if not path and not allow_root:
        raise ValueError("path must name a file")
    return path


def location(value: str) -> str:
    """Normalize an agent's bare path or path::Qualified.name entry."""
    if not isinstance(value, str):
        raise ValueError("each location must be a string")
    path, separator, name = value.partition("::")
    path = relative_path(path)
    if not separator:
        return path
    if not name or any(not p.isidentifier() for p in name.split(".")):
        raise ValueError("function locations use path::Qualified.name")
    return f"{path}::{name}"


def submission(values: object, *, maximum: int = MAX_LOCATIONS) -> tuple[str, ...]:
    """Validate syntax only; nonexistent targets are potentially wrong guesses."""
    if not isinstance(values, list) or len(values) > maximum:
        raise ValueError(
            f"locations must be an ordered array of at most {maximum} entries"
        )
    return tuple(location(value) for value in values)


@dataclass(frozen=True)
class Gold:
    files: frozenset[str]
    edit_functions: frozenset[str]
    added_functions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("a case must have at least one gold file")
        if self.edit_functions & self.added_functions:
            raise ValueError("edited and added function labels overlap")


def score(locations: Sequence[str], gold: Gold) -> dict[str, float | None]:
    """Macro-averagable per-case metrics. Empty submissions receive zero reward."""
    entries = tuple(location(entry) for entry in locations)
    paths = tuple(dict.fromkeys(entry.partition("::")[0] for entry in entries))
    functions = tuple(
        dict.fromkeys(
            entry
            for entry in entries
            if "::" in entry and entry not in gold.added_functions
        )
    )
    result: dict[str, float | None] = {}
    for k in FILE_K:
        found = gold.files.intersection(paths[:k])
        result[f"file_acc@{k}"] = float(found == gold.files)
        result[f"file_recall@{k}"] = len(found) / len(gold.files)
    for k in FUNCTION_K:
        found = gold.edit_functions.intersection(functions[:k])
        result[f"function_acc@{k}"] = (
            float(found == gold.edit_functions) if gold.edit_functions else None
        )
        result[f"function_recall@{k}"] = (
            len(found) / len(gold.edit_functions) if gold.edit_functions else None
        )
    result["reward"] = result["file_recall@5"]
    return result


def aggregate(records: Iterable[dict[str, float | None]]) -> dict[str, Any]:
    """Mean over cases, with explicit per-metric eligibility denominators."""
    rows = list(records)
    keys = sorted({key for row in rows for key in row})
    counts = {key: sum(row.get(key) is not None for row in rows) for key in keys}
    means = {
        key: sum(value for row in rows if (value := row.get(key)) is not None)
        / counts[key]
        if counts[key]
        else None
        for key in keys
    }
    return {"cases": len(rows), "means": means, "eligible_cases": counts}
