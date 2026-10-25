"""LocBench case records and private scoring labels."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import Gold, location, relative_path


def validate_source(repo: str, commit: str) -> None:
    if not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", repo
    ):
        raise ValueError(f"invalid GitHub repository: {repo!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("base_commit must be a full lowercase 40-character SHA")


def _patch_path(value: str) -> str | None:
    if value == "/dev/null":
        return None
    if value.startswith('"'):
        # Git quotes non-ASCII filename bytes using C-style octal escapes.
        value = ast.literal_eval("b" + value).decode("utf-8")
    if not value.startswith(("a/", "b/")):
        raise ValueError(f"unexpected patch path: {value!r}")
    return relative_path(value[2:])


def patch_files(patch: str) -> frozenset[str]:
    """Use base-side paths for edits/renames/deletions and new paths for additions.

    Only structural headers are read; test_patch is never passed here. Mode-only
    and binary changes have no ---/+++ pair, so retain the diff header fallback.
    """
    files: set[str] = set()
    for block in re.split(r"(?m)^diff --git ", patch)[1:]:
        header, _, body = block.partition("\n")
        match = re.fullmatch(r'("a/.*?"|a/.*?) ("b/.*"|b/.*)', header)
        if match is None:
            raise ValueError(f"unrecognized diff header: {header!r}")
        old = _patch_path(match[1])
        new = _patch_path(match[2])
        for line in body.splitlines():
            if line.startswith(("@@", "GIT binary patch")):
                break
            if line.startswith("--- "):
                old = _patch_path(line[4:])
            elif line.startswith("+++ "):
                new = _patch_path(line[4:])
        path = old or new
        if path is None:
            raise ValueError("patch file has neither a base nor destination path")
        files.add(path)
    if not files:
        raise ValueError("patch contains no file headers")
    return frozenset(files)


def _label(value: str) -> str:
    path, separator, name = value.partition(":")
    if not separator:
        raise ValueError(f"function label lacks path separator: {value!r}")
    return location(f"{path}::{name}")


@dataclass(frozen=True)
class Case:
    id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold: Gold

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Case:
        validate_source(row["repo"], row["base_commit"])
        return cls(
            id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            gold=Gold(
                patch_files(row["patch"]),
                frozenset(_label(v) for v in row["edit_functions"]),
                frozenset(_label(v) for v in row["added_functions"]),
            ),
        )

    def task_prompt(self) -> str:
        """The sole model-facing dataset projection; no hints, patch, or labels."""
        return (
            f"Repository: {self.repo}\nBase commit: {self.base_commit}\n\n"
            f"Issue to localize:\n{self.problem_statement}"
        )


def load_cases(path: Path) -> list[Case]:
    """Read user-supplied JSONL records and validate their case identities."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cases = [Case.from_row(row) for row in rows]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("duplicate instance_id in dataset")
    return cases
