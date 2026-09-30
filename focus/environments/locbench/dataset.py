"""Pinned Loc-Bench V1 download and private scoring labels."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import Gold, location, relative_path

DATASET = "czlll/Loc-Bench_V1"
REVISION = "c44cf3b74e07ca642cec841b471a9939907c12a7"
SHA256 = "8df0833c2c1276c5837aab923d489ab97d7654529abe759d0f59242c4978a662"
CONFIG = "default"
SPLIT = "test"
URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
    "/data/test-00000-of-00001.parquet"
)


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
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    else:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    cases = [Case.from_row(row) for row in rows]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("duplicate instance_id in dataset")
    return cases


def download(directory: Path) -> Path:
    """Fetch one pinned parquet atomically and record its content hash."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{SPLIT}-{REVISION}.parquet"
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temp = Path(output.name)
            try:
                with urllib.request.urlopen(URL, timeout=120) as source:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                output.flush()
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        try:
            # Check schema/labels before publishing the cached download.
            import pyarrow.parquet as pq

            if hashlib.sha256(temp.read_bytes()).hexdigest() != SHA256:
                raise ValueError("download does not match the pinned parquet SHA-256")
            rows = pq.read_table(temp).to_pylist()
            if len(rows) != 560:
                raise ValueError(f"expected 560 pinned cases, received {len(rows)}")
            for row in rows:
                Case.from_row(row)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    if hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
        raise ValueError("cached parquet does not match the pinned SHA-256")
    cases = load_cases(target)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "config": CONFIG,
        "split": SPLIT,
        "url": URL,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "cases": len(cases),
        "repositories": len({case.repo for case in cases}),
    }
    target.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return target
