"""Immutable base-commit access without checkout, execution, or history tools."""

from __future__ import annotations

import fcntl
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import validate_source
from .metrics import relative_path

LIST_LIMIT = 200
LIST_DEPTH = 2
READ_LINES = 60
MAX_HITS = 50
MAX_LINE_CHARS = 2000
MAX_BLOB_BYTES = 32 * 1024 * 1024
FETCH_ATTEMPTS = 3
FETCH_RETRY_SECONDS = 1.0


def _git(directory: Path, *args: str, timeout: float = 30) -> bytes:
    result = subprocess.run(
        ["git", "--literal-pathspecs", "-C", str(directory), *args],
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[:1000].strip())
    return result.stdout


def prepare(cache: Path, repo: str, commit: str) -> Repository:
    """Fetch only a requested base commit, with a per-repository preparation lock."""
    validate_source(repo, commit)
    owner, name = repo.split("/")
    directory = cache / owner / f"{name}.git"
    directory.parent.mkdir(parents=True, exist_ok=True)
    with directory.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not directory.exists():
            directory.mkdir()
            _git(directory, "init", "--bare")
        try:
            _git(directory, "cat-file", "-e", f"{commit}^{{commit}}")
        except ValueError:
            for attempt in range(FETCH_ATTEMPTS):
                try:
                    _git(
                        directory,
                        "-c",
                        "fetch.fsckObjects=true",
                        "fetch",
                        "--depth=1",
                        "--no-tags",
                        "--no-recurse-submodules",
                        f"https://github.com/{repo}.git",
                        f"{commit}:refs/locbench/{commit}",
                        timeout=300,
                    )
                    break
                except (ValueError, subprocess.TimeoutExpired):
                    if attempt + 1 == FETCH_ATTEMPTS:
                        raise
                    time.sleep(FETCH_RETRY_SECONDS * 2**attempt)
        return Repository(directory, commit)


@dataclass(frozen=True)
class Entry:
    path: str
    mode: str
    oid: str

    @property
    def readable(self) -> bool:
        return self.mode in {"100644", "100755"}


def _clip(text: str) -> str:
    if len(text) <= MAX_LINE_CHARS:
        return text
    return text[:MAX_LINE_CHARS] + " … [line truncated]"


class Repository:
    """All evidence is addressed by the full base SHA, never by HEAD or a branch."""

    def __init__(self, directory: Path, commit: str) -> None:
        validate_source("local/repository", commit)
        self.directory = directory.resolve()
        self.commit = commit
        entries: dict[str, Entry] = {}
        for record in _git(directory, "ls-tree", "-r", "-z", commit).split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, _, oid = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
            # Do not expose paths outside the tool vocabulary (.git, control chars).
            try:
                normalized = relative_path(path)
            except ValueError:
                continue
            if normalized == path:
                entries[path] = Entry(path, mode, oid)
        self.entries = entries

    def _scope(self, path: str | None) -> str:
        normalized = relative_path(path or "", allow_root=True)
        if (
            normalized
            and normalized not in self.entries
            and not any(entry.startswith(normalized + "/") for entry in self.entries)
        ):
            raise ValueError(
                f"no file or directory at {normalized!r} in the base commit"
            )
        return normalized

    def list(self, path: str | None = None) -> dict[str, Any]:
        scope = self._scope(path)
        selected: dict[str, str] = {}
        if scope in self.entries:
            selected = {scope: self.entries[scope].mode}
        else:
            prefix = scope + "/" if scope else ""
            depth = LIST_DEPTH if path is not None else 1
            for name, entry in self.entries.items():
                if not name.startswith(prefix):
                    continue
                parts = name[len(prefix) :].split("/")
                for index in range(1, min(len(parts), depth) + 1):
                    relative = prefix + "/".join(parts[:index])
                    selected[relative] = (
                        entry.mode if index == len(parts) else "directory"
                    )
        types = {
            "100644": "file",
            "100755": "file",
            "120000": "symlink",
            "160000": "submodule",
        }
        paths = sorted(selected)
        return {
            "path": scope or ".",
            "entries": [
                {"path": p, "type": types.get(selected[p], selected[p])}
                for p in paths[:LIST_LIMIT]
            ],
            "total_entries_at_depth": len(paths),
            "truncated": len(paths) > LIST_LIMIT,
            "hint": "List a narrower subtree to see omitted entries."
            if len(paths) > LIST_LIMIT
            else "",
        }

    def text(self, path: str) -> str:
        path = relative_path(path)
        entry = self.entries.get(path)
        if entry is None:
            raise ValueError(f"no file at {path!r} in the base commit")
        if not entry.readable:
            raise ValueError("symlinks and submodules cannot be read or followed")
        size = int(_git(self.directory, "cat-file", "-s", entry.oid))
        if size > MAX_BLOB_BYTES:
            raise ValueError(f"file exceeds the {MAX_BLOB_BYTES}-byte read limit")
        raw = _git(self.directory, "cat-file", "blob", entry.oid)
        if b"\0" in raw:
            raise ValueError("binary files cannot be read as source text")
        return raw.decode("utf-8", errors="replace")

    def read(
        self, path: str, start: int = 1, lines: int = READ_LINES
    ) -> dict[str, Any]:
        if type(start) is not int or start < 1:
            raise ValueError("start must be a positive 1-based line number")
        if type(lines) is not int or not 1 <= lines <= READ_LINES:
            raise ValueError(f"lines must be between 1 and {READ_LINES}")
        # Split on LF like git grep, preserving consistent line numbers on CRLF files.
        content = self.text(path)
        all_lines = content.split("\n")
        if content.endswith("\n") or not content:
            all_lines.pop()
        end = min(start - 1 + lines, len(all_lines))
        window = all_lines[start - 1 : end]
        return {
            "path": relative_path(path),
            "start": start,
            "total_lines": len(all_lines),
            "text": "\n".join(
                f"{i}: {_clip(line.rstrip(chr(13)))}"
                for i, line in enumerate(window, start)
            ),
            "next_start": end + 1 if end < len(all_lines) else None,
            "truncated_lines": any(len(line) > MAX_LINE_CHARS for line in window),
        }

    def grep(
        self, pattern: str, path: str | None = None, max_hits: int = MAX_HITS
    ) -> dict[str, Any]:
        if (
            not isinstance(pattern, str)
            or not pattern
            or "\0" in pattern
            or "\n" in pattern
        ):
            raise ValueError(
                "pattern must be a nonempty, single-line regular expression"
            )
        if len(pattern) > 2000:
            raise ValueError("pattern exceeds 2000 characters; narrow the expression")
        if type(max_hits) is not int or not 1 <= max_hits <= MAX_HITS:
            raise ValueError(f"max_hits must be between 1 and {MAX_HITS}")
        scope = self._scope(path)
        command = [
            "git",
            "--literal-pathspecs",
            "-C",
            str(self.directory),
            "grep",
            "--no-color",
            "--no-ext-grep",
            "-n",
            "-I",
            "-E",
            "-z",
            "-e",
            pattern,
            self.commit,
            "--",
        ]
        if scope:
            command.append(scope)
        # Count all hits without placing unbounded grep output into Python memory
        # or the model transcript. The process deadline also bounds regex work.
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(
                command, stdout=output, stderr=subprocess.PIPE, timeout=30, check=False
            )
            if result.returncode not in {0, 1}:
                raise ValueError(result.stderr.decode(errors="replace")[:1000].strip())
            output.seek(0)
            hits: list[str] = []
            total = 0
            for raw_line in output:
                source, line, text = raw_line.split(b"\0", 2)
                name = source.decode("utf-8").removeprefix(self.commit + ":")
                entry = self.entries.get(name)
                if entry is None or not entry.readable:
                    continue
                total += 1
                if len(hits) < max_hits:
                    snippet = text.decode("utf-8", errors="replace").rstrip("\r\n")
                    hits.append(f"{name}:{int(line)}: {_clip(snippet)}")
        return {
            "hits": hits,
            "total_hits": total,
            "truncated": total > max_hits,
            "hint": "Narrow the path or pattern to see more relevant hits."
            if total > max_hits
            else "",
        }
