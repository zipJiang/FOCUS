"""Complete rollout snapshots, independent of search checkpoint selection.

Stores contain trusted Python objects (cloudpickle), not portable untrusted input.
Every put commits one whole continuation, including nested protocol state. There is
deliberately no per-task journal. SQLite commits the snapshot and its metadata in one
transaction; interrupted writes leave the previous checkpoint available.
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import cloudpickle

from focus.harness.rollout import AnyRollout, RolloutState


class CorruptCheckpoint(ValueError):
    """A saved snapshot failed its integrity or decoding check."""


@dataclass(frozen=True)
class CheckpointInfo:
    id: str
    parent_id: str | None
    episode_id: str
    branch_id: str
    invocation_id: str
    format_version: int = 1


@dataclass(frozen=True)
class _Stored:
    info: CheckpointInfo
    payload: bytes
    digest: str


class Checkpointer(ABC):
    """Save complete states and load isolated copies, in commit order.

    One writer drives each branch; different branches may share a store. The caller
    must supply compatible policy, protocol, environment and runtime on recovery.
    Backends implement atomic insertion and newest-first lookup; serialization and
    validation are shared. No live runners, model clients or tasks are persisted.
    """

    @abstractmethod
    def _put(self, stored: _Stored) -> None: ...

    @abstractmethod
    def _get(self, checkpoint_id: str) -> _Stored: ...

    @abstractmethod
    def info(self, checkpoint_id: str) -> CheckpointInfo:
        """Metadata only; never reads or decodes the snapshot payload."""

    @abstractmethod
    def list(
        self, episode_id: str, branch_id: str, invocation_id: str = "root"
    ) -> tuple[CheckpointInfo, ...]:
        """Metadata in reverse commit order, for exactly one execution scope."""

    def put[S](self, rollout: RolloutState[S]) -> RolloutState[S]:
        """Atomically save a new snapshot; the input's checkpoint ID is its parent."""
        rollout.check_continuation()
        if not rollout.snapshot_available:
            raise ValueError("Cannot save an incomplete execution snapshot")
        if rollout.checkpoint_id is not None:
            parent = self.info(rollout.checkpoint_id)
            if (parent.episode_id, parent.invocation_id) != (
                rollout.episode_id,
                rollout.invocation_id,
            ):
                raise ValueError("Checkpoint parent belongs to another execution")
        info = CheckpointInfo(
            uuid4().hex,
            rollout.checkpoint_id,
            rollout.episode_id,
            rollout.branch_id,
            rollout.invocation_id,
        )
        snapshot = replace(rollout, checkpoint_id=info.id)
        payload = cloudpickle.dumps(snapshot)
        self._put(_Stored(info, payload, hashlib.sha256(payload).hexdigest()))
        return snapshot

    def get(self, checkpoint_id: str) -> AnyRollout:
        """Load an independent snapshot. Explicit IDs never silently fall back."""
        stored = self._get(checkpoint_id)
        if stored.info.format_version != 1:
            raise ValueError("Unsupported checkpoint storage format")
        if hashlib.sha256(stored.payload).hexdigest() != stored.digest:
            raise CorruptCheckpoint(f"Checksum mismatch for checkpoint {checkpoint_id}")
        try:
            snapshot = cloudpickle.loads(stored.payload)
        except Exception as exc:
            raise CorruptCheckpoint(
                f"Cannot decode checkpoint {checkpoint_id}"
            ) from exc
        if not isinstance(snapshot, RolloutState) or (
            snapshot.checkpoint_id,
            snapshot.episode_id,
            snapshot.branch_id,
            snapshot.invocation_id,
        ) != (
            stored.info.id,
            stored.info.episode_id,
            stored.info.branch_id,
            stored.info.invocation_id,
        ):
            raise CorruptCheckpoint(f"Invalid checkpoint record {checkpoint_id}")
        snapshot.check_continuation()
        if not snapshot.snapshot_available:
            raise CorruptCheckpoint(f"Incomplete checkpoint {checkpoint_id}")
        return snapshot

    def latest(
        self, episode_id: str, branch_id: str, invocation_id: str = "root"
    ) -> AnyRollout:
        """Load the last intact snapshot, skipping corrupt records in this scope."""
        last_error: CorruptCheckpoint | None = None
        for info in self.list(episode_id, branch_id, invocation_id):
            try:
                return self.get(info.id)
            except CorruptCheckpoint as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise KeyError((episode_id, branch_id, invocation_id))


class MemoryCheckpointer(Checkpointer):
    """Serialized snapshots for recovery within the current process."""

    def __init__(self) -> None:
        self._records: dict[str, _Stored] = {}

    def _put(self, stored: _Stored) -> None:
        self._records[stored.info.id] = stored

    def _get(self, checkpoint_id: str) -> _Stored:
        return self._records[checkpoint_id]

    def info(self, checkpoint_id: str) -> CheckpointInfo:
        return self._records[checkpoint_id].info

    def list(
        self, episode_id: str, branch_id: str, invocation_id: str = "root"
    ) -> tuple[CheckpointInfo, ...]:
        return tuple(
            record.info
            for record in reversed(self._records.values())
            if (
                record.info.episode_id,
                record.info.branch_id,
                record.info.invocation_id,
            )
            == (episode_id, branch_id, invocation_id)
        )


class SQLiteCheckpointer(Checkpointer):
    """Durable local snapshots. Each operation opens and closes its own connection.

    Commits are synchronous: a runner does not continue or emit a checkpoint before
    the commit succeeds. Use a file path, not SQLite's connection-local :memory:.
    """

    def __init__(self, path: str | Path) -> None:
        if str(path) == ":memory:":
            raise ValueError("Use MemoryCheckpointer for in-memory snapshots")
        self.path = str(Path(path).resolve())
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS rollout_checkpoints ("
                "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                "id TEXT UNIQUE NOT NULL, parent_id TEXT, episode_id TEXT NOT NULL, "
                "branch_id TEXT NOT NULL, invocation_id TEXT NOT NULL, "
                "format_version INTEGER NOT NULL, payload BLOB NOT NULL, "
                "digest TEXT NOT NULL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS rollout_checkpoint_scope ON "
                "rollout_checkpoints (episode_id, branch_id, invocation_id, seq)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _put(self, stored: _Stored) -> None:
        info = stored.info
        with self._connect() as db:
            db.execute(
                "INSERT INTO rollout_checkpoints "
                "(id, parent_id, episode_id, branch_id, invocation_id, "
                "format_version, payload, digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    info.id,
                    info.parent_id,
                    info.episode_id,
                    info.branch_id,
                    info.invocation_id,
                    info.format_version,
                    stored.payload,
                    stored.digest,
                ),
            )

    def _get(self, checkpoint_id: str) -> _Stored:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, parent_id, episode_id, branch_id, invocation_id, "
                "format_version, payload, digest FROM rollout_checkpoints WHERE id=?",
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            raise KeyError(checkpoint_id)
        return _Stored(CheckpointInfo(*row[:6]), row[6], row[7])

    def info(self, checkpoint_id: str) -> CheckpointInfo:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, parent_id, episode_id, branch_id, invocation_id, "
                "format_version FROM rollout_checkpoints WHERE id=?",
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            raise KeyError(checkpoint_id)
        return CheckpointInfo(*row)

    def list(
        self, episode_id: str, branch_id: str, invocation_id: str = "root"
    ) -> tuple[CheckpointInfo, ...]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, parent_id, episode_id, branch_id, invocation_id, "
                "format_version FROM rollout_checkpoints "
                "WHERE episode_id=? AND branch_id=? AND invocation_id=? "
                "ORDER BY seq DESC",
                (episode_id, branch_id, invocation_id),
            ).fetchall()
        return tuple(CheckpointInfo(*row) for row in rows)
