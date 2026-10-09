"""Managed Python cells with snapshots and host-mediated recursive calls."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import sys
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import cloudpickle

from focus.execution.pool import DEFAULT_OWNER, AdmissionPool, CapacityError, Owner


@dataclass(frozen=True)
class ProcessSnapshot:
    namespace: bytes = field(default_factory=lambda: cloudpickle.dumps({}))
    files: dict[str, bytes] = field(default_factory=dict)

    @classmethod
    def variables(cls, **values: Any) -> ProcessSnapshot:
        return cls(cloudpickle.dumps(values))


@dataclass(frozen=True)
class CellResult:
    snapshot: ProcessSnapshot | None
    output: str
    error: str = ""


type CallHandler = Callable[[str, list[str], str | None], Awaitable[list[str]]]


class ProcessExecutor:
    """Cells restore namespace bytes, never replay earlier source.

    Resident capacity is fail-fast. Active-cell capacity is released while the
    process waits for a model/child callback, preventing recursive pool deadlocks.
    This controls trusted research code; it is not a hostile-code security sandbox.
    """

    def __init__(self, capacity: int = 4, max_resident: int = 32):
        if max_resident < 1:
            raise ValueError("max_resident must be positive")
        self.pool = AdmissionPool(capacity)
        self.max_resident = max_resident
        self._resident = 0
        self._tasks: set[asyncio.Task[object]] = set()
        self._closed = False

    async def execute(
        self,
        code: str,
        snapshot: ProcessSnapshot | None = None,
        *,
        handler: CallHandler | None = None,
        owner: Owner = DEFAULT_OWNER,
        timeout: float = 30,
        max_output: int = 16000,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> CellResult:
        if self._closed:
            raise RuntimeError("process executor is closed")
        if self._resident >= self.max_resident:
            raise CapacityError(
                "resident process limit reached; recursive call refused"
            )
        self._resident += 1
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        snapshot = snapshot or ProcessSnapshot()
        process: asyncio.subprocess.Process | None = None
        with tempfile.TemporaryDirectory(prefix="focus-cell-") as directory:
            try:
                root = Path(directory).resolve()
                for name, content in snapshot.files.items():
                    path = root / name
                    if not path.resolve().is_relative_to(root):
                        raise ValueError("snapshot path escapes workspace")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)

                async def exchange(message: dict[str, Any] | None) -> dict[str, Any]:
                    nonlocal process
                    if process is None:
                        process = await asyncio.create_subprocess_exec(
                            sys.executable,
                            "-m",
                            "focus.execution.worker",
                            cwd=directory,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            start_new_session=True,
                            limit=max_bytes * 2,
                        )
                        message = {
                            "code": code,
                            "state": base64.b64encode(snapshot.namespace).decode(),
                            "max_output": max_output,
                            "max_bytes": max_bytes,
                        }
                    assert process.stdin is not None
                    assert process.stdout is not None
                    if message is not None:
                        process.stdin.write((json.dumps(message) + "\n").encode())
                        await process.stdin.drain()
                    line = await process.stdout.readline()
                    if not line:
                        raise RuntimeError("managed process exited without a result")
                    result: dict[str, Any] = json.loads(line)
                    return result

                reply: dict[str, Any] | None = None
                async with asyncio.timeout(timeout):
                    while True:
                        current_reply = reply
                        result = await self.pool.run(
                            partial(exchange, current_reply), owner=owner
                        )
                        if result["kind"] == "call":
                            # The pool permit has been returned before recursion.
                            try:
                                if handler is None:
                                    raise ValueError("model calls are unavailable")
                                values = await handler(
                                    result["operation"],
                                    result["prompts"],
                                    result["model"],
                                )
                                reply = {"result": values}
                            except (ValueError, CapacityError) as exc:
                                reply = {"error": str(exc)}
                            continue
                        if result["kind"] == "unsupported":
                            return CellResult(None, result["output"], result["error"])
                        return CellResult(
                            ProcessSnapshot(
                                base64.b64decode(result["state"]),
                                {
                                    name: base64.b64decode(value)
                                    for name, value in result["files"].items()
                                },
                            ),
                            result["output"],
                            result["error"],
                        )
            finally:
                if process is not None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.communicate()
                self._resident -= 1
                self._tasks.discard(task)

    async def aclose(self) -> None:
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.pool.aclose()
