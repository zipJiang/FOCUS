"""Serializable Python continuations; no worker is retained at a checkpoint."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from typing import Any

from pydantic_monty import (
    AsyncFunctionSnapshot,
    AsyncMonty,
    AsyncNameLookupSnapshot,
    CollectString,
    ExternalSettledResult,
    MontyComplete,
    MontyCrashedError,
    MontyError,
)

from focus.execution.pool import DEFAULT_OWNER, AdmissionPool, Owner

# Worker startup and session restore are infrastructure, not the cell's budget.
_SETUP_SECONDS = 30.0
TIMEOUT_ERROR = "TimeoutError: Python execution time limit exceeded"

_HELPERS = """
def llm_query(prompt, model=None):
    return __focus_query('llm', [prompt], model)[0]
def rlm_query(prompt, model=None):
    return __focus_query('rlm', [prompt], model)[0]
def llm_query_batched(prompts, model=None):
    return __focus_query('llm', prompts, model)
def rlm_query_batched(prompts, model=None):
    return __focus_query('rlm', prompts, model)
"""


@dataclass(frozen=True)
class MontySnapshot:
    data: bytes = b""
    context: str = ""
    suspended: bool = False
    # Python seconds a suspended cell has left; time parked outside the
    # interpreter, such as child generation, never consumes it.
    remaining: float = 0.0


@dataclass(frozen=True)
class Query:
    kind: Any
    prompts: Any
    model: Any


@dataclass(frozen=True)
class MontyResult:
    snapshot: MontySnapshot | None
    output: str = ""
    error: str = ""
    query: Query | None = None


class MontyExecutor:
    """Bound active interpreters; suspended calls occupy only snapshot bytes.

    Workers are scoped to one segment. This also makes cancellation and service
    shutdown terminate every owned worker without keeping resident waiting parents.
    """

    def __init__(self, pool: AdmissionPool | None = None):
        self.pool = pool or AdmissionPool()

    async def execute(
        self,
        code: str,
        snapshot: MontySnapshot,
        *,
        reply: ExternalSettledResult | None = None,
        owner: Owner = DEFAULT_OWNER,
        timeout: float = 30,
    ) -> MontyResult:
        """Run a cell for `timeout` Python seconds; a resumed cell has its remainder."""
        if snapshot.suspended:
            if reply is None:
                raise ValueError("suspended Monty call requires a reply")
            timeout = snapshot.remaining
        if timeout <= 0:
            return MontyResult(None, error=TIMEOUT_ERROR)
        return await self.pool.run(
            partial(self._segment, code, snapshot, reply, timeout),
            owner=owner,
            resource="monty",
        )

    async def _segment(
        self,
        code: str,
        snapshot: MontySnapshot,
        reply: ExternalSettledResult | None,
        timeout: float,
    ) -> MontyResult:
        output = CollectString(max_bytes=1_000_000)
        started: float | None = None
        try:
            # One deadline: setup gets a fixed allowance, then the cell's budget.
            async with (
                asyncio.timeout(_SETUP_SECONDS) as deadline,
                AsyncMonty(
                    max_processes=1, request_timeout=timeout + _SETUP_SECONDS
                ) as pool,
                pool.checkout(
                    limits={
                        "max_feed_duration_secs": timeout,
                        "max_memory": 256 * 1024 * 1024,
                    }
                ) as session,
            ):
                if snapshot.suspended:
                    state = await session.load_snapshot(
                        snapshot.data, print_callback=output
                    )
                elif snapshot.data:
                    await session.load_session(snapshot.data)
                else:
                    await session.feed_run(
                        _HELPERS, inputs={"context": snapshot.context}
                    )
                started = time.monotonic()
                deadline.reschedule(asyncio.get_running_loop().time() + timeout)
                try:
                    if snapshot.suspended:
                        assert isinstance(state, AsyncFunctionSnapshot)
                        assert reply is not None
                        state = await state.resume(reply)
                    else:
                        state = await session.feed_start(code, print_callback=output)
                    while not isinstance(state, MontyComplete):
                        if isinstance(state, AsyncFunctionSnapshot):
                            if (
                                state.function_name == "__focus_query"
                                and len(state.args) == 3
                                and not state.kwargs
                            ):
                                return MontyResult(
                                    MontySnapshot(
                                        state.dump(),
                                        suspended=True,
                                        remaining=timeout
                                        - (time.monotonic() - started),
                                    ),
                                    output.output,
                                    query=Query(*state.args),
                                )
                            state = await state.resume(
                                {
                                    "exc_type": "RuntimeError",
                                    "message": "Host operation is unavailable: "
                                    + state.function_name,
                                }
                            )
                        elif isinstance(state, AsyncNameLookupSnapshot):
                            state = await state.resume()
                        else:
                            raise ValueError("unmanaged Monty future")
                    return MontyResult(
                        MontySnapshot(await session.dump()), output.output
                    )
                except MontyError as exc:
                    if isinstance(exc, MontyCrashedError) and exc.timed_out:
                        return MontyResult(None, output.output, TIMEOUT_ERROR)
                    if isinstance(exc.exception(), (TimeoutError, MemoryError)):
                        return MontyResult(None, output.output, str(exc))
                    # Runtime errors may leave useful variables in the session.
                    # Never silently resume an older snapshot if this dump fails.
                    try:
                        saved = MontySnapshot(await session.dump())
                    except Exception:
                        saved = None
                    return MontyResult(saved, output.output, str(exc))
        except TimeoutError:
            if started is None:
                raise  # Setup stalled: an infrastructure failure, not the cell's.
            # The worker died with the pool, so the session state is gone.
            return MontyResult(None, output.output, TIMEOUT_ERROR)

    async def aclose(self) -> None:
        await self.pool.aclose()
