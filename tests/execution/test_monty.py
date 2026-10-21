import asyncio
import pickle
from dataclasses import replace

import pytest

from focus.execution.monty import TIMEOUT_ERROR, MontyExecutor, MontySnapshot
from focus.execution.pool import AdmissionPool


async def test_suspended_nested_python_frames_can_resume_twice():
    executor = MontyExecutor(AdmissionPool(1))
    try:
        paused = await executor.execute(
            "def outer():\n"
            "    items = [context]\n"
            "    def inner():\n"
            "        items.append('before')\n"
            "        answer = rlm_query('child')\n"
            "        items.append(answer)\n"
            "        print(items)\n"
            "    inner()\n"
            "outer()",
            MontySnapshot(context="input"),
        )
        assert paused.snapshot is not None
        assert paused.snapshot.suspended
        assert executor.pool.stats.active == 0
        snapshot = pickle.loads(pickle.dumps(paused.snapshot))
        left = await executor.execute("", snapshot, reply={"return_value": ["A"]})
        right = await executor.execute("", snapshot, reply={"return_value": ["B"]})
        assert left.output == "['input', 'before', 'A']\n"
        assert right.output == "['input', 'before', 'B']\n"
        assert executor.pool.stats.peak_active == 1
    finally:
        await executor.aclose()


async def test_idle_session_and_error_state_survive_restore():
    executor = MontyExecutor()
    try:
        first = await executor.execute(
            "x=3\ndef f(y):\n    return x+y", MontySnapshot()
        )
        assert first.snapshot is not None
        error = await executor.execute("x=7\n1/0", first.snapshot)
        assert "ZeroDivisionError" in error.error
        assert error.snapshot is not None
        final = await executor.execute("print(f(2))", error.snapshot)
        assert final.output == "9\n"
    finally:
        await executor.aclose()


async def test_cancellation_releases_worker_and_admission():
    executor = MontyExecutor(AdmissionPool(1))
    try:
        task = asyncio.create_task(
            executor.execute("while True:\n    pass", MontySnapshot(), timeout=10)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert executor.pool.stats.active == 0
        result = await executor.execute("print(42)", MontySnapshot())
        assert result.output == "42\n"
    finally:
        await executor.aclose()


async def test_unavailable_host_access_is_model_visible():
    executor = MontyExecutor()
    try:
        result = await executor.execute(
            "open('/tmp/not-allowed', 'w')", MontySnapshot()
        )
        assert result.error
        assert result.snapshot is not None
    finally:
        await executor.aclose()


async def test_runaway_cell_returns_timeout_instead_of_raising():
    executor = MontyExecutor(AdmissionPool(1))
    try:
        result = await executor.execute(
            "print('started')\nwhile True:\n    pass", MontySnapshot(), timeout=1
        )
        assert result.snapshot is None
        assert "TimeoutError" in result.error
        assert result.output == "started\n"
        assert executor.pool.stats.active == 0
    finally:
        await executor.aclose()


async def test_suspended_cell_carries_only_its_remaining_python_time():
    executor = MontyExecutor()
    try:
        paused = await executor.execute(
            "print(rlm_query('child'))", MontySnapshot(), timeout=5
        )
        assert paused.snapshot is not None
        # Worker startup and helper setup are not charged to the cell.
        assert 4 < paused.snapshot.remaining < 5
        spent = replace(paused.snapshot, remaining=0)
        timed_out = await executor.execute("", spent, reply={"return_value": ["A"]})
        assert timed_out.snapshot is None
        assert timed_out.error == TIMEOUT_ERROR
        assert executor.pool.stats.submitted == 1
    finally:
        await executor.aclose()
