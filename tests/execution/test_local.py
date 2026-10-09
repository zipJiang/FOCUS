import asyncio
from functools import partial

import pytest

from focus.execution import AdmissionPool, CapacityError, Owner
from focus.execution.process import ProcessExecutor, ProcessSnapshot


async def test_queue_is_bounded_and_cancel_releases_capacity():
    pool = AdmissionPool(1, max_pending=1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked():
        entered.set()
        await release.wait()

    first = asyncio.create_task(pool.run(blocked))
    await entered.wait()
    second = asyncio.create_task(pool.run(blocked))
    await asyncio.sleep(0)
    with pytest.raises(CapacityError):
        await pool.run(blocked)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert pool.stats.active == 0
    assert pool.stats.pending == 0
    await pool.aclose()


async def test_episode_fairness_and_queue_timeout():
    pool = AdmissionPool(1)
    ready, release = asyncio.Event(), asyncio.Event()
    order = []

    async def block():
        ready.set()
        await release.wait()

    async def record(value):
        order.append(value)

    first = asyncio.create_task(pool.run(block))
    await ready.wait()
    tasks = [
        asyncio.create_task(pool.run(partial(record, i), owner=Owner(episode)))
        for i, episode in [(1, "A"), (2, "A"), (3, "B"), (4, "B")]
    ]
    with pytest.raises(TimeoutError):
        await pool.run(block, queue_timeout=0.01)
    release.set()
    await asyncio.gather(first, *tasks)
    assert order == [1, 3, 2, 4]
    assert pool.stats.peak_active == 1


async def test_python_restores_without_replaying_and_branches_files():
    executor = ProcessExecutor(1)
    original = await executor.execute(
        "x += 1\nopen('value', 'w').write(str(x))",
        ProcessSnapshot.variables(x=1),
    )
    assert original.snapshot is not None
    branches = await asyncio.gather(
        *[executor.execute("x += 1\nprint(x)", original.snapshot) for _ in range(2)]
    )
    assert [b.output.strip() for b in branches] == ["3", "3"]
    assert original.snapshot.files["value"] == b"2"
    await executor.aclose()


async def test_recursive_callback_releases_process_capacity():
    executor = ProcessExecutor(1)

    async def child(kind, prompts, model):
        result = await executor.execute("print('child')")
        return [result.output.strip()]

    result = await executor.execute("print(rlm_query('question'))", handler=child)
    assert result.output.strip() == "child"
    assert executor.pool.stats.peak_active == 1
    await executor.aclose()


async def test_unserializable_state_and_timeout_are_explicit():
    executor = ProcessExecutor(1)
    result = await executor.execute("handle = open('file', 'w')")
    assert result.snapshot is None
    assert "handle" in result.error
    with pytest.raises(TimeoutError):
        await executor.execute("while True: pass", timeout=0.2)
    assert executor._resident == 0
    assert executor.pool.stats.active == 0
    await executor.aclose()
