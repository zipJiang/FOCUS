import asyncio
from functools import partial

import pytest

from focus.execution import AdmissionPool, CapacityError, Owner


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


async def test_backend_limits_and_active_deadline_do_not_leak_permits():
    pool = AdmissionPool(2, resource_limits={"slow": 1})
    entered, release = asyncio.Event(), asyncio.Event()

    async def wait():
        entered.set()
        await release.wait()

    first = asyncio.create_task(pool.run(wait, resource="slow"))
    await entered.wait()
    second = asyncio.create_task(pool.run(wait, resource="slow"))
    await asyncio.sleep(0)
    with pytest.raises(TimeoutError):
        await pool.run(wait, resource="other", execution_timeout=0.01)
    assert pool.stats.active == pool.stats.pending == 1
    await pool.aclose()
    await asyncio.gather(first, second, return_exceptions=True)
    assert pool.stats.active == pool.stats.pending == 0


async def test_owned_clients_all_close_even_if_one_fails():
    from focus.execution import LocalInferenceExecutor
    from focus.generation import Policy
    from tests.helpers import test_format

    closed = []

    class Client(Policy[str]):
        async def aclose(self):
            closed.append(self.model)
            if self.model == "bad":
                raise RuntimeError("close failed")

    executor = LocalInferenceExecutor(own_clients=True)
    executor.bind(Client(format=test_format(), model="bad"))
    executor.bind(Client(format=test_format(), model="good"))
    with pytest.raises(ExceptionGroup, match="cleanup failed"):
        await executor.aclose()
    assert closed == ["bad", "good"]
    await executor.aclose()
    assert len(closed) == 2


async def test_invocations_rotate_within_a_branch():
    pool = AdmissionPool(1)
    release = asyncio.Event()
    first = asyncio.create_task(pool.run(release.wait))
    await asyncio.sleep(0)
    order = []

    async def record(value):
        order.append(value)

    tasks = [
        asyncio.create_task(
            pool.run(partial(record, n), owner=Owner("ep", "branch", inv))
        )
        for n, inv in [(1, "a"), (2, "a"), (3, "b"), (4, "b")]
    ]
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, *tasks)
    assert order == [1, 3, 2, 4]
    await pool.aclose()
