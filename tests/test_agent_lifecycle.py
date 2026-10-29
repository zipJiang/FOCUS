"""The agent is the lifecycle scope for a standalone rollout harness."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from focus import Collector, Fifo, Phase, Search
from focus.branching import EveryDecision
from focus.execution import AdmissionPool, ExecutionServices
from tests.helpers import ScriptedGenerator, ToyEnv, complete, toy_runner


async def test_agent_scope_runs_forks_and_leaves_policy_to_caller(monkeypatch):
    agent = toy_runner(ScriptedGenerator(["go", "stop"]), branch_points=EveryDecision())
    close_policy = AsyncMock()
    monkeypatch.setattr(agent.policy, "aclose", close_policy)
    async with agent as entered:
        assert entered is agent
        checkpoint = await agent.advance(await agent.start("question"))
        left, right = await asyncio.gather(
            agent.run(checkpoint.fork()), agent.run(checkpoint.fork())
        )
        assert left.done
        assert right.done
        assert not checkpoint.done
    close_policy.assert_not_awaited()
    with pytest.raises(RuntimeError, match="executor is closed"):
        await agent.advance(checkpoint.fork())
    await agent.aclose()  # repeated cleanup is harmless


@pytest.mark.parametrize("exit_kind", ["normal", "error", "cancel"])
async def test_exit_cancels_and_awaits_running_and_queued_requests(exit_kind):
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    ready = asyncio.Event()

    class Blocked(ScriptedGenerator):
        async def agenerate_tokens(self, *args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

    services = ExecutionServices(pool=AdmissionPool(capacity=1))
    agent = toy_runner(Blocked([]), services=services)
    requests = []

    async def scope():
        async with agent:
            root = await agent.start("question")
            requests.append(asyncio.create_task(agent.advance(root.fork())))
            await entered.wait()
            requests.append(asyncio.create_task(agent.advance(root.fork())))
            await asyncio.sleep(0)
            assert services.pool.stats.active == services.pool.stats.pending == 1
            ready.set()
            if exit_kind == "error":
                raise ValueError("body failed")
            if exit_kind == "cancel":
                await asyncio.Event().wait()

    task = asyncio.create_task(scope())
    await asyncio.wait_for(ready.wait(), timeout=2)
    if exit_kind == "cancel":
        task.cancel()
    if exit_kind == "normal":
        await task
    else:
        expected = ValueError if exit_kind == "error" else asyncio.CancelledError
        with pytest.raises(expected):
            await task
    assert cleaned.is_set()
    assert all(request.cancelled() for request in requests)
    assert services.pool.stats.active == services.pool.stats.pending == 0


async def test_one_outer_agent_owns_services_for_siblings_and_borrowing_collector():
    agent = toy_runner(ScriptedGenerator(["stop"]))
    sibling = toy_runner(ScriptedGenerator(["stop"]), services=agent.services)
    derived = agent.derive(
        env=ToyEnv(), system_prompt="notes", max_steps=2, finish_prompt=None
    )
    collector = Collector(
        sibling,
        search=Search([Phase(Fifo(), 2, max_children=1)]),
        estimator=None,
        own=False,
    )
    async with agent:
        async with collector:
            batch = await collector.collect("question")
            assert not batch.failures
        assert (await complete(derived, "notes")).done
        assert (await complete(sibling, "another question")).done
        assert (await complete(agent, "last question")).done
    with pytest.raises(RuntimeError, match="executor is closed"):
        await complete(sibling, "closed")
