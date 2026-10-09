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


async def test_restored_functions_see_live_globals_and_callbacks():
    executor = ProcessExecutor(1)
    first = await executor.execute(
        "x=1\ndef f(): return x\ndef query(): return llm_query(str(x))"
    )

    async def handler(kind, prompts, model):
        return prompts

    second = await executor.execute(
        "x=2\nprint(f(), query())", first.snapshot, handler=handler
    )
    assert second.output.strip() == "2 2"
    third = await executor.execute("x=3\nprint(f())", second.snapshot)
    assert third.output.strip() == "3"
    await executor.aclose()


async def test_process_death_preserves_error_and_releases_capacity():
    executor = ProcessExecutor(1)
    with pytest.raises(RuntimeError, match="exited without a result"):
        await executor.execute("import os\nos._exit(3)")
    assert executor._resident == 0
    assert executor.pool.stats.active == 0
    await executor.aclose()


async def test_resident_exhaustion_rejects_descendant_without_deadlock():
    executor = ProcessExecutor(1, max_resident=1)

    async def child(kind, prompts, model):
        await executor.execute("print('unreachable')")
        return []

    result = await asyncio.wait_for(
        executor.execute("rlm_query('child')", handler=child), 3
    )
    assert "resident process limit" in result.error
    assert executor._resident == 0
    await executor.aclose()


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


async def test_managed_cell_terminates_descendant_in_separate_session():
    import os
    import sys

    executor = ProcessExecutor()
    result = await executor.execute(
        "import subprocess\n"
        f"p = subprocess.Popen([{sys.executable!r}, '-c', "
        "'import time; time.sleep(60)'], "
        "start_new_session=True)\nprint(p.pid)\ndel p"
    )
    pid = int(result.output.strip())
    async with asyncio.timeout(3):
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            # A terminated orphan may briefly remain as a zombie until reaped.
            process = await asyncio.create_subprocess_exec(
                "ps",
                "-p",
                str(pid),
                "-o",
                "stat=",
                stdout=asyncio.subprocess.PIPE,
            )
            status, _ = await process.communicate()
            if not status.strip() or status.strip().startswith(b"Z"):
                break
            await asyncio.sleep(0.02)
    await executor.aclose()


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
