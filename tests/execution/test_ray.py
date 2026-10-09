import asyncio
from uuid import uuid4

import pytest

from focus.execution import InferenceRequest
from focus.execution.ray import (
    RayInferenceExecutor,
    RayServiceConfig,
    create_ray_service,
)
from focus.generation import SamplingParams
from tests.execution.ray_support import worker

ray = pytest.importorskip("ray")
pytestmark = pytest.mark.ray


@pytest.fixture
def service():
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    config = RayServiceConfig(
        "focus-" + uuid4().hex,
        "focus-tests",
        ({"_target_": "tests.execution.ray_support.make_policy"},),
        capacity=1,
        max_pending=8,
        lease_seconds=0.6,
    )
    actor = create_ray_service(config)
    ray.get(actor.describe.remote())
    yield config, actor
    ray.kill(actor)
    ray.shutdown()


async def test_real_workers_share_one_limit_and_shutdown(service):
    config, actor = service
    remote = ray.remote(worker)
    results = await asyncio.gather(
        *[remote.remote(config.name, config.namespace, 3) for _ in range(2)]
    )
    assert all(len(group) == 3 for group in results)
    stats = await actor.stats.remote()
    assert stats["peak_active"] == 1
    assert stats["completed"] == 6
    assert stats["active"] == stats["pending"] == 0
    await actor.shutdown.remote()
    client = RayInferenceExecutor(config.name, config.namespace)
    try:
        with pytest.raises(Exception, match="closed"):
            await client.submit(InferenceRequest("custom:policy:test", "test", (1,)))
    finally:
        await client.aclose()


async def test_cancel_loss_and_coordinator_failure(service):
    config, actor = service
    client = RayInferenceExecutor(config.name, config.namespace)
    request = InferenceRequest(
        "custom:policy:test", "test", (1,), params=SamplingParams(temperature=10)
    )
    pending = asyncio.create_task(client.submit(request))
    await asyncio.sleep(0.1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await asyncio.sleep(0.1)
    assert (await actor.stats.remote())["active"] == 0
    # A worker disappearing without closing leaves no heartbeat. The service reaps it.
    with pytest.raises(Exception, match="lease expired"):
        await actor.submit.remote("lost-worker", request)
    pending = asyncio.create_task(client.submit(request))
    await asyncio.sleep(0.1)
    ray.kill(actor)
    with pytest.raises(ray.exceptions.RayError):
        await pending
    await client.aclose()
