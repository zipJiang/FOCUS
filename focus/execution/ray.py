"""Optional named Ray admission service. Only configuration and requests cross Ray."""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast
from uuid import uuid4

from focus.critic import Critic
from focus.execution.inference import (
    InferenceRequest,
    LocalInferenceExecutor,
    policy_key,
)
from focus.execution.pool import DEFAULT_OWNER, AdmissionPool, CapacityError, Owner
from focus.generation import GenerateResult, Policy


@dataclass(frozen=True)
class RayServiceConfig:
    name: str
    namespace: str
    backends: tuple[dict[str, Any], ...]
    capacity: int = 16
    max_pending: int = 256
    resource_limits: dict[str, int] = field(default_factory=dict)
    lease_seconds: float = 10
    request_seconds: float = 300
    critics: dict[str, dict[str, Any]] = field(default_factory=dict)


def _construct(config: dict[str, Any]) -> Any:
    """Call the ``_target_`` import path with the remaining keys as keyword arguments.

    Backends and value models are built inside the coordinator process, so the
    configuration names them rather than shipping live clients across Ray.
    """
    target = config.get("_target_")
    if not isinstance(target, str) or "." not in target:
        raise ValueError("_target_ must be an import path such as 'package.factory'")
    module_name, _, attribute = target.rpartition(".")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(**{k: v for k, v in config.items() if k != "_target_"})


class _Coordinator:
    def __init__(self, config: RayServiceConfig):
        if config.lease_seconds <= 0 or config.request_seconds <= 0:
            raise ValueError("lease and request deadlines must be positive")
        self.config = config
        self.executor = LocalInferenceExecutor(
            AdmissionPool(config.capacity, config.max_pending, config.resource_limits),
            own_clients=True,
        )
        for backend in config.backends:
            self.executor.bind(_construct(backend))
        self.clients: dict[str, float] = {}
        self.critics = {name: _construct(cfg) for name, cfg in config.critics.items()}

    async def describe(self) -> RayServiceConfig:
        return self.config

    async def pulse(self, client: str) -> None:
        self.clients[client] = time.monotonic()

    async def submit(self, client: str, request: InferenceRequest) -> GenerateResult:
        return await self._leased(
            client,
            self.executor.submit(
                replace(
                    request,
                    execution_timeout=min(
                        request.execution_timeout or self.config.request_seconds,
                        self.config.request_seconds,
                    ),
                )
            ),
        )

    async def predict(
        self, client: str, backend: str, contexts: tuple[str, ...], owner: Owner
    ) -> list[float]:
        critic = self.critics[backend]

        async def predict() -> list[float]:
            return [float(v) for v in await critic.predict(contexts)]

        return await self._leased(
            client,
            self.executor.pool.run(
                predict,
                owner=owner,
                resource="critic:" + backend,
                execution_timeout=self.config.request_seconds,
            ),
        )

    async def _leased[T](self, client: str, call: Coroutine[Any, Any, T]) -> T:
        self.clients.setdefault(client, time.monotonic())
        operation = asyncio.create_task(call)
        try:
            while not operation.done():
                await asyncio.wait({operation}, timeout=self.config.lease_seconds / 3)
                if time.monotonic() - self.clients[client] > self.config.lease_seconds:
                    raise TimeoutError("request owner lease expired; worker lost")
            return await operation
        finally:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)

    async def stats(self) -> dict[str, int]:
        stats = self.executor.pool.stats
        return {
            name: getattr(stats, name)
            for name in (
                "active",
                "pending",
                "peak_active",
                "completed",
                "failed",
            )
        }

    async def shutdown(self) -> None:
        await self.executor.aclose()
        for critic in self.critics.values():
            await critic.aclose()


def create_ray_service(config: RayServiceConfig) -> Any:
    """Called once by the job owner. Name collision is an error, never silent reuse."""
    import ray

    if not config.name or not config.namespace:
        raise ValueError("Ray execution requires explicit name and namespace")
    return (
        ray.remote(_Coordinator)
        .options(
            name=config.name,
            namespace=config.namespace,
            lifetime="detached",
            num_cpus=0,
            max_restarts=0,
            max_task_retries=0,
            max_concurrency=config.capacity + config.max_pending + 32,
        )
        .remote(config)
    )


class RayInferenceExecutor:
    """Connect-only client. Closing it never kills a service shared by other workers.

    Coordinator death fails in-flight work; neither actor tasks nor backend calls are
    retried. A heartbeat lease bounds abandoned requests after a worker is lost.
    """

    def __init__(self, name: str, namespace: str, max_outstanding: int = 256):
        import ray

        if max_outstanding < 1:
            raise ValueError("max_outstanding must be positive")
        self.max_outstanding = max_outstanding
        self._outstanding = 0
        self.actor = ray.get_actor(name, namespace=namespace)
        self.client = uuid4().hex
        self._pulse: asyncio.Task[None] | None = None
        self._refs: set[Any] = set()
        self._closed = False

    def bind(self, policy: Policy[Any]) -> str:
        return policy_key(policy)

    async def _heartbeat(self, interval: float) -> None:
        while True:
            await self.actor.pulse.remote(self.client)
            await asyncio.sleep(interval)

    async def submit(self, request: InferenceRequest) -> GenerateResult:
        self._reserve()
        try:
            await self._start()
            ref: Any = self.actor.submit.remote(self.client, request)
            return cast(GenerateResult, await self._result(ref))
        finally:
            self._outstanding -= 1

    async def predict(
        self, backend: str, contexts: Sequence[str], owner: Owner = DEFAULT_OWNER
    ) -> list[float]:
        self._reserve()
        try:
            await self._start()
            ref: Any = self.actor.predict.remote(
                self.client, backend, tuple(contexts), owner
            )
            return cast(list[float], await self._result(ref))
        finally:
            self._outstanding -= 1

    def _reserve(self) -> None:
        if self._outstanding >= self.max_outstanding:
            raise CapacityError("Ray client outstanding request limit exceeded")
        self._outstanding += 1

    async def _start(self) -> None:
        if self._closed:
            raise RuntimeError("Ray executor client is closed")
        if self._pulse is None:
            config = await self.actor.describe.remote()
            # No await between assignment and task creation: concurrent first requests
            # cannot leave an unowned heartbeat behind.
            if self._pulse is None:
                self._pulse = asyncio.create_task(
                    self._heartbeat(config.lease_seconds / 3)
                )
        if self._pulse.done():
            await self._pulse  # propagate coordinator failure

    async def _result(self, ref: Any) -> Any:
        import ray

        self._refs.add(ref)
        try:
            return await ref
        except asyncio.CancelledError:
            ray.cancel(ref, force=False)
            raise
        finally:
            self._refs.discard(ref)

    async def aclose(self) -> None:
        import ray

        self._closed = True
        for ref in list(self._refs):
            ray.cancel(ref, force=False)
        if self._pulse is not None:
            self._pulse.cancel()
            await asyncio.gather(self._pulse, return_exceptions=True)


class RayCritic(Critic):
    """A critic served by the coordinator, sharing its aggregate admission pool.

    ``backend`` names an entry of ``RayServiceConfig.critics``.
    """

    def __init__(
        self,
        executor: RayInferenceExecutor,
        backend: str,
        *,
        timeout: float | None = None,
    ) -> None:
        self.executor, self.backend = executor, backend
        self.timeout = timeout

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        return await self.executor.predict(self.backend, contexts)

    async def predict_owned(self, contexts: Sequence[str], owner: Owner) -> list[float]:
        return await self.executor.predict(self.backend, contexts, owner)
