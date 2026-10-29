"""Bounded, fair async admission. Waiting parents never own request permits."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


class CapacityError(RuntimeError):
    """Work could not be admitted without exceeding a configured bound."""


@dataclass(frozen=True)
class Owner:
    episode: str = "default"
    branch: str = "root"
    invocation: str = "root"


DEFAULT_OWNER = Owner()


@dataclass
class _Ticket:
    owner: Owner
    resource: str
    ready: asyncio.Future[None]
    granted: bool = False


@dataclass
class PoolStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    active: int = 0
    peak_active: int = 0
    pending: int = 0
    resources: dict[str, int] = field(default_factory=dict)


class AdmissionPool:
    """Round robin across episodes and branches, with per-resource capacity.

    Bound to one event loop. The queue is deliberately finite: callers get an explicit
    CapacityError rather than creating arbitrarily many waiters outside a semaphore.
    Resource limits measure client requests, not unknown server-side GPU scheduling.
    """

    def __init__(
        self,
        capacity: int = 16,
        max_pending: int = 256,
        resource_limits: dict[str, int] | None = None,
    ) -> None:
        if capacity < 1 or max_pending < 1:
            raise ValueError("capacity and max_pending must be positive")
        self.capacity = capacity
        self.max_pending = max_pending
        self.limits = dict(resource_limits or {})
        if any(value < 1 for value in self.limits.values()):
            raise ValueError("resource limits must be positive")
        self.stats = PoolStats()
        self._queue: OrderedDict[
            str, OrderedDict[str, OrderedDict[str, deque[_Ticket]]]
        ] = OrderedDict()
        self._running: dict[asyncio.Task[object], Owner] = {}
        self._closed = False

    def _pump(self) -> None:
        while self.stats.active < self.capacity and self.stats.pending:
            selected: _Ticket | None = None
            for episode in tuple(self._queue):
                branches = self._queue[episode]
                for branch in tuple(branches):
                    invocations = branches[branch]
                    for invocation in tuple(invocations):
                        queue = invocations[invocation]
                        for ticket in tuple(queue):
                            used = self.stats.resources.get(ticket.resource, 0)
                            if used < self.limits.get(ticket.resource, self.capacity):
                                selected = ticket
                                queue.remove(ticket)
                                break
                        if not queue:
                            del invocations[invocation]
                        elif selected is not None:
                            invocations.move_to_end(invocation)
                        if selected is not None:
                            break
                    if not invocations:
                        del branches[branch]
                    elif selected is not None:
                        branches.move_to_end(branch)
                    if selected is not None:
                        break
                if not branches:
                    del self._queue[episode]
                elif selected is not None:
                    self._queue.move_to_end(episode)
                if selected is not None:
                    break
            if selected is None:
                return
            self.stats.pending -= 1
            self.stats.active += 1
            self.stats.peak_active = max(self.stats.peak_active, self.stats.active)
            resource = selected.resource
            self.stats.resources[resource] = self.stats.resources.get(resource, 0) + 1
            selected.granted = True
            selected.ready.set_result(None)

    def _remove(self, ticket: _Ticket) -> None:
        if ticket.granted:
            self.stats.active -= 1
            self.stats.resources[ticket.resource] -= 1
        else:
            branches = self._queue[ticket.owner.episode]
            invocations = branches[ticket.owner.branch]
            queue = invocations[ticket.owner.invocation]
            queue.remove(ticket)
            self.stats.pending -= 1
            if not queue:
                del invocations[ticket.owner.invocation]
            if not invocations:
                del branches[ticket.owner.branch]
            if not branches:
                del self._queue[ticket.owner.episode]
        self._pump()

    async def run[T](
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        owner: Owner = DEFAULT_OWNER,
        resource: str = "default",
        queue_timeout: float | None = None,
        execution_timeout: float | None = None,
    ) -> T:
        if self._closed:
            raise RuntimeError("executor is closed")
        if self.stats.pending >= self.max_pending:
            raise CapacityError("executor queue is full")
        task = asyncio.current_task()
        assert task is not None
        if task in self._running:
            raise RuntimeError("nested admission would hold a parent permit")
        self._running[task] = owner
        ticket = _Ticket(owner, resource, asyncio.get_running_loop().create_future())
        branches = self._queue.setdefault(owner.episode, OrderedDict())
        invocations = branches.setdefault(owner.branch, OrderedDict())
        invocations.setdefault(owner.invocation, deque()).append(ticket)
        self.stats.submitted += 1
        self.stats.pending += 1
        self._pump()
        try:
            async with asyncio.timeout(queue_timeout):
                await asyncio.shield(ticket.ready)
            async with asyncio.timeout(execution_timeout):
                result = await operation()
            self.stats.completed += 1
            return result
        except BaseException:
            self.stats.failed += 1
            raise
        finally:
            self._running.pop(task, None)
            self._remove(ticket)

    async def cancel(self, episode: str) -> None:
        tasks = [
            task for task, owner in self._running.items() if owner.episode == episode
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        self._closed = True
        tasks = list(self._running)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["AdmissionPool", "CapacityError", "Owner"]
