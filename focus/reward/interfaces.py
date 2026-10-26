"""Reward model interfaces.

A :class:`RewardModel` ABC callable sync or async, and an async-primary
:class:`AsyncRewardModel` variant. A reward model turns a *serialized
state context* (a string) into a sequence-level scalar :class:`RewardResult`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Coroutine, Sequence
from typing import Any

from focus.reward.types import RewardResult


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Bridge an async scoring call into a sync ``score`` entrypoint.

    A ``Coroutine``, not any ``Awaitable``: ``asyncio.run`` rejects a non-coroutine
    awaitable outright (``ValueError: a coroutine was expected``), so the wider
    annotation promised something this never accepted.
    """

    return asyncio.run(coro)


class RewardModel(ABC):
    """Sequence-level reward for a serialized state context, callable sync or async.

    Sync-primary backends implement :meth:`score`; the default :meth:`ascore` offloads
    it to a worker thread. Async-primary backends subclass :class:`AsyncRewardModel`
    (implement ``ascore``, get sync ``score`` via :func:`run_async`). ``score_batch``
    scores many contexts at once -- a backend that can batch (one request for a whole
    search frontier) overrides it.
    """

    @abstractmethod
    def score(self, context: str) -> RewardResult:
        raise NotImplementedError

    def score_batch(self, contexts: Sequence[str]) -> list[RewardResult]:
        """Score many contexts. Default: one :meth:`score` per context."""

        return [self.score(c) for c in contexts]

    async def ascore(self, context: str) -> RewardResult:
        """Async score. Default offloads :meth:`score` to a worker thread."""

        return await asyncio.to_thread(self.score, context)

    async def ascore_batch(self, contexts: Sequence[str]) -> list[RewardResult]:
        """Async batch score. Default offloads :meth:`score_batch` to a thread."""

        return await asyncio.to_thread(self.score_batch, contexts)

    def startup_check(self) -> None:
        """Optional one-time readiness check (e.g. ping the server)."""

        return None

    def close(self) -> None:
        """Optional resource teardown."""

        return None


class AsyncRewardModel(RewardModel):
    """Async-primary reward model: implement :meth:`ascore`, inherit sync ``score``.

    Sync entrypoints must be called from a thread with no running event loop; callers
    already on a loop should await ``ascore`` / ``ascore_batch``.
    """

    def score(self, context: str) -> RewardResult:
        return run_async(self.ascore(context))

    def score_batch(self, contexts: Sequence[str]) -> list[RewardResult]:
        return run_async(self.ascore_batch(contexts))

    @abstractmethod
    async def ascore(self, context: str) -> RewardResult:
        raise NotImplementedError

    async def ascore_batch(self, contexts: Sequence[str]) -> list[RewardResult]:
        """Default: fan out :meth:`ascore` concurrently."""

        return list(await asyncio.gather(*(self.ascore(c) for c in contexts)))


__all__ = [
    "AsyncRewardModel",
    "RewardModel",
    "run_async",
]
