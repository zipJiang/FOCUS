"""Adapt FOCUS collection and export to Slime's generation callback."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol, cast

from focus.collector import Collector
from focus.export import CriticSampleFactory, SampleFactory
from focus.harness.workspace import Workspace
from focus.preparation.records import ESTIMATOR_KEY, LANE_KEY

logger = logging.getLogger(__name__)


class GenerateHook(Protocol):
    async def __call__(
        self, args: Any, sample: Any, sampling_params: Any = None
    ) -> list[Any]: ...
    async def aclose(self) -> None: ...


def make_generate(
    collector_factory: Callable[[Any], Collector],
    *,
    own: bool = True,
    to_prompt: Callable[[Any], Any] = lambda sample: sample,
    workspace_factory: Callable[[Any], Workspace] | None = None,
    sample_factory: SampleFactory[Any] | None = None,
    critic_sample_factory: CriticSampleFactory[Any] | None = None,
    min_abs_advantage: float | None = None,
) -> GenerateHook:
    """Build Slime's generation callback from a collector factory.

    The factory builds one :class:`~focus.collector.Collector` per distinct Slime
    ``args`` object; the hook collects and exports one group per sample with it.
    Sampling comes from the agent and the search phases; Slime's ``sampling_params``
    argument is accepted but does not override them.

    ``to_prompt`` extracts the prompt and ``workspace_factory`` creates each prompt's
    workspace. With ``own=True``, :meth:`aclose` closes the execution services of
    every collector the hook created, once per distinct service; pass ``False`` when an
    enclosing context owns them. Model clients and critics stay with the factory's
    owner. Close the hook after all active calls finish.

    Actor and critic factories customize exported records. ``min_abs_advantage``
    filters actor edges during export without changing collection or critic targets.
    """
    cached: tuple[Any, Collector] | None = None
    creation = asyncio.Lock()
    owned: list[Collector] = []

    async def generate(
        args: Any, sample: Any, sampling_params: Any = None
    ) -> list[Any]:
        nonlocal cached
        async with creation:
            if cached is None or cached[0] is not args:
                cached = (args, collector_factory(args))
                if own:
                    owned.append(cached[1])
            collector = cached[1]
        # Another call may replace the cache while this call awaits search.
        index = getattr(sample, "index", 0)
        # Each prompt receives its own workspace; tree branches fork that state.
        workspace = None if workspace_factory is None else workspace_factory(sample)
        batch = await collector.collect(to_prompt(sample), workspace=workspace)
        samples = batch.to_records(
            index,
            actor_factory=sample_factory,
            critic_factory=critic_sample_factory,
            min_abs_advantage=min_abs_advantage,
        )
        # Warn when a prompt contributes no samples; include collection failures.
        logger.log(
            logging.WARNING if not samples else logging.INFO,
            (
                "rollout finished group=%s samples=%d actor=%d critic=%d stats=%s "
                + "failures=%d"
            ),
            index,
            len(samples),
            len(batch.actor),
            len(batch.critic),
            dict(batch.stats),
            len(batch.failures),
        )
        return samples

    async def aclose() -> None:
        nonlocal cached
        async with creation:
            # Collectors built for different args may share one execution service;
            # close each service once, through a collector that uses it.
            closing = {id(c.agent.services): c for c in owned}.values()
            owned.clear()
            cached = None
            failures: list[Exception] = []
            for collector in closing:
                try:
                    await collector.aclose()
                except Exception as exc:
                    failures.append(exc)
            if failures:
                raise ExceptionGroup("Slime collector cleanup failed", failures)

    generate.__dict__["aclose"] = aclose
    return cast(GenerateHook, generate)


__all__ = ["ESTIMATOR_KEY", "LANE_KEY", "make_generate"]
