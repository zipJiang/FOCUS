"""Adapt FOCUS collection and export to Slime's generation callback."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol, cast

from focus.config import RolloutConfig, RolloutSpec
from focus.export import CriticSampleFactory, SampleFactory, to_samples
from focus.harness.workspace import Workspace
from focus.loop import Runtime, collect_samples
from focus.preparation import AdvantageEstimator
from focus.preparation.records import ESTIMATOR_KEY, LANE_KEY

logger = logging.getLogger(__name__)


class GenerateHook(Protocol):
    async def __call__(
        self, args: Any, sample: Any, sampling_params: Any = None
    ) -> list[Any]: ...
    async def aclose(self) -> None: ...


def make_generate(
    spec: RolloutSpec,
    runtime_factory: Callable[[Any], Runtime],
    *,
    estimator: AdvantageEstimator | None,
    own_services: bool = True,
    to_prompt: Callable[[Any], Any] = lambda sample: sample,
    workspace_factory: Callable[[Any], Workspace] | None = None,
    sample_factory: SampleFactory[Any] | None = None,
    critic_sample_factory: CriticSampleFactory[Any] | None = None,
    min_abs_advantage: float | None = None,
) -> GenerateHook:
    """Build Slime's generation callback from a runtime factory.

    The factory supplies the runner, model clients, execution services, and optional
    scorers. The hook applies spec.config to a new Runtime record, then collects and
    exports one group per sample. It preserves the factory's collaborators.

    to_prompt extracts the prompt; workspace_factory creates each prompt's workspace.
    Sampling comes from Runner and RolloutConfig.proposal. Slime's sampling_params
    argument is accepted by the callback but does not override those settings.

    With own_services=True, aclose closes the execution services returned by the
    factory’s runner. Use False when an enclosing context owns shared services.
    Model-client lifetime follows the executor's own_clients setting. Close the
    hook after all active calls finish.

    Actor and critic factories customize exported records. min_abs_advantage filters
    actor edges during export without changing collection or critic targets.
    """
    cached: tuple[Any, Runtime] | None = None
    creation = asyncio.Lock()
    owned: list[Runtime] = []

    async def generate(
        args: Any, sample: Any, sampling_params: Any = None
    ) -> list[Any]:
        nonlocal cached
        async with creation:
            if cached is None or cached[0] is not args:
                runtime = runtime_factory(args)
                cached = (args, replace(runtime, config=spec.config))
                owned.append(cached[1])
            runtime = cached[1]
        # Another call may replace the cache while this call awaits search.
        index = getattr(sample, "index", 0)
        # Each prompt receives its own workspace; tree branches fork that state.
        workspace = None if workspace_factory is None else workspace_factory(sample)
        result = await collect_samples(
            to_prompt(sample), runtime, estimator=estimator, workspace=workspace
        )
        samples = to_samples(
            result,
            group_index=index,
            sample_factory=sample_factory,
            critic_sample_factory=critic_sample_factory,
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
            len(result.actor),
            len(result.critic),
            dict(result.stats),
            len(result.failures),
        )
        return samples

    async def aclose() -> None:
        nonlocal cached
        async with creation:
            services = (
                {
                    id(runtime.runner.services): runtime.runner.services
                    for runtime in owned
                }
                if own_services
                else {}
            )
            owned.clear()
            cached = None
            failures: list[Exception] = []
            for service in services.values():
                try:
                    await service.aclose()
                except Exception as exc:
                    failures.append(exc)
            if failures:
                raise ExceptionGroup("Slime service cleanup failed", failures)

    generate.__dict__["aclose"] = aclose
    return cast(GenerateHook, generate)


__all__ = [
    "ESTIMATOR_KEY",
    "LANE_KEY",
    "RolloutConfig",
    "RolloutSpec",
    "make_generate",
    "to_samples",
]
