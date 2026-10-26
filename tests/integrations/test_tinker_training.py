"""The Tinker integration: next-token data, one update, and the one-step async loop.

The loop tests replace collection and the update with fakes that record ordering; the
data and update tests bind calls to the installed Tinker SDK's signatures.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from focus.integrations import Problem
from focus.integrations import tinker as tinker_integration
from focus.integrations.tinker import (
    CollectedBatch,
    next_token_inputs,
    train,
)

ROW = Problem("Solve 3x+7=22", r"\boxed{5}", "linear")


def _actor(reward: float = 0.3) -> dict[str, Any]:
    return {
        "tokens": [5, 6, 7, 8],
        "loss_mask": [False, False, True, True],
        "logprobs": [0.0, 0.0, -0.1, -0.2],
        "reward": reward,
        "metadata": {"lane": "actor"},
    }


def test_next_token_inputs_shift_targets_probabilities_and_advantages() -> None:
    inputs, loss = next_token_inputs(_actor())
    assert inputs == [5, 6, 7]
    assert loss == {
        "target_tokens": [6, 7, 8],
        "logprobs": [0.0, -0.1, -0.2],
        "advantages": [0.0, 0.3, 0.3],
    }
    for bad in (
        {**_actor(), "metadata": {"lane": "critic"}},
        {**_actor(), "loss_mask": [True, False, True, True]},
        {**_actor(), "logprobs": [0.0]},
    ):
        with pytest.raises(ValueError, match=r"actor lane|conditioning|aligned"):
            next_token_inputs(bad)


def test_to_datum_builds_tinker_s_record() -> None:
    pytest.importorskip("tinker")
    datum = tinker_integration.to_datum(_actor())
    assert datum.model_input.to_ints() == [5, 6, 7]
    assert list(datum.loss_fn_inputs["advantages"].data) == pytest.approx(
        [0.0, 0.3, 0.3]
    )


async def test_update_waits_for_gradients_before_the_optimizer() -> None:
    tinker = pytest.importorskip("tinker")
    events: list[Any] = []

    class Future:
        def __init__(self, label: str) -> None:
            self.label = label

        async def result_async(self) -> Any:
            events.append(self.label)
            return SimpleNamespace(metrics={})

    class Training:
        async def forward_backward_async(self, data, **kwargs):
            inspect.signature(tinker.TrainingClient.forward_backward_async).bind(
                self, data, **kwargs
            )
            events.append(("backward", len(data), kwargs["loss_fn"]))
            return Future("gradients ready")

        async def optim_step_async(self, params):
            inspect.signature(tinker.TrainingClient.optim_step_async).bind(self, params)
            events.append("optimizer submitted")
            return Future("weights ready")

    critic = {"metadata": {"lane": "critic"}}
    await tinker_integration.update_actor(
        Training(), [_actor(), critic, _actor(-0.2)], learning_rate=1e-5
    )
    assert events == [
        ("backward", 2, "ppo"),
        "gradients ready",
        "optimizer submitted",
        "weights ready",
    ]
    with pytest.raises(ValueError, match="No actor segments"):
        await tinker_integration.update_actor(Training(), [critic], learning_rate=1e-5)


class Loop:
    """A fake learner and collector that record what the loop asked for, in order."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.weights = 0
        self.snapshots: list[int] = []
        self.active = 0
        self.peak = 0

    def training_client(self) -> Any:
        owner = self

        class Training:
            def save_weights_and_get_sampling_client(self) -> Any:
                owner.snapshots.append(owner.weights)
                return SimpleNamespace(step=owner.weights)

        return Training()

    async def batch(self, sampler, problem, index, behavior_step, **kwargs):
        assert sampler.step == behavior_step
        assert problem is ROW
        self.events.append(("collect", index, sampler.step))
        record = {
            "metadata": {
                "lane": "critic",
                "training": {"batch_index": index, "behavior_step": behavior_step},
            },
            "target": 0.5,
        }
        return CollectedBatch(index, behavior_step, [record])

    async def update(self, training, records, **kwargs):
        self.events.append(("update", self.weights))
        self.weights += 1
        return SimpleNamespace(metrics={"loss": 0.1})

    async def run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        steps: int = 4,
        depth: int = 1,
        collector: Any = None,
        updater: Any = None,
    ) -> tuple[list[Any], list[Any]]:
        monkeypatch.setattr(
            tinker_integration, "_collect_batch", collector or self.batch
        )
        monkeypatch.setattr(tinker_integration, "update_actor", updater or self.update)
        sunk: list[Any] = []
        reports = await asyncio.wait_for(
            train(
                self.training_client(),
                model="test",
                problems=[ROW],
                collector_for=lambda policy, problem: None,  # type: ignore[arg-type,return-value]
                steps=steps,
                async_depth=depth,  # type: ignore[arg-type]
                learning_rate=1e-5,
                critic_sink=sunk.extend,
            ),
            timeout=5,
        )
        return reports, sunk


async def test_one_step_overlap_is_bounded_and_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = Loop()
    started = [asyncio.Event() for _ in range(4)]
    updated = [asyncio.Event() for _ in range(4)]

    async def collector(sampler, problem, index, behavior_step, **kwargs):
        loop.active += 1
        loop.peak = max(loop.peak, loop.active)
        try:
            result = await loop.batch(sampler, problem, index, behavior_step)
            started[index].set()
            if index:
                await updated[index - 1].wait()
            # An optimizer update never changes the in-flight sampling snapshot.
            assert sampler.step == result.behavior_step
            return result
        finally:
            loop.active -= 1

    async def updater(*args, **kwargs):
        step = loop.weights
        if step < 3:
            # Fails by timeout if collection and training are sequential.
            await started[step + 1].wait()
        result = await loop.update(*args, **kwargs)
        updated[step].set()
        return result

    reports, sunk = await loop.run(monkeypatch, collector=collector, updater=updater)
    assert [e for e in loop.events if e[0] == "collect"] == [
        ("collect", 0, 0),
        ("collect", 1, 0),
        ("collect", 2, 1),
        ("collect", 3, 2),
    ]
    assert loop.snapshots == [0, 1, 2]
    assert loop.weights == 4
    assert (loop.peak, loop.active) == (1, 0)
    assert [r["metadata"]["training"]["learner_step"] for r in sunk] == [0, 1, 2, 3]
    assert [r.policy_lag for r in reports] == [0, 1, 1, 1]
    assert reports[0].metrics == {"loss": 0.1}


async def test_fast_collection_cannot_run_more_than_one_batch_ahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = Loop()

    async def updater(*args, **kwargs):
        await asyncio.sleep(0)
        collections = [e for e in loop.events if e[0] == "collect"]
        assert len(collections) == min(loop.weights + 2, 4)
        return await loop.update(*args, **kwargs)

    await loop.run(monkeypatch, updater=updater)
    assert loop.weights == 4


async def test_synchronous_mode_refreshes_before_each_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = Loop()
    await loop.run(monkeypatch, depth=0)
    assert [e for e in loop.events if e[0] == "collect"] == [
        ("collect", i, i) for i in range(4)
    ]


async def test_single_step_has_no_unused_collection_or_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = Loop()
    await loop.run(monkeypatch, steps=1)
    assert loop.events == [("collect", 0, 0), ("update", 0)]
    assert loop.snapshots == [0]


async def test_training_failure_cancels_and_joins_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = Loop()
    started, closed = asyncio.Event(), asyncio.Event()

    async def collector(sampler, problem, index, behavior_step, **kwargs):
        if index == 0:
            return await loop.batch(sampler, problem, index, behavior_step)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def fail(*args, **kwargs):
        await started.wait()
        raise RuntimeError("optimizer failed")

    with pytest.raises(RuntimeError, match="optimizer failed"):
        await loop.run(monkeypatch, collector=collector, updater=fail)
    assert closed.is_set()
    assert loop.weights == 0


async def test_collection_failure_stops_before_another_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = Loop()

    async def collector(sampler, problem, index, behavior_step, **kwargs):
        if index == 1:
            raise RuntimeError("collection failed")
        return await loop.batch(sampler, problem, index, behavior_step)

    with pytest.raises(RuntimeError, match="collection failed"):
        await loop.run(monkeypatch, collector=collector)
    assert loop.weights == 1


@pytest.mark.parametrize(("index", "step"), [(99, 0), (0, -2), (0, 1)])
async def test_stale_or_out_of_order_batch_is_rejected(
    monkeypatch: pytest.MonkeyPatch, index: int, step: int
) -> None:
    loop = Loop()

    async def collector(*args, **kwargs):
        return CollectedBatch(index, step, [])

    with pytest.raises(ValueError, match="Unexpected batch or policy lag"):
        await loop.run(monkeypatch, collector=collector)
    assert loop.weights == 0


async def test_caller_cancellation_joins_prefetched_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = Loop()
    started, closed = asyncio.Event(), asyncio.Event()

    async def collector(sampler, problem, index, behavior_step, **kwargs):
        if index == 0:
            return await loop.batch(sampler, problem, index, behavior_step)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def updater(*args, **kwargs):
        await asyncio.Event().wait()

    task = asyncio.create_task(
        loop.run(monkeypatch, collector=collector, updater=updater)
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


async def test_invalid_loop_settings_are_refused() -> None:
    invalid: list[dict[str, Any]] = [{"steps": 0}, {"async_depth": 2}, {"problems": []}]
    for kwargs in invalid:
        settings: dict[str, Any] = {
            "model": "m",
            "problems": [ROW],
            "collector_for": lambda policy, problem: None,
            "steps": 1,
            "learning_rate": 1e-5,
            **kwargs,
        }
        with pytest.raises(ValueError, match="positive steps"):
            await train(object(), **settings)
