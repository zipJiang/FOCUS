from dataclasses import replace

import pytest

from focus import (
    AdvantageEstimator,
    GrpoEstimator,
    RolloutConfig,
    Runtime,
    VinePpoConfig,
    VinePpoEstimator,
    build,
    prepare_samples,
    run_search,
    to_samples,
)
from focus.generation import SamplingParams
from focus.scheduler import MixtureProposal
from focus.vine import VINE_READY, VINE_VALUE
from tests.helpers import ScriptedGenerator
from tests.test_loop import _runner


def runtime(group_size=2, value_rollouts_per_state=1, completions=None):
    generator = ScriptedGenerator(
        completions
        or [
            "stop-probe-root",  # the shared root probe starts before any actor
            "go",
            "stop-good",
            "go",
            "stop-bad",  # retained trajectories
            "stop-probe-left",
            "stop-good",  # disposable probes, in checkpoint order
        ],
        logprobs=-0.25,
    )
    return Runtime(
        runner=_runner(generator),
        config=RolloutConfig(
            vine=VinePpoConfig(
                group_size=group_size,
                value_rollouts_per_state=value_rollouts_per_state,
            ),
            max_concurrency=1,
        ),
    ), generator


async def test_independent_probes_only_and_stepwise_advantages(monkeypatch):
    rt, gen = runtime()
    horizons = []
    original = rt.runner.run

    async def track(rs, **kwargs):
        horizons.append(kwargs["max_steps"])
        return await original(rs, **kwargs)

    monkeypatch.setattr(rt.runner, "run", track)
    state = await run_search("question", rt)
    assert len(state.nodes) == 5  # root plus two retained two-edge trajectories
    assert state.nodes[0].metadata[VINE_VALUE] == 0.5  # fresh probe, not group mean
    assert horizons == [4, 2, 2]  # never grant a fresh horizon at an intermediate state
    batch = prepare_samples(
        state,
        estimator=VinePpoEstimator(),
        reward_config=rt.config.reward_config,
        behavior_version=rt.runner.policy.version,
    )
    assert [r.weight for r in batch.actor] == [0, 1.5, 1.5, -1.5]
    assert not batch.critic
    assert state.stats["rollouts"] == 5
    assert state.stats["turns"] == 7
    assert state.stats["vine_actor_rollouts"] == 2
    assert state.stats["vine_value_rollouts"] == 3
    assert state.stats["vine_actor_turns"] == 4
    assert state.stats["vine_value_turns"] == 3
    assert state.stats["vine_trainable_edges"] == 4
    assert state.stats["vine_actor_generated_tokens"] == sum(
        map(len, ["go", "stop-good", "go", "stop-bad"])
    )
    assert state.stats["vine_value_generated_tokens"] == sum(
        map(len, ["stop-probe-root", "stop-probe-left", "stop-good"])
    )
    assert len(gen.prefixes) == 7
    text = "".join(
        rt.runner.policy.decode(t.tokens)
        for n in state.nodes.values()
        for t in n.payload.turns
    )
    assert "probe" not in text
    samples = to_samples(batch, 0)
    emitted = [
        token
        for s in samples
        for token, mask in zip(s["tokens"], s["loss_mask"], strict=True)
        if mask
    ]
    assert rt.runner.policy.decode(emitted) == "gostop-goodgostop-bad"
    assert all(r.importance == 1 for r in batch.actor)
    # Preparation is repeatable and cannot repurpose probes for another estimator.
    again = prepare_samples(
        state,
        estimator=VinePpoEstimator(),
        reward_config=rt.config.reward_config,
        behavior_version=rt.runner.policy.version,
    )
    assert [r.weight for r in again.actor] == [r.weight for r in batch.actor]
    with pytest.raises(ValueError, match="require the vine_ppo"):
        prepare_samples(
            state,
            estimator=GrpoEstimator(),
            reward_config=rt.config.reward_config,
            behavior_version=rt.runner.policy.version,
        )


def test_config_registry_and_rejections():
    config = build(RolloutConfig, {"vine": {"group_size": 3}, "max_concurrency": 2})
    assert config.vine == VinePpoConfig(3)
    assert build(AdvantageEstimator, {"name": "vine_ppo"}).name == "vine_ppo"
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            build(VinePpoConfig, {"group_size": invalid})
        with pytest.raises(ValueError, match="positive integer"):
            build(VinePpoConfig, {"value_rollouts_per_state": invalid})
    with pytest.raises(ValueError, match="gamma=1"):
        replace(config, reward_config=replace(config.reward_config, gamma=0.9))
    with pytest.raises(ValueError, match="same anchor"):
        replace(config, proposal=MixtureProposal(SamplingParams(), epsilon=0.5))
    with pytest.raises(ValueError, match="concurrency"):
        replace(config, max_concurrency=0)


async def test_value_failure_never_becomes_zero_or_a_partial_batch(monkeypatch):
    rt, _ = runtime()

    async def broken(*args, **kwargs):
        raise RuntimeError("probe backend failed")

    monkeypatch.setattr(rt.runner, "run", broken)
    with pytest.raises(ExceptionGroup, match="TaskGroup") as caught:
        await run_search("q", rt)
    assert "probe backend failed" in str(caught.value.exceptions[0])


async def test_truncated_main_path_is_not_treated_as_an_outcome():
    rt, _ = runtime()
    rt.runner._max_steps = 2
    rt.runner._finish_prompt = None
    with pytest.raises(ExceptionGroup, match="TaskGroup") as caught:
        await run_search("q", rt)
    assert "truncated" in str(caught.value.exceptions[0])


async def test_incomplete_or_missing_values_cannot_be_exported():
    rt, _ = runtime()
    state = await run_search("q", rt)
    state.nodes[0].metadata[VINE_READY] = False
    with pytest.raises(ValueError, match="completed independent"):
        prepare_samples(
            state,
            estimator=VinePpoEstimator(),
            reward_config=rt.config.reward_config,
            behavior_version=rt.runner.policy.version,
        )
    state.nodes[0].metadata[VINE_READY] = True
    del state.nodes[1].metadata[VINE_VALUE]
    with pytest.raises(ValueError, match="missing"):
        prepare_samples(
            state,
            estimator=VinePpoEstimator(),
            reward_config=rt.config.reward_config,
            behavior_version=rt.runner.policy.version,
        )


async def test_group_size_controls_actor_data_and_probe_cost():
    rt, _ = runtime(group_size=1)
    state = await run_search("q", rt)
    assert len(state.nodes) == 3
    assert state.stats["vine_actor_rollouts"] == 1
    assert state.stats["vine_value_rollouts"] == 2
    assert state.stats["rollouts"] == 3


async def test_multiple_value_rollouts_are_averaged_and_discarded():
    rt, _ = runtime(
        value_rollouts_per_state=2,
        completions=[
            "stop-good",
            "stop-bad",  # root mean: 1.25
            "go",
            "stop-good",
            "go",
            "stop-bad",
            "stop-good",
            "stop-good",  # first successor mean: 2.0
            "stop-bad",
            "stop-bad",  # second successor mean: 0.5
        ],
    )
    state = await run_search("q", rt)
    live = [node for node in state.nodes.values() if not node.payload.done]
    assert [node.metadata[VINE_VALUE] for node in live] == [1.25, 2.0, 0.5]
    assert state.stats["vine_value_rollouts"] == 6
    batch = prepare_samples(
        state,
        estimator=VinePpoEstimator(),
        reward_config=rt.config.reward_config,
        behavior_version=rt.runner.policy.version,
    )
    assert all(
        record.diagnostics["parent_value_rollouts"] == 2.0 for record in batch.actor
    )


async def test_bounded_sampling_and_sibling_cancellation():
    import asyncio

    from focus.vine import _bounded_map

    active = peak = finished = 0

    async def work(index):
        nonlocal active, peak, finished
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            if index == 2:
                raise RuntimeError("broken")
            await asyncio.sleep(0)
            finished += 1
            return index
        finally:
            active -= 1

    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await _bounded_map(list(range(20)), work, 2)
    assert active == 0
    assert peak == 2
    assert finished < 20


async def test_probes_start_while_actor_paths_are_still_running():
    import asyncio

    from tests.helpers import ScriptedGenerator

    order = []
    release = asyncio.Event()

    class Gated(ScriptedGenerator):
        async def agenerate_tokens(self, prefix_tokens, sampling_params=None):
            result = await super().agenerate_tokens(prefix_tokens, sampling_params)
            order.append(result.text)
            if result.text == "stop-bad":
                # The actor's final step waits until an intermediate probe has run.
                await release.wait()
            if result.text == "stop-probe-left":
                release.set()
            return result

    rt, _ = runtime(group_size=1)
    gen = Gated(
        ["stop-probe-root", "go", "stop-bad", "stop-probe-left"], logprobs=-0.25
    )
    rt = Runtime(
        runner=_runner(gen),
        config=replace(rt.config, max_concurrency=2),
    )
    state = await asyncio.wait_for(run_search("q", rt), timeout=5)
    # The probe from the first post-fold checkpoint finished before the actor did.
    assert order.index("stop-probe-left") < len(order)
    assert release.is_set()
    assert len(state.nodes) == 3
    assert state.stats["vine_value_rollouts"] == 2
