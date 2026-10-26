from __future__ import annotations

import asyncio

import pytest

from focus import (
    DirectBranchTdEstimator,
    ExecutionServices,
    FifoAllocator,
    InMemoryWorkspace,
    RewardConfig,
    RolloutConfig,
    Runtime,
    SearchPass,
    WidthGating,
    collect_samples,
    to_samples,
)
from focus.generation.parsing import HermesToolCallParser
from focus.harness import StorageTool
from focus.preparation import AdvantageEstimator
from tests.helpers import DemoValue, ScriptedGenerator, SubmitEnv, tool_call, toy_runner


def runtime(*, storage=False, version="snapshot-7"):
    script = (
        [tool_call("storage", operate="put", key="note", value="evidence")]
        if storage
        else []
    )
    script += [tool_call("submit", answer="42")]
    generator = ScriptedGenerator(
        ["r</think>" + text for text in script], logprobs=-0.25
    )
    return Runtime(
        runner=toy_runner(
            generator,
            env=SubmitEnv([StorageTool()] if storage else []),
            parser=HermesToolCallParser(),
            version=version,
        ),
        value_model=DemoValue(),
        config=RolloutConfig(
            passes=(SearchPass(FifoAllocator(), budget=2, gating=WidthGating(1)),),
            reward_config=RewardConfig(beta_step=0.5),
            max_concurrency=1,
        ),
    )


@pytest.mark.parametrize("estimator", [DirectBranchTdEstimator(), None])
async def test_collection_uses_runtime_versions_and_keeps_lanes_separate(estimator):
    run = runtime()
    async with run.runner.services:
        batch = await collect_samples("question", run, estimator=estimator)
    assert batch.reward_config is run.config.reward_config
    assert batch.critic
    assert bool(batch.actor) == (estimator is not None)
    assert not batch.failures
    samples = to_samples(batch, group_index=12)
    assert all(sample["group_index"] == 12 for sample in samples)
    for sample in samples:
        if sample["metadata"]["lane"] == "actor":
            assert all(
                lp == -0.25
                for lp, keep in zip(
                    sample["logprobs"], sample["loss_mask"], strict=True
                )
                if keep
            )
        else:
            assert "context" in sample
            assert "target" in sample


async def test_collection_accepts_a_workspace_and_existing_checkpoint():
    run = runtime(storage=True)
    workspace = InMemoryWorkspace()
    async with run.runner.services:
        root = await run.runner.start("question", workspace=workspace)
        batch = await collect_samples(root, run, estimator=DirectBranchTdEstimator())
    text = "\n".join(
        run.runner.policy.decode(record["tokens"])
        for record in to_samples(batch, group_index=0)
        if record["metadata"]["lane"] == "actor"
    )
    assert "evidence" in text
    # Collection branches from the original checkpoint; it does not mutate it.
    assert not root.workspace.exists("note")


async def test_custom_estimator_and_export_factories_still_apply():
    class FixedWeight(AdvantageEstimator):
        name = "custom-weight"

        def assign(self, records, *, view, reward_config):
            for record in records:
                record.weight = 0.125

    run = runtime()
    async with run.runner.services:
        batch = await collect_samples("question", run, estimator=FixedWeight())
    actors = to_samples(
        batch,
        group_index=0,
        sample_factory=lambda **fields: (fields["lane"], fields["reward"]),
        critic_sample_factory=lambda **fields: ("critic", fields["target"]),
    )
    assert ("actor", 0.125) in actors
    assert any(lane == "critic" for lane, _ in actors)


async def test_caller_owns_services_across_multiple_collections():
    run = runtime()
    # Repeated collection can share execution services; the helper does not close them.
    async with run.runner.services:
        first = await collect_samples("first", run, estimator=None)
        second = await collect_samples("second", run, estimator=None)
    assert first.critic
    assert second.critic


async def test_collection_preserves_failure_statistics_for_the_callers_decision():
    from tests.helpers import DeadReward

    run = runtime()
    run.value_model = DeadReward()
    async with run.runner.services:
        batch = await collect_samples("question", run, estimator=None)
    assert batch.stats["score_failures"] > 0
    assert not batch.actor


async def test_two_policy_snapshots_share_services_and_keep_their_own_logprobs():
    """One-batch asynchronous training collects with the previous snapshot while the
    next one is published. Both snapshots carry the same model name and label, share
    one execution service, and each batch exports its own behavior logprobs."""
    services = ExecutionServices()

    def snapshot(logprob: float) -> Runtime:
        generator = ScriptedGenerator(["go", "go", "stop"], logprobs=logprob)
        return Runtime(
            runner=toy_runner(generator, services=services),
            value_model=DemoValue(),
            config=RolloutConfig(
                passes=(SearchPass(FifoAllocator(), budget=4, gating=WidthGating(2)),),
                max_concurrency=2,
            ),
        )

    old, new = snapshot(-0.25), snapshot(-0.75)
    assert old.runner.policy.version == new.runner.policy.version
    async with services:
        batches = await asyncio.gather(
            collect_samples("old", old, estimator=DirectBranchTdEstimator()),
            collect_samples("new", new, estimator=DirectBranchTdEstimator()),
        )
    for batch, expected in zip(batches, (-0.25, -0.75), strict=True):
        actor = [s for s in to_samples(batch, 0) if s["metadata"]["lane"] == "actor"]
        assert actor
        assert {
            lp
            for sample in actor
            for lp, keep in zip(sample["logprobs"], sample["loss_mask"], strict=True)
            if keep
        } == {expected}
