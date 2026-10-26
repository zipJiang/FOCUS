"""Collector: one call from a prompt to a training batch, and who owns what."""

from __future__ import annotations

import asyncio

import pytest

from focus import Collector, DirectBranchTd, Fifo, Phase, RewardConfig, Search
from focus.execution import ExecutionServices
from focus.extend import InMemoryWorkspace
from focus.generation.parsing import HermesToolCallParser
from focus.harness import StorageTool
from focus.preparation import Estimator
from tests.helpers import DemoValue, ScriptedGenerator, SubmitEnv, tool_call, toy_runner


def collector(*, storage=False, version="snapshot-7", **kwargs):
    script = (
        [tool_call("storage", operate="put", key="note", value="evidence")]
        if storage
        else []
    )
    script += [tool_call("submit", answer="42")]
    generator = ScriptedGenerator(
        ["r</think>" + text for text in script], logprobs=-0.25
    )
    kwargs.setdefault("critic", DemoValue())
    return Collector(
        toy_runner(
            generator,
            env=SubmitEnv([StorageTool()] if storage else []),
            parser=HermesToolCallParser(),
            version=version,
        ),
        search=Search([Phase(Fifo(), 2, max_children=1)], max_concurrency=1),
        reward=RewardConfig(beta_step=0.5),
        **kwargs,
    )


@pytest.mark.parametrize("estimator", [DirectBranchTd(), None])
async def test_collection_keeps_lanes_separate(estimator):
    async with collector(estimator=estimator) as run:
        batch = await run.collect("question")
    assert batch.reward_config is run.reward
    assert batch.critic
    assert bool(batch.actor) == (estimator is not None)
    assert not batch.failures
    samples = batch.to_records(12)
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


def test_the_paper_recipe_is_the_default_estimator():
    assert isinstance(collector().estimator, DirectBranchTd)
    assert collector(estimator=None).estimator is None


async def test_collection_accepts_a_workspace_and_existing_checkpoint():
    workspace = InMemoryWorkspace()
    async with collector(storage=True) as run:
        root = await run.agent.start("question", workspace=workspace)
        batch = await run.collect(root)
    text = "\n".join(
        run.agent.policy.decode(record["tokens"])
        for record in batch.to_records(0)
        if record["metadata"]["lane"] == "actor"
    )
    assert "evidence" in text
    # Collection branches from the original checkpoint; it does not mutate it.
    assert not root.workspace.exists("note")


async def test_custom_estimator_and_export_factories_still_apply():
    class FixedWeight(Estimator):
        name = "custom-weight"

        def advantages(self, edges, values):
            return [0.125 for _ in edges]

    async with collector(estimator=FixedWeight()) as run:
        batch = await run.collect("question")
    actors = batch.to_records(
        0,
        actor_factory=lambda **fields: (fields["lane"], fields["reward"]),
        critic_factory=lambda **fields: ("critic", fields["target"]),
    )
    assert ("actor", 0.125) in actors
    assert any(lane == "critic" for lane, _ in actors)


async def test_one_tree_prepares_under_several_recipes():
    async with collector() as run:
        tree = await run.search("question")
    paper = run.prepare(tree)
    critic_only = run.prepare(tree, estimator=None)
    assert paper.actor
    assert not critic_only.actor
    assert paper.critic == critic_only.critic
    assert tree.describe(paper).startswith("node   0")
    assert "value" in paper.describe()


async def test_a_collector_closes_what_it_owns():
    closed = []

    class Services(ExecutionServices):
        async def aclose(self):
            closed.append("services")
            await super().aclose()

    class Closing(DemoValue):
        async def aclose(self):
            closed.append("critic")

    run = collector(critic=Closing())
    run.agent.services = Services()
    async with run:
        first = await run.collect("first")
        second = await run.collect("second")
    assert first.critic
    assert second.critic
    assert closed == ["services", "critic"]


async def test_a_borrowing_collector_leaves_shared_services_open():
    closed = []

    class Services(ExecutionServices):
        async def aclose(self):
            closed.append("services")
            await super().aclose()

    run = collector(own=False)
    run.agent.services = Services()
    async with run:
        await run.collect("question")
    assert closed == []
    await run.aclose()
    assert closed == ["services"]


async def test_collection_preserves_failure_statistics_for_the_callers_decision():
    from tests.helpers import DeadReward

    async with collector(critic=DeadReward(), estimator=None) as run:
        batch = await run.collect("question")
    assert batch.stats["score_failures"] > 0
    assert not batch.actor


async def test_two_policy_snapshots_share_services_and_keep_their_own_logprobs():
    """One-batch asynchronous training collects with the previous snapshot while the
    next one is published. Both snapshots carry the same model name and label, share
    one execution service, and each batch exports its own behavior logprobs."""
    services = ExecutionServices()

    def snapshot(logprob: float) -> Collector:
        generator = ScriptedGenerator(["go", "go", "stop"], logprobs=logprob)
        return Collector(
            toy_runner(generator, services=services),
            critic=DemoValue(),
            search=Search([Phase(Fifo(), 4, max_children=2)], max_concurrency=2),
            own=False,
        )

    old, new = snapshot(-0.25), snapshot(-0.75)
    assert old.agent.policy.version == new.agent.policy.version
    async with services:
        batches = await asyncio.gather(old.collect("old"), new.collect("new"))
    for batch, expected in zip(batches, (-0.25, -0.75), strict=True):
        actor = [s for s in batch.to_records(0) if s["metadata"]["lane"] == "actor"]
        assert actor
        assert {
            lp
            for sample in actor
            for lp, keep in zip(sample["logprobs"], sample["loss_mask"], strict=True)
            if keep
        } == {expected}
