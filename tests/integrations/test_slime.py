"""The slime adapter: the generate callback built from a collector factory.

Flattening a result into trainer samples is the export's, not this module's -- see
``tests/test_export.py`` -- so the results here are the smallest thing
``make_generate`` can hand back.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from focus import Collector, Fifo, Phase, Search
from focus.codec import ChatCodec
from focus.generation.parsing import HermesToolCallParser
from focus.harness import (
    InMemoryWorkspace,
    PackedSequence,
    StorageTool,
)
from focus.integrations.slime import make_generate
from focus.preparation.records import ActorSample, Batch, CriticSample
from tests.helpers import (
    DemoValue,
    FakeChatTokenizer,
    ScriptedGenerator,
    SubmitEnv,
    tool_call,
    toy_runner,
)


def _collector(
    generator: ScriptedGenerator,
    *,
    tools: list[Any] | None = None,
    phases: int = 2,
    width: int = 3,
) -> Collector:
    """A collector over a submit-ending tool environment, small enough to run here."""
    return Collector(
        toy_runner(
            generator,
            env=SubmitEnv(tools or []),
            parser=HermesToolCallParser(),
        ),
        critic=DemoValue(),
        search=Search([Phase(Fifo(), phases, max_children=width)], max_concurrency=1),
        own=False,
    )


def _result() -> Batch:
    """One actor record and one critic record: these tests only count what came back."""
    span = PackedSequence(tokens=(0, 1), trainable=(False, True), logprobs=(0.0, -0.1))
    return Batch(
        actor=(
            ActorSample(
                node_id=1,
                spans=(span,),
                group_id=0,
                estimator="grpo",
            ),
        ),
        critic=(
            CriticSample(
                node_id=1,
                context="checkpoint",
                target=1.0,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_make_generate_builds_one_collector_per_actor_and_returns_a_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: list[object] = []

    class FakeArgs:
        pass

    class FakeSample:
        index = 9

    async def fake_collect(self, prompt, *, workspace=None):
        del prompt, workspace
        return _result()

    def factory(args: object) -> Collector:
        builds.append(args)
        return _collector(ScriptedGenerator(["stop"]))

    monkeypatch.setattr(Collector, "collect", fake_collect)
    generate = make_generate(factory)
    args = FakeArgs()
    first = await generate(args, FakeSample())
    second = await generate(args, FakeSample())

    assert len(builds) == 1
    assert first
    assert second
    assert all(s["group_index"] == 9 for s in first)


@pytest.mark.asyncio
async def test_concurrent_args_keep_the_collector_that_generated_each_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_ready = asyncio.Event()
    release_first = asyncio.Event()
    search = Collector.search

    async def paused_search(self, prompt, *, workspace=None):
        state = await search(self, prompt, workspace=workspace)
        if prompt == "first":
            first_ready.set()
            await release_first.wait()
        return state

    def factory(args: SimpleNamespace) -> Collector:
        generator = ScriptedGenerator(
            ["r</think>" + tool_call("submit", answer="42")],
            logprobs=args.logprob,
        )
        return _collector(generator, phases=1, width=1)

    monkeypatch.setattr(Collector, "search", paused_search)
    generate = make_generate(factory)
    async with asyncio.TaskGroup() as tasks:
        first = tasks.create_task(generate(SimpleNamespace(logprob=-0.1), "first"))
        await asyncio.wait_for(first_ready.wait(), timeout=5)
        try:
            second = await generate(SimpleNamespace(logprob=-0.2), "second")
        finally:
            release_first.set()

    for samples, expected in ((first.result(), -0.1), (second, -0.2)):
        assert samples
        assert {s["metadata"]["lane"] for s in samples} == {"actor", "critic"}
        for sample in samples:
            if sample["metadata"]["lane"] == "critic":
                continue
            generated = [
                lp
                for lp, keep in zip(
                    sample["logprobs"], sample["loss_mask"], strict=True
                )
                if keep
            ]
            assert generated
            assert set(generated) == {expected}


# -- the workspace: one per prompt, and only when the env has tools for one -----------


def _transcripts(samples: list[Any]) -> str:
    """Every sample's tokens, decoded -- i.e. the conversations that were exported.

    A tool result is not a field on a sample; it is text in the prefix the next turn was
    conditioned on, and the fake tokenizer round-trips ``ord``/``chr``, so reading it
    back is the same decode a trainer's own inspection would do.
    """
    codec = ChatCodec(FakeChatTokenizer())
    return "\n".join(
        codec.decode(sample["tokens"])
        for sample in samples
        if sample["metadata"]["lane"] == "actor"
    )


def _storage_generate(**kwargs: Any) -> Any:
    """`make_generate` over a real rollout: one ``storage`` put, then ``submit``.

    Not a fake collection like the tests above -- what is under test is precisely
    what the workspace reaches, which only a real dispatch can show.
    """
    gen = ScriptedGenerator(
        [
            "r</think>" + tool_call("storage", operate="put", key="k", value="v"),
            "r</think>" + tool_call("submit", answer="42"),
        ],
        logprobs=-0.1,
    )
    return make_generate(
        lambda args: _collector(gen, tools=[StorageTool()]),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_a_workspace_factory_gives_each_prompt_a_store_to_write_to() -> None:
    """The seam the adapter was missing: without one, every prompt ran on a
    `NullWorkspace` and each `storage` call came back to the model as an error string
    inside its own transcript -- which is what these samples are made of, so that is
    where the assertion is."""
    generate = _storage_generate(
        # one per prompt, never per actor: notes are a rollout's own, and the tree forks
        # them per branch
        workspace_factory=lambda sample: InMemoryWorkspace(),
    )

    samples = await generate(object(), "solve it")

    assert samples
    assert "Successfully saved k." in _transcripts(samples)


@pytest.mark.asyncio
async def test_adapter_filters_after_search_without_changing_critic_targets() -> None:
    full = await _storage_generate(
        workspace_factory=lambda _: InMemoryWorkspace(),
    )(object(), "solve it")
    filtered = await _storage_generate(
        workspace_factory=lambda _: InMemoryWorkspace(),
        min_abs_advantage=100.0,
    )(object(), "solve it")
    assert any(s["metadata"]["lane"] == "actor" for s in full)
    assert filtered
    assert filtered == [s for s in full if s["metadata"]["lane"] == "critic"]


@pytest.mark.asyncio
async def test_the_factory_sees_the_trainer_s_own_sample() -> None:
    """It takes the sample, as `to_prompt` does, so a workspace can start seeded from
    the case it belongs to rather than empty."""
    seen: list[Any] = []

    def factory(sample: Any) -> InMemoryWorkspace:
        seen.append(sample)
        workspace = InMemoryWorkspace()
        workspace.write("brief", str(sample))
        return workspace

    generate = _storage_generate(workspace_factory=factory)
    sample = "solve it"

    await generate(object(), sample)

    assert seen == [sample]


@pytest.mark.asyncio
async def test_an_env_with_no_workspace_tools_needs_no_factory() -> None:
    """The default stays the default: no factory is a `NullWorkspace`, which is exactly
    right for a world whose tools never touch a store."""
    gen = ScriptedGenerator(
        ["r</think>" + tool_call("submit", answer="42")], logprobs=-0.1
    )
    generate = make_generate(lambda args: _collector(gen))

    samples = await generate(object(), "solve it")

    assert samples


# -- what a config file turns into ----------------------------------------------------
