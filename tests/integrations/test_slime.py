"""The slime adapter: the picklable spec, and the generate function built from it.

Flattening a result into trainer samples is the export's, not this module's -- see
``tests/test_export.py`` -- so the results here are the smallest thing
``make_generate`` can hand back. What a config *file* turns into is
``tests/test_config.py``; this file is about the seam between an authored spec and the
live runtime an actor builds.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from focus import (
    ConcurrencyGating,
    FifoAllocator,
    RefinedTdEstimator,
    SearchPass,
    ValueRefinement,
    WidthGating,
)
from focus.codec import ChatCodec
from focus.config import RolloutConfig, RolloutSpec
from focus.generation.parsing import HermesToolCallParser
from focus.harness import (
    InMemoryWorkspace,
    PackedSequence,
    Runner,
    StorageTool,
)
from focus.integrations.slime import make_generate
from focus.loop import Runtime
from focus.preparation.records import ActorSample, CriticSample, PreparedBatch
from focus.registry import build
from focus.reward.config import RewardConfig
from focus.scheduler import MixtureProposal
from tests.helpers import (
    DemoValue,
    FakeChatTokenizer,
    ScriptedGenerator,
    SubmitEnv,
    tool_call,
    toy_runner,
)


def _runtime(**fields: Any) -> Runtime:
    """A runtime with a stand-in runner: nothing here ever runs a rollout."""
    fields.setdefault("value_model", DemoValue())
    return Runtime(
        runner=cast(
            Runner[Any, Any], SimpleNamespace(policy=SimpleNamespace(version="policy"))
        ),
        **fields,
    )


def _result() -> PreparedBatch:
    """One actor record and one critic record: these tests only count what came back."""
    span = PackedSequence(
        tokens=(0, 1), trainable=(False, True), logprobs={"policy": (0.0, -0.1)}
    )
    return PreparedBatch(
        actor=(
            ActorSample(
                node_id=1,
                spans=(span,),
                group_id=0,
                reward_config_id="rc-0",
                estimator="grpo",
            ),
        ),
        critic=(
            CriticSample(
                node_id=1,
                context="checkpoint",
                target=1.0,
                reward_config_id="rc-0",
                value_version="v-base",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_make_generate_builds_one_runtime_per_actor_and_returns_a_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: list[object] = []

    class FakeArgs:
        pass

    class FakeSample:
        index = 9

    async def fake_rollout(prompt, runtime, *, workspace=None):
        del prompt, runtime, workspace
        return _result()

    def factory(args: object) -> Runtime:
        builds.append(args)
        # Only the *identity* of what came back matters here: `fake_rollout` ignores
        # the runtime, and nothing on the way through dereferences its runner.
        return _runtime()

    monkeypatch.setattr("focus.loop.run_search", fake_rollout)
    monkeypatch.setattr("focus.loop.prepare_samples", lambda *a, **kw: _result())
    generate = make_generate(
        RolloutSpec(), runtime_factory=factory, estimator=RefinedTdEstimator()
    )
    args = FakeArgs()
    first = await generate(args, FakeSample())
    second = await generate(args, FakeSample())

    assert len(builds) == 1
    assert first
    assert second
    assert all(s["group_index"] == 9 for s in first)


@pytest.mark.asyncio
async def test_concurrent_args_keep_the_runtime_that_generated_each_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from focus.loop import run_search

    first_ready = asyncio.Event()
    release_first = asyncio.Event()

    async def paused_search(prompt, runtime, *, workspace=None):
        state = await run_search(prompt, runtime, workspace=workspace)
        if runtime.runner.policy.version == "v1":
            first_ready.set()
            await release_first.wait()
        return state

    def factory(args: SimpleNamespace) -> Runtime:
        generator = ScriptedGenerator(
            ["r</think>" + tool_call("submit", answer="42")],
            logprobs=args.logprob,
        )
        return Runtime(
            value_model=DemoValue(),
            runner=toy_runner(
                generator,
                env=SubmitEnv([]),
                parser=HermesToolCallParser(),
                version=args.version,
            ),
        )

    monkeypatch.setattr("focus.loop.run_search", paused_search)
    generate = make_generate(
        RolloutSpec(
            config=RolloutConfig(
                passes=(SearchPass(FifoAllocator(), budget=1, gating=WidthGating(1)),),
                max_concurrency=1,
            )
        ),
        runtime_factory=factory,
        estimator=RefinedTdEstimator(),
    )
    async with asyncio.TaskGroup() as tasks:
        first = tasks.create_task(
            generate(SimpleNamespace(version="v1", logprob=-0.1), "first")
        )
        await asyncio.wait_for(first_ready.wait(), timeout=5)
        try:
            second = await generate(
                SimpleNamespace(version="v2", logprob=-0.2), "second"
            )
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

    Not a fake `run_rollout` like the tests above -- what is under test is precisely
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
        RolloutSpec(
            config=RolloutConfig(
                max_concurrency=1,
                passes=(SearchPass(FifoAllocator(), budget=2, gating=WidthGating(3)),),
            )
        ),
        runtime_factory=lambda args: Runtime(
            value_model=DemoValue(),
            runner=toy_runner(
                gen, env=SubmitEnv([StorageTool()]), parser=HermesToolCallParser()
            ),
        ),
        **kwargs,
        estimator=RefinedTdEstimator(),
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
    generate = make_generate(
        RolloutSpec(
            config=RolloutConfig(
                max_concurrency=1,
                passes=(SearchPass(FifoAllocator(), budget=2, gating=WidthGating(3)),),
            )
        ),
        runtime_factory=lambda args: Runtime(
            value_model=DemoValue(),
            runner=toy_runner(gen, env=SubmitEnv([]), parser=HermesToolCallParser()),
        ),
        estimator=RefinedTdEstimator(),
    )

    samples = await generate(object(), "solve it")

    assert samples


def test_the_spec_carries_no_live_objects() -> None:
    """Picklable whole, which is what shipping one to a Ray actor amounts to.

    Read back field by field rather than with one ``==``: a policy is an ordinary
    object with no value equality, so comparing the records would be asserting
    identity -- which is exactly what a pickle round-trip does not preserve.
    """
    import pickle

    spec = RolloutSpec(
        env={"name": "toy"},
        system_prompt="sys",
        config=RolloutConfig(
            passes=(SearchPass(FifoAllocator(), budget=7, gating=WidthGating(4)),)
        ),
    )

    restored = pickle.loads(pickle.dumps(spec))

    assert restored.env == spec.env
    assert restored.system_prompt == spec.system_prompt
    assert restored.config.passes[0].budget == 7
    assert restored.config.reward_config == spec.config.reward_config
    assert isinstance(restored.config.passes[0].gating, WidthGating)


def test_a_fully_authored_spec_ships_too_rng_and_all() -> None:
    """The thin spec above is not the one an experiment writes. This is.

    Every policy field filled, a nested ``env`` config with its own compactor and
    trigger tree, and the one place a spec still carries a ``random.Random`` -- a
    mixture proposal's coin. An rng pickles by value, which is the right thing and
    worth pinning: each actor gets its own stream from wherever the seed had reached,
    so two actors do not draw the same explore coins. ``ChildRelative`` carries a plain
    ``seed`` instead, so every actor scores a given tree identically.
    """
    import pickle

    spec = build(
        RolloutSpec,
        {
            "env": {
                "name": "toy",
                "compactor": {
                    "name": "budget",
                    "trigger": {
                        "name": "any_of",
                        "triggers": [
                            {"name": "prompt_tokens", "limit": 10_000},
                            {"name": "state_counter"},
                        ],
                    },
                },
            },
            "system_prompt": "sys",
            "config": {
                "proposal": {"name": "mixture", "explore": {"temperature": 1.5}},
                "passes": [
                    {
                        "allocator": {"name": "fifo"},
                        "budget": 32,
                        "gating": {"name": "width", "width": 5},
                    },
                    {
                        "allocator": {"name": "value_refinement"},
                        "budget": 4,
                        "gating": {"name": "concurrency", "limit": 6},
                    },
                    {
                        "allocator": {"name": "fifo"},
                        "budget": 4,
                        "gating": {"name": "concurrency", "limit": 6},
                    },
                ],
            },
        },
    )

    restored = pickle.loads(pickle.dumps(spec))

    assert restored.env == spec.env
    assert isinstance(restored.config.proposal, MixtureProposal)
    assert [type(w) for w in [p.allocator for p in restored.config.passes[1:]]] == [
        ValueRefinement,
        FifoAllocator,
    ]
    # the coin came across at the state it had reached, not reseeded on the far side
    assert restored.config.proposal.draw() == spec.config.proposal.draw()


# -- the split, and the one rule that follows from it --------------------------------

#: The fields a spec may never carry: a live endpoint, a function over one, or a bound
#: on one -- how long *that* server may take is a fact about the server, so it is set
#: where the server is (as `Runner.generate_timeout` already is), not in a spec that
#: travels to actors holding different endpoints.
LIVE_FIELDS = {
    "runner",
    "scorer",
    "value_model",
    "value_serialize",
    "score_timeout",
}


def test_a_runtime_is_the_live_objects_and_the_authored_config_and_nothing_else() -> (
    None
):
    """The audit as a test: a new knob is a `RolloutConfig` field or a live one.

    Silence is the failure mode this guards. A knob added straight to `Runtime` would
    work in-process and be unreachable from a slime job -- unsayable in a config file,
    unshippable to an actor -- which nothing else would report. A timeout on a live
    endpoint is the one exception, and it is not unreachable: `make_generate` keeps
    every field the factory set except `config`, so the factory that built the reward
    model bounds it there, exactly as it bounds its generator.
    """
    from dataclasses import fields as dataclass_fields

    # Private fields are not knobs: `_weighters` is where `weighters_of` memoises what
    # `config.weighters` already says, and `replace(runtime, config=...)` drops it (it
    # is `init=False`), so a re-configured runtime resolves again rather than carrying
    # the previous config's estimators.
    runtime_fields = {
        f.name for f in dataclass_fields(Runtime) if not f.name.startswith("_")
    }

    assert runtime_fields - LIVE_FIELDS == {"config"}


@pytest.mark.asyncio
async def test_the_spec_s_config_replaces_the_factory_s_and_the_live_objects_stand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The precedence rule, whole, in one call.

    The spec owns the configuration -- an actor-local override would make two actors
    run different knobs while both claimed the same spec -- and the factory owns
    everything that holds a socket. There is no field-by-field transfer to get wrong:
    the config is one record, and it is the whole authored half.
    """
    seen: list[Runtime] = []

    async def fake_rollout(prompt, runtime, *, workspace=None):
        del prompt, workspace
        seen.append(runtime)
        return _result()

    monkeypatch.setattr("focus.loop.run_search", fake_rollout)
    monkeypatch.setattr("focus.loop.prepare_samples", lambda *a, **kw: _result())
    scorer, value_model = object(), object()
    spec = RolloutSpec(
        config=RolloutConfig(
            max_concurrency=1,
            passes=(
                SearchPass(FifoAllocator(), budget=5, gating=WidthGating(3)),
                SearchPass(FifoAllocator(), budget=2, gating=ConcurrencyGating(4)),
            ),
        )
    )
    generate = make_generate(
        spec,
        runtime_factory=lambda args: _runtime(
            scorer=cast(Any, scorer),
            value_model=cast(Any, value_model),
            config=RolloutConfig(
                max_concurrency=99,
                passes=(SearchPass(FifoAllocator(), budget=99, gating=WidthGating(3)),),
            ),
        ),
        estimator=RefinedTdEstimator(),
    )

    await generate(object(), object())

    assert seen[0].config is spec.config
    assert seen[0].scorer is scorer
    assert seen[0].value_model is value_model


def test_an_authored_empty_weighters_means_none_not_the_default_pair() -> None:
    """Explicit critic-only preparation is supported by the integration."""
    assert RolloutConfig().passes == ()
    assert build(RolloutConfig, {"passes": []}).passes == ()


# -- what a config file turns into ----------------------------------------------------


def test_a_yaml_shaped_mapping_builds_the_spec_it_describes() -> None:
    """What a loaded config file is: nested mappings and lists, no objects.

    Every policy field is *built* on the way in, because the config is what the
    runtime runs with -- there is no later pass to turn a leftover `dict` into the
    object its field is typed for.
    """
    built = build(
        RolloutSpec,
        {
            "env": {"name": "toy", "size": 3},
            "system_prompt": "sys",
            "config": {
                "reward_config": {"gamma": 0.95, "value_version": "v-1", "clip": 3.0},
                "proposal": {"name": "mixture", "explore": {"temperature": 1.5}},
                "continuation_id": "nu-7",
                "max_concurrency": 2,
                "passes": [
                    {
                        "allocator": {"name": "fifo"},
                        "budget": 4,
                        "gating": {"name": "width", "width": 5},
                    },
                    {
                        "allocator": {"name": "value_refinement"},
                        "budget": 4,
                        "gating": {"name": "concurrency", "limit": 6},
                    },
                    {
                        "allocator": {"name": "fifo"},
                        "budget": 4,
                        "gating": {"name": "concurrency", "limit": 6},
                    },
                ],
            },
        },
    )

    assert built.env == {"name": "toy", "size": 3}
    assert built.system_prompt == "sys"
    config = built.config
    assert config.passes[0].budget == 4
    assert config.reward_config == RewardConfig(
        gamma=0.95, value_version="v-1", clip=3.0
    )
    assert isinstance(config.passes[0].allocator, FifoAllocator)
    assert isinstance(config.proposal, MixtureProposal)
    assert isinstance(config.passes[0].gating, WidthGating)
    assert isinstance(config.passes[-1].gating, ConcurrencyGating)
    assert [type(w) for w in [p.allocator for p in config.passes[1:]]] == [
        ValueRefinement,
        FifoAllocator,
    ]
    assert config.continuation_id == "nu-7"
    assert config.max_concurrency == 2


def test_a_field_the_file_omits_is_the_default_it_documents() -> None:
    """There is no "unset" any more: what the record says is what the run does."""
    config = build(
        RolloutConfig,
        {
            "passes": [
                {
                    "allocator": {"name": "fifo"},
                    "budget": 12,
                    "gating": {"name": "width", "width": 3},
                }
            ]
        },
    )

    assert config.passes[0].budget == 12
    assert config.max_concurrency == 4
    assert config.reward_config == RewardConfig()


def test_a_misspelled_key_is_refused_rather_than_ignored() -> None:
    with pytest.raises(TypeError, match="unexpected config keys"):
        build(RolloutSpec, {"first_pass_turns": 4})
    with pytest.raises(TypeError, match="unexpected config keys"):
        build(RolloutConfig, {"pass_turn": 4})
