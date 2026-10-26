from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest

from focus.generation import SamplingParams
from focus.harness import (
    Agent,
    RolloutState,
    Turn,
    TurnTag,
)
from focus.harness.agent import Execution
from focus.protocols import AppendOnly
from focus.scheduler import (
    PASS_KEY,
    Budget,
    Expander,
    RolloutExpander,
    Scheduler,
    WidthGating,
)
from tests.helpers import (
    ScriptedGenerator,
    ToyEnv,
    rollout,
    toy_runner,
    turn,
)


def fold_marker() -> Turn[None]:
    """The zero-token turn a silent fold leaves -- the whole record of one.

    Spelled out rather than assembled inline because the empty behavior column is the
    load-bearing part: a turn that generated nothing is still *scored*, and its
    contribution is zero. The silent fold is this module's subject, so it lives here.
    """
    return Turn(prefix=(), tokens=(), logprobs=(), tag="fold")


def as_runner(fake: object) -> Agent[None, Any]:
    """Present a scripted stand-in as the ``Agent`` the strategies ask for.

    The fakes below script one region (or one turn) at a time, which a real ``Agent``
    has no way to do, so they reimplement its surface rather than subclass it. Casting
    once here says that in one place, instead of a directive on every call that passes
    one through.
    """
    return cast("Agent[None, Any]", fake)


EXPLORE = SamplingParams(temperature=1.5)


# -- annotations -------------------------------------------------------------


class ScriptedExpander(Expander[RolloutState[None]]):
    """Produces one child checkpoint per call, with no runner involved."""

    async def expand(
        self, payload: RolloutState[None]
    ) -> tuple[list[RolloutState[None]], Mapping[str, object]]:
        return [rollout(*payload.turns, turn([-1.0]))], {}


class StampedExpander(ScriptedExpander):
    async def expand(self, payload):
        chain, _ = await super().expand(payload)
        return chain, {PASS_KEY: 3}


@pytest.mark.asyncio
async def test_every_node_an_expansion_creates_carries_its_annotation() -> None:
    """The payload exists before the node does, so the expander hands the record to
    the scheduler rather than reaching for a node it cannot see."""
    scheduler: Scheduler[RolloutState[None]] = Scheduler(
        expander=StampedExpander(),
        termination=Budget(max_rollouts=2),
        gating=WidthGating(2),
        max_concurrency=1,
    )

    result = await scheduler.run(rollout())

    created = [n for n in result.nodes.values() if n.id != 0]
    assert created
    assert all(n.metadata[PASS_KEY] == 3 for n in created)
    assert PASS_KEY not in result.nodes[0].metadata  # nothing produced the root


@pytest.mark.asyncio
async def test_an_unstamped_expansion_leaves_metadata_alone() -> None:
    scheduler: Scheduler[RolloutState[None]] = Scheduler(
        expander=ScriptedExpander(),
        termination=Budget(max_rollouts=1),
        gating=WidthGating(1),
        max_concurrency=1,
    )

    result = await scheduler.run(rollout())

    assert all(not n.metadata for n in result.nodes.values())


# -- shared completion loop ----------------------------------------------------------


class ScriptedRunner(Agent[None, Any]):
    """Exercise the real drive loop with deterministic advances and workspace writes."""

    def __init__(self, *tags: TurnTag, max_steps: int = 8, done_at_end: bool = True):
        self.execution = Execution()
        self.protocol = AppendOnly()
        self._tags = tags
        self._max_steps = max_steps
        self._finish_prompt = "answer"
        self._done_at_end = done_at_end
        self.seen: list[SamplingParams | None] = []
        self.finished = 0
        self.advances = 0

    async def advance(self, rs, *, sampling_params=None):
        self.advances += 1
        self.seen.append(sampling_params)
        index = len(rs.turns)
        if index >= len(self._tags):
            return rs
        rs.workspace.write(f"step{index}", "x")
        tag = self._tags[index]
        added = (
            fold_marker()
            if tag == "fold"
            else turn([-1.0], done=self._done_at_end and index == len(self._tags) - 1)
        )
        return replace(
            rs,
            turns=(*rs.turns, added),
            checkpoint_at=index + 1 if tag == "fold" else -1,
        )

    async def finish(self, rs, *, sampling_params=None):
        self.finished += 1
        self.seen.append(sampling_params)
        return replace(rs, turns=(*rs.turns, turn([-1.0], done=True)), truncated=True)


@pytest.mark.asyncio
async def test_only_folds_and_endpoint_are_exposed_and_snapshots_are_isolated():
    runner = ScriptedRunner("task", "fold", "task", "fold", "task")
    root = rollout()
    chain, _ = await RolloutExpander(runner).expand(root)
    assert [len(s.turns) for s in chain] == [2, 4, 5]
    assert [s.forkable for s in chain] == [True, True, False]
    assert chain[-1].done
    assert runner.finished == 0
    assert root.workspace.snapshot() == {}
    assert len({id(s.workspace) for s in chain}) == 3
    for state, count in zip(chain, [2, 4, 5], strict=True):
        assert sorted(state.workspace.snapshot()) == [f"step{i}" for i in range(count)]


@pytest.mark.asyncio
async def test_folds_and_task_advances_share_one_budget():
    runner = ScriptedRunner("task", "fold", "task", "fold", "task", max_steps=3)
    chain = [s async for s in runner.iter_run(rollout())]
    assert runner.advances == 3
    assert [len(s.turns) for s in chain] == [2, 4]
    assert runner.finished == 1
    assert chain[-1].done
    assert chain[-1].truncated
    assert not chain[-1].forkable


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, 1, 3, -1])
async def test_direct_and_scheduler_runs_use_the_same_loop(budget):
    script: tuple[TurnTag, ...] = ("task", "fold", "task", "task")
    direct = ScriptedRunner(*script, max_steps=budget)
    search = ScriptedRunner(*script, max_steps=budget)
    result = await direct.run(rollout())
    chain, _ = await RolloutExpander(search).expand(rollout())
    assert result.turns == chain[-1].turns
    assert result.truncated == chain[-1].truncated
    assert direct.advances == search.advances
    assert direct.finished == search.finished
    assert result.workspace.snapshot() == chain[-1].workspace.snapshot()


@pytest.mark.asyncio
async def test_repeated_silent_folds_are_progress_but_are_bounded():
    tags: list[TurnTag] = ["fold"] * 10
    runner = ScriptedRunner(*tags, max_steps=3)
    chain = [s async for s in runner.iter_run(rollout())]
    assert runner.advances == 3
    assert len(chain) == 4
    assert all(s.turns_taken == 0 for s in chain[:-1])
    assert runner.finished == 1


@pytest.mark.asyncio
async def test_no_progress_forces_one_answer_instead_of_spinning():
    runner = ScriptedRunner(max_steps=-1)
    chain = [s async for s in runner.iter_run(rollout())]
    assert runner.advances == runner.finished == 1
    assert len(chain) == 1
    assert chain[0].done


@pytest.mark.asyncio
async def test_ended_inputs_do_not_generate():
    runner = ScriptedRunner("task")
    result = await runner.run(rollout(turn([-1.0], done=True)))
    assert result.done
    assert runner.advances == runner.finished == 0


@pytest.mark.asyncio
async def test_pass_sampling_reaches_every_advance_and_forced_answer():
    runner = ScriptedRunner("task", "fold", "task", max_steps=2)
    expander = RolloutExpander(
        runner, sampling_params=EXPLORE, annotation={PASS_KEY: 1}
    )
    chain, annotation = await expander.expand(rollout())
    assert runner.seen == [EXPLORE, EXPLORE, EXPLORE]
    assert chain[-1].truncated
    assert annotation == {PASS_KEY: 1}


@pytest.mark.asyncio
async def test_search_without_folds_branches_only_from_root():
    expander = RolloutExpander(
        toy_runner(ScriptedGenerator(["go"]), env=ToyEnv(done_at=3))
    )
    scheduler = Scheduler(
        expander=expander,
        termination=Budget(max_rollouts=4),
        gating=WidthGating(2),
        max_concurrency=1,
    )
    result = await scheduler.run(await expander.start("hi"))
    assert len(result.nodes) == 3
    assert all(n.payload.done for n in result.nodes.values() if n.parent_id is not None)
    assert all(
        n.parent_id == 0 for n in result.nodes.values() if n.parent_id is not None
    )


@pytest.mark.asyncio
async def test_legacy_checkpoints_are_analysis_only():
    root = rollout()
    root.__dict__["forkable"] = True  # serialized field in old checkpoints
    assert root.reward == 0  # historical analysis is still possible
    with pytest.raises(ValueError, match="analysis-only"):
        await ScriptedRunner("task").run(root)
    with pytest.raises(ValueError, match="analysis-only"):
        root.fork()
    with pytest.raises(ValueError, match="analysis-only"):
        root.snapshot().snapshot().fork()


@pytest.mark.parametrize("old_flag", [False, True])
async def test_legacy_pickle_version_survives_snapshots_and_dataclass_copies(old_flag):
    original = rollout()
    del original.__dict__["checkpoint_version"]
    original.__dict__["forkable"] = old_flag
    restored = pickle.loads(pickle.dumps(original))
    assert restored.checkpoint_version == 0

    copied = replace(restored).snapshot().snapshot()
    assert "forkable" not in copied.__dict__
    copies = [restored, copied, pickle.loads(pickle.dumps(copied))]
    for checkpoint in copies:
        assert checkpoint.reward == 0  # historical analysis still works
        assert checkpoint.checkpoint_version == 0
        with pytest.raises(ValueError, match="analysis-only"):
            await ScriptedRunner("task").run(checkpoint)
        with pytest.raises(ValueError, match="analysis-only"):
            checkpoint.fork()


async def test_current_pickle_and_copies_remain_resumable():
    restored = pickle.loads(pickle.dumps(rollout()))
    copied = replace(restored).snapshot().fork()
    assert copied.checkpoint_version == 3
    assert (await ScriptedRunner("task").run(copied)).done


@pytest.mark.parametrize("version", [0, 1])
def test_old_pickle_fields_support_reward_analysis_and_snapshots(version):
    original = rollout(turn([-0.5], done=True))
    # Match the serialized field sets before context-runtime schema 2 existed.
    old_turn_fields = {
        "prefix",
        "tokens",
        "logprobs",
        "exact",
        "transition",
        "tag",
        "compaction_penalty",
        "compaction_issue",
    }
    old_rollout_fields = {
        "messages",
        "seed",
        "state",
        "workspace",
        "turns",
        "truncated",
        "checkpoint_version",
    }
    expected = original.reward
    for key in tuple(original.turns[0].__dict__):
        if key not in old_turn_fields:
            del original.turns[0].__dict__[key]
    for key in tuple(original.__dict__):
        if key not in old_rollout_fields:
            del original.__dict__[key]
    if version:
        original.__dict__["checkpoint_version"] = version
    else:
        del original.__dict__["checkpoint_version"]
    restored = pickle.loads(pickle.dumps(original))
    assert restored.reward == expected
    assert restored.turns[0].reward_components == {}
    copied = replace(restored).snapshot()
    assert copied.reward == expected
    assert copied.checkpoint_version == version
    with pytest.raises(ValueError, match="analysis-only"):
        copied.fork()


async def test_zero_budget_without_finalization_is_no_progress_for_search():
    runner = toy_runner(ScriptedGenerator(["unused"]), max_steps=0, finish_prompt=None)
    root = await runner.start("q")
    expander = RolloutExpander(runner)
    chain, _ = await expander.expand(root)
    assert chain == []
    state = await Scheduler(expander=expander, termination=Budget(max_rollouts=1)).run(
        root
    )
    assert state.stats["failures"] == 1
    assert len(state.nodes) == 1
