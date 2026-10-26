"""Complete snapshots recover failed work without recording partial task results."""

import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from dataclasses import replace

import pytest

from focus.execution.budget import ResourceBudget
from focus.generation.parsing import HermesToolCallParser
from focus.harness import (
    CorruptCheckpoint,
    InMemoryWorkspace,
    MemoryCheckpointer,
    SQLiteCheckpointer,
    regions,
)
from focus.protocols import AppendOnly, Summary
from tests.harness.test_runner import QuietMemory, ToyCompactor
from tests.helpers import ScriptedGenerator, ToyEnv, tool_call, toy_runner
from tests.protocols.helpers import KeepRecent


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryCheckpointer()
    return SQLiteCheckpointer(tmp_path / "checkpoints.sqlite")


async def test_store_isolates_full_state_and_tracks_branch_ancestry(store):
    runner = toy_runner(ScriptedGenerator(["go"]), checkpointer=store)
    root = await runner.start("q", workspace=InMemoryWorkspace())
    assert root.checkpoint_id is not None
    child = await runner.advance(root)
    assert not child.forkable  # Sparse search policy does not select ordinary turns.
    assert child.checkpoint_id is not None
    loaded = store.get(child.checkpoint_id)
    assert loaded.turns == child.turns
    assert regions(loaded.turns) == regions(child.turns)
    loaded.messages[-1]["content"] = "modified"
    loaded.turns[0].logprobs["new"] = (0.0,)
    assert isinstance(loaded.workspace, InMemoryWorkspace)
    loaded.workspace.write("leak", "bad")
    clean = store.get(child.checkpoint_id)
    assert clean.messages[-1]["content"] != "modified"
    assert "new" not in clean.turns[0].logprobs
    assert clean.workspace.snapshot() == {}
    branch = await runner.advance(root.fork())
    history = store.list(branch.episode_id, branch.branch_id)
    assert len(history) == 2  # branch root, then its first complete advance
    assert history[-1].parent_id == root.checkpoint_id
    assert history[0].parent_id == history[-1].id
    assert len(store.list(root.episode_id, root.branch_id)) == 2
    assert runner.restore(child.checkpoint_id).branch_id == child.branch_id
    assert runner.restore(child.checkpoint_id).forkable is False
    await runner.services.aclose()


async def test_failed_write_keeps_previous_commit_and_reopening_works(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    store = SQLiteCheckpointer(path)
    gen = ScriptedGenerator(["go"])
    runner = toy_runner(gen, checkpointer=store)
    root = await runner.start("q")
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TRIGGER fail_commit BEFORE INSERT ON rollout_checkpoints "
            "BEGIN SELECT RAISE(ABORT, 'disk failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="disk failure"):
        await runner.advance(root)
    reopened = SQLiteCheckpointer(path)
    assert (
        reopened.latest(root.episode_id, root.branch_id).checkpoint_id
        == root.checkpoint_id
    )
    assert gen.calls == 1  # A commit failure must not retry model/tool execution.
    await runner.services.aclose()


async def test_corrupt_latest_falls_back_only_within_requested_scope(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    store = SQLiteCheckpointer(path)
    runner = toy_runner(ScriptedGenerator(["go"]), checkpointer=store)
    root = await runner.start("q")
    changed = await runner.advance(root)
    await runner.advance(root.fork())  # newer, unrelated branch must not be selected
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE rollout_checkpoints SET payload=? WHERE id=?",
            (b"broken", changed.checkpoint_id),
        )
    assert changed.checkpoint_id is not None
    with pytest.raises(CorruptCheckpoint, match="Checksum"):
        store.get(changed.checkpoint_id)
    recovered = runner.restore_latest(root.episode_id, root.branch_id)
    assert recovered.checkpoint_id == root.checkpoint_id
    assert recovered.turns == ()
    await runner.services.aclose()


async def test_incomplete_and_legacy_states_cannot_replace_last_intact_snapshot(store):
    runner = toy_runner(ScriptedGenerator(["stop"]), checkpointer=store)
    root = await runner.start("q")
    with pytest.raises(ValueError, match="incomplete"):
        store.put(replace(root, snapshot_available=False))
    with pytest.raises(ValueError, match="Legacy"):
        store.put(replace(root, checkpoint_version=0))
    assert (
        store.latest(root.episode_id, root.branch_id).checkpoint_id
        == root.checkpoint_id
    )
    await runner.services.aclose()


class FailsAfterMutation(ToyEnv):
    attempts = 0

    async def step(self, state, action, *, workspace=None):
        assert isinstance(workspace, InMemoryWorkspace)
        assert "half-written" not in workspace.snapshot()
        workspace.write("half-written", "value")
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("transient failure after mutation")
        return await super().step(state, action, workspace=workspace)


async def test_retry_restores_workspace_budget_trace_and_same_branch(store):
    env = FailsAfterMutation()
    gen = ScriptedGenerator(["stop"], logprobs=-0.1)
    runner = toy_runner(
        gen,
        env=env,
        checkpointer=store,
        resource_budget=ResourceBudget(model_calls=1),
    )
    root = await runner.start("q", workspace=InMemoryWorkspace())
    endpoint = await runner.run(root)
    assert endpoint.done
    assert endpoint.branch_id == root.branch_id
    assert endpoint.budget.model_calls == 0
    assert len(endpoint.turns) == 1
    assert gen.calls == runner.services.usage[root.episode_id].calls == 2
    assert gen.prefixes[0] == gen.prefixes[1]
    assert root.workspace.snapshot() == {}
    assert len(store.list(root.episode_id, root.branch_id)) == 2
    assert endpoint.checkpoint_id is not None
    assert store.get(endpoint.checkpoint_id).turns == endpoint.turns
    await runner.services.aclose()


@pytest.mark.parametrize(
    ("error", "attempts"),
    [(ConnectionError, 3), (ValueError, 1), (asyncio.CancelledError, 1)],
)
async def test_retry_is_bounded_and_does_not_swallow_programming_errors_or_cancellation(
    error, attempts
):
    class FailingGenerator(ScriptedGenerator):
        async def agenerate_tokens(self, tokens, sampling_params=None):
            await super().agenerate_tokens(tokens, sampling_params)
            raise error("failure")

    store = MemoryCheckpointer()
    gen = FailingGenerator(["unused"])
    runner = toy_runner(gen, checkpointer=store)
    root = await runner.start("q")
    with pytest.raises(error):
        await runner.advance(root)
    assert gen.calls == attempts
    assert (
        store.latest(root.episode_id, root.branch_id).checkpoint_id
        == root.checkpoint_id
    )
    await runner.services.aclose()


@pytest.mark.parametrize("approach", ["plain", "summary", "custom"])
async def test_all_protocols_restore_complete_state_with_fresh_runner(
    tmp_path, approach
):
    path = tmp_path / "checkpoints.sqlite"

    def protocol():
        return {
            "plain": lambda: AppendOnly(),
            "summary": lambda: Summary(ToyCompactor()),
            "custom": KeepRecent,
        }[approach]()

    initial = {
        "plain": "go",
        "summary": "go",
        "custom": tool_call("keep_recent", n=1),
    }[approach]
    subsequent = "stop"

    def make_runner(output):
        return toy_runner(
            ScriptedGenerator([output], logprobs=-0.1),
            parser=HermesToolCallParser(),
            protocol=protocol(),
            checkpointer=SQLiteCheckpointer(path),
        )

    runner = make_runner(initial)
    root = await runner.start("hi", workspace=QuietMemory())
    saved = await runner.advance(root)
    if approach == "summary":
        saved = await runner.compact(saved)
    assert saved.checkpoint_id is not None
    await runner.services.aclose()
    restarted = make_runner(subsequent)
    restored = restarted.restore(saved.checkpoint_id)
    assert restored.messages == saved.messages
    assert restored.effects == saved.effects
    assert restored.execution == saved.execution
    assert restored.budget == saved.budget
    result = await restarted.advance(restored)
    assert result.turns[: len(saved.turns)] == saved.turns
    assert regions(restored.turns) == regions(saved.turns)
    if approach == "summary":
        assert result.workspace.snapshot() == {"summary": "compacted at step 1"}
    await restarted.services.aclose()


async def test_storage_frequency_is_independent_of_sparse_search_and_finalization(
    store,
):
    runner = toy_runner(
        ScriptedGenerator(["go"]), checkpointer=store, max_steps=2, finish_prompt=None
    )
    root = await runner.start("q")
    emitted = [c async for c in runner.iter_run(root)]
    assert len(emitted) == 1
    assert emitted[0].truncated
    assert (
        len(store.list(root.episode_id, root.branch_id)) == 4
    )  # root, 2 turns, finish
    last = runner.restore_latest(root.episode_id, root.branch_id)
    assert last.truncated
    assert len(last.turns) == 2
    await runner.services.aclose()


@pytest.mark.parametrize("automatic", [True, False])
async def test_summary_failure_retries_whole_fold_without_partial_workspace(automatic):
    class FlakyCompactor(ToyCompactor):
        attempts = 0

        async def compact(self, compaction):
            assert "summary" not in compaction.workspace.snapshot()
            result = await super().compact(compaction)
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("summary failed after writing memory")
            return result

    compactor = FlakyCompactor()
    store = MemoryCheckpointer()
    runner = toy_runner(
        ScriptedGenerator(["go"]), compactor=compactor, checkpointer=store
    )
    root = await runner.start("hi", workspace=QuietMemory())
    current = await runner.advance(root)
    operation = runner.advance if automatic else runner.compact
    folded = await operation(current)
    assert compactor.attempts == 2
    assert len(folded.turns) == 2
    assert folded.folds == 1
    assert current.workspace.snapshot() == {}
    assert len(store.list(root.episode_id, root.branch_id)) == 3
    await runner.services.aclose()


async def test_continuation_horizon_survives_crash_and_restore():
    from focus.harness.checkpoint import EveryDecision

    store = MemoryCheckpointer()
    runner = toy_runner(
        ScriptedGenerator(["go"]),
        checkpointer=store,
        checkpoints=EveryDecision(),
        finish_prompt=None,
    )
    started = await runner.advance(await runner.start("q"))
    stream = runner.iter_run(started, max_steps=2)
    assert isinstance(stream, AsyncGenerator)
    checkpoint = await anext(stream)
    await stream.aclose()  # simulate stopping after a committed/yielded boundary
    assert checkpoint.steps_remaining == 1
    assert checkpoint.checkpoint_id is not None
    result = await runner.run(runner.restore(checkpoint.checkpoint_id))
    assert result.truncated
    assert len(result.turns) == 3
    await runner.services.aclose()


async def test_automatic_retry_uses_most_recent_complete_step():
    class FailsSecondStep(ToyEnv):
        seen: list[int] = []
        failed = False

        async def step(self, state, action, *, workspace=None):
            self.seen.append(state.n)
            if state.n == 1 and not self.failed:
                self.failed = True
                raise ConnectionError("second step failed")
            return await super().step(state, action, workspace=workspace)

    env = FailsSecondStep(done_at=2)
    runner = toy_runner(
        ScriptedGenerator(["go"]), env=env, checkpointer=MemoryCheckpointer()
    )
    result = await runner.run_from("q")
    assert env.seen == [0, 1, 1]
    assert len(result.turns) == 2
    assert result.done
    await runner.services.aclose()


async def test_finish_retries_without_duplicating_forced_answer_trace():
    class FailsFinish(ToyEnv):
        attempts = 0

        async def finish(self, state, action):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("finish failed")
            return await super().finish(state, action)

    runner = toy_runner(
        ScriptedGenerator(["stop"]),
        env=FailsFinish(),
        checkpointer=MemoryCheckpointer(),
        max_steps=0,
        resource_budget=ResourceBudget(model_calls=1),
    )
    endpoint = await runner.run_from("q")
    assert endpoint.done
    assert endpoint.truncated
    assert len(endpoint.turns) == 1
    assert endpoint.checkpoint_id is not None
    assert runner.restore(endpoint.checkpoint_id).truncated
    assert runner.services.usage[endpoint.episode_id].calls == 2
    await runner.services.aclose()
