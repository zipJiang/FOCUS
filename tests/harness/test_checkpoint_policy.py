"""Checkpoint eligibility uses the completed state and remains branch-local."""

from dataclasses import replace

import pytest

from focus import EveryDecision, SparseCheckpoints
from focus.harness import InMemoryWorkspace
from focus.harness.checkpoint import CheckpointPolicy
from focus.harness.rollout import AnyRollout
from tests.harness.test_runner import QuietMemory, TalkativeCompactor, ToyCompactor
from tests.helpers import ScriptedGenerator, ToyState, toy_runner


class AtProgress:
    def eligible(self, state: AnyRollout) -> bool:
        assert isinstance(state.workspace, InMemoryWorkspace)
        assert isinstance(state.state, ToyState)
        return str(state.state.n) == state.workspace.read("target")


async def test_custom_eligibility_reads_completed_state_and_forked_workspace():
    policy = AtProgress()
    assert isinstance(policy, CheckpointPolicy)
    runner = toy_runner(ScriptedGenerator(["go", "go", "go"]), checkpoints=policy)
    workspace = InMemoryWorkspace()
    workspace.write("target", "2")
    root = await runner.start("question", workspace=workspace)
    first = await runner.advance(root)
    assert first.state.n == 1
    assert not first.forkable
    second = await runner.advance(first)
    assert second.state.n == 2
    assert second.forkable
    branch = second.fork()
    assert isinstance(branch.workspace, InMemoryWorkspace)
    branch.workspace.write("target", "3")
    third = await runner.advance(branch)
    assert third.forkable
    assert isinstance(second.workspace, InMemoryWorkspace)
    assert second.workspace.read("target") == "2"
    assert not policy.eligible(replace(third, workspace=second.workspace))


@pytest.mark.parametrize("compactor", [ToyCompactor(), TalkativeCompactor()])
async def test_sparse_selects_context_boundary_but_not_the_following_task(compactor):
    runner = toy_runner(
        ScriptedGenerator(["go", "go"]),
        compactor=compactor,
        checkpoints=SparseCheckpoints(),
    )
    root = await runner.start("hi", workspace=QuietMemory())
    assert root.forkable
    first = await runner.advance(root)
    assert not first.forkable
    folded = await runner.compact(first)
    assert folded.forkable
    assert folded.effects[-1].turn_end == len(folded.turns)
    continued = await runner.advance(folded.fork())
    assert continued.state.n == 2
    assert not continued.forkable
    assert continued.effects == folded.effects


async def test_every_decision_still_excludes_completed_and_truncated_states():
    runner = toy_runner(ScriptedGenerator(["go", "stop"]), checkpoints=EveryDecision())
    first = await runner.advance(await runner.start("hi"))
    assert first.forkable
    assert not replace(first, truncated=True).forkable
    assert not replace(first, snapshot_available=False).forkable
    terminal = await runner.advance(first)
    assert terminal.done
    assert not terminal.forkable
