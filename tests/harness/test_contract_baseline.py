from dataclasses import replace

from focus.harness import InMemoryWorkspace, regions
from tests.helpers import rollout, turn


def test_live_context_replacement_preserves_training_history():
    first = turn([-0.5, -0.7])
    original = rollout(first)
    edited = replace(original, messages=({"role": "user", "content": "new"},))
    assert edited.turns == original.turns
    assert regions(edited.turns) == regions(original.turns)


def test_snapshot_owns_memory_before_continuation_mutates_it():
    workspace = InMemoryWorkspace()
    workspace.write("evidence", "original")
    original = replace(rollout(), workspace=workspace)
    frozen = original.snapshot()
    workspace.write("evidence", "changed")
    assert frozen.workspace.snapshot()["evidence"] == "original"
