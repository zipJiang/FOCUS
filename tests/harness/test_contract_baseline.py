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


def test_snapshot_also_isolates_mutable_message_and_execution_data():
    source = replace(
        rollout(),
        messages=({"role": "user", "content": "original"},),
        execution={"variables": {"x": [1]}},
    )
    child = source.fork()
    child.messages[0]["content"] = "changed"
    child.execution["variables"]["x"].append(2)
    assert source.messages[0]["content"] == "original"
    assert source.execution["variables"]["x"] == [1]


def test_same_prefix_in_different_invocations_packs_separately():
    first = turn([-0.5])
    second = replace(
        turn([-0.2]), prefix=first.prefix + first.tokens, invocation_id="child"
    )
    assert len(regions((first, second))) == 2
