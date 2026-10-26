from __future__ import annotations

import pytest

from focus.harness.workspace import (
    InMemoryWorkspace,
    NullWorkspace,
    WorkspaceError,
)


def test_snapshot_is_empty_unless_the_backend_is_a_store() -> None:
    assert NullWorkspace().snapshot() == {}
    ws = InMemoryWorkspace()
    ws.write("k", "v")
    assert dict(ws.snapshot()) == {"k": "v"}


def test_write_then_read_round_trips() -> None:
    ws = InMemoryWorkspace()
    ws.write("plan.md", "step 1: think")
    assert ws.read("plan.md") == "step 1: think"


def test_write_overwrites_existing_entry() -> None:
    ws = InMemoryWorkspace()
    ws.write("notes", "draft")
    ws.write("notes", "final")
    assert ws.read("notes") == "final"


def test_reading_a_missing_path_raises() -> None:
    ws = InMemoryWorkspace()
    with pytest.raises(WorkspaceError, match="no such path: 'ghost'"):
        ws.read("ghost")


def test_exists_reflects_writes_and_deletes() -> None:
    ws = InMemoryWorkspace()
    assert not ws.exists("a")
    ws.write("a", "x")
    assert ws.exists("a")
    ws.delete("a")
    assert not ws.exists("a")


def test_deleting_a_missing_path_raises() -> None:
    ws = InMemoryWorkspace()
    with pytest.raises(WorkspaceError):
        ws.delete("ghost")


def test_paths_lists_entries_in_write_order() -> None:
    ws = InMemoryWorkspace()
    ws.write("first", "1")
    ws.write("second", "2")
    ws.write("first", "1b")  # overwrite keeps original position
    assert ws.paths() == ("first", "second")


def test_fork_reproduces_contents_then_diverges_both_ways() -> None:
    parent = InMemoryWorkspace()
    parent.write("shared", "v0")
    child = parent.fork()
    assert child.read("shared") == "v0"  # equal at fork time

    child.write("shared", "v1")  # child edit invisible to parent
    child.write("child_only", "c")
    parent.write("parent_only", "p")  # parent edit invisible to child
    parent.delete("shared")

    assert parent.read("parent_only") == "p"
    assert not parent.exists("child_only")
    assert not parent.exists("shared")
    assert child.read("shared") == "v1"
    assert not child.exists("parent_only")


def test_clear_empties_the_workspace() -> None:
    ws = InMemoryWorkspace()
    ws.write("a", "1")
    ws.write("b", "2")
    ws.clear()
    assert ws.paths() == ()
    assert len(ws) == 0


def test_init_replaces_all_contents() -> None:
    ws = InMemoryWorkspace()
    ws.write("old", "gone")
    ws.init({"seed": "here", "task": "solve"})
    assert not ws.exists("old")
    assert ws.read("seed") == "here"
    assert ws.read("task") == "solve"
    ws.init()  # no args -> empty
    assert ws.paths() == ()


def test_update_merges_without_clearing() -> None:
    ws = InMemoryWorkspace()
    ws.write("keep", "v0")
    ws.update({"keep": "v1", "new": "n"})
    assert ws.read("keep") == "v1"  # overwritten
    assert ws.read("new") == "n"  # added


def test_serialize_is_an_independent_representation() -> None:
    ws = InMemoryWorkspace()
    ws.write("a", "1")
    state = ws.serialize()
    assert state == {"a": "1"}
    state["a"] = "mutated"  # mutating serialized state doesn't touch the workspace
    ws.write("b", "2")  # a later write isn't reflected in the already-taken state
    assert ws.read("a") == "1"
    assert state == {"a": "mutated"}


def test_serialize_deserialize_round_trips_and_is_independent() -> None:
    ws = InMemoryWorkspace()
    ws.init({"x": "1", "y": "2"})
    restored = InMemoryWorkspace.deserialize(ws.serialize())
    assert isinstance(restored, InMemoryWorkspace)
    assert restored.serialize() == {"x": "1", "y": "2"}
    restored.write("x", "changed")  # independent of the original
    assert ws.read("x") == "1"


def test_contains_and_len_ergonomics() -> None:
    ws = InMemoryWorkspace()
    ws.write("a", "1")
    ws.write("b", "2")
    assert "a" in ws
    assert "missing" not in ws
    assert len(ws) == 2
