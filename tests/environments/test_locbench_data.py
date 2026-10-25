"""User-supplied cases retain private labels and unique identities."""

import json
from pathlib import Path

import pytest

from focus.environments.locbench.dataset import load_cases


def test_jsonl_cases_keep_labels_out_of_the_task_prompt(tmp_path: Path) -> None:
    row = {
        "instance_id": "example__repo-1",
        "repo": "example/repo",
        "base_commit": "a" * 40,
        "problem_statement": "Find the faulty function.",
        "patch": (
            "diff --git a/secret.py b/secret.py\n--- a/secret.py\n+++ b/secret.py\n"
        ),
        "edit_functions": ["secret.py:target"],
        "added_functions": [],
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(row) + "\n\n")
    [case] = load_cases(path)
    assert case.gold.files == frozenset({"secret.py"})
    assert "Find the faulty function." in case.task_prompt()
    assert "secret.py" not in case.task_prompt()

    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate instance_id"):
        load_cases(path)
