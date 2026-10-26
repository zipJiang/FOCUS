"""Offline integration tests for the maintained benchmark environments."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from focus import (
    ChatCodec,
    Policy,
    PolicyFormat,
    QwenXMLToolCallParser,
)
from focus.environments.browsercomp_plus import (
    BrowserFind,
    BrowserOpen,
    RetrievalArchive,
    make_runner,
)
from focus.environments.browsercomp_plus import (
    Case as BrowserCase,
)
from focus.environments.deontic import Case as DeonticCase
from focus.environments.deontic import RunConfig as DeonticConfig
from focus.environments.deontic import build_world as build_deontic_world
from focus.environments.locbench import Case as LocCase
from focus.environments.locbench import Gold, Repository
from focus.environments.locbench import RunConfig as LocConfig
from focus.environments.locbench import build_world as build_locbench_world
from focus.environments.toolkit import SourceFile
from focus.generation.parsing import ToolTurn
from tests.helpers import FakeChatTokenizer, ScriptedGenerator


def call(name: str, **arguments: object) -> str:
    params = "\n".join(
        f"<parameter={key}>\n"
        f"{json.dumps(value) if isinstance(value, list) else value}\n</parameter>"
        for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>"


def policy(*completions: str) -> Policy[ToolTurn]:
    generator = ScriptedGenerator(completions, logprobs=-0.1)
    return Policy(
        format=PolicyFormat(
            "test", ChatCodec(FakeChatTokenizer()), QwenXMLToolCallParser()
        ),
        complete=generator.agenerate_tokens,
        version="environment-test",
    )


async def test_deontic_world_executes_tools_and_scores_submission() -> None:
    case = DeonticCase(
        id="airline-1",
        domain="airline",
        facts="Alice checked one bag.",
        question="What fee applies?",
        answer="100",
        answer_type="integer",
        statutes=(
            SourceFile(
                "statutes/bags.md",
                "# Fees\nA checked bag costs 100 dollars.\n",
            ),
        ),
    )
    config = DeonticConfig(
        policy=policy(call("grep", pattern="costs"), call("submit", answer="100")),
        tools=("list", "grep", "read", "python"),
        max_steps=4,
        call_budget=4,
        max_prompt_tokens=4096,
        python_timeout=2,
        no_workspace=True,
        no_compact=True,
        scorer=lambda answer: float(answer == "100"),
    )
    world = build_deontic_world(case, config)
    result = await world.runner.run_from(world.prompt, workspace=world.workspace)
    assert result.done
    assert result.state.answer == "100"
    assert result.reward == 1.0
    assert result.state.calls == ("grep",)


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    git(checkout, "init", "-q")
    git(checkout, "config", "user.name", "Test")
    git(checkout, "config", "user.email", "test@example.invalid")
    (checkout / "src").mkdir()
    (checkout / "src/target.py").write_text("def target():\n    return 1\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-qm", "base")
    return Repository(checkout, git(checkout, "rev-parse", "HEAD"))


async def test_locbench_world_reads_immutable_revision_and_scores(
    repository: Repository,
) -> None:
    assert "def target" in repository.read("src/target.py")["text"]
    case = LocCase(
        id="example__repo-1",
        repo="example/repo",
        base_commit=repository.commit,
        problem_statement="Find target.",
        gold=Gold(frozenset({"src/target.py"}), frozenset()),
    )
    config = LocConfig(
        policy=policy(call("submit", locations=["src/target.py"])),
        max_steps=4,
    )
    world = build_locbench_world(case, repository, config)
    result = await world.runner.run_from(world.prompt, workspace=world.workspace)
    assert result.done
    assert result.reward == 1.0
    assert result.state.locations == ("src/target.py",)


class Corpus:
    async def search(self, query: str, top_k: int):
        return [SimpleNamespace(docid="doc", rank=1, score=1.0)]

    async def documents(self, ids: list[str]):
        return {
            key: {
                "text": "Alice is the answer.\nFurther evidence.",
                "url": "https://example.invalid/doc",
            }
            for key in ids
            if key == "doc"
        }


async def test_browsercomp_navigation_and_runner_share_archive() -> None:
    archive = RetrievalArchive(Corpus())
    opened = json.loads(await BrowserOpen(archive).call({"docid": "doc"}))
    found = json.loads(
        await BrowserFind(archive).call({"docid": "doc", "pattern": "answer"})
    )
    assert opened["lines"][0]["text"].startswith("Alice")
    assert found["matches"][0]["line"] == 0

    runner, _, workspace, prompt, _, schemas = make_runner(
        BrowserCase("query", "Who is the answer?", ("Alice",)),
        policy(
            call("browser.search", query="answer"),
            call("submit", answer="Alice"),
        ),
        archive=archive,
        no_compact=True,
        max_steps=4,
    )
    result = await runner.run_from(prompt, workspace=workspace)
    assert result.done
    assert result.state.answer == "Alice"
    assert [schema["function"]["name"] for schema in schemas] == [
        "browser.search",
        "browser.open",
        "browser.find",
        "submit",
    ]
