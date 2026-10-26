"""Offline integration tests for the maintained benchmark environments."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from focus.backends import ChatCodec, Policy, PolicyFormat, QwenXMLToolCallParser
from focus.environments.browsercomp_plus import (
    BrowserFind,
    BrowserOpen,
    RetrievalArchive,
    make_agent,
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
    result = await world.agent.run_from(world.prompt, workspace=world.workspace)
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
    result = await world.agent.run_from(world.prompt, workspace=world.workspace)
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

    case = BrowserCase("query", "Who is the answer?", ("Alice",))
    agent = make_agent(
        policy(
            call("browser.search", query="answer"),
            call("submit", answer="Alice"),
        ),
        archive=archive,
        no_compact=True,
        max_steps=4,
    )
    result = await agent.run_from(case.question)
    assert result.done
    assert result.state.answer == "Alice"
    assert [schema["function"]["name"] for schema in agent.tools] == [
        "browser.search",
        "browser.open",
        "browser.find",
        "submit",
    ]


def _rollout(state: object, *, turns: int = 0, folds: int = 0) -> Any:
    return SimpleNamespace(state=state, turns_taken=turns, folds=folds)


async def test_locbench_ends_on_repeated_requests_and_its_fold_horizon(
    repository: Repository,
) -> None:
    from focus.environments.locbench import LocBenchEnv

    gold = Gold(frozenset({"src/target.py"}), frozenset())
    repeats = LocBenchEnv(repository, gold, repeat_limit=3)
    state = await repeats.initial_state()
    assert not repeats.should_finish(_rollout(state))
    assert repeats.should_finish(_rollout(replace(state, consecutive_requests=3)))

    horizon = LocBenchEnv(repository, gold, fold_horizon=4)
    assert not horizon.should_finish(_rollout(state, turns=2, folds=1))
    assert horizon.should_finish(_rollout(state, turns=2, folds=2))

    failed = repeats.on_compaction_failure(
        replace(state, locations=("a",), submitted=True), ("reply_cap",)
    )
    assert (failed.locations, failed.submitted) == ((), False)
    assert failed.compaction_failure == ("reply_cap",)


async def test_browsercomp_answers_before_its_task_limit_with_submit_only() -> None:
    from focus.environments.browsercomp_plus import SearchEnv

    env = SearchEnv(RetrievalArchive(Corpus()), task_limit=5)
    state = await env.initial_state()
    assert not env.should_finish(_rollout(state, turns=3))
    assert env.should_finish(_rollout(state, turns=4))
    names = [t["function"]["name"] for t in env.finish_tools(env.schemas)]
    assert names == ["submit"]
    failed = env.on_compaction_failure(replace(state, answer="x"), ("empty_notes",))
    assert failed.answer is None
    assert failed.compaction_failure == ("empty_notes",)


async def test_deontic_records_a_rejected_fold() -> None:
    from focus.environments.deontic import DeonticEnv

    env = DeonticEnv(tools=[], budget=2, scorer=lambda *a: 0.0)
    state = replace(await env.initial_state(), answer="x")
    failed = env.on_compaction_failure(state, ("reply_cap",))
    assert failed.answer is None
    assert failed.compaction_failure == ("reply_cap",)
