"""Task hooks the agent consults: early endings, final-turn tools, context limits, and
episodes ended by a rejected compaction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from focus.generation import SamplingParams
from focus.generation.parsing import HermesToolCallParser
from focus.harness import (
    Compaction,
    CompactionResult,
    ContextManager,
    IncompleteCompactionError,
    InMemoryWorkspace,
    StepResult,
    StorageTool,
    Turn,
)
from tests.helpers import (
    ScriptedGenerator,
    SubmitEnv,
    SubmitState,
    ToyEnv,
    ToyState,
    WhenTrigger,
    tool_call,
    toy_runner,
)


class StopAfter(ToyEnv):
    """Asks for the final answer once ``limit`` task turns are taken."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def should_finish(self, rollout: Any) -> bool:
        return bool(rollout.turns_taken >= self.limit)


async def test_should_finish_takes_the_final_answer_turn_early() -> None:
    runner = toy_runner(ScriptedGenerator(["go"]), env=StopAfter(2), max_steps=8)
    result = await runner.run(await runner.start("hi"))
    assert result.truncated
    assert result.turns_taken == 3  # two task turns, then the forced answer
    assert "out of steps" in result.messages[-2]["content"]


async def test_final_turn_tools_and_role_are_the_task_s() -> None:
    class SubmitOnlyAtTheEnd(SubmitEnv):
        def finish_tools(self, tools):
            return tuple(t for t in tools if t["function"]["name"] == "submit")

    gen = ScriptedGenerator([tool_call("storage", operate="list")])
    runner = toy_runner(
        gen,
        env=SubmitOnlyAtTheEnd([StorageTool()]),
        parser=HermesToolCallParser(),
        max_steps=1,
        finish_prompt={"role": "user", "content": "Answer now."},
    )
    offered: list[list[str]] = []
    prepare = runner.policy.prepare

    def spy(messages, tools=()):
        offered.append([t["function"]["name"] for t in tools])
        return prepare(messages, tools=tools)

    runner.policy.prepare = spy  # type: ignore[method-assign]
    root = await runner.start("hi", workspace=InMemoryWorkspace())
    result = await runner.run(root)
    assert result.messages[-2] == {"role": "user", "content": "Answer now."}
    assert offered[0] == ["storage", "submit"]
    assert offered[-1] == ["submit"]


async def test_context_limit_clamps_the_reply_and_refuses_an_overfull_prompt() -> None:
    gen = ScriptedGenerator(["go"])
    probe = toy_runner(gen)
    prompt = len(probe.context.prepare(await probe.start("hi"), probe).tokens)

    runner = toy_runner(
        gen,
        sampling_params=SamplingParams(max_tokens=100),
        context_limit=prompt + 50 + 1024 + 1,
    )
    await runner.advance(await runner.start("hi"))
    assert gen.params[-1] is not None
    assert gen.params[-1].max_tokens == 100  # room enough: untouched

    tight = toy_runner(
        ScriptedGenerator(["go"]),
        sampling_params=SamplingParams(max_tokens=100),
        context_limit=prompt + 10,
        finish_prompt=None,
    )
    ended = await tight.advance(await tight.start("hi"))
    assert ended.truncated  # no room for a full reply: the rollout ends instead

    overfull = toy_runner(ScriptedGenerator(["go"]), context_limit=prompt)
    root = await overfull.start("hi")
    with pytest.raises(ValueError, match="context_limit"):
        await overfull.finish(root)


class Failing(ContextManager[ToyState]):
    """A fold the model botched: two generated turns, then a refusal."""

    trigger = WhenTrigger(lambda prompt, state: state.n > 0)

    async def compact(self, compaction: Compaction[ToyState]) -> CompactionResult[Any]:
        # Generated in the fold's own world: task-tagged, with its transitions.
        turns: tuple[Turn[Any], ...] = tuple(
            Turn(
                prefix=(1,),
                tokens=(2,),
                logprobs=(-0.1,),
                transition=StepResult(next_state=compaction.state),
            )
            for _ in range(2)
        )
        raise IncompleteCompactionError(turns=turns, reasons=("reply_cap",))


class Keeps(Failing):
    async def compact(self, compaction: Compaction[ToyState]) -> CompactionResult[Any]:
        try:
            await super().compact(compaction)
        except IncompleteCompactionError as exc:
            return CompactionResult(compaction.seed, compaction.state, turns=exc.turns)
        raise AssertionError("unreachable")


class Records(ToyEnv):
    def on_compaction_failure(self, state, reasons):
        return replace(state, n=-len(reasons))


async def test_a_rejected_fold_ends_the_episode_when_asked() -> None:
    runner = toy_runner(
        ScriptedGenerator(["go"]),
        env=Records(),
        compactor=Failing(),
        on_compaction_failure="end",
    )
    stepped = await runner.advance(await runner.start("hi"))
    ended = await runner.advance(stepped)
    assert ended.done
    assert ended.reward_outcome == 0.0
    assert ended.state.n == -1  # the task recorded why
    assert [t.tag for t in ended.turns[-3:]] == ["fold", "fold", "task"]
    assert ended.turns[-3].transition is None  # the fold's own world stays out
    assert ended.turns[-1].tokens == ()  # the ending generated nothing

    raising = toy_runner(ScriptedGenerator(["go"]), compactor=Failing())
    with pytest.raises(IncompleteCompactionError):
        await raising.advance(await raising.advance(await raising.start("hi")))


async def test_the_agent_records_a_folds_turns_as_transition_free_fold_turns() -> None:
    runner = toy_runner(ScriptedGenerator(["go"]), compactor=Keeps())
    stepped = await runner.advance(await runner.start("hi"))
    folded = await runner.advance(stepped)
    added = folded.turns[len(stepped.turns) :]
    assert [t.tag for t in added] == ["fold", "fold"]
    assert all(t.transition is None for t in added)
    assert folded.turns_taken == stepped.turns_taken  # a fold is not a task step


def test_agent_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="on_compaction_failure"):
        toy_runner(ScriptedGenerator(["go"]), on_compaction_failure="ignore")
    with pytest.raises(ValueError, match="context_limit"):
        toy_runner(ScriptedGenerator(["go"]), context_limit=0)


def test_submit_state_is_untouched_by_default() -> None:
    state = SubmitState()
    assert SubmitEnv([]).on_compaction_failure(state, ("x",)) is state
