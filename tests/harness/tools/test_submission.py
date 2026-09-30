"""Single submission, stateless feedback, and forced endings through the runner."""

from __future__ import annotations

from typing import Any

import pytest

from focus.generation.parsing import HermesToolCallParser
from focus.harness import (
    DEFAULT_FINISH_PROMPT,
    InMemoryWorkspace,
    Runner,
    StorageTool,
    SubmitTool,
)
from tests.helpers import (
    ScriptedGenerator,
    SubmitEnv,
    complete,
    tool_call,
    toy_runner,
)


def submit(answer: str) -> str:
    """An assistant turn whose only content is a ``submit`` call."""
    return "r</think>" + tool_call("submit", answer=answer)


def store(value: str) -> str:
    """An assistant turn that calls a real tool -- i.e. goes back to work."""
    return "r</think>" + tool_call("storage", operate="put", key="k", value=value)


def _runner(gen: ScriptedGenerator, **env_kwargs: Any) -> Runner[Any, Any]:
    """The submit env over one real tool: `storage`, which writes to the run's store.

    So every rollout below is started on an `InMemoryWorkspace`. A `storage` tool under
    a `NullWorkspace` is refused at `Runner.root` -- its calls would come back to the
    model as error strings, and a `store(...)` turn here means the agent went back to
    work, not that it hit a broken tool.
    """
    return toy_runner(
        gen,
        env=SubmitEnv([StorageTool()], **env_kwargs),
        parser=HermesToolCallParser(),
    )


@pytest.mark.asyncio
async def test_a_submit_call_ends_the_rollout() -> None:
    gen = ScriptedGenerator([store("a"), submit("42")])
    result = await complete(_runner(gen), "q", workspace=InMemoryWorkspace())
    assert result.done
    assert not result.truncated  # the env ended it, not the step budget
    assert result.state.answer == "42"
    assert len(gen.prefixes) == 2


@pytest.mark.asyncio
async def test_the_terminal_turn_leaves_the_region_ending_on_trainable_tokens() -> None:
    # a done transition contributes no observation, so the region's last tokens are the
    # model's own -- the property `test_turn.py` pins for the old convention, preserved
    from focus.harness import regions

    gen = ScriptedGenerator([submit("42")])
    result = await complete(_runner(gen), "q", workspace=InMemoryWorkspace())
    assert result.done
    assert result.turns[-1].exact
    seq = regions(result.turns)[-1]
    assert seq.trainable[-1]
    assert len(seq.trainable) == len(seq.tokens)


@pytest.mark.asyncio
async def test_running_out_of_steps_still_recovers_the_submitted_answer() -> None:
    # the regression this whole change exists for: the finish turn ends on a tool call,
    # `finish` dispatches nothing, and the visible text is empty -- so reading the
    # answer out of the call is the only thing standing between this and a blank score
    gen = ScriptedGenerator([store("a"), store("b"), submit("42")])
    result = await complete(
        _runner(gen), "q", max_steps=2, workspace=InMemoryWorkspace()
    )
    assert result.done
    assert result.truncated  # over, and tellably not on its own terms
    assert result.state.answer == "42"


@pytest.mark.asyncio
async def test_a_rollout_that_never_submits_ends_unanswered_on_budget() -> None:
    gen = ScriptedGenerator(["r</think>I think it is 42"])  # never calls anything
    result = await complete(
        _runner(gen), "q", max_steps=6, workspace=InMemoryWorkspace()
    )
    assert result.done
    assert result.state.answer is None
    assert result.reward == 0
    assert result.truncated
    # Six reminders, then one final attempt; prose is never submitted.
    assert len(gen.prefixes) == 7


@pytest.mark.parametrize(
    "reply",
    [
        "r</think>42",
        "r</think>" + tool_call("submit"),
        'r</think><tool_call>{"name":"submit","arguments":{"answer":null}}</tool_call>',
        store("must not run"),
    ],
)
async def test_final_invalid_submission_is_unanswered_and_runs_no_tools(reply):
    gen = ScriptedGenerator([reply])
    workspace = InMemoryWorkspace()
    result = await complete(_runner(gen), "q", max_steps=0, workspace=workspace)
    assert result.done
    assert result.truncated
    assert result.state.answer is None
    assert result.reward_outcome == 0
    assert workspace.snapshot() == {}
    assert len(gen.prefixes) == 1


async def test_submit_preempts_siblings_and_no_second_generation_occurs():
    gen = ScriptedGenerator([submit("42") + store("must not run"), submit("wrong")])
    workspace = InMemoryWorkspace()
    result = await complete(_runner(gen), "q", workspace=workspace)
    assert result.state.answer == "42"
    assert result.reward_outcome == 1
    assert gen.calls == 1
    assert workspace.snapshot() == {}


async def test_reminders_do_not_change_state_or_consume_tool_budget():
    runner = _runner(ScriptedGenerator(["r</think>draft"]))
    checkpoint = await runner.start("q", workspace=InMemoryWorkspace())
    original = checkpoint.state
    for _ in range(3):
        checkpoint = await runner.advance(checkpoint)
        assert checkpoint.state is original
        assert not checkpoint.done
        assert checkpoint.messages[-1]["role"] == "tool"


@pytest.mark.parametrize("name", [None, "submit", "final_answer"])
async def test_default_finalization_supports_the_environments_terminal_action(name):
    terminal = SubmitTool(name=name) if name is not None else None
    reply = "42" if name is None else tool_call(name, answer="42")
    gen = ScriptedGenerator(["r</think>" + reply])
    result = await complete(
        _runner(gen, submit=terminal), "q", max_steps=0, workspace=InMemoryWorkspace()
    )

    assert result.done
    assert result.truncated
    assert result.state.answer == "42"
    assert gen.calls == 1
    feedback = result.messages[-2]
    assert feedback == {"role": "tool", "content": DEFAULT_FINISH_PROMPT}
    assert "submit" not in feedback["content"]
    assert "call that tool; otherwise answer directly in text" in feedback["content"]
