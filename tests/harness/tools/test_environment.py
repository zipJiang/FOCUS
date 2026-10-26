"""ToolEnv: dispatch, single submission, and stateless feedback.

Each branch of `step`/`finish` in isolation; `tools/test_submission.py` drives the same
convention end-to-end through the Agent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from focus.generation import GenerateResult
from focus.generation.parsing import HermesToolCallParser, ToolTurn
from focus.harness import (
    InMemoryWorkspace,
    RoundBudget,
    StorageTool,
    SubmitFields,
    ToolEnv,
)
from tests.helpers import SubmitEnv, SubmitState, tool_call


def _result(text: str) -> GenerateResult:
    return GenerateResult(text=text)


TOOL_CALL = tool_call("storage", operate="put", key="k", value="v")


def _submit(answer: str) -> str:
    return tool_call("submit", answer=answer)


def _turn(*calls: str):
    """One Hermes assistant turn making ``calls``."""
    return HermesToolCallParser().parse(_result("go</think>" + "\n".join(calls)))


@pytest.mark.asyncio
async def test_env_dispatches_tool_calls_into_tool_messages() -> None:
    env = SubmitEnv([StorageTool()])
    turn = HermesToolCallParser().parse(_result(f"go\n</think>\n\n{TOOL_CALL}"))
    step = await env.step(SubmitState(), turn)
    assert not step.done
    assert step.messages == ({"role": "tool", "content": "Successfully saved k."},)
    assert step.next_state.calls == 1


@pytest.mark.asyncio
async def test_env_ends_the_episode_on_a_submit_call() -> None:
    env = SubmitEnv([StorageTool()])
    turn = HermesToolCallParser().parse(_result(f"done\n</think>\n\n{_submit('42')}"))
    step = await env.step(SubmitState(), turn)
    assert step.done
    assert step.reward == 1.0
    assert step.next_state.answer == "42"  # read out of the call, not the visible text


@pytest.mark.asyncio
async def test_env_does_not_dispatch_a_submit_call() -> None:
    # `submit` is intercepted by name, so it must never reach a tool -- a workspace left
    # untouched here is what proves the interception happens before dispatch
    ws = InMemoryWorkspace()
    env = SubmitEnv([StorageTool()])
    turn = HermesToolCallParser().parse(_result(f"d</think>{_submit('x')}"))
    step = await env.step(SubmitState(), turn, workspace=ws)
    assert step.messages == ()  # a terminal transition contributes no observation
    assert ws.paths() == ()


@pytest.mark.asyncio
async def test_env_nudges_a_turn_that_called_nothing() -> None:
    # under this convention silence is not an answer, it is a turn that did nothing
    env = SubmitEnv([StorageTool()])
    turn = HermesToolCallParser().parse(_result("done\n</think>\n\nfinal answer"))
    step = await env.step(SubmitState(), turn)
    assert not step.done
    assert step.messages[0]["role"] == "tool"
    assert "submit" in step.messages[0]["content"]
    assert step.next_state == SubmitState()


@pytest.mark.asyncio
async def test_a_malformed_submit_is_handed_back_rather_than_ending_the_episode() -> (
    None
):
    env = SubmitEnv([StorageTool()])
    empty = tool_call("submit")
    turn = HermesToolCallParser().parse(_result(f"d</think>{empty}"))
    step = await env.step(SubmitState(), turn)
    assert not step.done  # ending on arguments nobody could read helps no one
    assert "answer" in step.messages[0]["content"]


@pytest.mark.asyncio
async def test_submit_none_restores_the_no_tool_call_convention() -> None:
    class _Legacy(SubmitEnv):
        def __init__(self, tools):
            super().__init__(tools, submit=None)

    env = _Legacy([StorageTool()])
    turn = HermesToolCallParser().parse(_result("done\n</think>\n\nfinal answer"))
    step = await env.step(SubmitState(), turn)
    assert step.done
    assert step.next_state.answer == "final answer"
    assert [s["function"]["name"] for s in env.schemas] == ["storage"]


def test_env_exposes_tool_schemas_including_submit() -> None:
    env = SubmitEnv([StorageTool()])
    assert [s["function"]["name"] for s in env.schemas] == ["storage", "submit"]


# -- finalization -----------------------------------------------------------


@pytest.mark.asyncio
async def test_finish_reads_the_answer_out_of_an_undispatched_submit_call() -> None:
    # the bug that motivated all of this: out of steps, the model still ends on a call,
    # `finish` runs no tools, and the visible text is empty -- so the answer must be
    # read from the call or the whole rollout scores blank
    env = SubmitEnv([StorageTool()])
    turn = HermesToolCallParser().parse(_result(f"d</think>{_submit('42')}"))
    step = await env.finish(SubmitState(), turn)
    assert step.done
    assert step.next_state.answer == "42"


@pytest.mark.asyncio
async def test_finish_falls_back_to_the_visible_text_when_nothing_was_submitted() -> (
    None
):
    env = SubmitEnv([StorageTool()])
    turn = HermesToolCallParser().parse(_result("d</think>my best guess"))
    step = await env.finish(SubmitState(), turn)
    assert step.done
    assert step.next_state.answer is None
    assert step.reward == 0


# -- the state contract: only what the configuration actually writes ------------------


@dataclass(frozen=True)
class _Budgeted(SubmitFields, RoundBudget):
    """The two harness mixins composed, which is how a task opts into both."""


class _BudgetEnv(ToolEnv[_Budgeted]):
    async def initial_state(self) -> _Budgeted:
        return _Budgeted(calls_remaining=5, calls_max=5)


@dataclass(frozen=True)
class _Plain:
    """A task state that carries none of the harness's bookkeeping."""

    n: int = 0


class _PlainEnv(ToolEnv[_Plain]):
    async def initial_state(self) -> _Plain:
        return _Plain()


@pytest.mark.asyncio
async def test_dispatching_charges_the_round_budget_one_per_call() -> None:
    """`RoundBudget` is spent here, which is what makes a `StateCounter` fold ever fire.

    Two calls in one turn cost two: the budget is a tool-call budget, not a turn budget,
    and a model that batches its calls has done the same amount of looking.
    """
    env = _BudgetEnv([StorageTool()])
    turn = _turn(TOOL_CALL, TOOL_CALL)

    step = await env.step(await env.reset(), turn, workspace=InMemoryWorkspace())

    assert len(turn.tool_calls) == 2
    assert step.next_state.calls_remaining == 3
    assert step.next_state.calls_max == 5  # the refill line is the trigger's to write


@pytest.mark.asyncio
async def test_a_state_without_the_budget_fields_is_left_alone() -> None:
    """A task naming its budget something else still charges it in its own override."""
    env = SubmitEnv([StorageTool()])

    ws = InMemoryWorkspace()

    step = await env.step(SubmitState(), _turn(TOOL_CALL), workspace=ws)

    assert not hasattr(step.next_state, "calls_remaining")


@pytest.mark.asyncio
async def test_a_state_with_no_submit_fields_can_dispatch_and_receive_feedback() -> (
    None
):
    """Submission and reminders require no bookkeeping on task state."""
    env = _PlainEnv([StorageTool()])
    state = await env.reset()  # the check passes: nothing is in play

    ran = await env.step(state, _turn(TOOL_CALL), workspace=InMemoryWorkspace())
    assert ran.next_state is state  # not even copied

    # No-tool feedback leaves this plain state unchanged.
    silent = await env.step(state, HermesToolCallParser().parse(_result("d</think>x")))
    assert not silent.done
    assert silent.next_state is state


@pytest.mark.asyncio
async def test_reset_validates_initial_state_supplied_by_a_mixin() -> None:
    """A shared initial-state hook still passes through the common validation."""

    class _ResetMixin:
        async def initial_state(self) -> _Plain:
            return _Plain()

    class _Mixed(_ResetMixin, ToolEnv[_Plain]):
        pass

    assert await _Mixed([StorageTool()]).reset() == _Plain()


@pytest.mark.asyncio
async def test_reset_builds_fresh_state_from_environment_configuration() -> None:
    class _Seeded(ToolEnv[_Plain]):
        def __init__(self, seed: int = 0) -> None:
            super().__init__([StorageTool()])
            self.seed = seed

        async def initial_state(self) -> _Plain:
            return _Plain(n=self.seed)

    env = _Seeded(seed=3)
    first, second = await asyncio.gather(env.reset(), env.reset())
    assert first.n == second.n == 3
    assert first is not second
    assert (await _Seeded().reset()).n == 0


@pytest.mark.asyncio
async def test_only_the_mixins_opt_a_state_into_charging_and_answers() -> None:
    """A field that merely shares a name is the task's own business."""

    @dataclass(frozen=True)
    class _Named:
        calls_remaining: int = 3
        answer: str | None = None

    class _NamedEnv(ToolEnv[_Named]):
        async def initial_state(self) -> _Named:
            return _Named()

    env = _NamedEnv([])
    state = await env.reset()
    assert env.on_tool_calls(state, ToolTurn(text="", tool_calls=())) is state
    assert env.on_submit(state, ToolTurn(text=""), {"answer": "x"}) is state
