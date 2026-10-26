from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from focus.generation import GenerateResult
from focus.generation.parsing import ActionParser, TextActionParser
from focus.harness import Env, StepResult


@dataclass(frozen=True)
class CounterState:
    count: int = 0
    log: tuple[str, ...] = ()


class ToyEnv(Env[CounterState, str]):
    """A pure env: each action appends to a log and increments; 'stop' ends it."""

    def __init__(self, *, horizon: int = 3) -> None:
        self._horizon = horizon

    async def reset(self) -> CounterState:
        return CounterState()

    async def step(
        self, state: CounterState, action: str, *, workspace=None
    ) -> StepResult[CounterState]:
        nxt = replace(state, count=state.count + 1, log=(*state.log, action))
        done = action == "stop" or nxt.count >= self._horizon
        return StepResult(
            next_state=nxt,
            messages=({"role": "user", "content": f"seen {action!r} ({nxt.count})"},),
            reward_outcome=1.0 if action == "stop" else 0.0,
            done=done,
        )


@pytest.mark.asyncio
async def test_reset_returns_initial_state() -> None:
    assert await ToyEnv().reset() == CounterState()


@pytest.mark.asyncio
async def test_step_is_a_pure_transition_no_model() -> None:
    env = ToyEnv(horizon=3)
    s0 = await env.reset()
    r = await env.step(s0, "look")
    assert r.next_state == CounterState(count=1, log=("look",))
    assert r.messages == ({"role": "user", "content": "seen 'look' (1)"},)
    assert r.reward == 0.0
    assert r.done is False
    assert s0.count == 0  # original state untouched -> branch-safe


@pytest.mark.asyncio
async def test_done_on_stop_and_on_horizon() -> None:
    env = ToyEnv(horizon=2)
    s = await env.reset()
    assert (await env.step(s, "stop")).done  # explicit stop
    r1 = await env.step(s, "go")
    assert not r1.done
    assert (await env.step(r1.next_state, "go")).done  # hit horizon (count 2)


@pytest.mark.asyncio
async def test_two_steps_from_one_state_do_not_alias() -> None:
    env = ToyEnv()
    s = await env.reset()
    a = await env.step(s, "a")
    b = await env.step(s, "b")
    assert a.next_state.log == ("a",)
    assert b.next_state.log == ("b",)  # independent branches


def test_text_action_parser_strips_completion() -> None:
    parser = TextActionParser()
    assert parser.parse(GenerateResult(text="  do x  ")) == "do x"


@pytest.mark.asyncio
async def test_parse_then_apply_seam_without_a_model() -> None:
    # The runner flow, minus generation: a fake result -> parse -> env.step.
    parser: ActionParser[str] = TextActionParser()
    env = ToyEnv()
    result = GenerateResult(text=" stop ")  # as if a model produced this
    action = parser.parse(result)
    step = await env.step(await env.reset(), action)
    assert action == "stop"
    assert step.done
    assert step.reward == 1.0
