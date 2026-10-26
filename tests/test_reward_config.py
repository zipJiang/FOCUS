from __future__ import annotations

from typing import Any

import pytest

from focus.harness import StepResult, Turn
from focus.values import RewardConfig
from tests.helpers import rollout, turn


def _transition(outcome: float = 0.0, step: float = 0.0, done: bool = False):
    return StepResult(
        next_state=None, reward_outcome=outcome, reward_step=step, done=done
    )


def _turns(*transitions: Any) -> tuple[Turn[Any], ...]:
    """One turn per transition -- the shape the runner appends."""
    return tuple(Turn(prefix=(), tokens=(), transition=t) for t in transitions)


# -- the raw fields stay raw ---------------------------------------------------------


def test_the_two_rewards_are_stored_apart_and_summed_only_on_request() -> None:
    """An env that pre-mixed them would bake in a weight nothing could revisit."""
    transition = _transition(outcome=1.0, step=0.25)

    assert transition.reward_outcome == 1.0
    assert transition.reward_step == 0.25
    assert transition.reward == 1.25  # the raw view, for ranking


def test_a_rollout_sums_each_channel_over_its_turns() -> None:
    rs = rollout(
        *_turns(
            _transition(step=0.5),
            _transition(step=0.25),
            _transition(outcome=1.0, done=True),
        )
    )

    assert rs.reward_outcome == 1.0
    assert rs.reward_step == 0.75
    assert rs.reward == 1.75


# -- beta_step: what shaping is worth against outcome --------------------------------


@pytest.mark.parametrize(
    ("beta", "expected"),
    [
        (0.0, 1.0),  # outcome only: shaping is ignored entirely
        (1.0, 1.75),  # the raw sum
        (0.5, 1.375),
    ],
)
def test_beta_step_weighs_shaping_against_outcome(beta: float, expected: float) -> None:
    turns = _turns(
        _transition(step=0.5),
        _transition(step=0.25),
        _transition(outcome=1.0, done=True),
    )

    got = RewardConfig(beta_step=beta).configured_return(turns)

    assert got == pytest.approx(expected)


# -- gamma: discounting is per transition --------------------------------------------


def test_gamma_discounts_once_per_transition() -> None:
    turns = _turns(*(_transition(step=1.0) for _ in range(3)))

    assert RewardConfig(gamma=0.5).configured_return(turns) == pytest.approx(1.75)


def test_gamma_one_is_the_plain_sum() -> None:
    turns = _turns(*(_transition(step=1.0) for _ in range(4)))

    assert RewardConfig(gamma=1.0).configured_return(turns) == pytest.approx(4.0)


def test_a_fold_is_neither_a_reward_nor_a_time_step() -> None:
    """Folding rewrites context; it does not act on the world, so it costs no discount.

    A rollout that folded between two turns must discount exactly like one that did not,
    or how often the compactor happened to fire would silently reweight the task work.
    """
    config = RewardConfig(gamma=0.5)
    plain = (turn([-1.0], reward=1.0), turn([-1.0], reward=1.0))
    folded = (plain[0], turn([-1.0], tag="fold"), plain[1])

    assert config.configured_return(plain) == pytest.approx(1.5)
    assert config.configured_return(folded) == pytest.approx(1.5)


def test_a_suffix_return_restarts_the_discount_at_its_own_start() -> None:
    """`start` is a suffix from a node, and that node is where its clock begins."""
    turns = _turns(*(_transition(step=1.0) for _ in range(3)))
    config = RewardConfig(gamma=0.5)

    assert config.configured_return(turns, start=1) == pytest.approx(1.0 + 0.5)
    assert config.configured_return(turns, start=3) == 0.0  # nothing left


def test_defaults_are_the_core_configuration() -> None:
    config = RewardConfig()

    assert (config.gamma, config.value_prior_strength, config.beta_step) == (
        1.0,
        1.0,
        1.0,
    )
