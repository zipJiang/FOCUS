"""What a return is worth at a prefix: the baseline, the advantage, the correction.

Statements about the tree only. Where the next rollout is worth spending -- the priors
and the score that answer it -- is an estimator's question, and its tests live with the
estimator in ``tests/test_preparation_estimators.py``.
"""

from __future__ import annotations

import math

import pytest

from focus.reward.config import RewardConfig
from focus.scheduler import SchedulerView
from focus.scheduler.core.returns import (
    edge_return,
    finished_below,
    finished_through,
    importance,
)
from tests.helpers import VERSION
from tests.helpers import rollout as _rollout
from tests.trees import region as _region
from tests.trees import root_state
from tests.trees import stamped_child as _child


def _tree():
    state = root_state(_rollout(), width=4)
    return state, SchedulerView(state)


# -- the return of one edge ----------------------------------------------------------


def test_an_edge_return_starts_where_its_parent_s_log_ended() -> None:
    """The shared history belongs to the fork, not to the edge leaving it.

    Siblings are compared by what happened *after* the branch point, so a return that
    swept up the reward their parent already collected would shift all of them by the
    same amount -- invisible in a difference, and wrong the moment two edges leave
    different parents.
    """
    state, _ = _tree()
    parent = _child(state, 0, _rollout(_region(reward=1.0)))
    child = _child(
        state,
        parent.id,
        _rollout(_region(reward=1.0), _region(reward=0.5, done=True)),
    )

    assert edge_return(child, parent, RewardConfig()) == pytest.approx(0.5)
    assert edge_return(child, state.nodes[0], RewardConfig()) == pytest.approx(1.5)


# -- importance ----------------------------------------------------------------------


def test_the_ratio_is_exactly_one_for_an_edge_drawn_from_the_anchor() -> None:
    state, _ = _tree()
    from_anchor = _child(state, 0, _rollout(_region()))
    same_channel = _child(state, 0, _rollout(_region()), proposal_id="explore")

    config = RewardConfig()
    assert importance(from_anchor, state.nodes[0], config, VERSION, config.clip) == 1.0
    same = RewardConfig(anchor_version=VERSION)
    assert importance(same_channel, state.nodes[0], same, VERSION, same.clip) == 1.0


def test_the_ratio_is_clipped_against_heavy_tails() -> None:
    state, _ = _tree()
    child = _child(state, 0, _rollout(_region(anchor=5.0)), proposal_id="explore")

    ratio = importance(child, state.nodes[0], RewardConfig(), VERSION, 10.0)
    uncapped = math.exp(5.0 - (-1.0))

    assert uncapped > 10.0
    assert ratio == 10.0


def test_the_clip_survives_a_gap_no_exponential_could_represent() -> None:
    """The heavy tail the clip is *for* is the one that used to kill the export.

    These are sequence log-ratios: a hundredth of a nat per token over a long edge
    already clears the ~709 nats above which ``math.exp`` raises ``OverflowError``. So
    a clip applied to the exponential's *result* never ran on the edges it was written
    for -- one off-anchor rollout took the whole prompt's export down with ``math range
    error``.
    """
    state, _ = _tree()
    child = _child(state, 0, _rollout(_region(anchor=5_000.0)), proposal_id="explore")

    assert importance(child, state.nodes[0], RewardConfig(), VERSION, 10.0) == 10.0


def test_a_ratio_the_anchor_would_never_have_produced_underflows_to_zero() -> None:
    """The other tail needs no clip: an edge off the anchor's support gets no weight."""
    state, _ = _tree()
    child = _child(state, 0, _rollout(_region(anchor=-5_000.0)), proposal_id="explore")

    assert importance(child, state.nodes[0], RewardConfig(), VERSION, 10.0) == 0.0


def test_the_ratio_follows_the_two_channels_when_they_differ() -> None:
    state, _ = _tree()
    child = _child(state, 0, _rollout(_region(anchor=-2.0)), proposal_id="explore")

    config = RewardConfig()
    assert importance(
        child, state.nodes[0], config, VERSION, config.clip
    ) == pytest.approx(math.exp(-2.0 - (-1.0)))


# -- finished paths ------------------------------------------------------------------


def test_finished_helpers_walk_the_whole_tree() -> None:
    """Under A/B symmetry, later-pass edges are part of the same tree walk."""
    state, view = _tree()
    fold = _child(state, 0, _rollout(_region()), pass_id="pass-1")
    leaf = _child(
        state,
        fold.id,
        _rollout(_region(), _region(reward=1.0, done=True)),
        pass_id="pass-1",
    )

    assert finished_through(view, fold)
    assert finished_below(view, fold) == [leaf]
