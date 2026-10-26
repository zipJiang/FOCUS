"""What a return is worth at a prefix: the baseline and the advantage.

Statements about the tree only. Where the next rollout is worth spending -- the priors
and the score that answer it -- is an estimator's question, and its tests live with the
estimator in ``tests/test_preparation_estimators.py``.
"""

from __future__ import annotations

import pytest

from focus.reward.config import RewardConfig
from focus.scheduler import SchedulerView
from focus.scheduler.core.returns import (
    edge_return,
    finished_below,
    finished_through,
)
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


# -- finished paths ------------------------------------------------------------------


def test_finished_helpers_walk_the_whole_tree() -> None:
    """Under A/B symmetry, later-pass edges are part of the same tree walk."""
    state, view = _tree()
    fold = _child(state, 0, _rollout(_region()), search_pass=1)
    leaf = _child(
        state,
        fold.id,
        _rollout(_region(), _region(reward=1.0, done=True)),
        search_pass=1,
    )

    assert finished_through(view, fold)
    assert finished_below(view, fold) == [leaf]
