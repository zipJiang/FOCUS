from __future__ import annotations

import math
from typing import Any

import pytest

from focus import (
    TokenEntropy,
)
from focus.allocation import RolloutLedger
from focus.harness import RolloutState, Turn
from focus.scheduler import SchedulerState, SchedulerView
from focus.search import _PrioritySession
from focus.values import RewardConfig
from tests.helpers import turn as _step
from tests.trees import root_state
from tests.trees import search_payload as _rollout


def _tree(
    parent: RolloutState[None], *children: RolloutState[None]
) -> SchedulerState[Any]:
    state = root_state(parent)
    for child in children:
        state.add_child(0, child)
    return state


def TokenEntropySession(window: int = 8, tags: Any = ("task",)) -> _PrioritySession:
    return _PrioritySession(TokenEntropy(window, tags), RolloutLedger(RewardConfig()))


def _score(state: SchedulerState[Any], policy: _PrioritySession) -> float:
    return policy.score(SchedulerView(state), state.nodes[0])


def _h(logprobs: list[float]) -> list[float]:
    return [math.exp(lp) * -lp for lp in logprobs]


# -- the formula, ported from the pre-rewrite engine's own oracles --------------------

LOGPROBS = [0.0, 0.0, 0.0, -1.0, 0.0, 0.0]


def test_a_small_window_localizes_the_uncertainty_spike() -> None:
    """The point of windowing: one uncertain token must not average away."""
    state = _tree(_rollout(), _rollout(_step(LOGPROBS)))

    small = _score(state, TokenEntropySession(window=1))
    whole = _score(state, TokenEntropySession(window=len(LOGPROBS)))

    assert small == pytest.approx(max(_h(LOGPROBS)))
    # a window no shorter than the region falls back to the plain mean
    assert whole == pytest.approx(sum(_h(LOGPROBS)) / len(LOGPROBS))
    assert small > whole


def test_masked_positions_do_not_dilute_the_score() -> None:
    """0.0 at an observation position is `p=1`, i.e. fabricated certainty."""
    bare = _tree(_rollout(), _rollout(_step(LOGPROBS)))
    padded = _tree(_rollout(), _rollout(_step(LOGPROBS, masked=20)))
    policy = TokenEntropySession(window=3)

    assert _score(padded, policy) == pytest.approx(_score(bare, policy))


def test_only_the_tokens_added_since_the_parent_count() -> None:
    """The edge is the tail past the parent's log, not the whole rollout."""
    shared = _step([0.0, 0.0, 0.0])
    parent = _rollout(shared)
    # the child is the parent's log plus one more, uncertain, turn
    child = _rollout(shared, _step([-1.0]))
    state = _tree(parent, child)

    score = _score(state, TokenEntropySession(window=8))

    assert score == pytest.approx(math.exp(-1.0) * 1.0)  # the added token alone


def test_a_run_that_would_split_into_packs_still_scores_every_turn() -> None:
    """The edge is turns, so how they happen to pack cannot hide one.

    Qwen3 stripping ``<think>`` makes each turn its own pack, and the reading that only
    looked at the first would score turn 1 alone -- here the only uncertain token sits
    in turn 2.
    """
    first, second = [0.0, 0.0], [-1.0, 0.0]
    turns: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3, 4), logprobs=tuple(first)),
        # prefix drops the generated 4 -> not an extension of turn 1 -> a second pack
        Turn(prefix=(1, 2, 3, 9), tokens=(7, 8), logprobs=tuple(second)),
    )
    from focus.harness import regions

    assert len(regions(turns)) == 2  # the fixture is only interesting if it splits
    state = _tree(_rollout(), _rollout(*turns))

    spike = _score(state, TokenEntropySession(window=1))
    whole = _score(state, TokenEntropySession(window=99))

    assert spike == pytest.approx(max(_h(first + second)))  # found in turn 2
    assert whole == pytest.approx(sum(_h(first + second)) / 4)
    # the first-pack-only reading would have seen nothing but certainty
    assert max(_h(first)) == 0.0
    assert spike > 0.0


def test_only_task_turns_count_by_default() -> None:
    """A fold checkpoint's edge is the work plus the fold that ended it; the fold's
    summary tokens are not uncertainty about the task, so they are left out."""
    task, fold = [0.0, -0.5], [-1.0, -2.0]
    state = _tree(
        _rollout(),
        _rollout(_step(task), _step(fold, tag="fold")),
    )

    default = _score(state, TokenEntropySession(window=99))  # plain mean over the edge
    every = _score(state, TokenEntropySession(window=99, tags=None))

    assert default == pytest.approx(sum(_h(task)) / 2)
    assert every == pytest.approx(sum(_h(task + fold)) / 4)


def test_an_edge_that_is_only_a_fold_is_no_evidence() -> None:
    """A fold generates nothing task-side; that reads as unscored (inf)."""
    state = _tree(_rollout(), _rollout(_step([-1.0], tag="fold")))
    assert _score(state, TokenEntropySession()) == math.inf


def test_children_are_averaged_unweighted_not_token_weighted() -> None:
    """A long child must not outvote a short one -- the 'average' in the name."""
    short, long = [-1.0], [0.0] * 40
    state = _tree(_rollout(), _rollout(_step(short)), _rollout(_step(long)))

    score = _score(state, TokenEntropySession(window=8))

    edges = [math.exp(-1.0), 0.0]  # short child's single token; long child's zeros
    assert score == pytest.approx(sum(edges) / 2)


def test_a_node_with_no_scored_children_is_optimistic() -> None:
    """Untried tips outrank everything, so they get expanded first."""
    childless = _tree(_rollout())
    assert _score(childless, TokenEntropySession()) == math.inf

    # ...and a child whose region carries no logprobs at all is no evidence either
    unscored = _tree(_rollout(), _rollout(Turn(prefix=(), tokens=(1,))))
    assert _score(unscored, TokenEntropySession()) == math.inf


# -- selection -----------------------------------------------------------------------


def test_select_takes_the_least_certain_node() -> None:
    state = root_state(_rollout())
    # two candidate parents, reached by identical confident regions...
    a = state.add_child(0, _rollout(_step([0.0])))
    b = state.add_child(0, _rollout(_step([0.0])))
    # ...one branched on into another confident region, the other into an uncertain one
    # (both finished, so the frontier is the two branch points, not their leaves)
    state.add_child(a.id, _rollout(_step([0.0]), _step([0.0], done=True)))
    state.add_child(b.id, _rollout(_step([0.0]), _step([-1.0], done=True)))

    chosen = TokenEntropySession(window=8).choose(SchedulerView(state))

    assert chosen is not None
    assert chosen.id == b.id


def test_without_logprobs_selection_is_deterministic_rather_than_broken() -> None:
    """No `SamplingParams(logprobs=...)` -> every node ties at inf, lowest id wins."""
    state = root_state(_rollout())
    a = state.add_child(0, _rollout())
    state.add_child(a.id, _rollout(Turn(prefix=(), tokens=(1,))))

    chosen = TokenEntropySession().choose(SchedulerView(state))

    assert chosen is not None
    assert chosen.id == 0


def test_validation() -> None:
    with pytest.raises(ValueError, match="window"):
        TokenEntropySession(window=0)
    assert TokenEntropy(window=16).name == "token_entropy"
    # it reads generated logprobs only: no critic requirement
    assert not TokenEntropy().requires_critic
