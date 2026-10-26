"""Behavior at the boundary between search phases and preparation."""

from __future__ import annotations

import pickle

import pytest

import focus
from focus import DirectBranchTd, ErrorReduction, Fifo, Phase, Search, Selective
from focus.allocation import Observations
from focus.extend import NodeInfo, Priority
from focus.preparation import prepare_samples
from focus.scheduler import PASS_KEY
from tests.estimators import FixedWeight
from tests.test_collector import _collector


class Capture(Priority):
    """Buys nothing; keeps the evidence it was shown, by reference."""

    name = "capture"

    def __init__(self) -> None:
        self.calls = 0
        self.ledger: dict[int, Observations] = {}

    def __call__(self, node: NodeInfo) -> None:
        self.calls += 1
        self.ledger[node.id] = node.observations


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_phase_budget_is_a_nonnegative_integer(budget) -> None:
    with pytest.raises((TypeError, ValueError), match="budget"):
        Phase(Fifo(), budget, max_children=1)


def test_branching_caps_and_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        Phase(Fifo(), 1, max_children=0)
    with pytest.raises(ValueError, match="max_concurrency"):
        Search([], max_concurrency=0)


def test_selective_is_a_fifo_warmup_then_error_reduction() -> None:
    search = Selective(32, warmup=16, max_pending_per_node=3)
    warmup, selective = search.phases
    assert isinstance(warmup.priority, Fifo)
    assert (warmup.budget, warmup.max_children, warmup.max_pending) == (16, 2, None)
    assert isinstance(selective.priority, ErrorReduction)
    assert (selective.budget, selective.max_children) == (32, None)
    assert selective.max_pending == 3
    assert search.requires_critic


async def test_empty_configuration_returns_only_the_initialized_root() -> None:
    collector = _collector(passes=())
    state = await collector.search("solve it")
    assert len(state.nodes) == 1
    assert state.stats["turns"] == state.stats["rollouts"] == 0
    for estimator in (None, DirectBranchTd()):
        batch = collector.prepare(state, estimator=estimator)
        assert not batch.actor
        assert not batch.critic


async def test_zero_budget_phases_are_skipped_and_indices_preserved() -> None:
    skipped, observed = Capture(), Capture()
    collector = _collector(
        passes=(
            Phase(skipped, 0, max_children=1),
            Phase(ErrorReduction(), 8, max_children=1),
            Phase(Fifo(), 4, max_children=1),
            Phase(observed, 1, max_pending=1),
        )
    )
    state = await collector.search("solve it")
    assert skipped.calls == 0
    assert observed.calls
    assert {
        n.metadata[PASS_KEY] for n in state.nodes.values() if PASS_KEY in n.metadata
    } <= {1, 2}
    assert all(c.m_pending == 0 for c in observed.ledger.values())
    assert all(c.returns.n >= 1 for c in observed.ledger.values())


async def test_gate_changes_reopen_nodes_between_phases() -> None:
    capture = Capture()
    collector = _collector(
        passes=(
            Phase(Fifo(), 4, max_children=1),
            Phase(Fifo(), 8, max_pending=1),
            Phase(capture, 1, max_pending=1),
        )
    )
    state = await collector.search("solve it")
    assert len(state.nodes[state.root_id].children) > 1
    assert capture.ledger
    root = capture.ledger[state.root_id]
    assert root.m_done == len(state.nodes[state.root_id].children)
    assert all(c.m_pending == 0 for c in capture.ledger.values())
    assert {
        n.metadata[PASS_KEY] for n in state.nodes.values() if PASS_KEY in n.metadata
    } == {0, 1}


async def test_one_tree_prepares_independent_recipes_without_mutating_search() -> None:
    collector = _collector(
        passes=(
            Phase(Fifo(), 4, max_children=1),
            Phase(Fifo(), 8, max_pending=1),
        )
    )
    state = await collector.search("solve it")
    before = pickle.dumps(state)
    gae = collector.prepare(state, estimator=DirectBranchTd())
    fixed = collector.prepare(state, estimator=FixedWeight())
    critic = collector.prepare(state, estimator=None)
    assert pickle.dumps(state) == before
    assert gae.actor
    assert fixed.actor
    assert {r.estimator for r in gae.actor} == {"direct_branch_td"}
    assert {r.estimator for r in fixed.actor} == {"test_fixed"}
    assert len(gae.actor) == len({r.node_id for r in gae.actor})
    assert gae.critic == fixed.critic == critic.critic
    assert not critic.actor
    # the paper's recipe is the default
    default = prepare_samples(state, reward_config=collector.reward)
    assert {r.estimator for r in default.actor} == {"direct_branch_td"}


def test_removed_api_is_not_retained_as_aliases() -> None:
    for name in (
        "Runtime",
        "RolloutConfig",
        "RolloutSpec",
        "SearchPass",
        "run_search",
        "collect_samples",
        "Allocator",
        "FifoAllocator",
        "ValueRefinement",
        "WidthGating",
        "ConcurrencyGating",
        "RewardModel",
        "MixtureProposal",
        "TokenScorer",
        "RefinedTdEstimator",
        "VinePpoConfig",
    ):
        assert not hasattr(focus, name)
    assert not hasattr(DirectBranchTd(), "build")
    assert not hasattr(Fifo(), "advantages")
