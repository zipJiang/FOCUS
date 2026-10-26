"""Behavior at the boundary between explicit search passes and preparation."""

from __future__ import annotations

import pickle
from copy import deepcopy
from typing import Any

import pytest

import focus
from focus import (
    AllocationSession,
    Allocator,
    BestFirstAllocator,
    ConcurrencyGating,
    FifoAllocator,
    RefinedTdEstimator,
    RolloutConfig,
    RolloutLedger,
    SearchPass,
    ValueRefinement,
    WidthGating,
    prepare_samples,
    run_search,
)
from focus.allocation import Observations
from focus.scheduler import PASS_KEY
from tests.estimators import FixedWeight
from tests.test_loop import _runtime


class Capture(Allocator):
    name = "capture"

    def __init__(self) -> None:
        self.calls = 0
        self.ledger: dict[int, Observations] = {}
        self.snapshots: list[dict[int, Observations]] = []

    def build(self, view, ledger) -> AllocationSession[Any] | None:
        self.calls += 1
        self.ledger = ledger.observations
        self.snapshots.append(deepcopy(ledger.observations))
        return None


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_pass_budget_is_a_nonnegative_integer(budget) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        SearchPass(FifoAllocator(), budget=budget, gating=WidthGating())


async def test_empty_configuration_returns_only_the_initialized_root() -> None:
    runtime = _runtime(passes=())
    state = await run_search("solve it", runtime)
    assert len(state.nodes) == 1
    assert state.stats["turns"] == state.stats["rollouts"] == 0
    for estimator in (None, RefinedTdEstimator()):
        batch = prepare_samples(
            state,
            estimator=estimator,
            reward_config=runtime.config.reward_config,
            behavior_version=runtime.runner.policy.version,
        )
        assert not batch.actor
        assert not batch.critic


async def test_skips_preserve_indices_without_building_zero_budget_allocators() -> None:
    skipped, observed = Capture(), Capture()
    runtime = _runtime(
        passes=(
            SearchPass(skipped, budget=0, gating=WidthGating()),
            SearchPass(ValueRefinement(), budget=8, gating=WidthGating()),
            SearchPass(FifoAllocator(), budget=4, gating=WidthGating(1)),
            SearchPass(observed, budget=1, gating=WidthGating()),
        )
    )
    state = await run_search("solve it", runtime)
    assert skipped.calls == 0
    assert observed.calls == 1
    assert observed.ledger
    assert {
        n.metadata[PASS_KEY] for n in state.nodes.values() if PASS_KEY in n.metadata
    } == {2}
    assert sum(c.m_done for c in observed.ledger.values()) == state.stats["rollouts"]
    assert all(c.m_pending == 0 for c in observed.ledger.values())
    assert all(c.returns.n >= 1 for c in observed.ledger.values())


async def test_gate_changes_reopen_nodes_and_policies_keep_their_backup() -> None:
    capture = Capture()
    runtime = _runtime(
        passes=(
            SearchPass(FifoAllocator(), budget=4, gating=WidthGating(1)),
            SearchPass(
                BestFirstAllocator(key="mean_return"),
                budget=8,
                gating=ConcurrencyGating(1),
            ),
            SearchPass(capture, budget=1, gating=ConcurrencyGating(1)),
        )
    )
    state = await run_search("solve it", runtime)
    assert len(state.nodes[state.root_id].children) > 1
    assert state.nodes[state.root_id].visits > 0
    assert capture.ledger
    assert sum(c.m_done for c in capture.ledger.values()) == state.stats["rollouts"]
    assert all(c.m_pending == 0 for c in capture.ledger.values())
    assert {
        n.metadata[PASS_KEY] for n in state.nodes.values() if PASS_KEY in n.metadata
    } == {0, 1}


async def test_one_tree_prepares_independent_recipes_without_mutating_search() -> None:
    runtime = _runtime(
        passes=(
            SearchPass(FifoAllocator(), budget=4, gating=WidthGating(1)),
            SearchPass(FifoAllocator(), budget=8, gating=ConcurrencyGating(1)),
        )
    )
    state = await run_search("solve it", runtime)
    before = pickle.dumps(state)
    gae = prepare_samples(
        state,
        estimator=RefinedTdEstimator(),
        reward_config=runtime.config.reward_config,
        behavior_version=runtime.runner.policy.version,
    )
    grpo = prepare_samples(
        state,
        estimator=FixedWeight(),
        reward_config=runtime.config.reward_config,
        behavior_version=runtime.runner.policy.version,
    )
    critic = prepare_samples(
        state,
        estimator=None,
        reward_config=runtime.config.reward_config,
        behavior_version=runtime.runner.policy.version,
    )
    assert pickle.dumps(state) == before
    assert gae.actor
    assert grpo.actor
    assert {r.estimator for r in gae.actor} == {"refined_td"}
    assert {r.estimator for r in grpo.actor} == {"test_fixed"}
    assert len(gae.actor) == len({r.node_id for r in gae.actor})
    assert gae.critic == grpo.critic == critic.critic
    assert not critic.actor
    with pytest.raises(TypeError, match="estimator"):
        prepare_samples(
            state,
            reward_config=runtime.config.reward_config,  # type: ignore[call-arg]
            behavior_version=runtime.runner.policy.version,
        )


def test_removed_api_is_not_retained_as_aliases() -> None:
    for name in (
        "run_rollout",
        "score_search",
        "RolloutResult",
        "Allocation",
        "SelectionPolicy",
        "ScoredSelection",
        "Backup",
        "PolicyAllocator",
        "MeanReturnBackup",
        "NoBackup",
        "ReservingBackup",
        "LedgerBackup",
    ):
        assert not hasattr(focus, name)
    assert not hasattr(RefinedTdEstimator(), "build")
    assert not hasattr(FifoAllocator(), "assign")
    with pytest.raises(TypeError):
        RolloutConfig(weighters=())  # type: ignore[call-arg]


def test_custom_policies_are_built_independently_for_each_search() -> None:
    from focus import RewardConfig, SchedulerView
    from tests.helpers import rollout
    from tests.trees import root_state

    allocator = BestFirstAllocator(key="mean_return")
    view = SchedulerView(root_state(rollout()))
    first = allocator.build(view, RolloutLedger(RewardConfig()))
    second = allocator.build(view, RolloutLedger(RewardConfig()))
    assert first is not second
    assert first.ledger is not second.ledger
    assert first.ledger.observations is not second.ledger.observations
