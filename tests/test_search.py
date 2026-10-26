"""Search phases and priorities: FIFO by arrival, caps per checkpoint."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import pytest

from focus import Fifo, Phase
from focus.scheduler import (
    AllocationLedger,
    Budget,
    ConcurrencyGating,
    Expander,
    FifoSession,
    Scheduler,
    SchedulerState,
    WidthGating,
)
from focus.search import _PhaseGating, _PrioritySession
from focus.values import RewardConfig
from tests.scheduler.fakes import Toy


class ToyLedger(AllocationLedger[Toy]):
    """Purchase counts only: toy payloads carry no turns to record returns from."""

    reward_config = RewardConfig()


class BranchingExpander(Expander[Toy]):
    """One or two children per expansion, chosen by a seeded coin, recorded in order."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.expanded: list[int] = []

    async def expand(self, payload: Toy) -> tuple[list[Toy], Mapping[str, object]]:
        self.expanded.append(payload.n)
        if payload.n >= 2:
            return [Toy(3)], {}
        return [Toy(payload.n + 1)] * self.rng.choice((1, 2)), {}


async def _run(session: Any, gating: Any, seed: int) -> list[list[int]]:
    state: SchedulerState[Toy] = SchedulerState.root(Toy(0), gating=gating)
    expander = BranchingExpander(seed)
    await Scheduler(
        expander=expander,
        allocation=session,
        termination=Budget(max_rollouts=12),
        max_concurrency=1,
    ).run_on(state)
    return [state.nodes[i].children for i in sorted(state.nodes)]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "gating", [WidthGating(2), ConcurrencyGating(1)], ids=["width", "pending"]
)
async def test_fifo_priority_matches_frontier_insertion_order(seed, gating) -> None:
    """Arrival order is frontier insertion order, including re-entry after a gate
    frees, so the priority reproduces the first-in-first-out session exactly."""
    expected = await _run(FifoSession(), gating, seed)
    actual = await _run(
        _PrioritySession(Fifo(), ToyLedger()),  # type: ignore[arg-type]
        gating,
        seed,
    )
    assert actual == expected


def test_phase_caps_map_onto_gates() -> None:
    state: SchedulerState[Toy] = SchedulerState.root(
        Toy(0), gating=_PhaseGating(Phase(Fifo(), 1, max_children=2, max_pending=1))
    )
    root = state.nodes[0]
    assert root.gate.available(root)
    root.gate.reserve()
    assert not root.gate.available(root)  # one running continuation at most
    root.gate.release()
    state.add_child(0, Toy(1))
    state.add_child(0, Toy(1))
    assert not root.gate.available(root)  # two branches in total at most
