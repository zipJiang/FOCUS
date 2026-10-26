"""Shared tree fixtures for the export, weighting, scheduler and slime tests.

The counterpart to :mod:`tests.helpers`: that module builds the *log* -- turns,
rollouts, a runner over the fakes -- and this one builds the *tree* over it.

There is no rollout builder here: :func:`tests.helpers.rollout` is the one, and a
checkpoint records its folds as tagged turns rather than as a counter beside them.
"""

from __future__ import annotations

from typing import Any

from focus import (
    WidthGating,
)
from focus.harness import RolloutState, Turn
from focus.scheduler import PASS_KEY, Node, SchedulerState
from tests.helpers import VERSION, turn


def stamp(node: Node[Any], search_pass: int = 0) -> None:
    """Mark ``node`` as produced by ``search_pass``, as an expansion would."""
    node.metadata[PASS_KEY] = search_pass


def root_state(payload: Any, *, width: int = 8) -> SchedulerState[Any]:
    return SchedulerState.root(payload, gating=WidthGating(width))


def training_root_state(
    payload: Any, *, width: int = 8, reward_config=None
) -> SchedulerState[Any]:
    """A tree holding ``payload`` as its root, branchable ``width`` ways."""
    state = SchedulerState.root(payload, gating=WidthGating(width))
    if isinstance(payload, RolloutState):
        from focus.allocation.evidence import RolloutLedger
        from focus.reward.config import RewardConfig
        from focus.scheduler import SchedulerView

        RolloutLedger(reward_config or RewardConfig()).bind(SchedulerView(state))
        state.nodes[0].metadata["test_training_fixture"] = True
        state.nodes[0].critic["v-base"] = 0.0
        state.nodes[0].metadata["value_contexts"] = {"v-base": "root context"}
    return state


def stamped_child(
    state: SchedulerState[Any],
    parent_id: int,
    payload: Any,
    *,
    search_pass: int = 0,
) -> Node[Any]:
    """Attach ``payload`` under ``parent_id`` as an expansion would."""
    node = state.add_child(parent_id, payload)
    stamp(node, search_pass)
    if state.nodes[0].metadata.get("test_training_fixture"):
        from focus.allocation.evidence import RolloutLedger
        from focus.scheduler import SchedulerView

        evidence = state.nodes[0].metadata["rollout_evidence"]
        ledger = RolloutLedger(evidence.reward_config)
        ledger.bind(SchedulerView(state))
        node.critic["v-base"] = 0.0
        node.metadata["value_contexts"] = {"v-base": f"checkpoint {node.id}"}
        ledger.record(state.nodes[parent_id], [node])
    return node


def region(reward: float = 0.0, done: bool = False) -> Turn[None]:
    """One task turn under the behavior channel."""
    return turn([-1.0], reward=reward, done=done, version=VERSION)


__all__ = [
    "VERSION",
    "region",
    "root_state",
    "stamp",
    "stamped_child",
]


class SearchPayload(RolloutState[None]):
    """Generic scheduler fixture: every live node can branch, independent of Runner.

    Allocation tests exercise arbitrary Payload trees with synthetic reward logs.
    Real runner checkpoint eligibility is covered by the harness/execution tests.
    """

    @property
    def forkable(self) -> bool:
        return not self.done and not self.truncated


def search_payload(*turns: Turn[None], truncated: bool = False) -> SearchPayload:
    from tests.helpers import rollout

    rs = rollout(*turns, truncated=truncated)
    return SearchPayload(
        messages=rs.messages,
        seed=rs.seed,
        state=rs.state,
        workspace=rs.workspace,
        turns=rs.turns,
        truncated=rs.truncated,
    )
