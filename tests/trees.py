"""Shared tree fixtures for the export, weighting, scheduler and slime tests.

The counterpart to :mod:`tests.helpers`: that module builds the *log* -- turns,
rollouts, a runner over the fakes -- and this one builds the *tree* over it. A root, a
stamped child, and the provenance stamp itself were copied into five test modules, and
a stamp that drifts between them is not a harmless difference: whether an edge is
exportable, and whether it needs an importance correction, are both read off it.

There is no rollout builder here: :func:`tests.helpers.rollout` is the one, and a
checkpoint records its folds as tagged turns rather than as a counter beside them.
"""

from __future__ import annotations

from typing import Any

from focus import (
    WidthGating,
)
from focus.harness import RolloutState, Turn
from focus.scheduler import (
    EdgeProvenance,
    Node,
    SchedulerState,
    stamp_provenance,
)
from tests.helpers import VERSION, turn

ANCHOR = "anchor"


def stamp(pass_id: str = "pass-0", proposal_id: str = ANCHOR) -> EdgeProvenance:
    """The provenance a first search pass drawing from the anchor would leave."""
    return EdgeProvenance(
        proposal_id=proposal_id,
        continuation_id="nu-0",
        behavior_version=VERSION,
        reward_config_id="rc-0",
        pass_id=pass_id,
    )


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
    pass_id: str = "pass-0",
    proposal_id: str = ANCHOR,
) -> Node[Any]:
    """Attach ``payload`` under ``parent_id`` as an expansion would: with provenance."""
    node = state.add_child(parent_id, payload)
    stamp_provenance(node, stamp(pass_id, proposal_id))
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


def region(
    reward: float = 0.0, done: bool = False, anchor: float | None = None
) -> Turn[None]:
    """One task turn under the behavior channel, optionally with an anchor logprob."""
    step = turn([-1.0], reward=reward, done=done, version=VERSION)
    if anchor is not None:
        step.logprobs[ANCHOR] = (anchor,)
    return step


__all__ = [
    "ANCHOR",
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
