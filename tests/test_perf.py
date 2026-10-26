"""A floor under the library's own CPU cost: the export path stays sub-quadratic.

Two things on this path go quadratic if written naively, and both once were. Packing a
run of turns compares each turn's prefix against everything packed so far, and that
prefix *is* the whole conversation -- so a comparison written as a per-token Python
loop costs O(turns x prefix) at interpreter speed. And every subtree walk asks each
payload whether it is ``done``, which rescans the turn log unless the answer is kept.
Neither shows up in a correctness test -- the numbers come out right either way, just
slowly -- so this is the test that notices.

It is a *shape* check, not a stopwatch. The body takes about 0.2s here and the bound is
2s, an order of magnitude of headroom: a loaded CI box, a debug build or a slower
laptop cannot fail it, and neither can an honest 30%. What cannot pass it is the thing
worth catching -- a per-token interpreted loop or a re-pack per node, which cost 10x to
100x, not 30%. Marked ``perf``, so it can be deselected (``-m "not perf"``) where even
a loose wall-clock assertion is unwanted.
"""

from __future__ import annotations

import time

import pytest

from focus import (
    prepare_samples,
)
from focus.harness import RolloutState, StepResult, Turn
from focus.harness.packing import regions
from focus.harness.workspace import NullWorkspace
from focus.scheduler import SchedulerView
from focus.values import RewardConfig
from tests.trees import stamped_child
from tests.trees import training_root_state as root_state

# A realistic agentic shape: a couple of thousand tokens of system prompt and tool
# schemas, then a few hundred tokens of generation plus observation per turn -- so the
# packed prefix a turn is compared against is thousands of tokens long, which is what
# makes the difference between the linear and the quadratic reading visible.
BASE_TOKENS = 2000
GENERATED = 200
OBSERVATION = 100
#: Turns in the longest log, and nodes per line: a line's k-th node holds the first
#: ``ROOT_TURNS + k`` turns of its log, exactly as a per-turn strategy checkpoints them.
ROOT_TURNS = 10
DEPTH = 20
BRANCHES = 20


def _log(count: int) -> tuple[Turn[None], ...]:
    """``count`` turns whose prefixes chain, as an appending chat template renders them.

    Each turn is conditioned on everything before it, so the packed prefix grows the way
    a conversation does. Only the last turn ends the episode: a line's deepest
    checkpoint is what makes every edge above it exportable.
    """
    turns: list[Turn[None]] = []
    prefix = tuple(range(BASE_TOKENS))
    for index in range(count):
        tokens = tuple(range(index * GENERATED, (index + 1) * GENERATED))
        turns.append(
            Turn(
                prefix=prefix,
                tokens=tokens,
                logprobs=(-0.5,) * GENERATED,
                transition=StepResult(
                    next_state=None,
                    reward_outcome=1.0,
                    done=index == count - 1,
                ),
            )
        )
        prefix = prefix + tokens + tuple(range(OBSERVATION))
    return tuple(turns)


def _payload(turns: tuple[Turn[None], ...]) -> RolloutState[None]:
    """A checkpoint over ``turns`` -- the payload shape the exporter reads."""
    return RolloutState(
        messages=(), seed=(), state=None, workspace=NullWorkspace(), turns=turns
    )


@pytest.mark.perf
def test_building_a_result_over_a_full_tree_stays_sub_quadratic() -> None:
    """400 stamped edges over 30-turn logs: export, then pack every log once more."""
    logs = [_log(ROOT_TURNS + DEPTH) for _ in range(BRANCHES)]
    state = root_state(_payload(logs[0][:ROOT_TURNS]), width=BRANCHES)
    for log in logs:
        parent = 0
        for depth in range(DEPTH):
            parent = stamped_child(
                state, parent, _payload(log[: ROOT_TURNS + depth + 1])
            ).id
    view: SchedulerView[RolloutState[None]] = SchedulerView(state)
    assert len(view.nodes()) == BRANCHES * DEPTH + 1

    start = time.perf_counter()
    result = prepare_samples(
        view._state,
        estimator=None,
        reward_config=RewardConfig(),
    )
    # The re-scoring pass's shape: every node's whole log packed, not just its edge.
    for node in view.nodes():
        regions(node.payload.turns)
    elapsed = time.perf_counter() - start

    assert len(result.critic) >= BRANCHES  # the work was real, not skipped
    assert elapsed < 2.0, f"export over {len(result.critic)} edges took {elapsed:.2f}s"
