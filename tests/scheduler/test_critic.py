from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from focus.critic import Critic
from focus.harness import AnyRollout
from focus.harness.transcript import format_transcript
from focus.scheduler import (
    Budget,
    Expander,
    FifoSession,
    NodeScorer,
    Scheduler,
    SchedulerState,
    WidthGating,
)
from focus.scheduler.core.critic import ValuePrediction
from tests.helpers import DeadReward, ShortBatchReward, StalledReward, rollout
from tests.scheduler.fakes import ChainExpander, Toy

# -- fakes ---------------------------------------------------------------------------


class CountingReward(Critic):
    """Score = the serialized payload read back as a number; counts batch calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.scored: list[str] = []

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        self.calls += 1
        self.scored.extend(contexts)
        return [float(c) for c in contexts]


class ToyCritic(Critic):
    """Render a toy payload as its number for any wrapped critic."""

    def __init__(self, inner: Critic, *, timeout: float | None = None) -> None:
        self.inner = inner
        self.timeout = timeout

    def context(self, payload) -> str:
        return str(payload.n)

    async def predict(self, contexts: Sequence[str]) -> Sequence[float]:
        return await self.inner.predict(contexts)


def _tree(*payloads: Toy) -> SchedulerState[Toy]:
    """A width-4 state whose root is the first payload, rest attached as a chain."""
    state: SchedulerState[Toy] = SchedulerState.root(payloads[0], gating=WidthGating(4))
    state.attach_chain(0, list(payloads[1:]))
    return state


# -- tests ---------------------------------------------------------------------------


# -- the default serializer -----------------------------------------------------------


class EchoingReward(Critic):
    """Keeps whatever it was handed, and scores by its length -- so a test can see
    *how* the payload was rendered, not just that some number came back."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        self.seen.extend(contexts)
        return [float(len(context)) for context in contexts]


@pytest.mark.asyncio
async def test_a_rollout_payload_needs_no_serializer_from_either_caller() -> None:
    """Rollouts are the payload in production; their transcript is the default."""
    payload = replace(rollout(), messages=({"role": "user", "content": "hi"},))
    rendered = format_transcript(payload.messages)
    assert "hi" in rendered

    bundled = EchoingReward()
    scorer: NodeScorer[AnyRollout] = NodeScorer(bundled)
    assert [(p.score, p.context) for p in await scorer.score([payload])] == [
        (float(len(rendered)), rendered)
    ]
    assert bundled.seen == [rendered]


# -- bundled scoring: the scheduler scores what an expansion just produced -------------


class TableReward(Critic):
    """Score from a ``payload.n -> score`` table, or ``default`` for anything unlisted.

    ``TableReward({}, default=x)`` is the constant-scoring case -- which must not be
    spelled by enumerating every payload, or the fake silently stops covering the last
    node the day a toy chain grows.
    """

    def __init__(self, table: dict[str, float], default: float = 0.0) -> None:
        self.table = table
        self.default = default

    async def predict(self, contexts: Sequence[str]) -> list[float]:
        return [self.table.get(context, self.default) for context in contexts]


class StepExpander(Expander[Toy]):
    """One child per expansion, recording what it was asked to expand."""

    def __init__(self, width: int = 1) -> None:
        self.seen: list[int] = []
        self._width = width

    async def expand(self, payload: Toy) -> tuple[list[Toy], Mapping[str, object]]:
        self.seen.append(payload.n)
        return [Toy(payload.n + i + 1) for i in range(self._width)], {}


def _scorer(critic: Critic) -> NodeScorer[Toy]:
    return NodeScorer(ToyCritic(critic))


def test_a_short_score_list_is_refused_rather_than_filed_off_by_one() -> None:
    """`write` pairs nodes with scores positionally, so a scorer that returned the
    wrong number of them would file every score on the wrong node from the gap on --
    and silently leave the tail unscored. A custom `NodeScorer.score` is the seam that
    can get this wrong, and a mismatch is a defect, not a partial result."""
    state = _tree(Toy(0), Toy(1), Toy(2))
    nodes = list(state.nodes.values())

    with pytest.raises(ValueError, match="argument 2 is shorter"):
        _scorer(CountingReward()).write(
            nodes, [ValuePrediction(1.0, "one"), ValuePrediction(2.0, "two")]
        )

    assert not nodes[-1].critic  # ...and the node the scores ran out on stays unscored


@pytest.mark.asyncio
async def test_a_miscounted_batch_is_refused_where_the_batch_is_fetched() -> None:
    """And so is catchable: `write` is called past every scoring-failure handler.

    A backend answering a batch with the wrong number of results is a fault of the
    reward server, which both callers already know how to survive -- but only around
    `score`. Refused there, the run keeps its rollouts and counts the fault; refused at
    `write`, the same fault would take the whole search down with it.
    """
    with pytest.raises(ValueError, match="returned 2 predictions for 3 payloads"):
        await _scorer(ShortBatchReward()).score([Toy(0), Toy(1), Toy(2)])


def _chain_scheduler(model: Critic, **kwargs) -> Scheduler[Toy]:
    """One chain rollout, scored -- the shared shape of the bundled-scoring tests."""
    return Scheduler(
        expander=ChainExpander(),
        termination=Budget(max_rollouts=1),
        gating=WidthGating(1),
        scorer=_scorer(model),
        max_concurrency=1,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_a_node_is_scored_before_selection_can_ever_see_it() -> None:
    """The whole point of bundling: no window where a frontier node has no value.

    A polling critic cannot promise this -- it races the selection that reads it.
    """
    seen_unscored: list[int] = []

    class Watching(FifoSession[Toy]):
        requires_scorer = True

        def choose(self, view):
            seen_unscored.extend(
                n.payload.n for n in view.frontier() if n.critic is None
            )
            return super().choose(view)

    scheduler = Scheduler(
        expander=StepExpander(),
        allocation=Watching(),
        termination=Budget(max_rollouts=3),
        gating=WidthGating(1),
        scorer=_scorer(CountingReward()),
        max_concurrency=1,
    )

    result = await scheduler.run(Toy(0))

    # the root is the one unscored node -- nothing expanded it into being
    assert seen_unscored == [0]
    scored = [n for n in result.nodes.values() if n.id != 0]
    assert all(n.critic == float(n.payload.n) for n in scored)


@pytest.mark.asyncio
async def test_scores_land_before_the_backup_runs() -> None:
    """Ordering inside _integrate: a critic-reading Backup must find them there."""
    seen: list[float | None] = []

    class SpySession(FifoSession[Toy]):
        def completed(self, reservation, state, chain) -> None:
            seen.extend(node.critic for node in chain)

    await _chain_scheduler(CountingReward(), allocation=SpySession()).run(Toy(0))

    assert seen == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_a_scoring_failure_keeps_the_rollout_and_is_counted() -> None:
    """Generation is the expensive half; a dead reward server must not discard it."""
    result = await _chain_scheduler(DeadReward()).run(Toy(0))

    assert len(result.nodes) == 4  # the chain was attached anyway
    assert not any(n.critic for n in result.nodes.values())  # just unscored
    assert result.stats["score_failures"] == 1.0
    assert not result.failures  # a scoring fault is not an expansion failure


@pytest.mark.asyncio
async def test_a_stalled_reward_server_is_bounded_by_the_scorer_timeout() -> None:
    """An expansion is generate *then* score, so both halves need their own bound.

    ``Agent.generate_timeout`` bounds the first; without a bound on the second a
    reward server that swallows a request holds the expansion, and -- since a pass
    drains what is in flight rather than cancelling it -- the whole pass behind it. The
    message carries what a reader of the node's ``score_error`` needs and the bare
    ``TimeoutError`` does not.
    """
    model = StalledReward()
    scorer: NodeScorer[Toy] = NodeScorer(ToyCritic(model, timeout=0.01))

    with pytest.raises(TimeoutError, match=r"scoring exceeded timeout=0\.01s"):
        await scorer.score([Toy(0), Toy(1)])

    assert model.cancelled == 1  # the request was cancelled, not left running


@pytest.mark.asyncio
async def test_a_scorer_timeout_is_a_scoring_failure_like_any_other() -> None:
    """A timeout reaches both callers through the ``except`` a 503 already reaches:
    the rollouts are kept and counted, only the estimate is missing."""
    scorer: NodeScorer[Toy] = NodeScorer(ToyCritic(StalledReward(), timeout=0.01))
    sched = Scheduler(
        expander=ChainExpander(),
        termination=Budget(max_rollouts=1),
        gating=WidthGating(1),
        scorer=scorer,
        max_concurrency=1,
    )

    result = await sched.run(Toy(0))

    assert len(result.nodes) == 4  # the chain was attached
    assert not any(n.critic for n in result.nodes.values())
    assert result.stats["score_failures"] == 1.0
    assert not result.failures
    assert "timeout=0.01s" in str(result.nodes[0].metadata["score_error"])
