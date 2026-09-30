from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from focus.harness import AnyRollout
from focus.registry import build
from focus.reward import RewardModel, RewardResult
from focus.scheduler import (
    AllNodes,
    BestFirstSession,
    Budget,
    CriticScheduler,
    CriticSelection,
    Expander,
    FifoSession,
    NodeScorer,
    Scheduler,
    SchedulerState,
    Terminals,
    WidthGating,
    rollout_text,
)
from focus.scheduler.core.critic import ValuePrediction
from tests.helpers import DeadReward, ShortBatchReward, StalledReward, rollout
from tests.scheduler.fakes import ChainExpander, Toy

# -- fakes ---------------------------------------------------------------------------


class CountingReward(RewardModel):
    """Score = the serialized payload read back as a number; counts batch calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.scored: list[str] = []

    def score(self, context: str) -> RewardResult:
        return RewardResult(score=float(context))

    def score_batch(self, contexts: Sequence[str]) -> list[RewardResult]:
        self.calls += 1
        self.scored.extend(contexts)
        return [self.score(c) for c in contexts]

    async def ascore_batch(self, contexts: Sequence[str]) -> list[RewardResult]:
        return self.score_batch(contexts)


def _serialize(toy: Toy) -> str:
    return str(toy.n)


def _tree(*payloads: Toy) -> SchedulerState[Toy]:
    """A width-4 state whose root is the first payload, rest attached as a chain."""
    state: SchedulerState[Toy] = SchedulerState.root(payloads[0], gating=WidthGating(4))
    state.attach_chain(0, list(payloads[1:]))
    return state


# -- tests ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scores_and_version_tags_only_targets() -> None:
    # root Toy(0) live; Toy(1), Toy(2) live; Toy(3) terminal (done at n>=3)
    state = _tree(Toy(0), Toy(1), Toy(2), Toy(3))
    model = CountingReward()
    critic = CriticScheduler(model=model, serialize=_serialize, version="rm-v1")

    await critic.run_on(state)  # no `until` -> one drain and return

    terminal = state.nodes[3]
    assert terminal.critic == {"rm-v1": 3.0}  # the only done node
    assert all(not n.critic for i, n in state.nodes.items() if i != 3)


@pytest.mark.asyncio
async def test_second_version_adds_without_clobbering() -> None:
    state = _tree(Toy(0), Toy(1), Toy(2), Toy(3))
    terminal = state.nodes[3]

    await CriticScheduler(
        model=CountingReward(), serialize=_serialize, version="rm-v1"
    ).run_on(state)
    await CriticScheduler(
        model=CountingReward(), serialize=_serialize, version="rm-v2"
    ).run_on(state)

    assert terminal.critic == {"rm-v1": 3.0, "rm-v2": 3.0}


@pytest.mark.asyncio
async def test_rescoring_same_version_is_skipped() -> None:
    state = _tree(Toy(0), Toy(1), Toy(2), Toy(3))
    model = CountingReward()
    critic = CriticScheduler(model=model, serialize=_serialize, version="rm-v1")

    await critic.run_on(state)
    calls_after_first = model.calls
    await critic.run_on(state)  # everything already scored for rm-v1

    assert model.calls == calls_after_first  # no second /pooling round


@pytest.mark.asyncio
async def test_all_nodes_selection_scores_every_node() -> None:
    state = _tree(Toy(0), Toy(1), Toy(2), Toy(3))
    critic = CriticScheduler[Toy](
        model=CountingReward(),
        serialize=_serialize,
        version="rm-v1",
        selection=AllNodes(),
    )

    await critic.run_on(state)

    assert all(n.critic == {"rm-v1": float(n.payload.n)} for n in state.nodes.values())


def test_selection_builds_via_registry() -> None:
    assert isinstance(build(CriticSelection, {"name": "terminals"}), Terminals)
    assert isinstance(build(CriticSelection, {"name": "all"}), AllNodes)


@pytest.mark.asyncio
async def test_batches_respect_batch_size() -> None:
    state = _tree(Toy(0), Toy(1), Toy(2), Toy(3))
    model = CountingReward()
    critic = CriticScheduler[Toy](
        model=model,
        serialize=_serialize,
        version="rm-v1",
        selection=AllNodes(),
        batch_size=2,
    )

    await critic.run_on(state)

    assert model.calls == 2  # 4 nodes / batch_size 2
    assert all(n.critic for n in state.nodes.values())


@pytest.mark.asyncio
async def test_parallel_critic_tracks_generation_then_drains() -> None:
    # generation grows the tree while the critic annotates it; the critic tracks
    # generation via `until` and stops only once generation is done and it has drained.
    state: SchedulerState[Toy] = SchedulerState.root(Toy(0), gating=WidthGating(1))
    generation = Scheduler(
        expander=ChainExpander(),
        termination=Budget(max_rollouts=1),
        gating=WidthGating(1),
        max_concurrency=1,
    )
    model = CountingReward()
    critic = CriticScheduler[Toy](
        model=model, serialize=_serialize, version="rm-v1", poll_interval=0.0
    )

    done = asyncio.Event()

    async def _gen() -> None:
        try:
            await generation.run_on(state)
        finally:
            done.set()  # let the critic drain, then stop

    await asyncio.gather(_gen(), critic.run_on(state, until=done))

    # generation built root + Toy(1..3); the critic scored the sole terminal Toy(3)
    assert len(state.nodes) == 4
    terminals = state.terminals()
    assert [n.payload.n for n in terminals] == [3]
    assert terminals[0].critic == {"rm-v1": 3.0}
    best = state.best()
    assert best is not None
    assert best.payload.n == 3  # tree intact


# -- the default serializer -----------------------------------------------------------


class EchoingReward(RewardModel):
    """Keeps whatever it was handed, and scores by its length -- so a test can see
    *how* the payload was rendered, not just that some number came back."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score(self, context: str) -> RewardResult:
        self.seen.append(context)
        return RewardResult(score=float(len(context)))


@pytest.mark.asyncio
async def test_a_rollout_payload_needs_no_serializer_from_either_caller() -> None:
    """Rollouts are the payload in production, so `rollout_text` is the default -- and
    it has to be the *same* default on both ways of scoring a tree. A required
    `serialize` on one and an optional one on the other made "no renderer given" mean
    two different things for one payload type."""
    payload = replace(rollout(), messages=({"role": "user", "content": "hi"},))
    state: SchedulerState[AnyRollout] = SchedulerState.root(
        payload, gating=WidthGating(1)
    )
    rendered = rollout_text(payload)
    assert "hi" in rendered

    scheduled = EchoingReward()
    await CriticScheduler[AnyRollout](
        model=scheduled, version="rm-v1", selection=AllNodes()
    ).run_on(state)

    assert scheduled.seen == [rendered]
    assert state.nodes[0].critic == {"rm-v1": float(len(rendered))}

    bundled = EchoingReward()
    scorer: NodeScorer[AnyRollout] = NodeScorer(model=bundled, version="rm-v2")
    assert [(p.score, p.context) for p in await scorer.score([payload])] == [
        (float(len(rendered)), rendered)
    ]
    assert bundled.seen == scheduled.seen


# -- bundled scoring: the scheduler scores what an expansion just produced -------------


class TableReward(RewardModel):
    """Score from a ``payload.n -> score`` table, or ``default`` for anything unlisted.

    ``TableReward({}, default=x)`` is the constant-scoring case -- which must not be
    spelled by enumerating every payload, or the fake silently stops covering the last
    node the day a toy chain grows.
    """

    def __init__(self, table: dict[str, float], default: float = 0.0) -> None:
        self.table = table
        self.default = default

    def score(self, context: str) -> RewardResult:
        return RewardResult(score=self.table.get(context, self.default))


class StepExpander(Expander[Toy]):
    """One child per expansion, recording what it was asked to expand."""

    def __init__(self, width: int = 1) -> None:
        self.seen: list[int] = []
        self._width = width

    async def expand(self, payload: Toy) -> tuple[list[Toy], Mapping[str, object]]:
        self.seen.append(payload.n)
        return [Toy(payload.n + i + 1) for i in range(self._width)], {}


def _scorer(model: RewardModel) -> NodeScorer[Toy]:
    return NodeScorer(model=model, serialize=_serialize, version="rm-v1")


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
    with pytest.raises(ValueError, match="returned 2 results for 3 payloads"):
        await _scorer(ShortBatchReward()).score([Toy(0), Toy(1), Toy(2)])


def _chain_scheduler(model: RewardModel, **kwargs) -> Scheduler[Toy]:
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
                n.payload.n for n in view.frontier() if "rm-v1" not in n.critic
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
    assert all(n.critic == {"rm-v1": float(n.payload.n)} for n in scored)


@pytest.mark.asyncio
async def test_best_first_follows_the_critic_not_the_reward() -> None:
    """The scores have to actually steer selection, and steer it somewhere reward won't.

    Toy(1) scores 5.0 and Toy(2) scores 1.0, the reverse of their rewards (1.0, 2.0), so
    a critic-guided selection expands Toy(1) where a reward-guided one takes Toy(2).
    """
    expander = StepExpander(width=2)  # root -> Toy(1) -> Toy(2), both live
    scheduler = Scheduler(
        expander=expander,
        allocation=BestFirstSession(key="critic", version="rm-v1"),
        termination=Budget(max_rollouts=2),
        # width 2, not 1: at width 1 the chain's own child fills Toy(1)'s gate the
        # moment it attaches, and only the leaf is left on the frontier to choose from
        gating=WidthGating(2),
        scorer=_scorer(TableReward({"1": 5.0, "2": 1.0})),
        max_concurrency=1,
    )

    await scheduler.run(Toy(0))

    assert expander.seen == [0, 1]  # root, then the critic's pick -- not Toy(2)


def test_critic_selection_needs_a_scorer_and_a_version() -> None:
    exp = StepExpander()
    with pytest.raises(ValueError, match="no scorer is registered"):
        Scheduler(
            expander=exp, allocation=BestFirstSession(key="critic", version="rm-v1")
        )
    # ...and it constructs with one
    Scheduler(
        expander=exp,
        allocation=BestFirstSession(key="critic", version="rm-v1"),
        scorer=_scorer(CountingReward()),
    )
    with pytest.raises(ValueError, match="needs the version"):
        BestFirstSession(key="critic")


@pytest.mark.asyncio
async def test_scores_land_before_the_backup_runs() -> None:
    """Ordering inside _integrate: a critic-reading Backup must find them there."""
    seen: list[dict[str, float]] = []

    class SpySession(FifoSession[Toy]):
        def completed(self, reservation, state, chain) -> None:
            seen.extend(node.critic.copy() for node in chain)

    await _chain_scheduler(CountingReward(), allocation=SpySession()).run(Toy(0))

    assert seen == [{"rm-v1": 1.0}, {"rm-v1": 2.0}, {"rm-v1": 3.0}]


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

    ``Runner.generate_timeout`` bounds the first; without a bound on the second a
    reward server that swallows a request holds the expansion, and -- since a pass
    drains what is in flight rather than cancelling it -- the whole pass behind it. The
    message carries what a reader of the node's ``score_error`` needs and the bare
    ``TimeoutError`` does not.
    """
    model = StalledReward()
    scorer = NodeScorer(
        model=model, serialize=_serialize, version="rm-v1", timeout=0.01
    )

    with pytest.raises(TimeoutError, match=r"scoring exceeded timeout=0\.01s"):
        await scorer.score([Toy(0), Toy(1)])

    assert model.cancelled == 1  # the request was cancelled, not left running


@pytest.mark.asyncio
async def test_a_scorer_timeout_is_a_scoring_failure_like_any_other() -> None:
    """A timeout reaches both callers through the ``except`` a 503 already reaches:
    the rollouts are kept and counted, only the estimate is missing."""
    scorer = NodeScorer(
        model=StalledReward(), serialize=_serialize, version="rm-v1", timeout=0.01
    )
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


@pytest.mark.asyncio
async def test_a_polling_pass_survives_a_dead_reward_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same fault on the *other* scoring path: counted, logged, pass completes.

    And it must not spin: selection re-targets whatever is unscored, so a batch that
    is retried after it fails is the same batch forever -- with `until` already set,
    that would hang here instead of returning.
    """
    state = _tree(Toy(0), Toy(1), Toy(2), Toy(3))
    critic = CriticScheduler[Toy](
        model=DeadReward(),
        serialize=_serialize,
        version="rm-v1",
        selection=AllNodes(),
        batch_size=1,
        poll_interval=0.0,
    )
    until = asyncio.Event()
    until.set()  # "generation is done": the pass may only drain, never wait

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(critic.run_on(state, until=until), timeout=5.0)

    # one failure per batch and then the pass ended: each node was tried exactly once
    assert state.stats["score_failures"] == 4.0
    assert not any(n.critic for n in state.nodes.values())  # simply unscored
    assert "critic scoring failed" in caplog.text
    assert "reward server unreachable" in caplog.text  # the traceback came with it


@pytest.mark.asyncio
@pytest.mark.parametrize("estimate", [0.0, 100.0])
async def test_critic_scores_never_leak_into_the_backed_up_value(
    estimate: float,
) -> None:
    """Two channels kept apart: `mean_return` is realized, `critic` is the estimate.

    Whatever the model says -- 0.0 or a 100.0 that no Toy reward could be confused with
    -- the backup folds the leaf's realized 3.0 up every ancestor, unchanged. A leak
    either way would move `mean_return` off it.
    """
    scheduler = _chain_scheduler(
        TableReward({}, default=estimate),
        allocation=BestFirstSession(key="mean_return"),
    )

    result = await scheduler.run(Toy(0))

    assert [n.mean_return for n in result.nodes.values()] == [3.0, 3.0, 3.0, 3.0]
    assert result.nodes[3].critic == {"rm-v1": estimate}
