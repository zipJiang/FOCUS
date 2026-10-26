"""Adversarial admission, settlement, cancellation, and evidence-complexity checks."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from focus import (
    ErrorReduction,
    Fifo,
    Phase,
    RewardConfig,
    TokenEntropy,
)
from focus.allocation import RolloutLedger
from focus.scheduler import (
    AllocationSession,
    Budget,
    ConcurrencyGating,
    Expander,
    FifoSession,
    Node,
    RankedSession,
    Reservation,
    Scheduler,
    SchedulerState,
    SchedulerView,
    edge_return,
)
from focus.search import _PrioritySession
from tests.helpers import rollout, turn
from tests.scheduler.fakes import Toy
from tests.trees import root_state, stamp


def _state() -> SchedulerState[Toy]:
    return SchedulerState.root(Toy(0), gating=ConcurrencyGating(4))


def _assert_settled(state, session) -> None:
    assert all(c.m_pending == 0 for c in session.ledger.observations.values())
    assert all(n.gate._in_use == 0 for n in state.nodes.values())
    assert set(state.frontier) == {
        n.id for n in state.nodes.values() if state.selectable(n)
    }
    assert not session._active
    assert not session._running


@pytest.mark.parametrize("outcome", ["complete", "fail", "cancel"])
def test_distinct_purchases_settle_out_of_order_exactly_once(outcome) -> None:
    state = _state()
    view = SchedulerView(state)
    session = FifoSession[Toy]()
    first, second = session.select(view), session.select(view)
    assert first is not None
    assert second is not None
    assert first is not second
    assert first.node is second.node
    assert session.ledger.counts(view.root).m_pending == 2
    if outcome == "complete":
        child = state.add_child(0, Toy(3))
        session.complete(second, state, [child])
    elif outcome == "fail":
        session.fail(second, RuntimeError("backend failed"))
    else:
        session.cancel(second)
    assert session.ledger.counts(view.root).m_pending == 1
    with pytest.raises(ValueError, match="already settled"):
        session.cancel(second)
    assert session.ledger.counts(view.root).m_pending == 1
    session.cancel(first)
    assert session.ledger.counts(view.root).m_done == int(outcome == "complete")
    _assert_settled(state, session)


def test_foreign_reservation_and_foreign_tree_do_not_change_accounting() -> None:
    state = _state()
    session = FifoSession[Toy]()
    reservation = session.select(SchedulerView(state))
    assert reservation is not None
    for foreign in (Reservation(reservation.node), reservation):
        other = FifoSession[Toy]()
        with pytest.raises(ValueError, match="foreign"):
            other.cancel(foreign)
        assert not other.ledger.observations
    with pytest.raises(ValueError, match="one search tree"):
        session.select(SchedulerView(_state()))
    assert session.ledger.counts(reservation.node).m_pending == 1
    session.cancel(reservation)


@pytest.mark.parametrize("foreign", [False, True])
def test_invalid_choice_is_rejected_before_reserving(foreign) -> None:
    state = _state()
    candidate = _state().nodes[0] if foreign else state.add_child(0, Toy(3))

    class Invalid(AllocationSession[Toy]):
        def choose(self, view):
            return candidate

    session = Invalid()
    with pytest.raises(ValueError, match="not expandable"):
        session.select(SchedulerView(state))
    assert not session.ledger.observations
    _assert_settled(state, session)


class ControlledExpander(Expander[Toy]):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.stopped = 0
        self.calls = 0

    async def expand(self, payload):
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
            return [Toy(3)], {}
        finally:
            self.stopped += 1


@pytest.mark.parametrize("same_scheduler", [False, True])
async def test_concurrent_reuse_is_rejected_without_interrupting_owner(
    same_scheduler,
) -> None:
    state = _state()
    session = FifoSession[Toy]()
    expander = ControlledExpander()
    scheduler = Scheduler(
        expander=expander, allocation=session, termination=Budget(max_rollouts=1)
    )
    owner = asyncio.create_task(scheduler.run_on(state))
    try:
        await asyncio.wait_for(expander.started.wait(), timeout=1)
        other = (
            scheduler
            if same_scheduler
            else Scheduler(expander=expander, allocation=session)
        )
        with pytest.raises(RuntimeError, match=r"concurrent|already running"):
            await other.run_on(state)
        assert session.ledger.counts(state.nodes[0]).m_pending == 1
    finally:
        expander.release.set()
        await asyncio.wait_for(owner, timeout=1)
    _assert_settled(state, session)
    assert state.stats["rollouts"] == 1


async def test_launch_error_undoes_gate_and_purchase(monkeypatch) -> None:
    state = _state()
    session = FifoSession[Toy]()
    expander = ControlledExpander()

    def broken_create_task(coro):
        raise RuntimeError("task factory broke")

    with monkeypatch.context() as scoped:
        scoped.setattr(asyncio, "create_task", broken_create_task)
        with pytest.raises(RuntimeError, match="task factory"):
            await Scheduler(expander=expander, allocation=session).run_on(state)
    _assert_settled(state, session)
    assert expander.calls == state.stats["failures"] == 0


@pytest.mark.parametrize("where", ["selected", "completed", "released", "attach"])
async def test_extension_error_releases_this_purchase_and_other_tasks(
    where, monkeypatch
) -> None:
    state = _state()

    class BrokenSession(FifoSession[Toy]):
        def selected(self, reservation, view):
            if where == "selected":
                raise RuntimeError("extension broke")

        def completed(self, reservation, state, chain):
            if where == "completed":
                raise RuntimeError("extension broke")

        def released(self, reservation, error):
            if where == "released":
                raise RuntimeError("extension broke")

    class OneReturns(Expander[Toy]):
        calls = 0
        stopped = 0

        async def expand(self, payload):
            self.calls += 1
            try:
                if self.calls == 1:
                    if where == "released":
                        raise ValueError("generation broke")
                    return [Toy(3)], {}
                await asyncio.Event().wait()
            finally:
                self.stopped += 1

    def broken_attach(self, parent_id, chain):
        raise RuntimeError("extension broke")

    if where == "attach":
        monkeypatch.setattr(SchedulerState, "attach_chain", broken_attach)
    expander = OneReturns()
    session = BrokenSession()
    with pytest.raises(RuntimeError, match="extension broke"):
        await asyncio.wait_for(
            Scheduler(expander=expander, allocation=session, max_concurrency=2).run_on(
                state
            ),
            timeout=1,
        )
    _assert_settled(state, session)
    assert expander.calls == expander.stopped
    assert state.stats["rollouts"] == int(where == "completed")
    assert state.stats["failures"] == int(where == "released")


async def test_repeated_cancel_waits_for_lock_release_and_task_shutdown() -> None:
    state = _state()
    session = FifoSession[Toy]()
    expander = ControlledExpander()
    run = asyncio.create_task(
        Scheduler(expander=expander, allocation=session, max_concurrency=3).run_on(
            state
        )
    )
    await asyncio.wait_for(expander.started.wait(), timeout=1)
    async with state.lock:
        run.cancel()
        await asyncio.sleep(0)
        run.cancel()
        await asyncio.sleep(0)
        run.cancel()
        await asyncio.sleep(0)
        assert not run.done()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, timeout=1)
    _assert_settled(state, session)
    assert expander.stopped == 3
    assert state.stats["failures"] == 0
    old_gates = [node.gate for node in state.nodes.values()]
    state.regate(ConcurrencyGating(2))
    await asyncio.sleep(0)
    assert all(gate._in_use == 0 for gate in old_gates)
    assert all(node.gate._in_use == 0 for node in state.nodes.values())


@pytest.mark.parametrize("gamma", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("beta", [-0.5, 0.0, 2.0])
def test_all_suffixes_match_existing_return_math_and_are_linear(
    gamma, beta, monkeypatch
) -> None:
    config = RewardConfig(gamma=gamma, beta_step=beta)
    prefix = (turn(reward=100.0), turn(tag="fold"))
    state = root_state(rollout(*prefix))
    turns = list(prefix)
    chain: list[Node[Any]] = []
    for index in range(40):
        step = turn(reward=float(index % 5), done=index == 39)
        assert step.transition is not None
        step = replace(step, transition=replace(step.transition, reward_step=-0.2))
        turns.extend((replace(turn(tag="fold"), compaction_penalty=-0.01), step))
        # Multiple checkpoints may share one turn boundary.
        for _ in range(2):
            chain.append(state.add_child(chain[-1].id if chain else 0, rollout(*turns)))
    root = state.nodes[0]
    expected = {
        node.id: edge_return(chain[-1], node, config) for node in (root, *chain)
    }
    calls = 0
    original = RewardConfig.train_reward

    def counted(self, transition):
        nonlocal calls
        calls += 1
        return original(self, transition)

    monkeypatch.setattr(RewardConfig, "train_reward", counted)
    ledger = RolloutLedger(config)
    for node in chain:
        stamp(node)
    ledger.record(root, chain)
    assert calls == 40  # one reward evaluation per new transition, regardless of nodes
    for node_id, value in expected.items():
        counts = ledger.observations[node_id]
        assert counts.returns.n == 1
        assert counts.returns.mean == pytest.approx(value)
        assert counts.m_done == counts.m_pending == 0


def test_ranked_scan_does_not_copy_frontier_and_scores_each_node_once() -> None:
    state = _state()
    for _ in range(100):
        state.add_child(0, Toy(1))

    class StreamingView(SchedulerView[Toy]):
        def frontier(self):
            pytest.fail("ranked choice must not copy the frontier")

    class Counted(RankedSession[Toy]):
        calls = 0

        def score(self, view, node):
            self.calls += 1
            return 1.0

    session = Counted()
    assert session.choose(StreamingView(state)) is state.nodes[0]
    assert session.calls == len(state.frontier)


@pytest.mark.parametrize(
    "allocator",
    [Fifo(), TokenEntropy(), ErrorReduction()],
)
def test_configuration_creates_isolated_sessions(allocator) -> None:
    config = RewardConfig()
    views = [SchedulerView(root_state(rollout())) for _ in range(2)]
    sessions = []
    for view in views:
        view.root.critic = 0.5
        sessions.append(_PrioritySession(allocator, RolloutLedger(config)))
    first, second = sessions
    assert first is not None
    assert second is not None
    assert first is not second
    assert first.ledger is not second.ledger
    first.ledger.counts(views[0].root).m_pending = 3
    assert not second.ledger.observations


async def test_task_factory_failure_also_cleans_up_earlier_launches() -> None:
    state = _state()
    session = FifoSession[Toy]()
    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    launched: list[asyncio.Task[Any]] = []

    def factory(loop, coroutine, **kwargs):
        if launched:
            raise RuntimeError("task factory failed after one launch")
        task = asyncio.Task(coroutine, loop=loop, **kwargs)
        launched.append(task)
        return task

    loop.set_task_factory(factory)
    try:
        with pytest.raises(RuntimeError, match="task factory failed"):
            await Scheduler(expander=ControlledExpander(), allocation=session).run_on(
                state
            )
    finally:
        loop.set_task_factory(old_factory)
    _assert_settled(state, session)
    assert len(launched) == 1
    assert launched[0].cancelled()


async def test_regating_an_active_search_is_rejected_before_changing_gates() -> None:
    state = _state()
    session = FifoSession[Toy]()
    expander = ControlledExpander()
    original = state.nodes[0].gate
    owner = asyncio.create_task(
        Scheduler(
            expander=expander, allocation=session, termination=Budget(max_rollouts=1)
        ).run_on(state)
    )
    try:
        await asyncio.wait_for(expander.started.wait(), timeout=1)
        async with state.lock:
            with pytest.raises(RuntimeError, match="cannot regate"):
                state.regate(ConcurrencyGating(1))
        assert state.nodes[0].gate is original
    finally:
        expander.release.set()
        await asyncio.wait_for(owner, timeout=1)
    state.regate(ConcurrencyGating(1))
    _assert_settled(state, session)


async def test_cancelled_observer_does_not_interrupt_other_releases() -> None:
    class BrokenObserver(FifoSession[Toy]):
        releases = 0

        def released(self, reservation, error):
            self.releases += 1
            raise asyncio.CancelledError

    state = _state()
    session = BrokenObserver()
    expander = ControlledExpander()
    run = asyncio.create_task(
        Scheduler(expander=expander, allocation=session, max_concurrency=3).run_on(
            state
        )
    )
    await asyncio.wait_for(expander.started.wait(), timeout=1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, timeout=1)
    _assert_settled(state, session)
    assert session.releases == expander.stopped == 3


def test_completion_rejects_a_different_tree_and_detached_chain() -> None:
    state = _state()
    session = FifoSession[Toy]()
    reservation = session.select(SchedulerView(state))
    assert reservation is not None
    other = _state()
    foreign = other.add_child(0, Toy(3))
    with pytest.raises(ValueError, match="one search tree"):
        session.complete(reservation, other, [foreign])
    with pytest.raises(ValueError, match="attached continuation"):
        session.complete(reservation, state, [foreign])
    assert session.ledger.counts(state.nodes[0]).m_pending == 1
    session.cancel(reservation)


def test_empty_completion_requires_failure_settlement() -> None:
    state = _state()
    session = FifoSession[Toy]()
    reservation = session.select(SchedulerView(state))
    assert reservation is not None
    with pytest.raises(ValueError, match="empty expansion"):
        session.complete(reservation, state, [])
    assert session.ledger.counts(state.nodes[0]).m_pending == 1
    session.fail(reservation, RuntimeError("no progress"))
    assert session.ledger.counts(state.nodes[0]).m_done == 0


def test_invalid_checkpoint_boundaries_do_not_partially_record_evidence() -> None:
    state = root_state(rollout())
    first = state.add_child(0, rollout(turn(), turn(), turn()))
    leaf = state.add_child(first.id, rollout(turn(done=True)))
    ledger = RolloutLedger(RewardConfig())
    for node in (first, leaf):
        stamp(node)
    with pytest.raises(ValueError, match="nondecreasing"):
        ledger.record(state.nodes[0], [first, leaf])
    assert not ledger.observations


async def test_concurrent_searches_share_configuration_without_sharing_evidence() -> (
    None
):
    from focus.allocation.evidence import EVIDENCE_KEY
    from tests.test_collector import _collector

    collector = _collector(passes=(Phase(Fifo(), 4, max_pending=2),))
    states = await asyncio.gather(collector.search("first"), collector.search("second"))
    first, second = (state.nodes[0].metadata[EVIDENCE_KEY] for state in states)
    assert first.observations is not second.observations
    for state, evidence in zip(states, (first, second), strict=True):
        assert all(n.gate.in_use == 0 for n in state.nodes.values())
        assert all(c.m_pending == 0 for c in evidence.observations.values())
        assert (
            sum(c.m_done for c in evidence.observations.values())
            == state.stats["rollouts"]
        )
