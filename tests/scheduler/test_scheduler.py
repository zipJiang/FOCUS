from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import pytest

from focus.harness import (
    Compaction,
    CompactionResult,
    Compactor,
    Env,
    RolloutState,
    Runner,
    StepResult,
)
from focus.scheduler import (
    AllocationSession,
    Budget,
    ConcurrencyGating,
    Expander,
    FifoSession,
    Node,
    Reservation,
    RolloutExpander,
    Scheduler,
    SchedulerState,
    SchedulerView,
    WidthGating,
)
from tests.helpers import (
    ScriptedGenerator,
    ToyEnv,
    ToyState,
    WhenTrigger,
    toy_runner,
)
from tests.scheduler.fakes import ChainExpander, Toy

# -- fakes ---------------------------------------------------------------------------


class WritingEnv(ToyEnv):
    """Writes each step's action into the workspace, to catch checkpoint leaks."""

    async def step(
        self, state: ToyState, action: str, *, workspace=None
    ) -> StepResult[ToyState]:
        result = await super().step(state, action, workspace=workspace)
        if workspace is not None:
            workspace.write(f"step{result.next_state.n}", action)
        return result


class LenCompactor(Compactor[ToyState]):
    """Fold to the seed once the prompt grows past a threshold (self-relieving)."""

    trigger = WhenTrigger(lambda prompt, state: len(prompt) > 12)

    async def compact(
        self, compaction: Compaction[ToyState]
    ) -> CompactionResult[ToyState]:
        # reset the conversation to the seed but keep the env state (n keeps climbing)
        return CompactionResult(compaction.seed, compaction.state)


def _runner(
    gen: ScriptedGenerator,
    *,
    env: Env[Any, Any] | None = None,
    compactor: Compactor[Any] | None = None,
) -> Runner[ToyState, str]:
    return toy_runner(gen, env=env, compactor=compactor)


def _expander(
    gen: ScriptedGenerator,
    *,
    env: Env[Any, Any] | None = None,
    compactor: Compactor[Any] | None = None,
) -> RolloutExpander[ToyState]:
    return RolloutExpander(_runner(gen, env=env, compactor=compactor))


# -- region granularity: no fold means one node ---------------------------------------


@pytest.mark.asyncio
async def test_no_compaction_is_one_region_to_a_single_terminal() -> None:
    # a whole episode with no fold is one append-only region -> one expansion, one node.
    exp = _expander(ScriptedGenerator(["go", "go", "stop"]))
    sched = Scheduler(
        expander=exp, termination=Budget(max_rollouts=8), gating=WidthGating(1)
    )
    result = await sched.run(await exp.start("hi"))

    assert len(result.nodes) == 2  # root + the terminal region
    assert result.nodes[0].children == [1]
    terminal = result.nodes[1]
    assert terminal.payload.done
    assert terminal.payload.folds == 0  # no fold, one node
    assert terminal.payload.turns_taken == 3  # go, go, stop, all on one log
    assert result.best() is result.nodes[1]


# -- fork at compaction ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_forks_the_chain_at_each_compaction_boundary() -> None:
    # go forever, folding whenever the prompt grows; the env ends at n==3. Each fold is
    # a region boundary -> the tree is a chain of post-fold checkpoints to the terminal.
    exp = _expander(
        ScriptedGenerator(["go"]), env=ToyEnv(done_at=3), compactor=LenCompactor()
    )
    sched = Scheduler(
        expander=exp, termination=Budget(max_rollouts=8), gating=WidthGating(1)
    )
    result = await sched.run(await exp.start("hi"))

    folds = [n.payload.folds for n in result.nodes.values() if n.id]
    assert any(folds)  # boundaries are folds
    assert (best := result.best()) is not None
    assert best.payload.state == ToyState(3)
    assert best.payload.done


@pytest.mark.asyncio
async def test_width_branches_at_a_compaction_boundary() -> None:
    exp = _expander(
        ScriptedGenerator(["go"]), env=ToyEnv(done_at=5), compactor=LenCompactor()
    )
    sched = Scheduler(
        expander=exp,
        termination=Budget(max_rollouts=2),
        gating=WidthGating(2),
        max_concurrency=1,
    )
    result = await sched.run(await exp.start("hi"))

    # Two branches off the root, each a whole chain -- so the second one's id is not 2.
    # The width counts children, and one of them is the continuation the first expansion
    # actually took: `WidthGating(2)` therefore means the realized line plus one
    # alternative, which is the same number `candidate_count` reports.
    root = result.nodes[0]
    assert len(root.children) == 2
    assert result.stats["rollouts"] == 2.0
    assert len({result.nodes[c].id for c in root.children}) == 2


@pytest.mark.asyncio
async def test_branches_have_isolated_workspaces() -> None:
    from tests.harness.test_runner import QuietMemory

    exp = _expander(
        ScriptedGenerator(["go"]), env=WritingEnv(done_at=5), compactor=LenCompactor()
    )
    sched = Scheduler(
        expander=exp,
        termination=Budget(max_rollouts=2),
        gating=WidthGating(2),
        max_concurrency=1,
    )
    # QuietMemory: writes stick, but no guidance prose that would make LenCompactor
    # fire on the seed alone.
    result = await sched.run(await exp.start("hi", workspace=QuietMemory()))

    a, b = (result.nodes[i].payload.workspace for i in (1, 2))
    assert a is not b  # each branch forked its own workspace
    root_ws = result.nodes[0].payload.workspace
    assert a is not root_ws
    assert b is not root_ws


@pytest.mark.asyncio
async def test_one_expansion_attaches_its_folds_as_a_line() -> None:
    """An expansion rolls to the end and keeps what it passed, so the folds arrive as a
    descendant line off the parent -- one branch, several branch points."""
    exp = _expander(
        ScriptedGenerator(["go"]), env=ToyEnv(done_at=3), compactor=LenCompactor()
    )
    sched = Scheduler(
        expander=exp, termination=Budget(max_rollouts=1), gating=WidthGating(1)
    )
    result = await sched.run(await exp.start("hi"))

    created = [n for n in result.nodes.values() if n.id != 0]
    assert len(created) > 1  # the folds, not just a leaf
    assert result.nodes[0].children == [created[0].id]  # one branch off the root
    assert [len(n.children) for n in created[:-1]] == [1] * (len(created) - 1)
    assert created[-1].payload.done  # and the line ends at the episode's end
    # one expansion, however many folds it passed through
    assert result.stats["rollouts"] == 1.0
    assert result.stats["nodes"] == float(len(created))


class AlwaysDies(Expander[RolloutState[Any]]):
    """Fails, then gives up failing -- so the run ends whether or not the budget works.

    Raising a guard from `expand` would not help: the scheduler catches whatever comes
    out of an expansion and records it as one more failure, which is the very loop this
    is here to catch. Falling back to success lets the run terminate and the assertion
    below report how many attempts it took.
    """

    def __init__(self, give_up_after: int = 200) -> None:
        self.calls = 0
        self._give_up_after = give_up_after

    async def expand(self, payload):
        self.calls += 1
        if self.calls > self._give_up_after:
            return [payload], {}
        raise RuntimeError("dead endpoint")


@pytest.mark.asyncio
async def test_a_turn_budget_depletes_against_an_expander_that_always_fails() -> None:
    """A dead expansion records no turns -- there is no payload to read them off -- so a
    cap counting successes alone would never deplete, and a released gate puts the node
    straight back on the frontier to be re-selected and re-failed forever.
    """
    expander = AlwaysDies()

    exp = _expander(ScriptedGenerator(["stop"]))
    result = await Scheduler(
        expander=expander, termination=Budget(max_turns=4), max_concurrency=1
    ).run(await exp.start("hi"))

    assert expander.calls <= 5  # the attempts themselves are what charged the budget
    assert result.stats["failures"] == float(expander.calls)


@pytest.mark.asyncio
async def test_a_dead_expansion_is_logged_with_its_traceback(caplog) -> None:
    """A failure that never sinks the run is a failure nothing else would report: the
    stats carry a count, the tree carries a repr, and neither reaches an operator
    watching a training job. One WARNING per dead expansion, with the traceback."""
    caplog.set_level(logging.WARNING, logger="focus")
    exp = _expander(ScriptedGenerator(["stop"]))

    await Scheduler(
        expander=AlwaysDies(), termination=Budget(max_turns=2), max_concurrency=1
    ).run(await exp.start("hi"))

    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert records
    assert all("node=" in r.getMessage() for r in records)
    assert all("stats=" in r.getMessage() for r in records)
    # the traceback rides along, whichever of the two failure paths got here
    assert all(r.exc_info is not None for r in records)
    assert any("dead endpoint" in str(r.exc_info[1]) for r in records if r.exc_info)


@pytest.mark.asyncio
async def test_the_turn_budget_counts_what_was_actually_generated() -> None:
    """Rollouts are the wrong unit once an expansion runs a whole episode; turns are the
    unit compute is spent in, and mean the same thing however the chain was shaped."""
    exp = _expander(
        ScriptedGenerator(["go"]), env=ToyEnv(done_at=3), compactor=LenCompactor()
    )
    sched = Scheduler(
        expander=exp, termination=Budget(max_turns=3), gating=WidthGating(2)
    )
    result = await sched.run(await exp.start("hi"))

    turns = result.stats["turns"]
    assert turns > 0
    # a stop condition, not a launch count: what the next expansion costs is unknown
    # before it runs, so in-flight work can overshoot by one episode per slot
    assert turns <= 3 + sched._max_concurrency * 8


# -- registry, budget, goal -----------------------------------------------------------


@pytest.mark.asyncio
async def test_max_rollouts_is_a_hard_cap() -> None:
    exp = _expander(
        ScriptedGenerator(["go"]), env=ToyEnv(done_at=9), compactor=LenCompactor()
    )
    sched = Scheduler(
        expander=exp,
        termination=Budget(max_rollouts=2),
        gating=WidthGating(8),
        max_concurrency=4,
    )
    result = await sched.run(await exp.start("hi"))
    assert result.stats["rollouts"] == 2.0  # never overshoots


@pytest.mark.asyncio
async def test_goal_predicate_stops_early() -> None:
    exp = _expander(ScriptedGenerator(["stop"]))
    goal = lambda view: any(n.payload.done for n in view.nodes())  # noqa: E731
    sched = Scheduler(
        expander=exp,
        termination=Budget(max_rollouts=100, goal=goal),
        gating=WidthGating(2),
        max_concurrency=1,
    )
    result = await sched.run(await exp.start("hi"))
    assert result.stats["rollouts"] == 1.0
    assert result.terminals()
    assert result.terminals()[0].payload.done


class OneFastThenSlow(RolloutExpander[ToyState]):
    """First expansion completes at once; later ones only after termination fires."""

    def __init__(self, delay: float = 0.02) -> None:
        super().__init__(_runner(ScriptedGenerator(["stop"])))
        self.calls = 0
        self.cancelled = 0
        self._delay = delay

    async def expand(self, start):
        i = self.calls
        self.calls += 1
        if i == 0:
            return await super().expand(start)  # scripted 'stop' -> a terminal
        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return await super().expand(start)


@pytest.mark.asyncio
async def test_in_flight_rollouts_are_drained_not_cancelled_on_termination() -> None:
    """Termination stops the launching; what is already running is finished and kept.

    The rollout is the expensive half -- a whole episode of backend calls -- and one in
    flight when the budget runs out has already been paid for. Cancelling it (which is
    what this used to do) threw away up to ``max_concurrency`` of them per pass, and a
    training harness runs a pass per prompt per step.
    """
    strategy = OneFastThenSlow()
    exp = strategy
    goal = lambda view: any(n.payload.done for n in view.nodes())  # noqa: E731
    sched = Scheduler(
        expander=exp,
        termination=Budget(goal=goal),
        gating=WidthGating(2),  # both slots launch from the root before the first await
        max_concurrency=2,
    )
    result = await sched.run(await exp.start("hi"))

    assert strategy.calls == 2  # both launched before the goal was met
    assert strategy.cancelled == 0  # and the slow one was drained, not thrown away
    assert result.stats["rollouts"] == 2.0  # its result was integrated like any other
    assert len(result.terminals()) == 2
    assert result.stats["failures"] == 0.0


@pytest.mark.asyncio
async def test_concurrency_gating_allows_unbounded_total_children() -> None:
    # ConcurrencyGating caps only in-flight branches, so the root is re-expanded every
    # rollout (here into three terminal children) -- unlike WidthGating(1), which caps
    # total children at one. 'stop' makes each region terminal, so only the root stays
    # selectable and every expansion lands on it.
    exp = _expander(ScriptedGenerator(["stop"]))
    sched = Scheduler(
        expander=exp,
        gating=ConcurrencyGating(1),
        termination=Budget(max_rollouts=3),
        max_concurrency=1,
    )
    result = await sched.run(await exp.start("hi"))
    assert result.nodes[0].children == [1, 2, 3]  # root re-expanded three times
    assert len(result.terminals()) == 3


# -- purity: drive the whole scheduler with no Runner / harness at all ---------------


@pytest.mark.asyncio
async def test_scheduler_is_pure_over_the_tree_without_a_runner() -> None:
    sched = Scheduler(
        expander=ChainExpander(),
        termination=Budget(max_rollouts=1),
        gating=WidthGating(1),
        max_concurrency=1,
    )
    result = await sched.run(Toy(0))

    assert len(result.nodes) == 4  # root Toy(0) + Toy(1), Toy(2), Toy(3)
    assert (best := result.best()) is not None
    assert best.payload.n == 3
    assert result.stats["rollouts"] == 1.0


@pytest.mark.asyncio
async def test_a_selection_off_the_frontier_says_which_node_and_why() -> None:
    """The guard exists for custom policies, so its message is written for their author.

    ``select`` may return any node in the tree, and the three reasons the tree refuses
    one are not visible from the node a policy holds -- so the error names the node and
    reads all three back, and points at the set that would have been safe to choose
    from.
    """

    class PicksTheTerminal(AllocationSession[Toy]):
        def choose(self, view):
            return next((n for n in view.nodes() if n.payload.done), view.frontier()[0])

    sched = Scheduler(
        expander=ChainExpander(),
        allocation=PicksTheTerminal(),
        termination=Budget(max_rollouts=2),
        gating=WidthGating(4),
        max_concurrency=1,
    )

    message = r"PicksTheTerminal returned node 3, .*done=True"
    with pytest.raises(ValueError, match=message):
        await sched.run(Toy(0))


@pytest.mark.asyncio
async def test_state_lock_serializes_readers_and_writers() -> None:
    state: SchedulerState[Toy] = SchedulerState.root(Toy(0), gating=WidthGating(4))
    order: list[str] = []

    async def writer() -> None:
        async with state.lock:
            order.append("w-start")
            await asyncio.sleep(0)
            state.add_child(0, Toy(1))
            order.append("w-end")

    async def reader() -> None:
        async with state.lock:
            order.append("r")
            assert all(
                n.parent_id is None or n.parent_id in state.nodes
                for n in state.nodes.values()
            )

    await asyncio.gather(writer(), reader())
    assert order == ["w-start", "w-end", "r"]
    assert state.nodes[0].children == [1]


# -- draining and aborting: what the loop keeps, and what it gives back ---------------


class OneFastThenBlockingChain(Expander[Toy]):
    """The first expansion lands at once; every later one blocks until it is aborted."""

    def __init__(self) -> None:
        self.calls = 0
        self.cancelled = 0

    async def expand(self, payload: Toy) -> tuple[list[Toy], Mapping[str, object]]:
        self.calls += 1
        if self.calls == 1:
            return [Toy(payload.n + 1)], {}
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return [], {}


class OneFastThenSlowChain(Expander[Toy]):
    """The first expansion lands at once; later ones land after termination fires.

    Bounded, unlike :class:`OneFastThenBlockingChain`: a drained pass waits for what is
    in flight, so an expansion that never returns would hang the loop rather than being
    cancelled by it -- which is exactly what the per-call timeouts
    (``Runner.generate_timeout``, ``NodeScorer(timeout=...)``) exist to prevent in
    production.
    """

    def __init__(self, delay: float = 0.02) -> None:
        self.calls = 0
        self.cancelled = 0
        self.launched_after: list[float] = []
        self._delay = delay

    async def expand(self, payload: Toy) -> tuple[list[Toy], Mapping[str, object]]:
        self.calls += 1
        if self.calls == 1:
            return [Toy(payload.n + 1)], {}
        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return [Toy(payload.n + 1)], {}


class CountingSession(FifoSession[Toy]):
    """Counts what the loop hands back: one ``on_failure`` per released reservation."""

    def __init__(self) -> None:
        super().__init__()
        self.updates = 0
        self.releases = 0

    def completed(
        self,
        reservation: Reservation[Toy],
        state: SchedulerState[Toy],
        chain: list[Node[Toy]],
    ) -> None:
        self.updates += 1

    def released(
        self, reservation: Reservation[Toy], exc: BaseException | None
    ) -> None:
        self.releases += 1


@pytest.mark.asyncio
async def test_a_pass_that_hits_its_budget_drains_what_is_in_flight() -> None:
    """The budget stops the launching; the rollouts already running are kept.

    Read three ways, because "drained" is three separate promises: nothing new is
    launched after the budget is hit (the expander's own call count), what was in
    flight is integrated rather than cancelled (its nodes are on the tree and its turns
    are charged), and nothing is handed back as if it had died (no ``on_failure``, no
    reserved gate slot left behind).
    """
    exp = OneFastThenSlowChain()
    backup = CountingSession()
    state: SchedulerState[Toy] = SchedulerState.root(Toy(0), gating=WidthGating(4))
    goal = lambda view: view.stats["rollouts"] >= 1.0  # noqa: E731

    await Scheduler(
        expander=exp,
        allocation=backup,
        termination=Budget(goal=goal),
        max_concurrency=3,
    ).run_on(state)

    assert exp.calls == 3  # the three launched in the first fill, and no more
    assert exp.cancelled == 0
    assert state.stats["rollouts"] == 3.0  # every one of them was integrated
    assert backup.updates == 3
    assert backup.releases == 0  # nothing was released as a failure
    assert state.stats["failures"] == 0.0
    assert len(state.nodes[0].children) == 3  # the drained work is on the tree
    root = state.nodes[0]
    assert root.gate.available(root)  # and its slots came back through `_integrate`


@pytest.mark.asyncio
async def test_a_drained_turn_budget_overshoots_by_at_most_one_episode_per_slot() -> (
    None
):
    """The documented bound, exercised: a turn budget is a stop condition, not a cap.

    What the next expansion will cost is unknown before it runs and the drain finishes
    what is in flight, so the total may pass ``max_turns`` -- by at most one episode
    (``Runner.max_steps`` turns) per concurrency slot, since nothing launches after the
    budget is hit.
    """
    exp = _expander(
        ScriptedGenerator(["go"]), env=ToyEnv(done_at=3), compactor=LenCompactor()
    )
    sched = Scheduler(
        expander=exp,
        termination=Budget(max_turns=2),
        gating=WidthGating(4),
        max_concurrency=4,
    )
    result = await sched.run(await exp.start("hi"))

    assert result.stats["rollouts"] == 4.0  # one whole episode per slot, all kept
    assert result.stats["turns"] > 2  # the cap really was passed...
    assert result.stats["turns"] <= 2 + 4 * 8  # max_turns + max_concurrency * max_steps


class PeakExpander(Expander[Toy]):
    """Records how many expansions were in flight at once."""

    def __init__(self) -> None:
        self.inflight = 0
        self.peak = 0

    async def expand(self, payload: Toy) -> tuple[list[Toy], Mapping[str, object]]:
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        await asyncio.sleep(0)  # let a sibling launched in the same fill reach here too
        self.inflight -= 1
        return [Toy(payload.n + 1)], {}


class AlwaysTheRoot(AllocationSession[Toy]):
    """Expand the root, or nothing -- so what is measured is the root's own gate."""

    def choose(self, view: SchedulerView[Toy]) -> Node[Toy] | None:
        return next((n for n in view.frontier() if n.id == 0), None)


@pytest.mark.asyncio
async def test_an_aborted_expansion_hands_its_gate_slot_back() -> None:
    """A run cancelled with rollouts in flight must not cost the tree its slots.

    Gates are minted per node and live until ``regate``, which the search loop calls
    once between passes -- so a slot an aborted later-pass expansion kept was
    kept for the rest of the call, and a node that lost them all left the frontier for
    good. Read behaviourally: the second pass fills both of the root's slots from the
    root, which it can only do if the first pass gave them back.

    Cancellation, not a spent budget: a budget drains instead of aborting, so this is
    now the only way to end a pass with work still in flight.
    """
    exp = OneFastThenBlockingChain()
    state: SchedulerState[Toy] = SchedulerState.root(
        Toy(0), gating=ConcurrencyGating(2)
    )
    sched = Scheduler(
        expander=exp, termination=Budget(max_rollouts=8), max_concurrency=2
    )

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sched.run_on(state), timeout=0.05)

    assert exp.cancelled  # what was in flight was cancelled with the run
    assert state.stats["failures"] == 0.0  # an abort is not a death
    assert state.failures == []

    peak = PeakExpander()
    await Scheduler(
        expander=peak,
        allocation=AlwaysTheRoot(),
        # the tree already spent one rollout on the pass above, and this budget counts
        # the whole tree's -- two more is what the root's two slots are worth
        termination=Budget(max_rollouts=3),
        max_concurrency=2,
    ).run_on(state)

    assert peak.peak == 2  # both of the root's slots were free to launch into


@pytest.mark.asyncio
async def test_cancelling_the_search_is_not_recorded_as_a_failure() -> None:
    """A caller's timeout stops the search rather than filling the tree with failures.

    ``CancelledError`` is a ``BaseException``, so it passes through the ``except
    Exception`` that records an expansion failure -- and it has to, or a training job
    that gave a prompt a deadline would find the deadline logged as one more flaky
    backend, and the frontier re-selected until the budget ran out.
    """
    exp = OneFastThenBlockingChain()
    state: SchedulerState[Toy] = SchedulerState.root(
        Toy(0), gating=ConcurrencyGating(2)
    )
    sched = Scheduler(
        expander=exp, termination=Budget(max_rollouts=8), max_concurrency=2
    )

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sched.run_on(state), timeout=0.05)

    assert exp.cancelled  # everything in flight was cancelled, not left behind
    assert state.stats["failures"] == 0.0
    assert not state.lock.locked()  # the tree is inspectable after the cancellation
    assert state.nodes[0].gate.available(state.nodes[0])


@pytest.mark.asyncio
async def test_the_release_survives_a_second_cancellation_on_a_contended_lock() -> None:
    """The abort's releases run shielded, so a re-delivered cancel cannot eat them.

    The tree lock is shared -- a ``CriticScheduler`` scoring the same state holds it in
    the ordinary configuration -- so the abort can find it contended. Unshielded, a
    second cancellation delivered while waiting for it raises out of the ``async with``
    and every release below is skipped: gates are minted per node and live until
    ``regate``, so the slot is gone for the rest of the call and the node quietly leaves
    the frontier forever.
    """
    exp = OneFastThenBlockingChain()
    backup = CountingSession()  # one `on_failure` per reservation the abort gives back
    state: SchedulerState[Toy] = SchedulerState.root(
        Toy(0), gating=ConcurrencyGating(2)
    )
    sched = Scheduler(
        expander=exp,
        allocation=backup,
        termination=Budget(max_rollouts=8),
        max_concurrency=2,
    )
    run = asyncio.create_task(sched.run_on(state))
    await asyncio.sleep(0.02)  # both slots launched; the blocking ones are in flight

    async with state.lock:  # stand in for the concurrent critic
        run.cancel()
        await asyncio.sleep(0.02)  # the abort reaches the lock and waits there
        run.cancel()  # ... and is cancelled again while it waits
        await asyncio.sleep(0)
        assert not run.done()  # return only after cleanup can finish
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run, timeout=1.0)

    root = state.nodes[0]
    assert exp.cancelled  # the expansions themselves were cancelled either way
    # both in-flight parents got their reservation back, though the cancellation that
    # arrived mid-release still came out of the run
    assert backup.releases == 2
    assert all(node.gate.available(node) for node in state.nodes.values())
    assert root.gate.available(root)
    assert state.stats["failures"] == 0.0


class CancellingExpander(Expander[Toy]):
    """An expansion that cancels itself -- the innermost form of the same rule."""

    async def expand(self, payload: Toy) -> tuple[list[Toy], Mapping[str, object]]:
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_a_cancelled_expansion_stops_the_run_rather_than_dying_alone() -> None:
    state: SchedulerState[Toy] = SchedulerState.root(Toy(0), gating=WidthGating(2))
    sched = Scheduler(
        expander=CancellingExpander(),
        termination=Budget(max_rollouts=4),
        max_concurrency=1,
    )

    with pytest.raises(asyncio.CancelledError):
        await sched.run_on(state)

    assert state.stats["failures"] == 0.0
