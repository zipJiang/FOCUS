"""Randomized concurrency stress over the scheduler, the critic pass, and the loop.

Why a module of its own: every other scheduler test pins *one* interleaving -- a fake
that answers all at once, one that blocks forever, one that answers on a staircase --
because a named interleaving is how you name a behaviour. What none of them reach is
the space between: expansions landing in an order nobody chose, while the budget runs
out, while a second scheduler holds the same lock. The defects that live there are
accounting ones -- a gate slot never handed back, a reservation charged twice, a stat
that disagrees with the tree it was counted off -- and each is invisible in any single
interleaving while being a permanent loss of capacity in a training job.

So a test here fixes a *seed* rather than a schedule: it drives one of the concurrent
entry points with a fake that jitters its own completion order, and asserts the
invariants that have to hold whatever order things landed in. The seed is in the test
id, so a failure names the interleaving that produced it and replays it exactly.

**Jitter is ``await asyncio.sleep(0)``, repeated a random number of times, not a real
delay.** The event loop's ordering is then a function of the seed alone: a failure
reproduces on any machine and under any load, and the whole module costs no wall time.
Real (tiny) sleeps appear only where the thing under test *is* wall-clock -- the
cancellation deadline of :func:`asyncio.wait_for`.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from focus import (
    Allocator,
    ConcurrencyGating,
    FifoAllocator,
    SearchPass,
    WidthGating,
)
from focus.allocation import (
    AllocationSession,
    NodeWorth,
    Observations,
    RolloutLedger,
    ScoredSession,
)
from focus.codec import ChatCodec
from focus.config import RolloutConfig
from focus.export import to_samples
from focus.generation import (
    GenerateResult,
    Policy,
    PolicyFormat,
    SamplingParams,
)
from focus.generation.parsing import HermesToolCallParser, ToolTurn
from focus.harness import (
    Env,
    RoundBudget,
    Runner,
    StepResult,
    Workspace,
    build_budget,
)
from focus.harness.turn import Turn
from focus.loop import Runtime
from focus.preparation.records import PreparedBatch
from focus.reward import RewardModel, RewardResult
from focus.reward.config import RewardConfig
from focus.scheduler import (
    Budget,
    CriticScheduler,
    Expander,
    GatingPolicy,
    Node,
    NodeScorer,
    RankedSession,
    Scheduler,
    SchedulerState,
    SchedulerView,
    Termination,
)
from tests.estimators import FixedWeight
from tests.helpers import FakeChatTokenizer, prepared

#: Ten interleavings per scenario. Each is a whole run of the loop under test, so the
#: count is a trade against the module's own budget (< 3s); ten has been enough to
#: separate the orderings that matter here (a chain landing before/after the budget
#: latch, two expansions of one node returning out of order).
SEEDS = tuple(range(10))


# -- the payload and the expander that jitters ---------------------------------------


@dataclass(frozen=True)
class Step:
    """A minimal :class:`~focus.scheduler.Payload` that costs turns.

    ``turns_taken`` is what the scheduler's turn accounting reads off a payload (see
    ``scheduler._cost``), and it is the only reason this is not ``tests.helpers``'
    rollout: a turn budget over a payload that keeps no count never fires, so the
    overshoot bound could not be exercised at all.
    """

    n: int
    turns_taken: int
    depth: int
    terminal: bool = False
    forkable: bool = True

    @property
    def turns(self) -> tuple[Turn[Any], ...]:
        """No turn log, so every suffix return over this payload is ``0.0``."""
        return ()

    @property
    def done(self) -> bool:
        return self.terminal

    @property
    def reward(self) -> float:
        # deterministic, node-dependent, and not monotone in depth, so a reward-ranking
        # selection has something to disagree about
        return float((self.n * 7) % 11) / 11.0


async def _jitter(rng: random.Random, *, most: int = 4) -> None:
    """Yield to the loop a seed-determined number of times.

    A zero-delay yield rather than a sleep: what randomizes the completion order is how
    many times a coroutine goes back to the ready queue, and doing it this way keeps the
    order a pure function of the seed rather than of the machine's clock.
    """
    for _ in range(rng.randrange(most + 1)):
        await asyncio.sleep(0)


class JitterExpander(Expander[Step]):
    """Chains of random length, in random completion order, failing at random.

    The three outcomes an expansion actually has in production, at roughly the rates a
    flaky backend produces them: a chain (whatever length the strategy rolled), a raise
    (a 503, a tool that threw), and *no progress at all* -- which the scheduler treats
    as a failure of its own, and which is the one an all-succeeding fake never covers.
    """

    def __init__(
        self,
        rng: random.Random,
        *,
        fail: float = 0.15,
        empty: float = 0.10,
        max_chain: int = 4,
        max_depth: int = 6,
        jitter: int = 4,
    ) -> None:
        self._rng = rng
        self._fail = fail
        self._empty = empty
        self._jitter = jitter
        self.max_chain = max_chain
        self.max_depth = max_depth
        #: Expansions started, finished (however), and in flight right now.
        self.launched = 0
        self.finished = 0
        self.inflight = 0
        self.peak = 0
        self._counter = 0

    async def expand(self, payload: Step) -> tuple[list[Step], Mapping[str, object]]:
        self.launched += 1
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            await _jitter(self._rng, most=self._jitter)
            draw = self._rng.random()
            if draw < self._fail:
                raise RuntimeError(f"expansion of node n={payload.n} died")
            if draw < self._fail + self._empty:
                return [], {}
            chain: list[Step] = []
            cursor = payload
            for index in range(self._rng.randint(1, self.max_chain)):
                self._counter += 1
                last = index == self.max_chain - 1
                cursor = Step(
                    n=self._counter,
                    turns_taken=cursor.turns_taken + 1,
                    depth=cursor.depth + 1,
                    # a depth cap as well as a random ending, so a run with budget left
                    # over still settles instead of growing forever
                    terminal=cursor.depth + 1 >= self.max_depth
                    or (last and self._rng.random() < 0.35),
                )
                chain.append(cursor)
                if cursor.terminal:
                    break
            return chain, {"launch": self.launched}
        finally:
            self.inflight -= 1
            self.finished += 1


class WatchedBudget(Termination[Any]):
    """A :class:`Budget` that records what had been launched when it first said stop.

    The drain's promise is that termination ends the *launching*: nothing may start
    after ``remaining()`` first returns 0. That is a claim about a moment, so it needs
    a witness at that moment -- the expander's launch count -- rather than anything
    readable off the finished tree.
    """

    def __init__(self, inner: Termination[Any], expander: JitterExpander) -> None:
        self._inner = inner
        self._expander = expander
        self.launched_at_stop: int | None = None

    def remaining(self, view: SchedulerView[Any]) -> int | None:
        left = self._inner.remaining(view)
        if left == 0 and self.launched_at_stop is None:
            self.launched_at_stop = self._expander.launched
        return left


def _in_flight(node: Node[Any]) -> int:
    """The gate's reserved-but-unreleased count -- the leak this module hunts."""
    return node.gate._in_use


def _edge_turns(state: SchedulerState[Any]) -> float:
    """Turns summed over every edge of the tree, read off the payloads themselves.

    The scheduler charges one expansion the difference between its leaf's count and its
    parent's; summed over a chain that telescopes, so the tree's own total must equal
    ``stats["turns"]`` exactly -- for any number of expansions, in any order, however
    many of them were still in flight when the budget stopped the launching.
    """
    return float(
        sum(
            node.payload.turns_taken - state.nodes[node.parent_id].payload.turns_taken
            for node in state.nodes.values()
            if node.parent_id is not None
        )
    )


def assert_tree_invariants(
    state: SchedulerState[Any], expander: JitterExpander
) -> None:
    """Everything that must be true of a finished run, whatever order it landed in."""
    nodes = state.nodes
    assert all(node_id == node.id for node_id, node in nodes.items())  # no id reused
    for node in nodes.values():
        if node.id == state.root_id:
            assert node.parent_id is None
            assert node.depth == 0
            continue
        assert node.parent_id in nodes, f"node {node.id} has no parent in the tree"
        parent = nodes[node.parent_id]
        assert node.depth == parent.depth + 1
        assert node.id in parent.children
    for node in nodes.values():
        for child_id in node.children:
            assert nodes[child_id].parent_id == node.id

    stats = state.stats
    assert stats["nodes"] == float(len(nodes) - 1)
    assert stats["failures"] == float(len(state.failures))
    # `rollouts` counts integrated expansions and `failures` the dead ones; together
    # they are every expansion the loop ever started, which is what the expander saw.
    assert stats["rollouts"] + stats["failures"] == float(expander.launched)
    assert stats["turns"] == _edge_turns(state)

    assert expander.inflight == 0  # nothing is still running behind `run_on`'s return
    assert expander.finished == expander.launched
    assert not state.lock.locked()
    for node in nodes.values():
        assert _in_flight(node) == 0, f"node {node.id} kept a gate slot"


def _scheduler(
    expander: Expander[Any],
    termination: Termination[Any],
    *,
    max_concurrency: int,
    **kwargs: Any,
) -> Scheduler[Any]:
    """A scheduler over the stress payload, typed ``Any``.

    The annotation is the whole reason this exists: the shipped allocator pieces
    (:class:`ScoredSession`, :class:`RolloutLedger`) are typed for rollout
    payloads, and this module drives them over a payload of its own to reach the
    accounting without dragging a whole harness behind every test.
    """
    sched: Scheduler[Any] = Scheduler(
        expander=expander,
        termination=termination,
        max_concurrency=max_concurrency,
        **kwargs,
    )
    return sched


def _root(gating: GatingPolicy) -> SchedulerState[Any]:
    """A fresh tree at turn zero, branchable under ``gating``."""
    state: SchedulerState[Any] = SchedulerState.root(
        Step(n=0, turns_taken=0, depth=0), gating=gating
    )
    return state


# -- scenario 1: the scheduler under a jittered expander ------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_concurrency", [1, 4, 16])
@pytest.mark.parametrize("kind", ["turns", "rollouts", "nodes", "goal"])
async def test_a_jittered_search_keeps_its_books(
    seed: int, max_concurrency: int, kind: str
) -> None:
    """The whole account of a run agrees with the tree, whatever order it landed in.

    Node parentage and depth, the three counters against what the expander saw, the
    turn total against the edges it was summed from, and -- the ones a single
    interleaving cannot check -- that nothing is still running, that no gate kept a
    slot, and that nothing launched after the budget latched.
    """
    expander = JitterExpander(random.Random(seed))
    # Every branch of the shipped `Budget`, because they stop the loop differently:
    # rollouts are charged before launch (a hard cap), turns and nodes are read off what
    # landed (a stop condition, hence the overshoot), and a goal is a predicate over the
    # tree -- the one a drained result could flip back, which the latch forbids.
    budgets: dict[str, Termination[Any]] = {
        "turns": Budget(max_turns=12),
        "rollouts": Budget(max_rollouts=12),
        "nodes": Budget(max_nodes=12),
        "goal": Budget(goal=lambda view: view.stats["rollouts"] >= 4.0),
    }
    termination = WatchedBudget(budgets[kind], expander)
    # Both gating policies, keyed off the seed rather than parametrized: they differ in
    # the one thing that matters under concurrency -- whether the children a node
    # already has count against its cap -- and every seed is a whole run either way.
    width = seed % 2 == 0
    state = _root(WidthGating(3) if width else ConcurrencyGating(2))

    await _scheduler(expander, termination, max_concurrency=max_concurrency).run_on(
        state
    )

    assert_tree_invariants(state, expander)
    assert expander.peak <= max_concurrency
    if width:
        # A width cap counts children *and* in-flight branches, so it is a promise about
        # the finished tree and not only about the moment of launching: two expansions
        # of one node checking the cap before either has attached anything would
        # otherwise both pass and leave the node one branch over.
        assert all(len(node.children) <= 3 for node in state.nodes.values())
    if termination.launched_at_stop is not None:
        # the latch is what makes this checkable: once stop is said it is never
        # re-asked, so no launch may follow it
        assert expander.launched == termination.launched_at_stop
    if kind == "turns":
        # a stop condition, not a launch count: one whole chain per slot may land after
        assert state.stats["turns"] <= 12 + max_concurrency * expander.max_chain
    elif kind == "rollouts":
        # a rollout budget is exact -- in-flight expansions are charged before launch
        assert state.stats["rollouts"] + state.stats["failures"] <= 12
    elif kind == "nodes":
        assert len(state.nodes) <= 12 + max_concurrency * expander.max_chain
    elif termination.launched_at_stop is not None:
        # the goal really was reached, and the drain that followed it did not re-open
        # the pass however many rollouts it went on to integrate
        assert state.stats["rollouts"] >= 4.0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_concurrency", [1, 4, 16])
async def test_a_reserving_allocator_gets_every_reservation_back(
    seed: int, max_concurrency: int
) -> None:
    """A scored allocator's charge-at-selection survives every way an expansion ends.

    ``m_pending`` is charged when a node is selected and settled when the rollout comes
    back -- and "comes back" has four shapes here: integrated, raised, produced nothing,
    and (with concurrency) two of them outstanding on one node at once. A leak of any of
    them makes the node look permanently over-sampled, so the allocator stops buying
    evidence there for the rest of the run and nothing says why.
    """
    rng = random.Random(seed)
    expander = JitterExpander(rng)
    observations: dict[int, Observations] = {}
    state = _root(ConcurrencyGating(2))

    await _scheduler(
        expander,
        Budget(max_rollouts=16),
        max_concurrency=max_concurrency,
        allocation=_LeastSampled(observations),
    ).run_on(state)

    assert_tree_invariants(state, expander)
    assert observations  # the allocator really did charge something
    assert all(counts.m_pending == 0 for counts in observations.values())
    assert sum(counts.m_done for counts in observations.values()) == pytest.approx(
        state.stats["rollouts"]
    )


class _LeastSampled(RankedSession[Any]):
    """Buy evidence where least has been bought -- the shape every allocator has.

    A stand-in for ``ChildRelative``'s selection, reduced to the part this test is
    about: a positive score (so :meth:`ScoredSession.eligible` lets it through) and
    a reservation charged at selection rather than at backup.
    """

    def __init__(self, observations: dict[int, Observations]) -> None:
        super().__init__(RolloutLedger(RewardConfig(), observations))
        self._observations = observations

    def score(self, view: SchedulerView[Any], node: Node[Any]) -> float:
        if node.payload.done or not node.payload.forkable:
            return -math.inf
        counts = self._observations.setdefault(node.id, Observations())
        return 1.0 / (1.0 + counts.m_alloc)


# -- scenario 2: the same seed grows the same tree ------------------------------------


def _payload_trace(state: SchedulerState[Any]) -> list[tuple[int, int, int, bool]]:
    """The tree as a sequence, in the order the nodes were created."""
    return [
        (node.id, node.payload.n, node.payload.turns_taken, node.payload.terminal)
        for node in sorted(state.nodes.values(), key=lambda n: n.id)
    ]


@pytest.mark.parametrize("seed", SEEDS)
async def test_an_unjittered_serial_search_is_reproducible(seed: int) -> None:
    """No jitter and one slot: two runs of the same seed grow the same tree, node for
    node. The scheduler's own ordering (frontier insertion order, the tie-break toward
    the lower id) carries the reproducibility -- nothing here is left to the loop."""
    trees = []
    for _ in range(2):
        expander = JitterExpander(random.Random(seed), jitter=0)
        state = _root(WidthGating(3))
        await _scheduler(expander, Budget(max_rollouts=10), max_concurrency=1).run_on(
            state
        )
        assert_tree_invariants(state, expander)
        trees.append(_payload_trace(state))

    assert trees[0] == trees[1]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_concurrency", [4, 16])
async def test_a_jittered_parallel_search_is_reproducible_too(
    seed: int, max_concurrency: int
) -> None:
    """Concurrency reorders *completions*; it must not reorder the tree.

    This is the one that caught a real defect. ``asyncio.wait`` answers with a **set**,
    and the loop used to fold that set into the tree by iterating it -- so every
    expansion that finished in the same round was attached in hash, i.e. memory
    address, order. Two runs of one seed against one backend therefore grew different
    trees, with different node ids and hence different id tie-breaks in selection and
    export, and nothing in the run said so (see
    :func:`test_simultaneous_expansions_attach_in_the_order_they_were_launched` for the
    defect on its own). The scheduler is a search whose every other ordering decision --
    frontier insertion, the tie-break toward the lower id -- is spelled out to be
    reproducible; the integration order is the last one, and it belongs to the launch
    sequence, not to the allocator.

    The envelope is checked alongside, since a budget bounds the run whatever the order:
    every launch is charged, and a chain is at most ``max_chain`` nodes.
    """
    trees = []
    for _ in range(2):
        expander = JitterExpander(random.Random(seed))
        state = _root(WidthGating(3))
        await _scheduler(
            expander, Budget(max_rollouts=10), max_concurrency=max_concurrency
        ).run_on(state)
        assert_tree_invariants(state, expander)
        launched = state.stats["rollouts"] + state.stats["failures"]
        assert launched <= 10
        assert len(state.nodes) - 1 <= launched * expander.max_chain
        trees.append(_payload_trace(state))

    assert trees[0] == trees[1]


class Simultaneous(Expander[Step]):
    """Every expansion answers on the same loop turn, tagged with its launch order.

    The minimal shape of the defect above: one ``asyncio.wait`` round returns *all* of
    them, so which one is folded first is decided purely by how the loop walks the set
    it was handed.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def expand(self, payload: Step) -> tuple[list[Step], Mapping[str, object]]:
        self.calls += 1
        launched = self.calls
        await asyncio.sleep(0)
        return [Step(n=launched, turns_taken=payload.turns_taken + 1, depth=1)], {}


async def test_simultaneous_expansions_attach_in_the_order_they_were_launched() -> None:
    """Four expansions, one wait round, one answer: launch order, every time.

    Regression. Iterating ``asyncio.wait``'s ``done`` set attached them in address
    order, which varied between processes *and* between two runs inside one process
    (ids get reused), so this assertion failed for most orderings and passed for some.
    """
    state = _root(WidthGating(4))

    await _scheduler(Simultaneous(), Budget(max_rollouts=4), max_concurrency=4).run_on(
        state
    )

    attached = [node.payload.n for node in state.nodes.values() if node.id]
    assert attached == [1, 2, 3, 4]


# -- scenario 3: a critic pass scoring the tree a generation pass is growing ----------


def _name(payload: Step) -> str:
    """Serialize a payload for the reward model -- one distinct string per node."""
    return f"n{payload.n}"


class FlakyReward(RewardModel):
    """A reward server that answers late and 503s one batch in five.

    Batched-only on purpose: the default :meth:`RewardModel.ascore_batch` offloads to a
    worker *thread*, which would make the interleaving the OS scheduler's rather than
    the seed's -- and this module's whole claim is that a failure replays.
    """

    def __init__(self, rng: random.Random, *, fail: float = 0.2) -> None:
        self._rng = rng
        self._fail = fail
        #: How many times each serialized payload was sent -- the double-scoring check.
        self.calls: Counter[str] = Counter()
        self.dropped: set[str] = set()
        self.batches = 0
        self.failures = 0

    def score(self, context: str) -> RewardResult:  # pragma: no cover - batched only
        raise AssertionError("this model is only ever asked for batches")

    async def ascore_batch(self, contexts: Sequence[str]) -> list[RewardResult]:
        self.batches += 1
        self.calls.update(contexts)
        await _jitter(self._rng)
        if self._rng.random() < self._fail:
            self.failures += 1
            self.dropped.update(contexts)
            raise RuntimeError("reward server 503")
        return [RewardResult(score=len(context) / 10.0) for context in contexts]


@pytest.mark.parametrize("seed", SEEDS)
async def test_a_critic_pass_scores_a_tree_that_is_still_growing(seed: int) -> None:
    """Two schedulers, one tree, one lock: what the critic owes when generation stops.

    The critic tracks generation through ``until`` and drains after it, so the promise
    is that no finished node is left silently unscored: each one carries the version, or
    it was in a batch the server dropped -- and every drop is on the count a trainer
    reads. Nothing is scored twice (which would double a batch's cost and, with a
    non-deterministic server, overwrite a real estimate), nothing escapes as an
    exception, and the lock both of them take is free at the end.
    """
    expander = JitterExpander(random.Random(seed))
    model = FlakyReward(random.Random(seed + 1000))
    state = _root(WidthGating(3))
    generated = asyncio.Event()
    critic: CriticScheduler[Any] = CriticScheduler(
        model=model,
        serialize=_name,
        version="rm-v1",
        batch_size=3,
        poll_interval=0.0,  # a bare yield: the wait is for the seed, not for a clock
    )

    async def _generate() -> None:
        try:
            await _scheduler(
                expander, Budget(max_rollouts=12), max_concurrency=4
            ).run_on(state)
        finally:
            generated.set()

    await asyncio.gather(_generate(), critic.run_on(state, until=generated))

    assert_tree_invariants(state, expander)
    terminals = state.terminals()
    assert terminals  # the pass had something to score
    for node in terminals:
        assert "rm-v1" in node.critic or _name(node.payload) in model.dropped
    assert state.stats["score_failures"] == float(model.failures)
    assert model.calls  # ... and it really did call the server
    assert max(model.calls.values()) == 1  # no node was ever scored twice


def _reason_above(state: SchedulerState[Any], node: Node[Any]) -> bool:
    """Whether any strict ancestor of ``node`` recorded a scoring failure."""
    cursor = node.parent_id
    while cursor is not None:
        if "score_error" in state.nodes[cursor].metadata:
            return True
        cursor = state.nodes[cursor].parent_id
    return False


@pytest.mark.parametrize("seed", SEEDS)
async def test_two_scorers_on_one_tree_stay_on_their_own_channels(seed: int) -> None:
    """The production shape: a value head scoring inside the expansion, a critic pass
    scoring the same tree from outside, on a second version.

    They write the same dict on the same nodes under the same lock, so what has to hold
    is that neither can reach the other's key and that the failure account is the sum of
    both -- ``stats["score_failures"]`` is one counter and two writers increment it.
    """
    expander = JitterExpander(random.Random(seed))
    bundled = FlakyReward(random.Random(seed + 20))
    external = FlakyReward(random.Random(seed + 30))
    state = _root(WidthGating(3))
    generated = asyncio.Event()
    critic: CriticScheduler[Any] = CriticScheduler(
        model=external,
        serialize=_name,
        version="rm-v2",
        batch_size=4,
        poll_interval=0.0,
    )
    scorer: NodeScorer[Any] = NodeScorer(
        model=bundled, serialize=_name, version="rm-v1"
    )

    async def _generate() -> None:
        try:
            await _scheduler(
                expander,
                Budget(max_rollouts=12),
                max_concurrency=4,
                scorer=scorer,
            ).run_on(state)
        finally:
            generated.set()

    await asyncio.gather(_generate(), critic.run_on(state, until=generated))

    assert_tree_invariants(state, expander)
    assert state.stats["score_failures"] == float(bundled.failures + external.failures)
    for node in state.nodes.values():
        # A bundled scoring failure keeps the whole chain and files its reason on the
        # node the expansion *left* -- so an unscored node is one some ancestor of which
        # launched a dropped batch. Walked rather than checked on the parent: a chain is
        # attached as a line, and only its first node's parent is that launch point.
        if node.id != state.root_id:
            assert "rm-v1" in node.critic or _reason_above(state, node)
    for node in state.terminals():
        assert "rm-v2" in node.critic or _name(node.payload) in external.dropped


# -- the harness fakes the whole-loop scenarios run on --------------------------------


@dataclass(frozen=True)
class Round(RoundBudget):
    """A task state with the per-round tool budget the ``budget`` compactor folds on."""

    n: int = 0


class RoundEnv(Env[Round, ToolTurn]):
    """Spends one call of the round per step; ends on ``stop`` or after four steps.

    The counter is what makes :func:`~focus.harness.build_budget`'s
    ``StateCounter`` half live -- without a state that is charged, only the token half
    of that trigger ever fires and the fold pattern under test is a different one.

    Over :class:`ToolTurn` rather than plain text because the fold is a rollout through
    the *same* parser: an agentic compactor drives a
    :class:`~focus.harness.compaction.fold_env.FoldEnv`, which reads tool
    calls off the turn, so a text parser here would fail inside the first fold.
    """

    async def reset(self) -> Round:
        return Round(calls_remaining=2, calls_max=2)

    async def step(
        self, state: Round, action: ToolTurn, *, workspace: Workspace | None = None
    ) -> StepResult[Round]:
        del workspace
        nxt = Round(
            calls_remaining=state.calls_remaining - 1,
            calls_max=state.calls_max,
            n=state.n + 1,
        )
        done = nxt.n >= 4
        return StepResult(
            next_state=nxt,
            messages=({"role": "user", "content": f"obs{nxt.n}"},),
            reward_outcome=(
                (2.0 if action.text == "stop-good" else 0.5) if done else 0.0
            ),
            reward_step=0.1,
            done=done,
        )


class JitterGenerator(Policy[Any]):
    """A cycling generator that answers after a seed-chosen number of loop turns.

    The staggering is the point: with several expansions in flight, whose generate
    returns first decides which rollout reaches the budget line first, so an all-at-once
    fake never puts anything *in* flight at the moment the budget stops the launching.
    """

    def __init__(
        self,
        rng: random.Random,
        completions: Sequence[str] = ("go", "stop-good", "go", "stop-bad"),
        *,
        jitter: int = 3,
        delay: float = 0.0,
    ) -> None:
        self._rng = rng
        self._completions = list(completions)
        self._jitter = jitter
        self._delay = delay
        self.calls = 0
        #: Set once :attr:`trip_after` generations have run -- the cancellation trigger.
        self.tripped = asyncio.Event()
        self.trip_after: int | None = None

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        del prefix_tokens, sampling_params
        self.calls += 1
        if self.trip_after is not None and self.calls >= self.trip_after:
            self.tripped.set()
        if self._delay:
            await asyncio.sleep(self._delay)
        else:
            await _jitter(self._rng, most=self._jitter)
        text = self._completions[(self.calls - 1) % len(self._completions)]
        return GenerateResult(
            text=text,
            tokens=tuple(ord(c) for c in text),
            logprobs=[-0.5 - 0.1 * i for i in range(len(text))],
        )


class _UnitWorth(NodeWorth):
    def worth(self, view, node, counts: Observations) -> float:
        del view, counts
        return float(len(node.payload.turns))


class _PreferDepth(Allocator):
    """Expand the deepest forkable node -- a later pass without a ranking recipe."""

    name = "prefer_depth"

    def build(self, view, ledger) -> AllocationSession[Any]:
        del view
        return ScoredSession(_UnitWorth(ledger.reward_config), ledger)


def _runtime(
    generator: JitterGenerator,
    *,
    value_model: RewardModel | None = None,
    max_concurrency: int = 8,
) -> Runtime:
    """The whole loop over the fakes: folding runner, every-turn strategy, both
    default-shaped weighters at budgets this module can afford."""
    runner: Runner[Round, ToolTurn] = Runner(
        policy=Policy(
            format=PolicyFormat(
                "custom", ChatCodec(FakeChatTokenizer()), HermesToolCallParser()
            ),
            complete=(generator).agenerate_tokens,
            exact=(generator).provides_exact_tokens,
            native=(generator).provides_native_tool_calls,
            version="policy",
        ),
        env=RoundEnv(),
        system_prompt="sys",
        compactor=build_budget(max_prompt_tokens=64, max_interactions=2),
        max_steps=6,
    )
    return Runtime(
        runner=runner,
        value_model=value_model,
        config=RolloutConfig(
            max_concurrency=max_concurrency,
            passes=(
                SearchPass(FifoAllocator(), budget=8, gating=WidthGating(3)),
                SearchPass(_PreferDepth(), budget=8, gating=ConcurrencyGating(4)),
                SearchPass(_PreferDepth(), budget=8, gating=ConcurrencyGating(4)),
            ),
        ),
    )


@pytest.fixture
def trees(monkeypatch: pytest.MonkeyPatch) -> list[SchedulerState[Any]]:
    """Every tree :func:`run_rollout` seeds, captured as it is seeded.

    The loop returns records, not the tree they came off -- deliberately, since a result
    is meant to outlive the search. But the invariants this module is about (a gate that
    kept a slot, a lock still held, a node an exported record names) are properties of
    that tree, and a cancelled run never returns anything at all. So they are read off
    the state itself, captured at the one place every run makes one.
    """
    seen: list[SchedulerState[Any]] = []
    seed_root = SchedulerState.root

    def spy(payload: Any, *, gating: GatingPolicy) -> SchedulerState[Any]:
        state: SchedulerState[Any] = seed_root(payload, gating=gating)
        seen.append(state)
        return state

    monkeypatch.setattr(SchedulerState, "root", spy)
    return seen


def assert_settled(state: SchedulerState[Any]) -> None:
    """The tree is inspectable and holds nothing back: no lock, no reserved slot."""
    assert not state.lock.locked()
    for node in state.nodes.values():
        assert _in_flight(node) == 0, f"node {node.id} kept a gate slot"


# -- scenario 4: the whole loop, end to end -------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
async def test_the_whole_loop_exports_records_that_match_the_tree_they_came_off(
    seed: int, trees: list[SchedulerState[Any]]
) -> None:
    """A first pass, two later passes and export, with a value head that 503s one call
    in
    five: the records must name real nodes, and the account that ships with them must
    be the tree's own -- turns summed off the edges, failures matching their reasons."""
    generator = JitterGenerator(random.Random(seed))
    value_model = FlakyReward(random.Random(seed + 500))
    runtime = _runtime(generator, value_model=value_model)

    result = await prepared("solve it", runtime, estimator=FixedWeight())

    assert isinstance(result, PreparedBatch)
    assert result.actor
    assert result.critic
    assert {record.estimator for record in result.actor} == {"test_fixed"}
    (state,) = trees
    assert_settled(state)
    # the compactor really did fire: the `budget` composition folds on the round counter
    # this env spends, and a run with no fold is a different (and much tamer) tree
    assert any(node.payload.folds for node in state.nodes.values())
    assert state.stats["turns"] == _edge_turns(state)
    assert result.stats["failures"] == float(len(result.failures))
    assert result.failures == tuple(state.failures)
    # a scoring failure is a kept rollout with a missing estimate, so it is counted
    # apart from the dead ones -- and every one of them came from the flaky server
    assert result.stats["score_failures"] == float(value_model.failures)
    for record in result.actor:
        node = state.nodes[record.node_id]
        assert node.parent_id == record.group_id
    for sample in result.critic:
        assert sample.node_id in state.nodes


@pytest.mark.parametrize("seed", SEEDS)
async def test_the_same_seed_without_jitter_exports_the_same_samples(
    seed: int, trees: list[SchedulerState[Any]]
) -> None:
    """Given a deterministic tree, the export is a function of it -- estimator draws
    included, which is what ``ChildRelative``'s per-call seeded generator buys."""
    runs = []
    for _ in range(2):
        runtime = _runtime(
            JitterGenerator(random.Random(seed), jitter=0),
            value_model=FlakyReward(random.Random(seed + 500)),
            max_concurrency=4,
        )
        runs.append(
            to_samples(
                await prepared("solve it", runtime, estimator=FixedWeight()),
                0,
            )
        )

    assert runs[0] == runs[1]
    assert runs[0]
    for state in trees:
        assert_settled(state)


# -- scenario 5: cancellation under load ----------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
async def test_cancelling_a_loaded_run_leaves_nothing_behind(
    seed: int, trees: list[SchedulerState[Any]]
) -> None:
    """A deadline lands mid-flight, in a different place each seed. What must survive it
    is not the run -- it is the tree and the loop: the cancellation comes out as the
    caller asked for it, every in-flight expansion is finished rather than orphaned, no
    gate slot is kept, and the lock is free so the state can still be read."""
    rng = random.Random(seed)
    generator = JitterGenerator(random.Random(seed), delay=0.0015)
    runtime = _runtime(generator, value_model=FlakyReward(random.Random(seed + 500)))
    before = asyncio.all_tasks()

    if rng.random() < 0.5:
        # a wall-clock deadline, which is the shape a training job's timeout has
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                prepared("solve it", runtime), rng.uniform(2e-3, 6e-3)
            )
    else:
        # ... and a cancel keyed to the work itself, so the deadline can land inside a
        # generate, an env step, a fold, or the scheduler's own accounting
        generator.trip_after = rng.randint(1, 8)
        run = asyncio.create_task(prepared("solve it", runtime))
        await generator.tripped.wait()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    for _ in range(20):  # let the cancelled expansions wind down
        await asyncio.sleep(0)
    assert asyncio.all_tasks() - before == set()
    (state,) = trees
    assert_settled(state)
    assert state.stats["failures"] == 0.0  # an abort is not a death
    assert state.failures == []


@pytest.mark.parametrize("seed", SEEDS)
async def test_cancelling_a_jittered_search_gives_back_what_selection_took(
    seed: int,
) -> None:
    """The abort path, entered from a different point of the loop on every seed.

    A cancelled pass is the one case where the loop ends with work still in flight, and
    what it holds at that moment is not its own: a gate slot per in-flight parent, and
    an allocator's ``m_pending`` charged at selection. Gates are minted per node and
    live until ``regate``, so a slot kept here is capacity lost for the rest of the
    call -- and a node that loses them all leaves the frontier permanently. The proof is
    behavioural as well as arithmetic: a second pass over the same tree still grows it.
    """
    rng = random.Random(seed)
    expander = JitterExpander(random.Random(seed))
    observations: dict[int, Observations] = {}
    state = _root(ConcurrencyGating(2))
    before = asyncio.all_tasks()

    run = asyncio.create_task(
        _scheduler(
            expander,
            Budget(max_rollouts=64),  # more than the cancellation will ever reach
            max_concurrency=4,
            allocation=_LeastSampled(observations),
        ).run_on(state)
    )
    for _ in range(rng.randrange(1, 25)):  # a different point in the loop each seed
        await asyncio.sleep(0)
    assert not run.done()
    # Read in the same loop turn as the cancel, so it is the count at the moment the
    # abort begins: the expander really does fail sometimes, and what is being asserted
    # is that *the abort* charges nothing, not that nothing ever died.
    dead = state.stats["failures"]
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    for _ in range(20):  # let the cancelled expansions wind down
        await asyncio.sleep(0)

    assert asyncio.all_tasks() - before == set()
    assert expander.inflight == 0
    assert_settled(state)
    assert all(counts.m_pending == 0 for counts in observations.values())
    assert state.stats["failures"] == dead  # an abort is not a death
    assert len(state.failures) == int(dead)

    grown = len(state.nodes)
    resumed = JitterExpander(random.Random(seed + 1), fail=0.0, empty=0.0)
    # A rollout budget counts the whole tree's spend, so two *more* is what this pass
    # is worth -- the cancelled one already charged its own against the same counter.
    spent = int(state.stats["rollouts"] + state.stats["failures"])
    await _scheduler(resumed, Budget(max_rollouts=spent + 2), max_concurrency=2).run_on(
        state
    )
    assert len(state.nodes) > grown  # the tree still had slots to launch from


@pytest.mark.parametrize("seed", SEEDS)
async def test_cancelling_generation_leaves_the_critic_a_consistent_tree(
    seed: int,
) -> None:
    """The abort runs while a second scheduler is actively mutating the same state.

    A critic pass tracks generation through ``until``, so a cancelled generation is
    also the critic's signal to drain -- and it drains over the very tree the abort is
    handing reservations back into. Both have to come out clean: the critic stops on its
    own rather than spinning on an event that was set by a cancellation, and the tree it
    is left with holds no slot and no reservation.
    """
    rng = random.Random(seed)
    expander = JitterExpander(random.Random(seed))
    model = FlakyReward(random.Random(seed + 40))
    observations: dict[int, Observations] = {}
    state = _root(WidthGating(3))
    generated = asyncio.Event()
    critic: CriticScheduler[Any] = CriticScheduler(
        model=model,
        serialize=_name,
        version="rm-v1",
        batch_size=2,
        poll_interval=0.0,
    )
    before = asyncio.all_tasks()

    async def _generate() -> None:
        try:
            await _scheduler(
                expander,
                Budget(max_rollouts=64),
                max_concurrency=4,
                allocation=_LeastSampled(observations),
            ).run_on(state)
        finally:
            generated.set()

    generation = asyncio.create_task(_generate())
    scoring = asyncio.create_task(critic.run_on(state, until=generated))
    for _ in range(rng.randrange(2, 30)):
        await asyncio.sleep(0)
    assert not generation.done()
    generation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generation
    await scoring  # drains what generation left, and stops: no exception escapes
    for _ in range(20):
        await asyncio.sleep(0)

    assert asyncio.all_tasks() - before == set()
    assert_settled(state)
    assert all(counts.m_pending == 0 for counts in observations.values())
    for node in state.terminals():
        assert "rm-v1" in node.critic or _name(node.payload) in model.dropped
