from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import pytest

from focus import (
    AllocationSession,
    Allocator,
    ConcurrencyGating,
    DirectBranchTdEstimator,
    FifoAllocator,
    SearchPass,
    WidthGating,
    prepare_samples,
    run_search,
)
from focus.allocation import (
    NodeWorth,
    Observations,
    RolloutLedger,
    ScoredSession,
    ValueRefinement,
)
from focus.codec import ChatCodec
from focus.config import RolloutConfig
from focus.generation import (
    GenerateResult,
    Policy,
    PolicyFormat,
    SamplingParams,
)
from focus.generation.parsing import TextActionParser
from focus.harness import (
    Compaction,
    CompactionResult,
    Compactor,
    Env,
    Runner,
    StepResult,
    Workspace,
)
from focus.integrations.slime import to_samples
from focus.loop import Runtime, _budget
from focus.preparation.records import PreparedBatch
from focus.protocols import Summary
from focus.reward import RewardModel, RewardResult
from focus.scheduler import (
    PASS_KEY,
    SchedulerState,
    SchedulerView,
)
from focus.scheduler.core.tree import STAT_FAILURES, STAT_TURNS
from tests.estimators import FixedWeight
from tests.helpers import (
    DeadReward,
    FakeChatTokenizer,
    InexactBridge,
    ShortBatchReward,
    StalledGenerator,
    StalledReward,
    WhenTrigger,
    prepared,
    rollout,
)


class CyclingGenerator(Policy[Any]):
    """Deterministic: cycles fixed completions, with per-token logprobs."""

    def __init__(self, completions: Sequence[str]) -> None:
        self._completions = list(completions)
        self.calls = 0

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        text = self._completions[self.calls % len(self._completions)]
        self.calls += 1
        return GenerateResult(
            text=text,
            tokens=tuple(ord(c) for c in text),
            logprobs=[-0.5 - 0.1 * i for i in range(len(text))],
        )


@dataclass(frozen=True)
class CountState:
    n: int = 0


class CountEnv(Env[CountState, str]):
    """'stop' ends with an outcome reward keyed to the action; everything else steps."""

    async def reset(self) -> CountState:
        return CountState()

    async def step(
        self, state: CountState, action: str, *, workspace: Workspace | None = None
    ) -> StepResult[CountState]:
        nxt = CountState(state.n + 1)
        done = action.startswith("stop") or nxt.n >= 6
        return StepResult(
            next_state=nxt,
            messages=({"role": "user", "content": f"obs{nxt.n}"},),
            reward_outcome=(2.0 if action == "stop-good" else 0.5) if done else 0.0,
            reward_step=0.1,
            done=done,
        )


class EveryTurnCompactor(Compactor[CountState]):
    """Folds after every turn, so the rollout offers real branch points to work with."""

    trigger = WhenTrigger(lambda prompt, state: state.n > 0)

    async def compact(
        self, compaction: Compaction[CountState]
    ) -> CompactionResult[CountState]:
        return CompactionResult(compaction.seed, replace(compaction.state, n=0))


def _runner(
    generator: Policy[Any] | None = None,
    version: str = "policy",
    **kwargs: Any,
) -> Runner[CountState, str]:
    """The folding toy runner every runtime here drives.

    ``generator`` is a parameter rather than a fixture because what a *backend* does --
    answer, fail, stall -- is the one thing a robustness test varies about the runner.
    """
    generator = generator or CyclingGenerator(["go", "stop-good", "go", "stop-bad"])
    return Runner(
        policy=Policy(
            format=PolicyFormat(
                "custom", ChatCodec(FakeChatTokenizer()), TextActionParser()
            ),
            complete=generator.agenerate_tokens,
            exact=generator.provides_exact_tokens,
            native=generator.provides_native_tool_calls,
            version=version,
        ),
        env=CountEnv(),
        system_prompt="sys",
        protocol=Summary(EveryTurnCompactor()),
        max_steps=4,
        **kwargs,
    )


#: What a runtime *holds* -- plus the bound on the endpoint it holds -- rather than what
#: it is configured with: the only kwargs below that are not `RolloutConfig` knobs.
LIVE_FIELDS = frozenset(
    {"runner", "scorer", "value_model", "value_serialize", "score_timeout"}
)


def _runtime(**kwargs) -> Runtime:
    """A runtime around the fake runner, with this file's small search budgets.

    Keyword arguments are routed by what they are: a live object goes on the
    `Runtime`, everything else onto its `RolloutConfig`.
    """
    defaults: dict[str, Any] = {
        "passes": (
            SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
            SearchPass(FifoAllocator(), budget=48, gating=ConcurrencyGating(4)),
            SearchPass(FifoAllocator(), budget=8, gating=ConcurrencyGating(4)),
        ),
        "max_concurrency": 1,
    }
    defaults.update({k: v for k, v in kwargs.items() if k not in LIVE_FIELDS})
    live = {k: v for k, v in kwargs.items() if k in LIVE_FIELDS}
    live.setdefault("value_model", ConstantReward(0.0))
    runner = live.pop("runner", None) or _runner()
    return Runtime(runner=runner, config=RolloutConfig(**defaults), **live)


def test_search_has_no_implicit_passes() -> None:
    assert RolloutConfig().passes == ()
    assert not hasattr(DirectBranchTdEstimator(), "build")
    assert not hasattr(FifoAllocator(), "assign")


@pytest.mark.asyncio
async def test_one_call_runs_every_pass_and_returns_training_records() -> None:
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    assert isinstance(result, PreparedBatch)
    assert result.reward_config is not None
    assert len(result) > 0
    assert result.critic
    assert result.actor
    assert {r.estimator for r in result.actor} == {"direct_branch_td"}


@pytest.mark.asyncio
async def test_a_prompt_that_is_neither_text_nor_a_checkpoint_is_refused() -> None:
    """The trainer's sample object arriving instead of its prompt fails here, once.

    It used to reach the chat template, which raised inside every expansion; the
    scheduler charged each as a dead rollout and the caller got an empty sample list
    with nothing in it naming the cause.
    """

    @dataclass
    class TrainerSample:
        prompt: str

    with pytest.raises(TypeError, match="prompt must be"):
        await prepared(
            TrainerSample("solve it"), _runtime(), estimator=DirectBranchTdEstimator()
        )


# -- A/B symmetry --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_later_pass_edges_are_actor_data() -> None:
    """A/B symmetry: later-pass edges are actor data too.

    A later-pass edge is drawn under the same generative law as a first-pass edge, so
    nothing about how it was scheduled disqualifies it from training the actor.
    """
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    assert result.actor
    pass_ids = {r.search_pass for r in result.actor}
    assert 0 in pass_ids
    assert any(p != 0 for p in pass_ids)

    later_ids = {r.node_id for r in result.actor if r.search_pass != 0}
    assert later_ids
    # later-pass edges that finished are eligible for the actor lane
    assert later_ids & {r.node_id for r in result.actor}


@pytest.mark.asyncio
async def test_exported_spans_keep_their_mask_aligned_to_their_tokens() -> None:
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    records = result.actor
    spans = [span for r in records for span in r.spans]
    assert spans
    for span in spans:
        assert len(span.trainable) == len(span.tokens)
        # the behavior column, when present, is aligned to the pack
        assert span.logprobs is None or len(span.logprobs) == len(span.tokens)


@pytest.mark.asyncio
async def test_a_fold_and_a_leaf_on_one_chain_score_comparably() -> None:
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    weights = [abs(r.weight) for r in result.actor if r.estimator == "direct_branch_td"]
    assert len(weights) > 2
    assert all(math.isfinite(w) for w in weights)


@pytest.mark.asyncio
async def test_one_expansion_grows_a_line_of_branch_points() -> None:
    state = await run_search("solve it", _runtime())

    interior = [n for n in state.nodes.values() if n.children and n.id != state.root_id]
    assert interior
    assert all(not n.payload.done for n in interior)
    assert all(n.payload.done for n in state.nodes.values() if not n.children)
    assert state.stats["nodes"] > state.stats["rollouts"]
    assert state.stats["turns"] >= state.stats["nodes"]


@pytest.mark.asyncio
async def test_fifo_records_birth_evidence_for_the_next_pass() -> None:
    class Capture(FifoAllocator):
        def build(self, view, ledger):
            self.observations = ledger.observations
            return None

    capture = Capture()
    runtime = _runtime(
        passes=(
            SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
            SearchPass(capture, budget=1, gating=ConcurrencyGating(4)),
        )
    )
    state = await run_search("solve it", runtime)
    created = [n for n in state.nodes.values() if n.parent_id is not None]
    assert created
    assert set(capture.observations) == set(state.nodes)
    assert (
        sum(c.m_done for c in capture.observations.values()) == state.stats["rollouts"]
    )
    for node in created:
        counts = capture.observations[node.id]
        assert counts.returns.n == 1 + counts.m_done
        assert counts.m_pending == 0


def _unfolding_runtime(**kwargs) -> Runtime:
    """A runtime whose rollouts never fold, so the only branch points on a line are the
    ones the strategy itself decided to keep."""
    runner: Runner[CountState, str] = Runner(
        policy=Policy(
            format=PolicyFormat(
                "custom", ChatCodec(FakeChatTokenizer()), TextActionParser()
            ),
            complete=(CyclingGenerator(["go"])).agenerate_tokens,
            exact=(CyclingGenerator(["go"])).provides_exact_tokens,
            native=(CyclingGenerator(["go"])).provides_native_tool_calls,
            version="policy",
        ),
        env=CountEnv(),
        system_prompt="sys",
        max_steps=4,
    )
    defaults: dict[str, Any] = {
        "passes": (SearchPass(FifoAllocator(), budget=4, gating=WidthGating(1)),),
        "max_concurrency": 1,
    }
    defaults.update(kwargs)
    return Runtime(runner=runner, config=RolloutConfig(**defaults))


@pytest.mark.asyncio
async def test_the_runtime_picks_which_checkpoints_become_branch_points() -> None:
    """Without compaction the tree contains a root and completed trajectories."""
    folded = await run_search("solve it", _unfolding_runtime())
    assert len(folded.nodes) == 2  # root + the one terminal: no fold, no branch point


@pytest.mark.asyncio
async def test_a_node_ends_with_as_many_children_as_its_width() -> None:
    state = await run_search(
        "solve it",
        _runtime(
            passes=(SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),)
        ),
    )

    branched = [n for n in state.nodes.values() if len(n.children) > 1]
    assert branched
    for node in branched:
        assert len(node.children) <= 2


@pytest.mark.asyncio
async def test_the_weight_an_estimator_wrote_is_the_one_that_goes_on_the_wire() -> None:
    """Export composes nothing: what `assign` wrote reaches the sample, times the ratio.

    There used to be a multiplication here -- ``weight = confidence * advantage`` --
    and it was the one step between what an estimator meant and what the actor trained
    on that no estimator could see.
    """
    result = await prepared(
        "solve it",
        _runtime(
            passes=(
                SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
                SearchPass(FifoAllocator(), budget=4, gating=ConcurrencyGating(4)),
            )
        ),
        estimator=DirectBranchTdEstimator(),
    )

    assert result.actor
    rewards = [
        s["reward"] for s in to_samples(result, 0) if s["metadata"]["lane"] == "actor"
    ]
    assert rewards == [r.weight for r in result.actor for _ in r.spans]


@pytest.mark.asyncio
async def test_a_constant_weight_reaches_every_slot_untouched() -> None:
    result = await prepared(
        "solve it",
        _runtime(
            passes=(SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),)
        ),
        estimator=FixedWeight(0.5),
    )

    assert result.actor
    # the one weighter configured owns every slot, and every weight is the constant
    assert {r.estimator for r in result.actor} == {"test_fixed"}
    assert all(r.weight == 0.5 for r in result.actor)


@pytest.mark.asyncio
async def test_the_estimator_weight_is_the_exported_reward() -> None:
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    by_node = {(r.estimator, r.node_id): r for r in result.actor}
    checked = 0
    for sample in to_samples(result, 0):
        if sample["metadata"]["lane"] != "actor":
            continue
        meta = sample["metadata"]
        record = by_node[(meta["estimator"], meta["node_id"])]
        assert sample["reward"] == pytest.approx(record.weight)
        # the recipe's own account of the number rides in the metadata, as a plain
        # dict a JSON encoder can write, and never as a column beside `reward`
        assert meta["diagnostics"] == dict(record.diagnostics)
        assert isinstance(meta["diagnostics"], dict)
        assert "confidence" not in sample
        assert "advantage" not in sample
        checked += 1
    assert checked


@pytest.mark.asyncio
async def test_no_exported_span_is_empty() -> None:
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    records = result.actor
    assert records
    for record in records:
        assert record.spans
        assert all(span.tokens for span in record.spans)


@pytest.mark.asyncio
async def test_the_advantage_estimator_reaches_the_exported_weight() -> None:
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    advantages = [r for r in result.actor if r.estimator == "direct_branch_td"]
    assert advantages
    assert any(r.weight != 0.0 for r in advantages)
    assert all(r.group_id != r.node_id for r in advantages)


@pytest.mark.asyncio
async def test_every_expansion_records_which_pass_produced_it() -> None:
    runtime = _runtime()
    state = await run_search("solve it", runtime)

    created = [n for n in state.nodes.values() if n.id != state.root_id]
    assert created
    assert {n.metadata[PASS_KEY] for n in created} <= {0, 1, 2}


@pytest.mark.asyncio
async def test_one_scheduler_pass_per_weighter_that_asks_for_evidence() -> None:
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())
    passes = {r.search_pass for r in result.actor}
    assert passes == {0, 1, 2}

    quiet = await prepared(
        "solve it",
        _runtime(
            passes=(SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),)
        ),
        estimator=FixedWeight(),
    )
    assert {r.search_pass for r in quiet.actor} == {0}
    assert quiet.actor
    assert all(r.weight == 1.0 for r in quiet.actor)


@pytest.mark.asyncio
async def test_each_pass_gets_its_own_budget_on_the_shared_tree() -> None:
    runtime = _runtime(
        passes=(
            SearchPass(FifoAllocator(), budget=6, gating=WidthGating(2)),
            SearchPass(FifoAllocator(), budget=4, gating=ConcurrencyGating(4)),
            SearchPass(FifoAllocator(), budget=2, gating=ConcurrencyGating(4)),
        )
    )

    result = await prepared("solve it", runtime, estimator=DirectBranchTdEstimator())

    slack = runtime.config.max_concurrency * runtime.runner.max_steps
    turns = {0: 6, 1: 4, 2: 2}
    spent: dict[int | None, int] = {}
    for record in result.actor:
        spent[record.search_pass] = spent.get(record.search_pass, 0) + 1
    for search_pass, cap in turns.items():
        assert spent.get(search_pass, 0) <= cap + slack
    assert spent.get(0, 0) >= 1
    assert spent.get(1, 0) >= 1


class _UnitWorth(NodeWorth):
    """Any forkable node is worth buying -- a scored pass without a selection recipe."""

    def worth(self, view: SchedulerView[Any], node: Any, counts: Observations) -> float:
        del view, node, counts
        return 1.0


class _CollectSpy(Allocator):
    """A scored allocator that keeps what it was built against, and what it bought.

    The fold-back between passes is the loop's, so the only place its effect shows is
    the next pass's ``build`` -- which is exactly what this records. ``bought`` keeps
    the live counts by reference: they are what the pass fills as its rollouts return.
    """

    name = "spy"

    def __init__(self) -> None:
        self.seen: list[dict[int, Observations]] = []
        self.bought: list[Mapping[int, Observations]] = []

    def build(
        self,
        view: SchedulerView[Any],
        ledger: RolloutLedger,
    ) -> AllocationSession[Any]:
        del view
        self.seen.append(deepcopy(ledger.observations))
        self.bought.append(ledger.observations)
        return ScoredSession(_UnitWorth(ledger.reward_config), ledger)


@pytest.mark.asyncio
async def test_a_later_pass_starts_from_what_an_earlier_one_bought() -> None:
    """The fold-back is the loop's, so it happens after *every* pass.

    It used to be an estimator hook, which meant it happened only for the estimators
    that implemented it: a GAE pass could buy a hundred completed trajectories and the
    child-relative pass behind it would still read every fork as untouched and re-buy
    them.
    """
    spy = _CollectSpy()

    await prepared(
        "solve it",
        _runtime(
            passes=(
                SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
                SearchPass(_CollectSpy(), budget=8, gating=ConcurrencyGating(4)),
                SearchPass(spy, budget=8, gating=ConcurrencyGating(4)),
            )
        ),
        estimator=DirectBranchTdEstimator(),
    )

    assert spy.seen
    assert sum(counts.m_done for counts in spy.seen[0].values()) > 0


@pytest.mark.asyncio
async def test_passes_share_one_ledger_without_copying_totals() -> None:
    first, second, third = (_CollectSpy() for _ in range(3))
    runtime = _runtime(
        passes=(
            SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
            *(
                SearchPass(a, budget=8, gating=ConcurrencyGating(4))
                for a in (first, second, third)
            ),
        )
    )
    await run_search("solve it", runtime)
    assert first.bought[0] is second.bought[0] is third.bought[0]
    totals = [sum(c.m_done for c in a.seen[0].values()) for a in (first, second, third)]
    assert 0 < totals[0] < totals[1] < totals[2]
    assert sum(c.m_done for c in third.bought[0].values()) > totals[2]


@pytest.mark.asyncio
async def test_search_and_preparation_log_their_own_work(caplog) -> None:
    caplog.set_level(logging.INFO)
    await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())
    passes = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("search finished")
    ]
    assert len(passes) == 3
    assert all(
        "allocator=" in m and "spent=" in m and "turn_budget=" in m for m in passes
    )
    assert any("allocator=fifo" in m for m in passes)
    batches = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("samples prepared")
    ]
    assert len(batches) == 1
    assert "estimator=direct_branch_td" in batches[0]


@pytest.mark.asyncio
async def test_the_result_reports_what_the_search_cost() -> None:
    """Enough to log throughput or alert on a dead endpoint without the tree."""
    result = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    assert result.stats["rollouts"] > 0
    assert result.stats["nodes"] > 0
    assert result.stats["turns"] > 0
    assert result.stats["failures"] == 0.0
    assert result.failures == ()
    assert not isinstance(result.stats, dict)  # a frozen copy, not the live counters


# -- the tree as a deliverable of its own ---------------------------------------------


@pytest.mark.asyncio
async def test_run_rollout_is_run_search_and_then_an_export() -> None:
    """The split has to be exactly that, or a caller doing it by hand gets other data.

    Searching once and exporting twice is the reason ``run_search`` exists -- comparing
    two estimators on one tree, or re-scoring a tree from disk -- and it is only worth
    anything if the by-hand path lands on the same records the one-call path does.
    """
    whole = await prepared("solve it", _runtime(), estimator=DirectBranchTdEstimator())

    # a fresh runtime, so the scripted generator starts where the other one did
    piecewise = _runtime()
    state = await run_search("solve it", piecewise)

    assert (
        prepare_samples(
            state,
            estimator=DirectBranchTdEstimator(),
            reward_config=piecewise.config.reward_config,
        ).actor
        == whole.actor
    )
    assert (
        prepare_samples(
            state,
            estimator=DirectBranchTdEstimator(),
            reward_config=piecewise.config.reward_config,
        ).critic
        == whole.critic
    )


# -- what a search refuses to spend on ------------------------------------------------


class _InexactGenerator(InexactBridge, CyclingGenerator):
    """The hosted chat bridge's claim over a cycling script -- see `InexactBridge`."""


def _untrainable_runtime(**kwargs: Any) -> Runtime:
    generator = _InexactGenerator(["go", "stop"])
    return _runtime(runner=_runner(generator=generator), **kwargs)


@pytest.mark.asyncio
async def test_search_can_run_inexact_models_but_preparation_refuses_them() -> None:
    generator = _InexactGenerator(["go", "stop-good"])
    runtime = _runtime(runner=_runner(generator=generator))
    state = await run_search("solve it", runtime)
    assert generator.calls > 0
    with pytest.raises(ValueError, match="not trainable"):
        prepare_samples(
            state,
            estimator=DirectBranchTdEstimator(),
            reward_config=runtime.config.reward_config,
        )


@pytest.mark.asyncio
async def test_an_eval_only_tree_on_an_untrainable_policy_runs() -> None:
    """``weighters=()`` is how a caller says the tree is for looking at.

    Authored emptiness, not the default: ``weighters=None`` resolves to the shipped
    pair, so an unconfigured search *is* a training search by that rule.
    """
    runtime = _untrainable_runtime()

    state = await run_search("solve it", runtime)

    assert state.stats["turns"] > 0


@pytest.mark.asyncio
async def test_a_single_trajectory_on_an_untrainable_policy_is_untouched() -> None:
    """``run_from`` weights nothing, so it has nothing to refuse."""
    runner = _runner(generator=_InexactGenerator(["go", "stop-good"]))

    rs = await runner.run_from("solve it")

    generated = [turn for turn in rs.turns if turn.tag == "task"]
    assert generated
    assert not any(turn.exact for turn in generated)


@pytest.mark.asyncio
async def test_a_labelled_policy_s_logprobs_reach_the_trainer() -> None:
    """A policy's label is a diagnostic; its logprobs are the behavior column."""
    runtime = _runtime(runner=_runner(version="pi-7"))
    state = await run_search("solve it", runtime)

    result = prepare_samples(
        state,
        estimator=DirectBranchTdEstimator(),
        reward_config=runtime.config.reward_config,
    )

    assert result.actor
    assert all(all(span.logprobs for span in r.spans) for r in result.actor)
    assert {span.policy_id for r in result.actor for span in r.spans} == {"pi-7"}
    samples = to_samples(result, 0)
    assert samples
    assert all(
        sample["logprobs"]
        for sample in samples
        if sample["metadata"]["lane"] == "actor"
    )


@pytest.mark.asyncio
async def test_one_tree_scores_under_two_estimators_without_searching_twice() -> None:
    """The documented use: search with no weighters, then export once per recipe."""
    runtime = _runtime(
        passes=(SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),)
    )
    state = await run_search("solve it", runtime)

    gae = prepare_samples(
        state,
        estimator=DirectBranchTdEstimator(),
        reward_config=runtime.config.reward_config,
    )
    grpo = prepare_samples(
        state,
        estimator=FixedWeight(),
        reward_config=runtime.config.reward_config,
    )

    assert [r.node_id for r in gae.actor] == [r.node_id for r in grpo.actor]
    assert {r.estimator for r in gae.actor} == {"direct_branch_td"}
    assert {r.estimator for r in grpo.actor} == {"test_fixed"}
    assert [r.weight for r in gae.actor] != [r.weight for r in grpo.actor]


@pytest.mark.asyncio
async def test_a_settled_tree_ships_even_though_its_lock_does_not() -> None:
    """Saving a searched tree is the point; the lock is what stopped it.

    An ``asyncio.Lock`` that has ever been waited on holds a ``Future`` and a reference
    to the running loop, neither of which pickles -- so whether a tree could be saved
    depended on whether anything had ever contended it. The state drops the lock on the
    way out and mints a fresh one on the way in, and what comes back exports the same
    records.
    """
    import pickle

    runtime = _runtime(max_concurrency=4)
    state = await run_search("solve it", runtime)

    async def waiter() -> None:
        async with state.lock:
            pass

    async with state.lock:  # one holder and one waiter: a genuinely contended lock
        pending = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        with pytest.raises((TypeError, pickle.PicklingError)):
            pickle.dumps(state.lock)
    await pending

    restored = pickle.loads(pickle.dumps(state))

    assert isinstance(restored.lock, asyncio.Lock)
    assert restored.lock is not state.lock
    assert not restored.lock.locked()
    assert restored.nodes.keys() == state.nodes.keys()
    assert dict(restored.stats) == dict(state.stats)
    assert (
        prepare_samples(
            restored,
            estimator=DirectBranchTdEstimator(),
            reward_config=runtime.config.reward_config,
        ).actor
        == prepare_samples(
            state,
            estimator=DirectBranchTdEstimator(),
            reward_config=runtime.config.reward_config,
        ).actor
    )
    assert (
        prepare_samples(
            restored,
            estimator=DirectBranchTdEstimator(),
            reward_config=runtime.config.reward_config,
        ).critic
        == prepare_samples(
            state,
            estimator=DirectBranchTdEstimator(),
            reward_config=runtime.config.reward_config,
        ).critic
    )


@pytest.mark.asyncio
async def test_one_value_head_is_built_for_the_whole_call(monkeypatch) -> None:
    """Every search pass and the root file through one object.

    Two heads would be two objects writing the same channel -- harmless today, and
    exactly the seam a per-pass ``version`` or a per-pass timeout would drift through
    tomorrow.
    """
    import focus.loop as loop_module

    built: list[Any] = []
    real = loop_module._value_head

    def counting(runtime: Runtime) -> Any:
        head = real(runtime)
        built.append(head)
        return head

    monkeypatch.setattr(loop_module, "_value_head", counting)

    result = await prepared(
        "solve it", _value_runtime(LengthReward()), estimator=DirectBranchTdEstimator()
    )

    assert result.actor
    assert len(built) == 1


# -- the value baseline ---------------------------------------------------------------


class LengthReward(RewardModel):
    """A value head with an opinion: a longer context scores higher.

    Deterministic and *not* constant, so a run's baselines differ node by node -- a
    constant one would shift every advantage alike and hide a baseline read off the
    wrong node.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score(self, context: str) -> RewardResult:
        self.seen.append(context)
        return RewardResult(score=len(context) / 100.0)


class ConstantReward(RewardModel):
    """The same estimate everywhere -- so a baseline's effect is one known number."""

    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, context: str) -> RewardResult:
        return RewardResult(score=self.value)


def _value_runtime(model: RewardModel, **kwargs: Any) -> Runtime:
    """A runtime with a value head: the model alone, no channel to repeat."""
    return _runtime(value_model=model, **kwargs)


@pytest.mark.asyncio
async def test_every_node_the_search_makes_carries_a_baseline() -> None:
    """Including the root, which no expansion produces and every first edge is read
    against."""
    state = await run_search("solve it", _value_runtime(LengthReward()))

    assert len(state.nodes) > 1
    assert all(node.critic is not None for node in state.nodes.values())
    assert (state.nodes[state.root_id].critic or 0.0) > 0.0


@pytest.mark.asyncio
async def test_later_passes_score_what_they_add_too() -> None:
    """One scorer for the whole call: every search pass files into the same channel, so
    the count of scored payloads is every node the search made, plus the root."""
    model = LengthReward()

    result = await prepared(
        "solve it", _value_runtime(model), estimator=DirectBranchTdEstimator()
    )

    assert result.stats["nodes"] > 0
    assert len(model.seen) == result.stats["nodes"] + 1  # + the root
    assert "solve it" in model.seen[0]  # the root, scored before the first expansion
    assert result.stats["score_failures"] == 0.0


@pytest.mark.asyncio
async def test_network_prior_changes_td_through_the_refined_values() -> None:
    base = await prepared(
        "solve it",
        _runtime(value_model=ConstantReward(0.0)),
        estimator=DirectBranchTdEstimator(),
    )
    shifted = await prepared(
        "solve it",
        _runtime(value_model=ConstantReward(0.5)),
        estimator=DirectBranchTdEstimator(),
    )
    assert any(
        before.weight != after.weight
        for before, after in zip(base.actor, shifted.actor, strict=True)
    )
    for record in shifted.actor:
        d = record.diagnostics
        assert d["parent_prior"] == 0.5
        assert record.weight == pytest.approx(
            d["edge_reward"] + d["successor_value"] - d["parent_value"]
        )


@pytest.mark.asyncio
async def test_the_value_head_carries_the_runtime_s_bound_on_its_reward_server() -> (
    None
):
    """The bound belongs to the endpoint, so it is set beside the model that holds it.

    A pass drains what is in flight rather than cancelling it, so an unbounded reward
    call holds the expansion, the pass and the training step behind it. Read through
    the head the loop actually builds: a stalled server costs the baseline (one counted
    scoring failure) and nothing else.
    """
    model = StalledReward()
    runtime = _value_runtime(model, score_timeout=0.01)

    result = await asyncio.wait_for(
        prepared("solve it", runtime, estimator=FixedWeight()), timeout=30
    )

    assert result.actor  # exported, on raw returns
    assert result.stats["score_failures"] > 0.0
    assert result.stats["failures"] == 0.0  # a scoring fault is not an expansion death
    assert model.cancelled > 0  # each stalled request was cancelled, not left running


@pytest.mark.asyncio
async def test_the_run_stamps_and_reads_the_runner_s_own_behavior_channel() -> None:
    """The export reads a turn's logprobs from the channel the *runner* wrote them to.

    One name, and it is the runner's, so a runner filing under ``pi-7`` is exported
    under ``pi-7``. Two authored strings -- a ``behavior_version`` on the runtime beside
    ``Policy.version`` -- would raise nothing when they disagreed: every span
    would come out with its tokens, its mask and an *empty* logprob list, stamped with a
    provenance claiming the channel it was missing.
    """
    runtime = _runtime()
    runtime.runner = _runner(version="pi-7")

    result = await prepared("solve it", runtime, estimator=DirectBranchTdEstimator())

    assert result.actor
    # and the logprobs are there, not an empty column beside a claim that they are
    assert all(all(span.logprobs for span in r.spans) for r in result.actor)


@pytest.mark.asyncio
async def test_a_dead_value_head_costs_the_baseline_not_the_run(caplog) -> None:
    """The rollouts are the expensive half; only the estimate is missing -- and a whole
    run trained against a dead reward server has to be visible in the log."""
    caplog.set_level(logging.WARNING)

    result = await prepared(
        "solve it", _value_runtime(DeadReward()), estimator=FixedWeight()
    )

    assert result.actor  # still exported, on raw returns
    assert result.stats["score_failures"] > 0.0
    # a scoring fault is not an expansion failure
    assert result.stats["failures"] == 0.0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith("root scoring failed") for m in warnings)
    assert any(m.startswith("expansion scoring failed") for m in warnings)


@pytest.mark.asyncio
async def test_a_value_head_that_miscounts_its_batch_costs_the_baseline_not_the_run(
    caplog,
) -> None:
    """A short batch is a scoring fault like any other, not a dead run.

    It cannot be paired with the nodes, so it has to be refused -- but refusing it
    where the scores are *filed* would raise past both handlers and throw away rollouts
    that already finished. So the batch is checked where it is fetched, inside the try
    that keeps the chain and counts the fault.
    """
    caplog.set_level(logging.WARNING)

    result = await prepared(
        "solve it", _value_runtime(ShortBatchReward()), estimator=FixedWeight()
    )

    assert result.actor
    assert result.stats["score_failures"] > 0.0
    assert result.stats["failures"] == 0.0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("root scoring failed" in m for m in warnings)
    assert any("expansion scoring failed" in m for m in warnings)


# -- failure policy: degrade, raise, or propagate -------------------------------------


@pytest.mark.asyncio
async def test_a_stalled_backend_costs_expansions_not_the_whole_prompt(caplog) -> None:
    """The first half of the failure policy, end to end: a per-expansion fault degrades.

    Nothing else bounds a generate, so without ``generate_timeout`` this prompt waits on
    the first stalled request forever -- a pass budget is spent in turns, and a turn
    that never returns spends none. With one, each stall is a dead expansion: counted,
    charged a turn so the budget still depletes, and left behind. The prompt then ends
    with nothing at all, which is legal and is exactly why it is said at WARNING.
    """
    caplog.set_level(logging.WARNING)
    gen = StalledGenerator()
    runtime = _runtime(
        runner=_runner(generator=gen, generate_timeout=0.01),
        passes=(
            SearchPass(FifoAllocator(), budget=4, gating=WidthGating(2)),
            SearchPass(FifoAllocator(), budget=4, gating=ConcurrencyGating(4)),
        ),
    )

    result = await asyncio.wait_for(
        prepared("solve it", runtime, estimator=FixedWeight()), timeout=30
    )

    assert not result.actor
    assert not result.critic
    assert result.stats["failures"] > 0.0
    assert gen.cancelled == gen.calls  # every stalled request was cancelled, not left
    assert all("generate_timeout" in reason for _, reason in result.failures)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    exported = [m for m in warnings if m.startswith("samples prepared")]
    assert len(exported) == 1  # a prompt that yielded nothing says so above INFO
    assert "actor=0" in exported[0]
    assert "failures=" in exported[0]


class SlowGenerator(Policy[Any]):
    """Answers, but not before the caller's deadline can arrive."""

    def __init__(self) -> None:
        self.inflight = 0
        self.cancelled = 0

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        self.inflight += 1
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.inflight -= 1
        return GenerateResult(text="go", tokens=(103, 111), logprobs=[-0.5, -0.5])


@pytest.mark.asyncio
async def test_cancelling_a_rollout_stops_it_rather_than_failing_its_expansions() -> (
    None
):
    """The third: a caller's deadline is not a flaky backend.

    A trainer that gives a prompt a wall clock cancels the task running this, and that
    cancellation has to come out -- swallowed into the expansion-failure path it would
    be logged as one more dead rollout and the frontier re-selected until the pass
    budget ran out, which is the opposite of what a deadline asked for.
    """
    gen = SlowGenerator()
    runtime = _runtime(runner=_runner(generator=gen), max_concurrency=2)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(prepared("solve it", runtime), timeout=0.05)

    assert gen.cancelled > 0
    assert gen.inflight == 0  # nothing is left running behind the cancelled call


class BrokenAllocator(Allocator):
    """An allocator with a bug in the half that reads the tree."""

    name = "broken"

    def build(self, *args: Any, **kwargs: Any) -> AllocationSession[Any]:
        raise ValueError("estimator bug")


@pytest.mark.asyncio
async def test_allocator_errors_propagate_after_the_first_pass() -> None:
    """The second: a programmer error is wrong for every prompt, not for this rollout.

    Exporting the first-pass tree with a WARNING would be the friendlier-looking answer
    and
    the worse one -- the same bug fires on every prompt of every step, so the run would
    train on quietly unscored batches for as long as nobody read the log.
    """
    runtime = _runtime(
        passes=(
            SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
            SearchPass(BrokenAllocator(), budget=1, gating=ConcurrencyGating(4)),
        )
    )

    with pytest.raises(ValueError, match="estimator bug"):
        await prepared("solve it", runtime, estimator=DirectBranchTdEstimator())


class StaggeredGenerator(CyclingGenerator):
    """Like :class:`CyclingGenerator`, but each successive call answers later.

    Enough to make the expansions of one pass finish at *different* times, which is
    what puts work in flight at the moment the budget runs out -- the case the drain
    exists for and an all-at-once fake never reaches.
    """

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        await asyncio.sleep(0.005 * self.calls)
        return await super().agenerate_tokens(prefix_tokens, sampling_params)


@pytest.mark.asyncio
async def test_a_search_pass_keeps_the_rollouts_that_were_still_in_flight() -> None:
    """End to end: a pass that runs out of budget mid-flight keeps what it paid for.

    Three expansions launch, the first one alone spends the pass's turns, and the other
    two are still running -- whole episodes of backend calls, already bought. They are
    drained: their nodes land on the tree and their turns are counted, at the cost of an
    overshoot bounded by one episode per slot. Nothing new is launched after the budget
    is hit, and nothing is left reserved behind them.
    """
    gen = StaggeredGenerator(["go", "stop-good", "go", "stop-bad"])
    runtime = _runtime(
        runner=_runner(generator=gen),
        max_concurrency=3,
        passes=(SearchPass(FifoAllocator(), budget=1, gating=WidthGating(4)),),
    )

    state = await run_search("solve it", runtime)

    assert state.stats["rollouts"] == 3.0  # all three landed; none was thrown away
    assert state.stats["failures"] == 0.0
    assert len(state.nodes[state.root_id].children) == 3
    assert state.stats["turns"] > 1.0  # the drain overshoots the cap...
    assert state.stats["turns"] <= 1 + 3 * runtime.runner.max_steps  # ...but not past
    root = state.nodes[state.root_id]
    assert root.gate.available(root)  # every reservation came back


def test_a_pass_gets_its_own_share_beyond_whatever_the_last_one_overshot() -> None:
    """The overshoot is carried, not compounded, and never charged to the next pass.

    A drained pass can land a few turns past its own cap. Because the next cap is read
    off the tree's *realized* spend rather than off a running plan, that overshoot is
    already in the baseline -- so the next pass still gets its full ``pass_turns``, and
    only the run's grand total is longer.
    """
    state: SchedulerState[Any] = SchedulerState.root(rollout(), gating=WidthGating(1))
    state.stats[STAT_TURNS] = 13.0  # a 12-turn pass that drained one turn over
    view: SchedulerView[Any] = SchedulerView(state)

    budget = _budget(state, 4)

    state.stats[STAT_TURNS] = 16.0
    assert budget.remaining(view) != 0  # still inside its own four turns
    state.stats[STAT_TURNS] = 17.0
    assert budget.remaining(view) == 0  # ...which start where the overshoot ended


def test_a_pass_budget_is_not_eaten_by_an_earlier_pass_s_failures() -> None:
    """Each pass gets its own share of turns, whatever the passes before it lost.

    ``Budget`` charges a turn per dead expansion so a broken backend depletes a budget
    instead of being retried forever -- so a cap computed from realized turns alone came
    out already exceeded on a flaky run, and every later pass bought nothing at exactly
    the moment the tree most needed more evidence.
    """
    state: SchedulerState[Any] = SchedulerState.root(rollout(), gating=WidthGating(1))
    state.stats[STAT_TURNS] = 10.0
    state.stats[STAT_FAILURES] = 5.0
    view: SchedulerView[Any] = SchedulerView(state)

    budget = _budget(state, 4)

    assert budget.remaining(view) != 0  # the pass may still launch
    state.stats[STAT_TURNS] = 14.0
    assert budget.remaining(view) == 0  # ...and stops after its own four turns


# -- pure allocators ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_pure_allocator_grows_the_tree_and_claims_no_export_slot() -> None:
    """It spends real budget, and every exported record still carries a weighter's
    name -- which is the whole point of the `Allocator`/`AdvantageEstimator` split."""
    runtime = _value_runtime(
        ConstantReward(0.0),
        passes=(
            SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
            SearchPass(FifoAllocator(), budget=4, gating=ConcurrencyGating(4)),
            SearchPass(ValueRefinement(), budget=8, gating=ConcurrencyGating(4)),
        ),
    )
    without = await prepared(
        "solve it",
        _value_runtime(
            ConstantReward(0.0),
            passes=(
                SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
                SearchPass(FifoAllocator(), budget=4, gating=ConcurrencyGating(4)),
            ),
        ),
        estimator=DirectBranchTdEstimator(),
    )

    result = await prepared("solve it", runtime, estimator=DirectBranchTdEstimator())

    assert {record.estimator for record in result.actor} == {"direct_branch_td"}
    assert result.stats["rollouts"] > without.stats["rollouts"]


@pytest.mark.asyncio
async def test_refinement_sits_out_a_run_with_no_value_head() -> None:
    runtime = _runtime(
        value_model=None,
        passes=(
            SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
            SearchPass(FifoAllocator(), budget=4, gating=ConcurrencyGating(4)),
            SearchPass(ValueRefinement(), budget=8, gating=ConcurrencyGating(4)),
        ),
    )
    without = await prepared(
        "solve it",
        _runtime(
            passes=(
                SearchPass(FifoAllocator(), budget=12, gating=WidthGating(2)),
                SearchPass(FifoAllocator(), budget=4, gating=ConcurrencyGating(4)),
            )
        ),
        estimator=FixedWeight(),
    )

    result = await prepared("solve it", runtime, estimator=FixedWeight())

    assert result.stats["rollouts"] == without.stats["rollouts"]


def test_passes_are_authored_as_a_tuple() -> None:
    assert RolloutConfig(passes=()).passes == ()
