from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import pytest

from focus import (
    ConcurrencyGating,
)
from focus.allocation import (
    BetaBernoulli,
    CalibrationModel,
    Gaussian,
    NodeWorth,
    Observations,
    ResidualWorth,
    RolloutLedger,
    ScoredSession,
    ValueRefinement,
    retraining_gain,
)
from focus.harness import RolloutState
from focus.reward.config import RewardConfig
from focus.scheduler import (
    Budget,
    Expander,
    FifoSession,
    Scheduler,
    SchedulerState,
    SchedulerView,
)
from focus.statistics import RunningStats
from tests.trees import region, root_state, stamp, stamped_child
from tests.trees import search_payload as rollout


def test_a_session_updates_the_search_owned_counts() -> None:
    """Existing purchase counts are updated in place by the session."""
    state, view = _root()
    run = {state.root_id: Observations(m_done=3)}

    bundle = ScoredSession(
        _UnitWorth(RewardConfig()), RolloutLedger(RewardConfig(), run)
    )

    assert bundle is not None
    assert run[state.root_id].m_done == 3  # what it starts from
    reservation = bundle.select(view)
    assert reservation is not None
    child = stamped_child(state, 0, rollout(region(reward=1.0, done=True)))
    bundle.complete(reservation, state, [child])
    assert run[state.root_id].m_done == 4  # and what it just bought


def _root():
    """A bare root with room to branch -- the tree an allocator scores over."""
    state = root_state(rollout(), width=4)
    return state, SchedulerView(state)


class _UnitWorth(NodeWorth):
    """Any forkable node is worth buying -- isolates reservation from a score recipe."""

    def worth(self, view, node, counts: Observations) -> float:
        del view, node, counts
        return 1.0


class ContinuationExpander(Expander[RolloutState[None]]):
    def __init__(self) -> None:
        self.calls = 0

    async def expand(
        self, payload: RolloutState[None]
    ) -> tuple[list[RolloutState[None]], Mapping[str, object]]:
        self.calls += 1
        return [rollout(*payload.turns, region(reward=1.0, done=True))], {}


class DeadExpander(Expander[RolloutState[None]]):
    async def expand(self, payload):
        raise RuntimeError("continuation died")


@dataclass
class _Collection:
    observations: dict[int, Observations]
    session: ScoredSession


def _collection() -> _Collection:
    ledger = RolloutLedger(RewardConfig())
    return _Collection(
        ledger.observations, ScoredSession(_UnitWorth(ledger.reward_config), ledger)
    )


def _collected() -> tuple[SchedulerState[Any], _Collection]:
    state, _ = _root()
    for reward in (1.0, 0.0):
        fold = stamped_child(state, 0, rollout(region(reward=reward)))
        stamped_child(
            state,
            fold.id,
            rollout(*fold.payload.turns, region(reward=reward, done=True)),
        )
    collection = _collection()
    state.regate(ConcurrencyGating(4))
    return state, collection


@pytest.mark.asyncio
async def test_a_scored_session_counts_scheduler_purchases() -> None:
    state, collection = _collected()
    expander = ContinuationExpander()

    await Scheduler(
        expander=expander,
        allocation=collection.session,
        termination=Budget(max_rollouts=3),
        max_concurrency=1,
    ).run_on(state)

    assert expander.calls == 3
    assert sum(s.m_done for s in collection.observations.values()) == 3
    assert all(s.m_pending == 0 for s in collection.observations.values())


@pytest.mark.asyncio
async def test_a_dead_scored_rollout_releases_its_reservation() -> None:
    state, collection = _collected()

    await Scheduler(
        expander=DeadExpander(),
        allocation=collection.session,
        termination=Budget(max_rollouts=2),
        max_concurrency=1,
    ).run_on(state)

    assert all(s.m_pending == 0 for s in collection.observations.values())
    assert all(s.m_done == 0 for s in collection.observations.values())


def test_an_earlier_pass_observation_counts_for_the_advantage_statistic_too() -> None:
    """A new session reads the evidence already shared by earlier passes."""
    run = {}
    run[0] = _measured(2.0, 0.5, 4)

    bundle = ScoredSession(
        _UnitWorth(RewardConfig()), RolloutLedger(RewardConfig(), run)
    )

    assert bundle is not None
    seeded = run[0]
    assert seeded.m_done == 4
    assert seeded.returns.n == 4
    assert seeded.returns.mean == pytest.approx(2.0)


def test_allocation_uses_the_search_owned_ledger() -> None:
    _, view = _root()
    observations = {0: _measured(2.0, 0.5, 4)}
    allocation = ScoredSession(
        _UnitWorth(RewardConfig()), RolloutLedger(RewardConfig(), observations)
    )
    chosen = allocation.select(view)
    assert chosen is not None
    assert observations[chosen.node.id].m_pending == 1
    allocation.fail(chosen, RuntimeError("failed"))
    assert observations[chosen.node.id].m_pending == 0


def test_pending_is_charged_at_selection() -> None:
    state, collection = _collected()

    chosen = collection.session.select(SchedulerView(state))

    assert chosen is not None
    assert chosen.node.id == 0
    assert collection.observations[0].m_pending == 1


def test_running_stats_tracks_the_mean_and_the_sample_variance() -> None:
    stats = RunningStats()

    for value in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
        stats.add(value)

    assert stats.n == 8
    assert stats.mean == pytest.approx(5.0)
    assert stats.variance == pytest.approx(32.0 / 7)  # sample, not population


def test_one_observation_has_no_spread_to_report() -> None:
    stats = RunningStats()
    stats.add(3.0)

    assert stats.variance == 0.0


def test_unfinished_purchases_settle_without_return_evidence() -> None:
    """A chain that stopped early is a completed rollout but not a completed
    trajectory, and its suffix return holds shaping alone -- so it settles the
    reservation without dragging the node's mean toward zero."""
    state, _ = _root()
    root = state.nodes[0]
    collection = _collection()
    state.regate(ConcurrencyGating(4))
    first = collection.session.select(SchedulerView(state))
    second = collection.session.select(SchedulerView(state))
    assert first is not None
    assert second is not None
    unfinished = stamped_child(state, 0, rollout(region(reward=9.0)))
    finished = stamped_child(state, 0, rollout(region(reward=2.0, done=True)))

    collection.session.complete(first, state, [unfinished])
    collection.session.complete(second, state, [finished])

    counts = collection.observations[root.id]
    assert counts.m_done == 2
    assert counts.m_pending == 0
    assert counts.returns.n == 1  # honestly fewer than m_done
    assert counts.returns.mean == pytest.approx(2.0)


def test_the_shared_selection_refuses_a_node_that_cannot_be_forked() -> None:
    state, view = _root()
    done = stamped_child(state, 0, rollout(region(reward=1.0, done=True)))
    collection = _collection()

    assert collection.session.score(view, done) == -math.inf


def _stats(*values: float) -> RunningStats:
    return RunningStats.of(values)


def test_the_beta_posterior_is_the_hand_computed_conjugate_update() -> None:
    """κ=2 pseudo-samples at V=0.25, then three wins and a loss: α=3.5, β=2.5."""
    e2, outcome_var = BetaBernoulli().posterior(0.25, _stats(1.0, 0.0, 1.0, 1.0), 2.0)

    # mu = 3.5/6; E[e2] = (V - mu)^2 + Var(p), with Var(p) = mu(1-mu)/(total+1)
    assert e2 == pytest.approx(1.0 / 9.0 + (3.5 / 6) * (2.5 / 6) / 7)
    # ...and the next draw is Bernoulli with P(X=1) = mu, so Var(X) = mu(1-mu)
    assert outcome_var == pytest.approx((3.5 / 6) * (2.5 / 6))


def test_the_beta_outcome_variance_is_the_predictive_spread_of_the_next_draw() -> None:
    """`σ² = μ(1−μ)`: sampling noise `E[p(1−p)]` *plus* what is still unknown about `p`.

    The epistemic half is counted here as well as in `e²`, and that surcharge is the
    whole reason an unmeasured node scores zero rather than scoring its own uncertainty.
    Priced at `E[p(1−p)]` instead the two would be one coherent ledger and the gain
    could never reach zero — see `test_the_conditional_variance_would_price_away_both`.
    """
    model = BetaBernoulli()

    for value, ledger, kappa in (
        (0.5, _stats(), 1.0),
        (0.25, _stats(1.0, 0.0, 1.0, 1.0), 2.0),
        (0.9, _stats(0.0, 0.0), 0.5),
    ):
        e2, outcome_var = model.posterior(value, ledger, kappa)
        total = kappa + ledger.n
        mu = (kappa * value + ledger.n * ledger.mean) / total
        posterior_var = mu * (1.0 - mu) / (total + 1.0)

        assert outcome_var == pytest.approx(mu * (1.0 - mu))
        assert e2 == pytest.approx((value - mu) ** 2 + posterior_var)


def test_the_beta_model_reads_no_spread_at_all() -> None:
    """Two ledgers with the same count and mean and wildly different variances are the
    same posterior. σ² = p(1−p) is a function of the mean, which is what makes the
    two-lucky-wins degeneracy structurally impossible rather than floored away."""
    model = BetaBernoulli()

    assert model.posterior(0.0, _stats(0.4, 0.6), 1.0) == model.posterior(
        0.0, _stats(0.0, 1.0), 1.0
    )


def test_the_beta_model_clamps_a_value_its_premise_does_not_cover() -> None:
    """Outside [0, 1] the arithmetic still runs -- it has just stopped being a model of
    the returns, which is what `Gaussian` is for."""
    model = BetaBernoulli()

    assert model.posterior(3.0, _stats(1.0), 1.0) == model.posterior(
        1.0, _stats(1.0), 1.0
    )


def test_the_gaussian_model_charges_its_prior_below_two_samples() -> None:
    """There is nothing to estimate a spread from, so `v_prior` stands in -- and it is a
    real input, not a formality: doubling it doubles the noise term."""
    ledger = _stats(1.0)

    small = Gaussian(v_prior=0.25).posterior(0.0, ledger, 1.0)
    large = Gaussian(v_prior=1.0).posterior(0.0, ledger, 1.0)

    # the error term takes the spread itself; the predictive one adds the spread of the
    # mean the next draw is taken around, `sigma2/(kappa + n)` -- here a factor of 3/2
    assert small[0] == pytest.approx(0.25 + 0.25 / 2)
    assert large[0] == pytest.approx(0.25 + 1.0 / 2)
    assert small[1] == pytest.approx(0.25 * 1.5)
    assert large[1] == pytest.approx(1.0 * 1.5)


def test_the_gaussian_model_floors_the_measured_spread() -> None:
    """A node that happened to return one value twice is not infinitely precise."""
    e2, outcome_var = Gaussian(var_floor=0.01).posterior(0.0, _stats(0.5, 0.5), 1.0)

    assert e2 == pytest.approx((1.0 / 3.0) ** 2 + 0.01 / 3.0)
    assert outcome_var == pytest.approx(0.01 * (1.0 + 1.0 / 3.0))


def test_the_gaussian_model_refuses_priors_it_cannot_divide_by() -> None:
    with pytest.raises(ValueError, match="var_floor"):
        Gaussian(var_floor=0.0)
    with pytest.raises(ValueError, match="v_prior"):
        Gaussian(v_prior=-1.0)


def test_the_gain_is_affine_in_the_squared_error() -> None:
    """The property the whole rewrite rests on: the myopically best node can be found
    from `E[e²]` alone, so no test statistic and no significance floor is needed."""
    base = retraining_gain(0.0, 0.7, 4, 1.0)
    single = retraining_gain(0.3, 0.7, 4, 1.0)

    assert retraining_gain(0.6, 0.7, 4, 1.0) == pytest.approx(2 * single - base)


def test_the_gain_falls_away_as_a_node_accumulates_evidence() -> None:
    """A marginal value, not a detector: the twentieth sample at a node moves the
    blended estimate far less than the first, and the score says so on its own rather
    than through a damping factor multiplied in beside it."""
    scores = [retraining_gain(0.5, 0.2, n, 1.0) for n in range(9)]

    assert all(later < earlier for earlier, later in pairwise(scores))


def test_a_critic_known_to_be_right_is_worth_negative_below_kappa() -> None:
    """The veto, and it is a real branch of the formula rather than a threshold: below
    `n = κ` retraining a correct critic toward a mean built from fewer samples than the
    estimate it replaces *raises* the MSE. `eligible` refuses the whole range."""
    assert retraining_gain(0.0, 1.0, 0, 1.0) < 0.0
    assert retraining_gain(0.0, 1.0, 1, 1.0) == pytest.approx(0.0)
    # ...and above κ the same term turns into the noise-dilution tail: averaging one
    # more draw into the blend is a small real gain, falling off like 1/n².
    tail = [retraining_gain(0.0, 1.0, n, 1.0) for n in (2, 4, 8)]
    assert all(value > 0.0 for value in tail)
    assert all(later < earlier for earlier, later in pairwise(tail))


MODELS = [BetaBernoulli(), Gaussian()]


MODEL_IDS = [type(model).__name__ for model in MODELS]


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_an_unmeasured_node_is_worth_exactly_nothing(model: CalibrationModel) -> None:
    """The anti-flood theorem, and it is an identity rather than a tendency: at the
    default `κ = 1` a node with an empty ledger scores exactly `0.0` whatever its critic
    says, under either model. `eligible` refuses it, so an allocator facing nothing but
    empty ledgers cannot make a first purchase at all — which is what makes birth-sample
    seeding load-bearing rather than merely helpful."""
    for value in (0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 1.0):
        e2, outcome_var = model.posterior(value, _stats(), 1.0)

        assert retraining_gain(e2, outcome_var, 0, 1.0) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_a_critic_worth_less_than_one_sample_vetoes_its_own_unmeasured_node(
    model: CalibrationModel,
) -> None:
    """Below `κ = 1` the same node is worth strictly *less* than nothing: replacing an
    estimate worth half a sample with a mean built from zero of them raises the MSE."""
    e2, outcome_var = model.posterior(0.5, _stats(), 0.5)

    assert retraining_gain(e2, outcome_var, 0, 0.5) < 0.0


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_both_models_put_the_sign_at_the_same_place(model: CalibrationModel) -> None:
    """For a node the model reads as calibrated the sign is `κ² − κ + n² + n` exactly,
    and the two models agree on it despite sharing no arithmetic. That expression is
    negative only at `n = 0, κ < 1`, so the veto region is precisely *an unmeasured node
    whose critic is worth less than one sample*, and `κ = 1, n = 0` is the boundary."""
    for kappa in (0.25, 0.5, 0.75, 1.0, 1.5, 3.0):
        for n in range(5):
            calibrated = _stats(*([0.5] * n))  # the ledger agrees with the critic
            e2, outcome_var = model.posterior(0.5, calibrated, kappa)

            gain = retraining_gain(e2, outcome_var, n, kappa)
            predicted = kappa**2 - kappa + n**2 + n
            assert math.copysign(1.0, gain) == math.copysign(1.0, predicted)
            assert (gain == 0.0) == (predicted == 0.0)


def test_the_conditional_variance_would_price_away_both() -> None:
    """Why the surcharge is a *choice*. Priced with the sampling noise conditioned on
    the truth — `E[p(1−p)]`, smaller than the predictive `μ(1−μ)` by exactly `Var(p)` —
    `e2` and `σ²` become one coherent posterior ledger, which ties `E[e²] ≥ σ²/(κ+n)`
    and makes the gain positive everywhere: no zero at `κ = 1`, no veto, only ranking.
    """
    for kappa in (0.5, 1.0, 2.0):
        for n in range(4):
            e2, _ = BetaBernoulli().posterior(0.5, _stats(*([0.5] * n)), kappa)
            total = kappa + n
            conditional = 0.25 * total / (total + 1.0)  # E[p(1-p)] at mu = 0.5

            assert retraining_gain(e2, conditional, n, kappa) > 0.0


def _measured(mean: float, spread: float, n: int) -> Observations:
    """Counts carrying `n` purchased returns: the given mean, a symmetric spread."""
    counts = Observations(m_done=n)
    for index in range(n):
        counts.returns.add(mean + (spread if index % 2 else -spread))
    return counts


def _ledger(*returns: float, pending: int = 0) -> Observations:
    """Counts carrying exactly these evidence samples, bought or not."""
    counts = Observations(m_pending=pending)
    for value in returns:
        counts.returns.add(value)
    return counts


def _child(state: SchedulerState[Any], value: float | None) -> Any:
    """A forkable child whose critic says `value` -- or says nothing at all."""
    node = stamped_child(state, 0, rollout(region(reward=0.0)))
    if value is not None:
        node.critic = value
    return node


def _refinement(model: CalibrationModel | None = None) -> ResidualWorth:
    return ResidualWorth(RewardConfig(), model or BetaBernoulli())


def test_a_node_with_nothing_left_to_learn_scores_zero_and_is_refused() -> None:
    """Three ways the score bottoms out, and none is worth a rollout: a node nothing has
    been sampled at yet (whatever its critic says — the anti-flood identity at `κ = 1`),
    a node no critic ever scored, and one whose critic and every sample agree on the
    same endpoint."""
    state, view = _root()
    unmeasured = _child(state, 0.5)
    silent = _child(state, None)
    settled = _child(state, 1.0)
    worth = _refinement()
    selection = ScoredSession(worth, RolloutLedger(worth.reward_config))

    assert worth.worth(view, unmeasured, _ledger()) == pytest.approx(0.0, abs=1e-15)
    assert worth.worth(view, silent, _ledger()) == 0.0
    assert worth.worth(view, settled, _ledger(1.0, 1.0, 1.0)) == 0.0
    assert not selection.eligible(0.0)


def test_evidence_against_the_critic_outbids_evidence_for_it() -> None:
    """The ranking that replaces the old significance threshold, at equal evidence: one
    birth sample apiece, and the score grows monotonically with how far the critic sits
    from what came back. The agreeing node is refused outright."""
    state, view = _root()
    worth = _refinement()

    scores = [
        worth.worth(view, _child(state, value), _ledger(1.0))
        for value in (1.0, 0.7, 0.3, 0.0)
    ]

    assert scores[0] == 0.0
    assert all(earlier < later for earlier, later in pairwise(scores))


def test_two_lucky_wins_no_longer_dominate_the_frontier() -> None:
    """The old pathology, priced. A node at V=0.5 that returned two wins used to report
    σ̂=0, hit the variance floor and score as an enormous certainty; now its residual is
    read against a Beta posterior that never saw a spread, and it ranks below a node
    with the same evidence and a bigger residual."""
    state, view = _root()
    worth = _refinement()

    lucky = worth.worth(view, _child(state, 0.5), _ledger(1.0, 1.0))
    adrift = worth.worth(view, _child(state, 0.0), _ledger(1.0, 1.0))

    assert math.isfinite(lucky)
    assert lucky < adrift


def test_purchases_in_flight_count_as_evidence_that_has_not_landed() -> None:
    """`n_eff` is the ledger plus `m_pending`, which is both halves of one fact: the
    derivation's `n`, and `ScoredSession`'s promise that a node cannot win the
    whole budget before its first rollout returns."""
    state, view = _root()
    adrift = _child(state, 0.0)
    worth = _refinement()

    idle = worth.worth(view, adrift, _ledger(1.0))
    charged = worth.worth(view, adrift, _ledger(1.0, pending=2))

    assert charged < idle


def test_carried_returns_let_refinement_discriminate_on_its_first_selection() -> None:
    """Prior direct evidence distinguishes calibrated and drifted nodes immediately.

    The new pass uses the existing ledger without buying replacement observations.
    """
    state, view = _root()
    calibrated = _child(state, 1.0)
    adrift = _child(state, 0.0)
    state.regate(ConcurrencyGating(4))
    run = {calibrated.id: _measured(1.0, 0.1, 6), adrift.id: _measured(1.0, 0.1, 6)}

    bundle = ValueRefinement().build(view, RolloutLedger(RewardConfig(), run))

    assert bundle is not None
    assert isinstance(bundle, ScoredSession)
    assert bundle.score(view, calibrated) == 0.0
    assert bundle.score(view, adrift) > 0.0


def test_the_gaussian_model_ranks_a_drifted_node_over_a_calibrated_one() -> None:
    """The debiasing claim, on returns the Beta premise does not cover: a node whose
    mean sits where its critic says scores below one the same evidence contradicts,
    even though the contradicted node's measured spread is identical."""
    state, view = _root()
    coin = [0.0, 1.0] * 4
    worth = _refinement(Gaussian())

    calibrated = worth.worth(view, _child(state, 0.5), _ledger(*coin))
    drifted = worth.worth(view, _child(state, 1.5), _ledger(*coin))

    assert calibrated < drifted


def test_refinement_refuses_a_pass_when_nothing_filled_the_value_channel() -> None:
    state, view = _root()
    stamped_child(state, 0, rollout(region(reward=5.0, done=True)))

    assert ValueRefinement().build(view, RolloutLedger(RewardConfig(), {})) is None

    state.nodes[0].critic = 0.25
    assert ValueRefinement().build(view, RolloutLedger(RewardConfig(), {}))


def test_refinement_takes_its_model_by_config_or_already_built() -> None:
    """A caller passes a built model or nothing."""
    built = Gaussian(v_prior=0.5)

    assert isinstance(ValueRefinement().model, BetaBernoulli)
    assert ValueRefinement(model=built).model is built


def test_refinement_refuses_a_prior_strength_it_cannot_divide_by() -> None:
    with pytest.raises(ValueError, match="value_prior_strength"):
        RewardConfig(value_prior_strength=0.0)


def _chain(state: SchedulerState[Any], *rewards: float) -> list[Any]:
    """A line of nodes under the root, the last one finished."""
    chain: list[Any] = []
    parent = state.nodes[0]
    for index, reward in enumerate(rewards):
        node = stamped_child(
            state,
            parent.id,
            rollout(
                *parent.payload.turns,
                region(reward=reward, done=index == len(rewards) - 1),
            ),
        )
        chain.append(node)
        parent = node
    return chain


def test_a_created_node_records_the_return_of_the_rollout_that_created_it() -> None:
    """One free unbiased draw per node: nothing was selected at or below it conditional
    on its prefix, so the leaf's suffix return is exactly what `returns` holds."""
    state, _ = _root()
    chain = _chain(state, 0.0, 1.0)
    observations: dict[int, Observations] = {}

    RolloutLedger(RewardConfig(), observations).record(state.nodes[0], chain)

    assert observations[chain[0].id].returns.n == 1
    assert observations[chain[0].id].returns.mean == pytest.approx(1.0)
    # ...and no purchase was invented to pay for it
    assert all(counts.m_done == 0 for counts in observations.values())


def test_an_unfinished_chain_leaves_no_birth_sample() -> None:
    """Its suffix return holds shaping alone -- the outcome reward lands on the
    episode-ending transition -- so folding it in would drag every node toward zero in
    proportion to how often the budget ran out mid-episode."""
    state, _ = _root()
    unfinished = stamped_child(state, 0, rollout(region(reward=9.0)))
    observations: dict[int, Observations] = {}

    RolloutLedger(RewardConfig(), observations).record(state.nodes[0], [unfinished])

    assert observations == {}


def test_a_scored_pass_records_births_beside_the_return_it_bought() -> None:
    """Settlement records both the purchased return and new nodes' birth evidence."""
    state, _ = _root()
    root = state.nodes[0]
    observations: dict[int, Observations] = {}
    session = FifoSession(RolloutLedger(RewardConfig(), observations))
    reservation = session.select(SchedulerView(state))
    assert reservation is not None
    chain = _chain(state, 0.0, 1.0)
    for node in chain:
        stamp(node)
    session.complete(reservation, state, chain)

    assert observations[root.id].m_done == 1
    assert observations[root.id].returns.n == 1
    assert observations[chain[0].id].returns.n == 1
    assert observations[chain[0].id].m_done == 0


def test_birth_samples_alone_let_the_first_pass_discriminate() -> None:
    """The reason the search carries a ledger at all. With nothing but the samples the
    first pass's own rollouts left behind -- no scored purchase anywhere -- the
    first refinement pass already spends where the critic disagrees with what came
    back."""
    state, view = _root()
    observations: dict[int, Observations] = {}
    ledger = RolloutLedger(RewardConfig(), observations)
    forks = []
    for value in (1.0, 0.0):
        chain = _chain(state, 0.0, 1.0)  # a first-pass rollout: a fork, then its finish
        ledger.record(state.nodes[0], chain)
        chain[0].critic = value
        forks.append(chain[0])
    state.regate(ConcurrencyGating(4))
    calibrated, adrift = forks

    bundle = ValueRefinement().build(view, RolloutLedger(RewardConfig(), observations))

    assert bundle is not None
    assert all(counts.m_done == 0 for counts in observations.values())
    assert isinstance(bundle, ScoredSession)
    assert bundle.score(view, calibrated) == 0.0
    assert bundle.score(view, adrift) > 0.0
