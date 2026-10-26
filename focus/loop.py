"""Run explicitly configured allocation passes and return the settled search tree."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from focus.allocation import RolloutLedger
from focus.config import RolloutConfig
from focus.execution import Usage
from focus.generation import TokenId
from focus.harness import AnyRollout, RolloutState, Runner
from focus.harness.rescoring import PolicyScorer, reevaluate
from focus.harness.workspace import Workspace
from focus.preparation import AdvantageEstimator, PreparedBatch, prepare_samples
from focus.reward import RewardModel
from focus.reward.config import RewardConfig
from focus.scheduler.core.critic import NodeScorer
from focus.scheduler.core.execution import (
    EdgeProvenance,
    RolloutExpander,
    drawn_from_anchor,
    provenance_of,
)
from focus.scheduler.core.gating import ConcurrencyGating
from focus.scheduler.core.policies import Budget
from focus.scheduler.core.scheduler import Scheduler
from focus.scheduler.core.tree import (
    STAT_FAILURES,
    STAT_SCORE_FAILURES,
    STAT_TURNS,
    SchedulerState,
    SchedulerView,
)

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Connect a runner, optional scorers, and a search configuration.

    Construction preserves each collaborator's configuration and service ownership.
    Pass the same ExecutionServices to Runner and TokenScorer when generation and
    rescoring should share capacity. Each search creates its own tree and workspace.
    """

    runner: Runner[Any, Any]
    #: Supplies anchor-policy log probabilities for off-anchor continuations.
    scorer: PolicyScorer | None = None
    #: Predicts remaining return at checkpoints. Estimates are stored under
    #: config.reward_config.value_version.
    value_model: RewardModel | None = None
    #: Renders a checkpoint for the value model; None uses rollout_text.
    value_serialize: Callable[[AnyRollout], str] | None = None
    #: Maximum seconds for one critic batch; None leaves it unbounded.
    score_timeout: float | None = None
    config: RolloutConfig = field(default_factory=RolloutConfig)


def _expander(runtime: Runtime, pass_id: str) -> RolloutExpander[Any]:
    """The pass's expander: one runner, one law -- its own pass stamp.

    Every pass shares the same runner and proposal law. Only its
    allocation session, budget, gating, and provenance stamp vary.
    """
    return RolloutExpander(
        runtime.runner,
        proposal=runtime.config.proposal,
        provenance=EdgeProvenance(
            # the law the pass will draw from; a draw that lands elsewhere (a mixture
            # coming up `explore`) overwrites it on the edge it actually produced
            proposal_id=runtime.config.proposal.proposal_id,
            continuation_id=runtime.config.continuation_id,
            # the runner's own channel, read off the runner rather than authored
            # beside it: it is the key every generation is actually filed under, and a
            # second name for it is a name that can be wrong
            behavior_version=runtime.runner.policy.version,
            reward_config_id=runtime.config.reward_config.config_id,
            pass_id=pass_id,
        ),
    )


def _value_head(runtime: Runtime) -> NodeScorer[AnyRollout] | None:
    """Build this runtime's value head, on the channel the baseline is read from.

    The two halves of a value head that have to agree -- the key it files estimates
    under and the key
    :func:`~focus.scheduler.core.returns.value_baseline` looks them up by --
    are one name, ``reward_config.value_version``, spelled here. Two authored strings
    would cost nothing at runtime to name differently: every baseline would silently
    read 0.0, exactly as if no critic had been configured at all.
    """
    if runtime.value_model is None:
        return None
    return NodeScorer(
        model=runtime.value_model,
        serialize=runtime.value_serialize,
        version=runtime.config.reward_config.value_version,
        timeout=runtime.score_timeout,
        services=runtime.runner.services,
    )


def _budget(state: SchedulerState[AnyRollout], pass_turns: int) -> Budget[AnyRollout]:
    """A cap for one pass: what the tree has already spent, plus this pass's share.

    "Spent" is read the way :class:`Budget` reads it -- realized turns *plus* failures,
    since a dead expansion is charged a turn so a broken backend depletes a budget
    instead of being retried forever. Counting only the turns here made every earlier
    pass's failures come out of the next pass's share: a pass that lost 20
    expansions to a flaky server handed the next pass an already-exceeded cap.
    Later passes silently bought nothing at exactly the moment the tree most needed
    more evidence.

    The same arithmetic is what makes the drain-at-budget overshoot harmless. A pass
    ends by draining its in-flight expansions, so it can land a few turns past its own
    cap; because the next cap is read off the tree's *realized* spend rather than off a
    running plan, that overshoot is already in the baseline and the next pass still gets
    ``pass_turns`` of its own. Each pass is promised its share beyond whatever has been
    spent -- overshoot included -- not a slice of a fixed total.
    """
    spent = state.stats.get(STAT_TURNS, 0.0) + state.stats.get(STAT_FAILURES, 0.0)
    return Budget(max_turns=int(spent) + pass_turns)


async def _score_root(
    critic: NodeScorer[AnyRollout] | None, state: SchedulerState[AnyRollout]
) -> None:
    """Score the root prior before sampling its first continuation.

    Scoring failures preserve search data and are counted here. Refined TD later
    refuses actor preparation if a required nonterminal prediction is missing.
    """
    if critic is None:
        return
    node = state.nodes[state.root_id]
    try:
        scores = await critic.score([node.payload])
    except Exception as exc:
        state.stats[STAT_SCORE_FAILURES] += 1.0
        node.metadata["score_error"] = repr(exc)
        logger.warning("root scoring failed node=%d error=%r", node.id, exc)
        return
    critic.write([node], scores)


class _CountingScorer:
    """A scorer that also counts the teacher-forced passes it ran.

    The pack count is the number of backend round-trips the re-scoring cost -- the only
    part of the rescoring a training job pays for in latency. It cannot be read from
    the tree afterwards, because :func:`reevaluate` skips packs another node filled.
    So it is counted where it happens.
    """

    def __init__(self, inner: PolicyScorer) -> None:
        self._inner = inner
        self.packs = 0

    async def score(
        self, tokens: tuple[TokenId, ...], trainable: tuple[bool, ...]
    ) -> tuple[float, ...]:
        self.packs += 1
        return await self._inner.score(tokens, trainable)


def _off_anchor(view: SchedulerView[AnyRollout], config: RewardConfig) -> bool:
    """Whether any edge here was drawn from a law other than the anchor.

    Read off provenance, like :func:`importance` itself: the root has no incoming edge
    and no stamp, and an edge's need for a correction is a fact about the law that drew
    it, not about which channels happen to be filled.
    """
    return any(
        provenance_of(node) is not None
        and not drawn_from_anchor(node, config.anchor_version)
        for node in view.nodes()
    )


async def rescore_anchor(runtime: Runtime, view: SchedulerView[AnyRollout]) -> None:
    """Fill the anchor logprob channel when an off-anchor draw made it load-bearing.

    Off-anchor is read off the stamp, not off the policy's class: an edge is off-anchor
    exactly when its ``provenance.proposal_id`` differs from
    ``reward_config.anchor_version``. So the two are one name written twice, and a run
    that renames one of them alone arrives here with every edge looking off-anchor --
    which is what the error below says, rather than blaming the scorer alone.

    A mixture proposal buys coverage and owes an importance ratio on everything it drew
    off-anchor, and that ratio is
    :math:`\\exp(\\log\\pi_{\\mathrm{anchor}} - \\log\\pi_{\\mathrm{behavior}})` over
    tokens the anchor never generated -- so someone has to teacher-force them. Nobody
    below can: the export reads the channel strictly (an absent anchor would otherwise
    read as ``exp(0) = 1``, "the policies agree exactly"), so a missing scorer surfaces
    here as a configuration error rather than as a ``KeyError`` two layers down.

    *Every* turn is scored, folds included, because :func:`importance` measures the
    whole edge (``tags=None``): a fold's summary tokens were generated under the
    proposal like any others, and an edge missing them is not the edge the ratio is for.

    A scorer of the caller's that raises -- notably a ``TokenScorer`` whose
    ``timeout`` fired -- comes out of here unchanged, which is the honest answer rather
    than the friendly one: the export reads the anchor channel strictly, so leaving a
    pack unscored would not degrade a ratio, it would silently make it ``exp(0) = 1``.

    Nodes are walked longest turn log first so the work is not repeated: turns are
    shared objects across forked checkpoints, so a leaf's pass fills every ancestor's
    turns too, and :func:`reevaluate` then skips a region whose turns all carry the
    version. Scoring is per *region*, so a region straddling scored and unscored turns
    is still re-scored as a whole -- ``setdefault`` keeps the columns already there.
    """
    config = runtime.config.reward_config
    if config.anchor_version == runtime.runner.policy.version:
        return  # the anchor *is* the behavior policy; every ratio is exactly 1
    if not _off_anchor(view, config):
        return
    if runtime.scorer is None:
        # The message names both halves of the pairing, because a missing scorer is
        # only one of the two ways to get here and the other is far likelier: whether
        # an edge counts as off-anchor is `provenance.proposal_id == anchor_version`,
        # so renaming one of those two and not the other turns an anchor-only run into
        # a fully off-anchor one. Saying only "set a scorer" sends that reader to fix
        # the wrong thing -- and a scorer would indeed silence it, at the cost of
        # teacher-forcing the whole tree to compute ratios that are all 1.
        raise ValueError(
            "proposal drew off-anchor edges but Runtime.scorer is None; importance "
            + "ratios need the anchor logprobs. An edge is off-anchor when its "
            + "provenance proposal_id differs from reward_config.anchor_version="
            + f"{config.anchor_version!r} -- this run's proposal draws under "
            + f"{runtime.config.proposal.proposal_id!r}, so if those two are meant "
            + "to be the same law, name them the same rather than adding a scorer"
        )
    nodes = sorted(view.nodes(), key=lambda n: len(n.payload.turns), reverse=True)
    pending = {
        id(turn)
        for node in nodes
        for turn in node.payload.turns
        if config.anchor_version not in turn.logprobs
    }
    scorer = _CountingScorer(runtime.scorer)
    for node in nodes:
        await reevaluate(node.payload, version=config.anchor_version, scorer=scorer)
    logger.info(
        "anchor rescored version=%s turns=%d packs=%d",
        config.anchor_version,
        len(pending),
        scorer.packs,
    )


def _spent(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    """What one pass added to each running stat -- its cost, not the tree's total.

    The stats are cumulative over the whole tree, so the raw dict after a pass
    says nothing about that pass. The difference does, and it is the number a budget is
    read against.
    """
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


async def run_search(
    prompt: str | AnyRollout,
    runtime: Runtime,
    *,
    workspace: Workspace | None = None,
) -> SchedulerState[AnyRollout]:
    """Initialize a root, execute the authored passes, and fill anchor logprobs."""
    if not isinstance(prompt, (str, RolloutState)):
        raise TypeError(
            "prompt must be a str or a RolloutState, not " + type(prompt).__name__
        )
    root = (
        prompt
        if isinstance(prompt, RolloutState)
        else await runtime.runner.start(prompt, workspace=workspace)
    )
    before_usage = asdict(runtime.runner.services.usage.get(root.episode_id, Usage()))
    state: SchedulerState[AnyRollout] = SchedulerState.root(
        root, gating=ConcurrencyGating(1)
    )
    critic = _value_head(runtime)
    await _score_root(critic, state)
    ledger = RolloutLedger(runtime.config.reward_config)
    view = SchedulerView(state)
    ledger.bind(view)
    for index, search_pass in enumerate(runtime.config.passes):
        pass_id = f"pass-{index}"
        allocator = search_pass.allocator
        if search_pass.budget == 0:
            logger.info(
                "search skipped pass=%s allocator=%s budget=0", pass_id, allocator.name
            )
            continue
        async with state.lock:
            state.regate(search_pass.gating)
        allocation = allocator.build(view, ledger)
        if allocation is None:
            logger.info("search skipped pass=%s allocator=%s", pass_id, allocator.name)
            continue
        before = dict(state.stats)
        await Scheduler(
            expander=_expander(runtime, pass_id),
            allocation=allocation,
            termination=_budget(state, search_pass.budget),
            scorer=critic,
            max_concurrency=runtime.config.max_concurrency,
        ).run_on(state)
        logger.info(
            "search finished pass=%s allocator=%s turn_budget=%d spent=%s",
            pass_id,
            allocator.name,
            search_pass.budget,
            _spent(before, state.stats),
        )
    await rescore_anchor(runtime, view)
    _resource_stats(state, runtime, root.episode_id, before_usage)
    return state


async def collect_samples(
    prompt: str | AnyRollout,
    runtime: Runtime,
    *,
    estimator: AdvantageEstimator | None,
    workspace: Workspace | None = None,
) -> PreparedBatch:
    """Collect a tree and prepare actor and critic samples for a training batch.

    Use the runtime's reward configuration and policy version for both collection
    and preparation. ``estimator`` remains explicit: choose a training recipe, or
    pass ``None`` to prepare only critic targets. Inspect ``batch.stats`` and
    ``batch.failures`` before training; successful samples from a partial collection
    are retained. Export with ``to_samples`` or consume the typed records directly.

    The caller owns execution services and model clients. Keep those services open
    around a batch or several concurrent collections, then close them when finished.
    This function returns the prepared batch without retaining the rollout tree.
    Use ``run_search`` followed by ``prepare_samples`` when you need the tree for
    inspection or want to compare several estimators on the same evidence.
    """
    tree = await run_search(prompt, runtime, workspace=workspace)
    return prepare_samples(
        tree,
        estimator=estimator,
        reward_config=runtime.config.reward_config,
        behavior_version=runtime.runner.policy.version,
    )


def _resource_stats(
    state: SchedulerState[AnyRollout],
    runtime: Runtime,
    episode: str,
    before: dict[str, Any],
) -> None:
    current = asdict(runtime.runner.services.usage.get(episode, Usage()))
    for name, value in current.items():
        state.stats["resource_" + name] = float(value - before[name])


__all__ = ["Runtime", "collect_samples", "rescore_anchor", "run_search"]
