"""A critic scheduler: annotate tree nodes with version-tagged critic values.

Runs *concurrently* with the generation
:class:`~focus.scheduler.core.scheduler.Scheduler` over the same
:class:`SchedulerState`. Where generation grows the tree, the critic reads it: it
selects nodes, scores each node's serialized payload with a
:class:`~focus.reward.RewardModel`, and writes the scalar back as
``node.critic[version]`` -- keyed by the reward-model version so several
passes/models coexist.

Both schedulers share one ``state`` and mutate it under ``state.lock`` (the lock the
tree exposes for exactly this). The critic *tracks* generation via an ``until``
event: it keeps scoring new nodes as they appear, then drains and stops once
generation is done.

    state = SchedulerState.root(root, gating=WidthGating(4))
    done = asyncio.Event()
    async def _gen():
        try: return await generation.run_on(state)
        finally: done.set()
    gen_result, _ = await asyncio.gather(_gen(), critic.run_on(state, until=done))
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from focus.execution import AdmissionPool
from focus.harness.transcript import format_transcript
from focus.registry import register, registrable
from focus.reward import RewardModel, RewardResult
from focus.scheduler.core.tree import (
    STAT_SCORE_FAILURES,
    Node,
    Payload,
    SchedulerState,
    SchedulerView,
)

if TYPE_CHECKING:
    from focus.harness import AnyRollout

logger = logging.getLogger(__name__)

P = TypeVar("P", bound=Payload)


# -- scoring a payload -----------------------------------------------------------------


@dataclass(frozen=True)
class ValuePrediction:
    score: float
    context: str


VALUE_CONTEXTS_KEY = "value_contexts"


class NodeScorer[P: Payload]:
    """A reward model, how to serialize a payload for it, and the version to file under.

    The three things that always travel together when anything scores a node, in one
    object so both callers share them: the :class:`CriticScheduler` below (scoring an
    existing tree, after the fact) and the
    :class:`~focus.scheduler.core.scheduler.Scheduler` itself (scoring what an
    expansion just produced, before the node reaches the frontier).

    Scores land in ``node.critic[version]`` and *only* there -- never in
    ``node.mean_return``, which is backup bookkeeping over the realized return. Keeping
    the learned estimate and the observed return in separate fields is what makes one
    auditable against the other; a scorer that folded into ``mean_return`` would launder
    them together.

    ``serialize`` is optional and defaults to :func:`rollout_text`: rollout payloads are
    the only ones in production, so a value head is ``NodeScorer(model=rm,
    version=cfg.value_version)`` and nothing more. Pass one for any other payload type.

    ``timeout`` bounds the reward-model call in seconds (``None``, the default, waits as
    long as the backend does). It is the scoring half of what
    ``Runner.generate_timeout`` is for generation, and for the same reason: nothing else
    bounds one HTTP call, a scheduler stops on a *budget*, and a reward server that
    never answers spends none of it -- so one swallowed request stalls the expansion,
    the prompt, and the training step behind it. When it fires the batch is a scoring
    failure on the ordinary terms (counted in ``stats[score_failures]``, the rollout
    kept and only the estimate missing), because a timeout reaches both callers through
    the same ``except`` a 503 does.
    """

    def __init__(
        self,
        *,
        model: RewardModel,
        serialize: Callable[[P], str] | None = None,
        version: str,
        timeout: float | None = None,
        pool: AdmissionPool | None = None,
    ) -> None:
        self._pool = pool or AdmissionPool()
        self._model = model
        self._timeout = timeout
        # rollout payloads are the only ones in production, so their renderer is the
        # default rather than something every call site has to repeat
        self._serialize: Callable[[P], str] = serialize or rollout_text  # type: ignore[assignment]
        self.version = version

    async def score(self, payloads: Sequence[P]) -> list[ValuePrediction]:
        """Score ``payloads`` in one batched call (the order is preserved).

        A backend that answers with the wrong number of results raises *here*, which is
        the only place both callers guard: the :class:`Scheduler` runs this inside the
        try that turns a scoring fault into a kept-but-unscored chain, and
        ``_score_root`` inside the one that counts it. Left to be caught by
        :meth:`write`'s pairing instead, the same fault would escape past both and sink
        a run whose rollouts -- the expensive half -- were already finished and valid.
        """
        if not payloads:
            return []
        contexts = [self._serialize(p) for p in payloads]
        results = await self._score_batch(contexts)
        if len(results) != len(payloads):
            raise ValueError(
                f"reward model returned {len(results)} results for "
                + f"{len(payloads)} payloads; the scores cannot be paired with the "
                + "nodes"
            )
        return [
            ValuePrediction(r.score, context)
            for r, context in zip(results, contexts, strict=True)
        ]

    async def _score_batch(self, contexts: list[str]) -> list[RewardResult]:
        """The reward-model round trip, bounded by ``timeout`` when one is configured.

        The unbounded path is the call itself rather than ``wait_for(..., None)``: every
        expansion of every rollout scores through here, and a run that asked for no
        timeout should not pay a task wrapper per batch to be told so. Mirrors
        ``Runner._generate``, deliberately -- the two bounds a scheduler leans on to
        make a stalled backend cost one expansion instead of the whole job should not
        differ in shape.

        The re-raise carries what the bare :class:`TimeoutError` does not and what a
        reader of a node's ``score_error`` needs: how long was waited, and how big the
        batch was. ``wait_for`` has already cancelled the request by then.
        """
        if self._timeout is None:
            return await self._pool.run(lambda: self._model.ascore_batch(contexts))
        try:
            return await asyncio.wait_for(
                self._pool.run(lambda: self._model.ascore_batch(contexts)),
                self._timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"scoring exceeded timeout={self._timeout}s "
                + f"(batch={len(contexts)}); the batch is abandoned and counts as one "
                + "scoring failure -- the rollouts are kept, only the estimate is "
                + "missing"
            ) from exc

    def write(
        self, nodes: Sequence[Node[P]], scores: Sequence[ValuePrediction]
    ) -> None:
        """File ``scores`` on ``nodes`` under this scorer's version, pairwise.

        The one place a critic score reaches a node, so both callers spell the version
        key the same way -- and so ``node.mean_return``, the backup's realized-return
        channel, is visibly not written here.

        ``strict``: a short list would otherwise file every score one node early and
        leave the tail silently unscored. It should be unreachable -- :meth:`score`
        checks the batch where a failure is still catchable -- so reaching it means the
        caller paired a list this scorer did not produce.
        """
        for node, score in zip(nodes, scores, strict=True):
            node.critic[self.version] = score.score
            node.metadata.setdefault(VALUE_CONTEXTS_KEY, {})[self.version] = (
                score.context
            )


# -- which nodes to score ------------------------------------------------------------


@registrable(slot="critic_selection")
class CriticSelection[P: Payload](ABC):
    """Name the nodes a critic pass should score (candidates; the scheduler skips ones
    already scored for the current version)."""

    @abstractmethod
    def targets(self, view: SchedulerView[P]) -> list[Node[P]]: ...


@register(CriticSelection, "terminals")
class Terminals(CriticSelection[P]):
    """Score terminal (done) nodes -- outcome scoring. The default."""

    def targets(self, view: SchedulerView[P]) -> list[Node[P]]:
        return view.terminals()


@register(CriticSelection, "all")
class AllNodes(CriticSelection[P]):
    """Score every node -- a process/per-node critic over the whole tree."""

    def targets(self, view: SchedulerView[P]) -> list[Node[P]]:
        return view.nodes()


# -- the critic scheduler ------------------------------------------------------------


class CriticScheduler[P: Payload]:
    """Score selected nodes with a reward model and tag each with a version.

    Batch-per-round: one :meth:`RewardModel.ascore_batch` scores a whole batch.
    Selecting and writing happen under ``state.lock``; scoring (the slow HTTP) happens
    *outside* it, so generation keeps mutating the tree meanwhile.

    ``serialize`` is optional here for the same reason it is on :class:`NodeScorer`, and
    means the same thing -- the two ways of scoring a tree must not disagree about how a
    payload is rendered just because one of them made the argument mandatory.
    ``timeout`` likewise: seconds per batch, a fire counted as the scoring failure it
    is, which for a pass tracking generation via ``until`` is the difference between a
    slow reward server and one that never lets the pass drain at all.
    """

    def __init__(
        self,
        *,
        model: RewardModel,
        serialize: Callable[[P], str] | None = None,
        version: str,
        selection: CriticSelection[P] | None = None,
        batch_size: int = 16,
        poll_interval: float = 0.05,
        timeout: float | None = None,
    ) -> None:
        # the (model, serialize, version) triple lives on the scorer, so this scheduler
        # and the generation scheduler score a payload the same way -- including how a
        # payload is rendered when the caller names no renderer, and how long one batch
        # may take (see `NodeScorer`); this scheduler builds the scorer, so a caller
        # could not otherwise bound it at all
        self._scorer: NodeScorer[P] = NodeScorer(
            model=model, serialize=serialize, version=version, timeout=timeout
        )
        self._selection = selection or Terminals()
        self._batch_size = max(1, batch_size)
        self._poll_interval = poll_interval

    async def run_on(
        self, state: SchedulerState[P], *, until: asyncio.Event | None = None
    ) -> None:
        """Score nodes until drained. With ``until``, keep scoring new nodes until it
        is set (generation done) *and* nothing is left; without it, one drain and
        return.

        A scoring fault is survived exactly the way the bundled path survives one (see
        :class:`~focus.scheduler.core.scheduler.Scheduler`): counted in
        ``stats[score_failures]``, logged at WARNING with a traceback, and the pass
        carries on with the next batch. The nodes it hit are *given up on for this run*
        -- selection re-targets whatever is still unscored, so retrying them is what a
        dead reward server would turn into an infinite loop, spinning on the same batch
        while generation waits for a pass that can never drain.
        """
        view = SchedulerView(state)
        # Node ids this run has already failed to score. Local to the call, not the
        # tree: another pass (a second version, a retry after the server is back) starts
        # with a clean slate, and nothing outside this loop should read a transient.
        abandoned: set[int] = set()
        while True:
            async with state.lock:
                batch = self._next_batch(view, abandoned)
            if not batch:
                if until is None or until.is_set():
                    return  # nothing left and generation is done -> stop
                await asyncio.sleep(self._poll_interval)  # wait for new nodes
                continue
            try:
                scores = await self._scorer.score([node.payload for node in batch])
                async with state.lock:
                    self._scorer.write(batch, scores)
            except Exception as exc:
                abandoned.update(node.id for node in batch)
                # `.get`: `SchedulerState.root` seeds every counter, but this scheduler
                # also runs on a state some other caller assembled, and a missing key
                # must not turn a reward-server outage into a KeyError.
                state.stats[STAT_SCORE_FAILURES] = (
                    state.stats.get(STAT_SCORE_FAILURES, 0.0) + 1.0
                )
                logger.warning(
                    "critic scoring failed version=%s nodes=%s",
                    self._scorer.version,
                    [node.id for node in batch],
                    exc_info=exc,
                )

    def _next_batch(self, view: SchedulerView[P], abandoned: set[int]) -> list[Node[P]]:
        """Up to ``batch_size`` selected nodes not yet scored for this version.

        ``abandoned`` names nodes a scoring fault already ate this run; without it the
        same unscored nodes would be selected again on the very next turn of the loop.
        """
        unscored = [
            node
            for node in self._selection.targets(view)
            if self._scorer.version not in node.critic and node.id not in abandoned
        ]
        return unscored[: self._batch_size]


# -- payload serialization -----------------------------------------------------------


def rollout_text(rollout: AnyRollout) -> str:
    """Default ``serialize`` for rollout payloads: flattened ``role: content`` join.

    For exactness a reward model should eventually see its own chat-templated context.
    """
    return format_transcript(rollout.messages)


__all__ = [
    "AllNodes",
    "CriticScheduler",
    "CriticSelection",
    "NodeScorer",
    "Terminals",
    "rollout_text",
]
