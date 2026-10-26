"""Score search checkpoints with a value model and record the predictions on nodes.

The scheduler calls :class:`NodeScorer` on every chain it attaches. Each prediction is
written as ``node.critic[version]`` together with the serialized context it scored.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from focus.execution import ExecutionServices, Owner
from focus.harness.transcript import format_transcript
from focus.reward import RewardModel, RewardResult
from focus.scheduler.core.tree import (
    Node,
    Payload,
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
        services: ExecutionServices | None = None,
    ) -> None:
        self._services = services or ExecutionServices()
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
        payload = payloads[0]
        owner = Owner(getattr(payload, "episode_id", "default"), "value")
        results = await self._score_batch(contexts, owner)
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

    async def _score_batch(
        self, contexts: list[str], owner: Owner
    ) -> list[RewardResult]:
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
            return await self._services.score(self._model, contexts, owner=owner)
        try:
            return await asyncio.wait_for(
                self._services.score(self._model, contexts, owner=owner),
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


def rollout_text(rollout: AnyRollout) -> str:
    """Default ``serialize`` for rollout payloads: flattened ``role: content`` join.

    For exactness a reward model should eventually see its own chat-templated context.
    """
    return format_transcript(rollout.messages)


__all__ = [
    "NodeScorer",
    "rollout_text",
]
