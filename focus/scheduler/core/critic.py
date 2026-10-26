"""Score search checkpoints with the critic and record the predictions on nodes.

The scheduler calls :class:`NodeScorer` on every chain it attaches. Each prediction is
written as ``node.critic`` together with the context it scored.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from focus.execution import ExecutionServices, Owner
from focus.scheduler.core.tree import Node, Payload

if TYPE_CHECKING:
    from focus.critic import Critic

logger = logging.getLogger(__name__)

P = TypeVar("P", bound=Payload)


@dataclass(frozen=True)
class ValuePrediction:
    score: float
    context: str


class NodeScorer[P: Payload]:
    """Batch a critic's predictions for new checkpoints through the execution services.

    Predictions land in ``node.critic`` and *only* there, separate from the realized
    returns the evidence ledger records.

    ``critic.timeout`` bounds one prediction batch. When it fires the batch is a
    scoring failure on the ordinary terms (counted in ``stats[score_failures]``, the
    rollout kept and only the estimate missing).
    """

    def __init__(
        self, critic: Critic, *, services: ExecutionServices | None = None
    ) -> None:
        self.critic = critic
        self._services = services or ExecutionServices()

    async def score(self, payloads: Sequence[P]) -> list[ValuePrediction]:
        """Predict values for ``payloads`` in one batched call, preserving order.

        A critic that answers with the wrong number of predictions raises here, inside
        the callers' guards, so the fault costs an estimate rather than the run.
        """
        if not payloads:
            return []
        contexts = [self.critic.context(p) for p in payloads]  # type: ignore[arg-type]
        owner = Owner(getattr(payloads[0], "episode_id", "default"), "value")
        values = await self._predict(contexts, owner)
        if len(values) != len(payloads):
            raise ValueError(
                f"critic returned {len(values)} predictions for {len(payloads)} "
                + "payloads; the predictions cannot be paired with the nodes"
            )
        return [
            ValuePrediction(float(v), context)
            for v, context in zip(values, contexts, strict=True)
        ]

    async def _predict(self, contexts: list[str], owner: Owner) -> Sequence[Any]:
        timeout = self.critic.timeout
        if timeout is None:
            return await self._services.predict(self.critic, contexts, owner=owner)
        try:
            return await asyncio.wait_for(
                self._services.predict(self.critic, contexts, owner=owner), timeout
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"scoring exceeded timeout={timeout}s (batch={len(contexts)}); the "
                + "batch is abandoned and counts as one scoring failure -- the "
                + "rollouts are kept, only the estimate is missing"
            ) from exc

    def write(
        self, nodes: Sequence[Node[P]], scores: Sequence[ValuePrediction]
    ) -> None:
        """File ``scores`` on ``nodes``, pairwise."""
        for node, score in zip(nodes, scores, strict=True):
            node.critic = score.score
            node.value_context = score.context


__all__ = ["NodeScorer", "ValuePrediction"]
