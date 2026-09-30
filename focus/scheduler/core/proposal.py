"""Which law an expansion draws its macro-action from -- the design's :math:`\\mu_k`.

Discovery does not have to sample from the anchor, and *which* law produced an edge is
a fact about that edge, not something to be reconstructed later from a temperature. So a
proposal returns the identity of what it drew alongside the parameters, and the expander
folds that identity into the edge's
:class:`~focus.scheduler.core.execution.EdgeProvenance`.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from focus.generation import SamplingParams
from focus.registry import register, registrable
from focus.reward.config import RewardConfig

#: The anchor law's default name, taken from the config field that decides which draws
#: need no importance correction rather than spelled again here. The two are one
#: question -- "was this edge drawn from the anchor?" -- answered by comparing a stamp
#: against ``reward_config.anchor_version``, so a proposal defaulting to some other
#: literal would make the out-of-box configuration correct itself against itself.
ANCHOR_ID: str = RewardConfig().anchor_version


@registrable(slot="proposal_policy")
class ProposalPolicy(ABC):
    """Draws the law one expansion samples under -- the design's :math:`\\mu_k`.

    Discovery does not have to sample from the anchor. A broader proposal buys
    coverage, and costs an importance correction on everything derived from those draws;
    :meth:`draw` returns the identity of what it drew alongside the parameters so the
    correction can be *recorded* rather than inferred later from a temperature.
    """

    #: The law this policy draws from by default -- the name a caller building an
    #: :class:`~focus.scheduler.core.execution.EdgeProvenance` up front stamps
    #: on it, before any draw has happened. Declared here so that caller reads an
    #: attribute of the interface rather than probing for one that might not exist.
    proposal_id: str = ANCHOR_ID

    @abstractmethod
    def draw(self) -> tuple[str, SamplingParams | None]:
        """A ``(proposal_id, sampling override)`` pair for one expansion."""


@register(ProposalPolicy, "anchor")
class AnchorProposal(ProposalPolicy):
    """Sample from the anchor itself -- :math:`\\mu_k = \\pi_k`. The v1 default.

    Worth stating what this collapses: with the proposal identical to the anchor, every
    importance ratio is exactly 1, the empirical draw distribution is uniform, and the
    anchor logprob channel never needs filling -- so the re-scoring pass is skipped
    entirely rather than computed and multiplied in as ones.
    """

    def __init__(
        self,
        proposal_id: str = ANCHOR_ID,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        self.proposal_id = proposal_id
        self._sampling_params = sampling_params

    def draw(self) -> tuple[str, SamplingParams | None]:
        return self.proposal_id, self._sampling_params


@register(ProposalPolicy, "mixture")
class MixtureProposal(ProposalPolicy):
    """``1 - epsilon`` anchor, ``epsilon`` a wider exploration law.

    The drawn component is returned, not reconstructed: which law produced an edge
    decides whether that edge needs an importance correction, and a temperature read off
    the sampling params afterwards would not say *whether the coin came up explore*.

    Using this makes the anchor logprob channel load-bearing -- see
    :func:`~focus.harness.rescoring.reevaluate`, and note that a backend
    without
    prompt logprobs cannot fill it.
    """

    def __init__(
        self,
        explore: SamplingParams,
        epsilon: float = 0.1,
        anchor: SamplingParams | None = None,
        *,
        rng: random.Random | None = None,
        proposal_id: str = ANCHOR_ID,
        explore_id: str = "explore",
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self._explore = explore
        self._epsilon = epsilon
        self._anchor = anchor
        self._rng = rng or random.Random()
        self.proposal_id = proposal_id
        self.explore_id = explore_id

    def draw(self) -> tuple[str, SamplingParams | None]:
        if self._rng.random() < self._epsilon:
            return self.explore_id, self._explore
        return self.proposal_id, self._anchor


__all__ = ["ANCHOR_ID", "AnchorProposal", "MixtureProposal", "ProposalPolicy"]
