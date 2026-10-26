"""Collect checkpoint suffix evidence where reducing value error looks useful.

The score estimates the marginal MSE benefit of a blended value target. Under
refined TD this is an allocation heuristic, not a proof of reduced actor-gradient
error: successor-value error affects the actor's return estimate directly.
The default BetaBernoulli calibration model requires returns in [0, 1].
"""

from __future__ import annotations

import logging

from focus.allocation.base import Allocator
from focus.allocation.calibration import BetaBernoulli, CalibrationModel
from focus.allocation.evidence import (
    NodeWorth,
    Observations,
    RolloutLedger,
    ScoredSession,
)
from focus.harness import AnyRollout
from focus.reward.config import RewardConfig
from focus.scheduler.core.allocation import AllocationSession
from focus.scheduler.core.returns import value_baseline
from focus.scheduler.core.tree import Node, SchedulerView

logger = logging.getLogger(__name__)


def retraining_gain(e2: float, outcome_var: float, n_eff: int, kappa: float) -> float:
    """:math:`(2\\kappa^2 e^2 + (n - \\kappa)\\sigma^2)/(\\kappa + n)^3` -- the MSE the
    next sample removes from the critic's target at this node.

    The derivative of the blended estimator's mean squared error, derived in the module
    docstring, evaluated at the node's current evidence count. ``e2`` is a posterior
    mean :math:`E[(V - V^*)^2]` and ``outcome_var`` the *predictive* variance of the
    next sample, both from a
    :class:`~focus.allocation.calibration.CalibrationModel`.

    **Negative is a real answer**, not an error: below :math:`n = \\kappa` retraining
    toward a mean built from fewer samples than the estimate it replaces *raises* the
    MSE, and
    :meth:`~focus.allocation.evidence.ScoredSession.eligible` already
    refuses the whole non-positive range. For a node the model reads as calibrated,
    both shipped models put the sign at :math:`\\kappa^2 - \\kappa + n^2 + n` exactly --
    so the veto region is precisely *an unmeasured node whose critic is worth less than
    one sample*, and the boundary is the identity worth remembering:

    **At** :math:`\\kappa = 1` **an unmeasured node scores exactly zero**, whatever the
    critic says, under either model. That is what makes birth-sample seeding
    (:func:`~focus.allocation.evidence.RolloutLedger.record`) load-bearing
    rather than merely helpful: with an empty ledger everywhere the pass is ineligible
    everywhere and could never make its first purchase.

    Both properties are consequences of pricing with the *predictive* variance --
    :meth:`~focus.allocation.calibration.CalibrationModel.posterior` carries
    the argument, and what pricing with the coherent alternative would cost.

    Above :math:`n = \\kappa` the :math:`(n - \\kappa)\\sigma^2` term goes positive on
    its own, which is why a perfectly calibrated node still scores *something*:
    averaging in one more draw dilutes the noise already in the blend. It is a real
    gain, it falls off like :math:`1/n^2`, and it sits below any genuine miscalibration
    at the same count.
    """
    return (2.0 * kappa**2 * e2 + (n_eff - kappa) * outcome_var) / (kappa + n_eff) ** 3


class ResidualWorth(NodeWorth):
    """What retraining on one more sample here would buy the critic.

    One branch and no thresholds: ask the model what the node's ledger says about the
    critic's error, and price it with :func:`retraining_gain`. The "is the critic wrong
    *enough*" question that a studentized detector had to answer with a floor does not
    arise, because the gain is affine in :math:`E[e^2]` -- a node the evidence cannot
    convict scores the small noise-dilution tail and loses to one it can, and a node
    with no evidence at all scores nothing and is refused outright.

    The count the gain is evaluated at is
    :attr:`~focus.allocation.evidence.Observations.n_eff`, which is where
    "an in-flight purchase is an evidence sample that has not landed yet" is argued.
    """

    def __init__(
        self,
        reward_config: RewardConfig,
        model: CalibrationModel,
    ) -> None:
        super().__init__(reward_config)
        self.model = model
        self.kappa = reward_config.value_prior_strength

    def worth(
        self,
        view: SchedulerView[AnyRollout],
        node: Node[AnyRollout],
        counts: Observations,
    ) -> float:
        del view  # everything this worth reads is on the node and its counts
        stats = counts.returns
        e2, outcome_var = self.model.posterior(
            value_baseline(node, self.reward_config), stats, self.kappa
        )
        return retraining_gain(e2, outcome_var, counts.n_eff, self.kappa)


class ValueRefinement(Allocator):
    """Allocate where another sample is expected to improve the critic target.

    The owning SearchPass supplies budget and gating. The calibration model defaults
    to BetaBernoulli; use Gaussian for returns outside its bounded-outcome premise.
    RewardConfig.value_prior_strength sets the network prior's strength. This
    allocator gathers data; it does not retrain the network during search.
    """

    name = "value_refinement"

    def __init__(
        self,
        *,
        model: CalibrationModel | None = None,
    ) -> None:
        self.model = model if model is not None else BetaBernoulli()

    def build(
        self,
        view: SchedulerView[AnyRollout],
        ledger: RolloutLedger,
    ) -> AllocationSession[AnyRollout] | None:
        """Refuse the pass outright when nothing filled the value channel.

        ``value_baseline`` reads a missing estimate as ``0.0``, which is the honest
        answer for an advantage -- a coarser one, not a hole. Here it is not: with no
        critic anywhere, every residual becomes ``|mean|`` and this allocator quietly
        turns into "spend where the return is largest", which is a different recipe
        wearing this one's name.

        A check on the tree's data rather than on the run's configuration, which is the
        weaker of the two: ``Scheduler.__init__`` *refuses* a selection that declares
        ``requires_scorer`` with none registered, where this skips. The stronger form
        needs the value head, which reaches a pass but not a ``collect``.
        """
        reward_config = ledger.reward_config
        if not any(reward_config.value_version in node.critic for node in view.nodes()):
            logger.info(
                (
                    "value_refinement skipped: no node carries a critic "
                    + "estimate "
                    + "under version=%r, so every residual would be the raw return"
                ),
                reward_config.value_version,
            )
            return None
        return ScoredSession(ResidualWorth(reward_config, self.model), ledger)


__all__ = ["ResidualWorth", "ValueRefinement", "retraining_gain"]
