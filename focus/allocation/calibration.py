"""How wrong the critic is at a node, and how noisy one more draw there would be.

:class:`~focus.allocation.refinement.ResidualWorth` needs exactly two numbers
per node -- the posterior mean of the squared calibration error :math:`E[(V - V^*)^2
\\mid \\text{data}]`, and the variance of one more sampled return -- and nothing else
about the tree. Both are statements about a *distribution over outcomes*, and which
distribution that is is a fact about the task rather than a tuning: a {0,1} verifier
outcome and a real-valued score are different models, not different constants in one. So
the answer comes from a registrable slot, and the allocation score is the same function
of it either way.

Nothing here reads the view or the node. A posterior is a function of ``(value, stats,
kappa)`` alone, which is what lets every claim below be checked against a hand
computation instead of against a tree -- and what lets a project with its own outcome
family register a third model without touching the allocator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from focus.registry import register, registrable
from focus.statistics import RunningStats, refined_mean

logger = logging.getLogger(__name__)


@registrable(slot="calibration")
class CalibrationModel(ABC):
    """A posterior over :math:`V^*(h)`, summarised for the allocation score.

    The critic's estimate enters as a *prior*: :attr:`kappa` pseudo-samples centred on
    ``value``, so the same number that is being audited is also what the model falls
    back on where the ledger is empty. That is deliberate. The alternative -- an
    uninformative prior -- would make every unmeasured node report the same maximal
    error and the allocator would rank them by nothing at all.

    Two numbers rather than a whole distribution, because the score
    (:func:`~focus.allocation.refinement.retraining_gain`) is affine in the
    first and linear in the second. Handing back a distribution object would let a model
    express things no caller could read.
    """

    @abstractmethod
    def posterior(
        self, value: float, stats: RunningStats, kappa: float
    ) -> tuple[float, float]:
        """``(E[(V - V*)^2 | data], predictive variance of the next sample)``.

        ``value`` is what the critic says at this node, ``stats`` the evidence samples
        recorded there, ``kappa > 0`` the prior strength.

        The second element is **predictive**: the variance of the next draw with the
        posterior's own spread folded in, which is the noise a purchase actually
        imports. Not the sampling noise conditioned on the truth -- that quantity
        (:math:`E[p(1-p)]` for the Beta) is smaller by exactly
        :math:`\\mathrm{Var}(V^*)`, which the first element already carries, so pricing
        with it would make the two elements one coherent ledger and
        :func:`~...refinement.retraining_gain` non-negative everywhere. Charging the
        epistemic part twice is deliberate: it is
        the surcharge that makes an unmeasured node worth *nothing* rather than worth
        its own uncertainty, and it is what puts a sign on the veto below
        :math:`n = \\kappa`.
        """


@register(CalibrationModel, "beta_bernoulli")
class BetaBernoulli(CalibrationModel):
    """The default: a Beta prior over a Bernoulli outcome, centred on the critic.

    The premise is that **returns lie in** :math:`[0, 1]`.
    Terminal-only reward from a verifier is the case this was written for, and there the
    model earns its place structurally rather than by fitting better. Under Bernoulli
    outcomes the spread is a *function of the mean*, so nothing here ever reads
    :attr:`~...allocation.RunningStats.variance`. A node that returned the same value
    twice therefore cannot report itself as certain, which is precisely the degeneracy
    that made a studentized detector pour budget into the sparsest nodes.

    The predictive variance is where that identity pays twice: the next draw is itself
    Bernoulli with :math:`P(X = 1) = \\mu` marginally, so :math:`\\mathrm{Var}(X) =
    \\mu(1-\\mu)` *exactly*. No approximation, and nothing estimated -- the mean the
    model already computed is the whole answer.

    Outside :math:`[0, 1]` the arithmetic still runs -- ``value`` and the ledger mean
    are clamped, and one warning per model says so -- but it has stopped being a model
    of the returns, and :class:`Gaussian` is the honest choice.
    """

    def __init__(self) -> None:
        self._warned = False

    def posterior(
        self, value: float, stats: RunningStats, kappa: float
    ) -> tuple[float, float]:
        if not self._warned and not (0.0 <= value <= 1.0 and 0.0 <= stats.mean <= 1.0):
            # Once per model, not per node: the premise is a property of the run's
            # returns, and a violated one misranks quietly after the clamp below.
            self._warned = True
            logger.warning(
                (
                    "beta_bernoulli saw a critic value or ledger mean outside [0, 1] "
                    + "(value=%g, mean=%g): clamping, but these returns are not "
                    + "Bernoulli "
                    + 'outcomes -- author model={"name": "gaussian"} instead'
                ),
                value,
                stats.mean,
            )
        v = min(1.0, max(0.0, value))
        # The ledger keeps a mean, not the sample, so the wins are reconstructed from
        # it. Exact for {0,1} returns and the natural reading for anything in between.
        wins = stats.n * min(1.0, max(0.0, stats.mean))
        alpha = kappa * v + wins
        beta = kappa * (1.0 - v) + (stats.n - wins)
        total = alpha + beta  # == kappa + stats.n, so > 0 whenever kappa > 0
        mu = refined_mean(
            v, RunningStats(n=stats.n, mean=min(1.0, max(0.0, stats.mean))), kappa
        )
        # E[(v - p)^2] = (v - E p)^2 + Var(p), with Var(p) = mu(1-mu)/(total+1).
        e2 = (v - mu) ** 2 + mu * (1.0 - mu) / (total + 1.0)
        # The predictive variance of the next draw, and for a binary outcome it is
        # exact rather than an approximation: X is Bernoulli with P(X=1) = E[p] = mu
        # marginally, so Var(X) = mu(1-mu). Equivalently E[p(1-p)] + Var(p), the
        # sampling noise plus what is still unknown about p.
        outcome_var = mu * (1.0 - mu)
        return e2, outcome_var


@register(CalibrationModel, "gaussian")
class Gaussian(CalibrationModel):
    """Conjugate normal: for returns that are not outcomes of a coin.

    The alternative, and the one that has to be *configured*, because a Gaussian must
    estimate its spread from the spread -- there is no mean-variance identity to lean
    on. So the two priors the Beta model does not need live here as this model's own
    fields rather than as knobs on the allocator: they answer a question only this model
    asks.

    The spread is the ledger's own sample variance once there are two samples to compute
    one from, floored so that a node which happened to return one value twice does not
    report itself as infinitely precise. Below two samples there is nothing to estimate
    and ``v_prior`` stands in. The *predictive* variance adds the spread of the mean the
    next draw is taken around, :math:`\\hat\\sigma^2/(\\kappa+n)`, which is the same
    posterior variance the error term carries -- see :meth:`CalibrationModel.posterior`
    for why it is counted in both.
    """

    def __init__(self, *, v_prior: float = 0.25, var_floor: float = 1e-3) -> None:
        if v_prior < 0.0 or var_floor <= 0.0:
            raise ValueError("v_prior must be >= 0 and var_floor must be > 0")
        #: Spread charged to a node with fewer than two samples. A *variance*,
        #: defaulting to a fair coin's -- the most a terminal-only outcome in [0, 1] can
        #: have.
        self.v_prior = v_prior
        #: Floor under the measured spread.
        self.var_floor = var_floor

    def posterior(
        self, value: float, stats: RunningStats, kappa: float
    ) -> tuple[float, float]:
        sigma2 = max(stats.variance, self.var_floor) if stats.n >= 2 else self.v_prior
        mu = refined_mean(value, stats, kappa)
        e2 = (value - mu) ** 2 + sigma2 / (kappa + stats.n)
        # Predictive: the draw's own spread plus the spread of the mean it is drawn
        # around, `sigma2/(kappa + n)` -- the same posterior variance that sits in `e2`.
        return e2, sigma2 * (1.0 + 1.0 / (kappa + stats.n))


__all__ = ["BetaBernoulli", "CalibrationModel", "Gaussian"]
