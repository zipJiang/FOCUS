"""Collect checkpoint suffix evidence where reducing value error looks useful.

The score estimates the marginal MSE benefit of a blended value target. Under
refined TD this is an allocation heuristic, not a proof of reduced actor-gradient
error: successor-value error affects the actor's return estimate directly.
The default BetaBernoulli calibration model requires returns in [0, 1].
"""

from __future__ import annotations


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
    MSE, and :class:`~focus.search.ErrorReduction` refuses the whole non-positive
    range. For a node the model reads as calibrated,
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


__all__ = ["retraining_gain"]
