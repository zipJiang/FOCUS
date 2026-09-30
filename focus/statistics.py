"""Running return statistics and the shared network-prior blend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class RunningStats:
    """Welford mean and variance of the returns bought at one node.

    Welford rather than a list: an allocator asks for the mean and the spread, never for
    the sample, and keeping the sample would make :class:`Observations` grow with the
    budget for no reader.

    :attr:`n` is deliberately its own count rather than reusing ``m_done``, and neither
    bounds the other: they are two ledgers about one node. ``m_done`` counts *purchases*
    -- rollouts this allocator paid for and must settle -- while :attr:`n` counts
    *evidence*, every valid policy suffix observed at this checkpoint.
    A purchase that came back unfinished settles without contributing a return (its
    suffix would be shaping-only), which puts ``m_done`` ahead; a node an earlier pass
    created contributes the return of the rollout that created it without anyone having
    bought it (:func:`~focus.allocation.evidence.RolloutLedger.record`),
    which puts :attr:`n` ahead.
    """

    n: int = 0
    mean: float = 0.0
    #: Sum of squared deviations from the running mean -- Welford's ``M2``.
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @classmethod
    def of(cls, values: Sequence[float]) -> RunningStats:
        """The stats of a finished list -- the one spelling of ``sum((v-m)**2)/(n-1)``.

        For a caller that already holds every value and only wants the summary, so that
        the ``(n-1)`` denominator and the "what does one sample report" convention live
        here rather than once per estimator.
        """
        stats = cls()
        for value in values:
            stats.add(value)
        return stats

    @property
    def variance(self) -> float:
        """Sample variance, or ``0.0`` below two observations.

        Zero rather than a prior: this record reports what was measured, and what an
        unmeasured node should be *charged* at is a policy each allocator states for
        itself (see :class:`~focus.allocation.calibration.Gaussian`'s
        ``v_prior``).
        """
        return self.m2 / (self.n - 1) if self.n >= 2 else 0.0


def refined_mean(value: float, stats: RunningStats, strength: float) -> float:
    """Posterior mean with ``strength`` prior observations at ``value``."""
    return (strength * value + stats.n * stats.mean) / (strength + stats.n)
