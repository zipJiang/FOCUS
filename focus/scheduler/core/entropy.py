"""Branch where the model was least certain: windowed token-entropy node selection.

A verifier-free selection criterion, ported from the pre-rewrite traversal engine
(``windowed_token_entropy`` there). It ranks a frontier node by how uncertain the model
was while generating that node's *children*, so the search spends its budget where the
policy's own distribution says the outcome is still open -- no critic, no ground
truth, nothing but the logprobs generation already returned.

The per-token value is ARES-style ``p * -log p`` over the **sampled** token only. It is
not the entropy of the distribution (which would need the full logits): it peaks at
``p = 1/e`` and falls to zero at *both* ends, so a token the model was certain of and a
token it got badly wrong both score low. That is the quantity the original experiments
measured, kept as-is rather than "improved" into something unmeasured.

Windowing is what the results turned on: a single uncertain token is noise, so the score
is the highest *sliding-window mean* over the region, which localizes the most uncertain
span instead of averaging it away across a long generation.

What counts as an edge -- and getting only the *new* tokens out of a checkpoint that
carries the whole rollout -- belongs to the harness
(:meth:`~focus.harness.rollout.RolloutState.generated_since`); this module is
the windowing and the aggregation, which is the part that is a scheduler policy. That
edge is every generated token of every turn on it, in generation order, so turn
boundaries are invisible to the window and a window may straddle two turns.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import accumulate


def token_entropies(logprobs: Iterable[float]) -> list[float]:
    """The per-token proxy ``p * -log p``, one per finite logprob.

    Not the entropy of the distribution -- that needs the full logits, which no backend
    here returns. This is the sampled token's own term, which peaks at ``p = 1/e`` and
    falls to zero at *both* ends, and it is the quantity the experiments that justified
    this criterion actually measured. The one definition, so a criterion and an
    estimator built on it cannot drift apart.

    Non-finite logprobs are dropped rather than zeroed: ``0.0`` is a logprob meaning
    *certain*, so zeroing would invent evidence instead of recording its absence.
    """
    return [math.exp(lp) * -lp for lp in logprobs if math.isfinite(lp)]


def _edge_entropy(logprobs: Sequence[float], window: int) -> float | None:
    """The highest mean of ``p * -log p`` over stride-1 windows of ``window`` tokens.

    ``None`` when there is nothing to score. A window longer than the sequence clamps
    to its length -- one window, i.e. the plain mean -- so a large enough window
    degenerates to a flat average-entropy criterion.
    """
    values = token_entropies(logprobs)
    n = len(values)
    if n == 0:
        return None
    width = min(window, n)
    cumulative = list(accumulate(values, initial=0.0))
    return (
        max(cumulative[i + width] - cumulative[i] for i in range(n - width + 1)) / width
    )


__all__ = ["token_entropies"]
