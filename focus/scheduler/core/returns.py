"""Edge rewards, importance corrections, and terminal-path inspection."""

from __future__ import annotations

import math

from focus.harness import AnyRollout
from focus.reward.config import RewardConfig
from focus.scheduler.core.execution import drawn_from_anchor
from focus.scheduler.core.tree import Node, SchedulerView


def value_baseline(node: Node[AnyRollout], config: RewardConfig) -> float:
    """:math:`V_{\\mathrm{base}}(h)` -- the critic's estimate under the value channel.

    Absent means zero, not "skip this node": an advantage against a missing baseline is
    just the raw return, which is a coarser but honest estimate rather than a hole.
    """
    return node.critic.get(config.value_version, 0.0)


def edge_return(
    node: Node[AnyRollout], parent: Node[AnyRollout], config: RewardConfig
) -> float:
    """:math:`G` over the edge into ``node``: the return of what followed ``parent``.

    The one statistic every estimator starts from, and the one the critic regresses on,
    so it is spelled once here: a suffix of the turn log, starting where the parent's
    log ended. Writing it out by hand is where an off-by-one turns into a return that
    quietly includes the shared history a fork's siblings all have.
    """
    return config.configured_return(node.payload.turns, len(parent.payload.turns))


def importance(
    node: Node[AnyRollout],
    parent: Node[AnyRollout],
    config: RewardConfig,
    behavior_version: str,
    clip: float,
) -> float:
    """:math:`\\widetilde\\varpi` -- the clipped anchor/behavior ratio for one edge.

    Exactly 1 for an edge drawn from the anchor -- the whole v1 configuration. That is
    read off the edge's own provenance rather than off the channel names, because the
    configuration that needs no correction is also the one that never fills the anchor
    channel: deciding by name would demand logprobs that by design do not exist.

    Neither the behavior channel nor the clip has a default: both are properties of the
    run this edge came out of (the policy's ``version``, the batch's
    ``reward_config.clip``), and a default here is a second answer to a question the
    caller already has the answer to -- which is how a ratio comes to be computed
    against a channel nobody wrote, or capped at a bound nobody authored.

    The clip is applied **in log space**, before the exponential rather than after it.
    ``math.exp`` raises ``OverflowError`` above ~709 nats, and these are *sequence*
    log-ratios: a few hundredths of a nat per token over a four-thousand-token edge
    clears that comfortably. Clipping the result would therefore be the one thing that
    never ran on the edges the clip exists for -- the whole export dying with ``math
    range error`` on exactly the heavy tail it was written to survive.
    """
    if config.anchor_version == behavior_version or drawn_from_anchor(
        node, config.anchor_version
    ):
        return 1.0
    log_ratio = node.payload.edge_logprob(
        parent.payload, config.anchor_version
    ) - node.payload.edge_logprob(parent.payload, behavior_version)
    if log_ratio >= math.log(clip):
        return clip
    # Underflows to 0.0 rather than raising, which is the honest answer: an edge the
    # anchor would never have produced gets no weight. A ``nan`` (both channels
    # ``-inf``) fails the comparison above and comes through as ``nan`` -- export
    # refuses it there, where the record it spoils can be named.
    return math.exp(log_ratio)


def finished_below(
    view: SchedulerView[AnyRollout], node: Node[AnyRollout]
) -> list[Node[AnyRollout]]:
    """The finished trajectories reachable from ``node``.

    Every statistic about a prefix is measured over a path that *ended*: a suffix
    return stopping at a fold contains shaping only, because the outcome reward lands on
    the episode-ending transition alone.
    """
    return [n for n in view.descendants(node) if n.payload.done]


def finished_through(view: SchedulerView[AnyRollout], node: Node[AnyRollout]) -> bool:
    """Whether any path *through* ``node`` ended, counting ``node`` itself if it did.

    The subtree walk is lazy, so this stops at the first ending rather than collecting
    every one -- which is the whole difference from :func:`finished_below`.
    """
    return node.payload.done or any(n.payload.done for n in view.descendants(node))


__all__ = [
    "finished_below",
    "finished_through",
    "importance",
    "value_baseline",
]
