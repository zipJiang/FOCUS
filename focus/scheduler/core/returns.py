"""Edge rewards and terminal-path inspection."""

from __future__ import annotations

from focus.harness import AnyRollout
from focus.scheduler.core.tree import Node, SchedulerView
from focus.values import RewardConfig


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


def edge_duration(node: Node[AnyRollout], parent: Node[AnyRollout]) -> int:
    """:math:`d`, the task transitions on the edge into ``node``: its discount time."""
    return node.payload.turns_taken - parent.payload.turns_taken


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
    "edge_duration",
    "finished_below",
    "finished_through",
]
