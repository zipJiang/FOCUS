"""Admission control for the scheduling loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from focus.scheduler.core.tree import (
    STAT_FAILURES,
    STAT_ROLLOUTS,
    STAT_TURNS,
    Payload,
    SchedulerView,
)


class Termination[P: Payload](ABC):
    """Admission control for the scheduling loop: how many more rollouts may launch.

    :meth:`remaining` is the whole interface, and deliberately the only one: a separate
    ``done()`` would be a second answer to the same question, free to disagree with the
    first. ``0`` *is* done.
    """

    @abstractmethod
    def remaining(self, view: SchedulerView[P]) -> int | None:
        """Rollouts still launchable now: ``0`` = stop, positive = cap, ``None`` = none.

        The scheduler counts in-flight rollouts against a returned cap, so a rollout
        budget is a *hard* cap -- completed + in-flight never exceeds it.

        ``0`` ends the *launching*, not the run: the scheduler then drains what is
        already in flight and integrates it normally (finished rollouts are the
        expensive half, and cancelling them at the budget threw away up to one episode
        per slot). The first ``0`` is latched -- this is never asked again on that run,
        so a predicate a drained result would flip back cannot re-open the pass -- and
        the stats a caller reads afterwards include everything the drain integrated.
        """


class Budget[P: Payload](Termination[P]):
    """Stop on a spent budget (turns / rollout count / tree size), or on a goal.

    ``goal`` is an optional predicate over the read-only view (e.g. "a terminal with
    reward >= 1 exists"); when it returns ``True`` the loop ends early.

    ``max_rollouts`` bounds *attempts*: completed expansions plus failed ones.

    ``max_turns`` bounds realized env transitions -- the unit compute is actually spent
    in, and the only one that means the same thing whether an expansion rolled one
    region or a whole episode. It is a stop condition rather than a launch count,
    because what the *next* expansion will cost is not known before it runs. Total turns
    are therefore bounded by ``max_turns + max_concurrency * Agent.max_steps``: the
    check happens before launching, and expansions already in flight are drained rather
    than cancelled -- they finish and are counted, since a finished rollout thrown away
    at the budget line is the most expensive thing this loop could discard.
    ``max_nodes`` has the same overshoot, for the same reason. Only ``max_rollouts``
    is exact, because a rollout can be counted before it runs.

    That bound assumes the agent *has* a step budget. Under ``Agent.max_steps = -1``
    one expansion is unlimited, so a turn or node budget still stops the loop
    *launching* but bounds nothing: an env that never ends the episode makes the pass as
    long as its slowest in-flight expansion, which is then bounded only by
    ``Agent.generate_timeout``. A run that wants a guaranteed ceiling needs a
    ``max_rollouts`` (exact whatever the agent does) or a finite ``max_steps``.

    The overshoot is *carried*, not compounded: the next pass's cap is computed from
    what the tree has already spent (:meth:`turns_spent`), so a pass
    that ran 3 turns over hands the next one a baseline 3 higher and still gets its own
    full share -- the overshoot is paid once, by the whole run's total, never by the
    following pass's evidence.
    """

    def __init__(
        self,
        max_rollouts: int | None = None,
        *,
        max_turns: int | None = None,
        max_nodes: int | None = None,
        goal: Callable[[SchedulerView[P]], bool] | None = None,
    ) -> None:
        self._max_rollouts = max_rollouts
        self._max_turns = max_turns
        self._max_nodes = max_nodes
        self._goal = goal

    @staticmethod
    def turns_spent(view: SchedulerView[P]) -> float:
        """Turns realized, plus one per failure.

        A dead expansion records no turns -- it produced no payload to read them off --
        so a turn cap counting successes alone would never deplete against an expander
        that always fails, and the loop would re-select and re-fail forever. Charging a
        turn for the attempt is the same protection ``max_rollouts`` gets by counting
        failures, and it is a floor: a rollout that died partway spent at least that.
        """
        return view.stats.get(STAT_TURNS, 0.0) + view.stats.get(STAT_FAILURES, 0.0)

    def remaining(self, view: SchedulerView[P]) -> int | None:
        if self._goal is not None and self._goal(view):
            return 0
        if self._max_turns is not None and self.turns_spent(view) >= self._max_turns:
            return 0
        if self._max_nodes is not None and len(view) >= self._max_nodes:
            return 0
        if self._max_rollouts is None:
            return None  # only a rollout budget makes launches a hard cap
        # Failures count against the budget too. A rollout that died still spent the
        # compute, and -- load-bearing -- a budget that only counted successes would
        # never deplete against an expander that always fails, so the loop would
        # re-select and re-fail forever instead of ending the pass.
        spent = view.stats.get(STAT_ROLLOUTS, 0.0) + view.stats.get(STAT_FAILURES, 0.0)
        return max(0, self._max_rollouts - int(spent))


__all__ = ["Budget", "Termination"]
