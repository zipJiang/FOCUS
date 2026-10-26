"""The scheduler: a bounded-parallel loop that grows a search tree, pure over the tree.

One scheduling step selects a forkable node, asks the injected :class:`Expander` for its
children (``expand(payload) -> (payloads, metadata)``), attaches those to the tree, and
lets a :class:`AllocationSession` update bookkeeping -- then selects again, until a
:class:`Termination` says stop. Up to ``max_concurrency`` expansions run at once via
slot-filling (``asyncio.wait(FIRST_COMPLETED)``): as each finishes it is integrated and
a freed slot is refilled. An expansion that raises is recorded as a failure, never
sinking the run.

**Termination drains, it does not abort.** When ``remaining()`` reaches 0 the loop stops
*launching* and then awaits what is already in flight, integrating each result the
ordinary way -- nodes attached, turns charged, backup run. The alternative (cancelling
them) threw away up to ``max_concurrency`` rollouts' worth of finished LLM calls every
time a pass hit its budget, which for a training harness is the most expensive compute
in the system discarded at the exact moment the budget said it had been paid for. The
price is a bounded overshoot: a turn/node budget is a stop condition, not a launch
count, so totals may exceed it by at most one episode per slot -- the same bound the
check-before-launch already carried, now realized in the tree instead of thrown away.
:class:`~focus.scheduler.core.policies.Budget` states the size of it, and the
one configuration where it is not a bound at all. A *rollout* budget stays a hard cap:
in-flight expansions are charged against it before launch. Draining also makes the
per-call bounds load-bearing (``Agent.generate_timeout``, ``NodeScorer(timeout=...)``):
a pass ends no sooner than its slowest in-flight expansion.

Cancellation is the one thing that is *not* a failure. ``CancelledError`` is a
``BaseException``, so it passes straight through the ``except Exception`` that records a
dead expansion: cancelling the task that runs the loop stops the search and propagates,
rather than being logged as one more dead rollout and re-selected. That is the one path
that still aborts: in-flight expansions are cancelled and the gate slots and backup
reservations they hold are handed back (see :meth:`Scheduler._abort`).

The scheduler holds only the injected ``expander`` and the pure tree policies
(selection, termination, backup); it never touches a ``Agent``, generation, or how a
payload is made. All mutable search state lives in the :class:`SchedulerState` per run,
and every tree access runs under ``state.lock`` -- so external code can inspect or
mutate the tree mid-run by holding the same lock (``async with state.lock:``).

**Scoring.** With a :class:`~focus.scheduler.core.critic.NodeScorer`
registered, one expansion is *expand then score*: children come back already scored, so
a node's critic value is on it before selection can ever see it on the frontier. That
ordering is the point -- a value-guided selection needs it, and a separate scoring loop
cannot give it, since it races the selection that reads it. Scores go to
``node.critic[version]``; ``node.mean_return`` remains the :class:`AllocationSession`'s,
fed by
the realized return alone. To score an *existing* tree instead (post-hoc, a second
model, another version), run a
:class:`~focus.scheduler.core.critic.CriticScheduler` over the same state.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from focus.scheduler.core.allocation import (
    AllocationSession,
    FifoSession,
    Reservation,
)
from focus.scheduler.core.critic import NodeScorer, ValuePrediction
from focus.scheduler.core.execution import Expander
from focus.scheduler.core.gating import GatingPolicy, WidthGating
from focus.scheduler.core.policies import Budget, Termination
from focus.scheduler.core.tree import (
    STAT_NODES,
    STAT_ROLLOUTS,
    STAT_SCORE_FAILURES,
    STAT_TURNS,
    Payload,
    SchedulerState,
    SchedulerView,
)

logger = logging.getLogger(__name__)

P = TypeVar("P", bound=Payload)


@dataclass(frozen=True)
class Expansion[P: Payload]:
    """What one expansion task produced, ready to fold into the tree.

    Three outcomes, each its own field rather than one nullable meaning several things:
    no scorer at all (``scores`` and ``score_error`` both ``None``), a scored chain, or
    a chain whose scoring raised -- which keeps the chain, since generation is the
    expensive half and the estimate is the only thing missing.
    """

    chain: list[P]
    scores: list[ValuePrediction] | None = None
    score_error: BaseException | None = None
    #: What to record on every node this expansion creates -- how the edge was drawn
    #: (see :class:`~focus.scheduler.core.execution.EdgeProvenance`). Merged
    #: into ``Node.metadata``, which exists for exactly this and stays the expander's to
    #: define: the scheduler carries it without reading it.
    metadata: Mapping[str, object] = field(default_factory=dict)


#: One in-flight expansion.
ExpansionTask = asyncio.Task[Expansion[P]]


class Scheduler[P: Payload]:
    """Drive many rollouts as a parallel tree search, pure over the tree/state."""

    def __init__(
        self,
        *,
        expander: Expander[P],
        allocation: AllocationSession[P] | None = None,
        termination: Termination[P] | None = None,
        gating: GatingPolicy | None = None,
        scorer: NodeScorer[P] | None = None,
        max_concurrency: int = 8,
    ) -> None:
        self._expander = expander
        self._allocation = allocation
        self._running = False
        self._termination = termination or Budget(max_rollouts=8)
        self._gating = gating or WidthGating(1)
        self._scorer = scorer
        self._max_concurrency = max(1, max_concurrency)
        if (
            allocation is not None
            and allocation.requires_scorer
            and self._scorer is None
        ):
            raise ValueError(
                "selection reads node.critic but no scorer is registered, so critic "
                + "values stay empty and selection degenerates; pass a scorer such as "
                + "NodeScorer(model=..., serialize=..., version=...)."
            )

    async def run(self, root: P) -> SchedulerState[P]:
        """Search from a root payload until termination, and return the tree it grew."""
        return await self.run_on(SchedulerState.root(root, gating=self._gating))

    async def _expand(self, payload: P) -> Expansion[P]:
        """One unit of work: produce the children, and (with a scorer) score them.

        Bundled deliberately. A node is scored *before* it is attached, so by the time
        selection can see it on the frontier its critic value is already there -- the
        invariant a value-guided :class:`AllocationSession` needs, and the one a
        separate
        polling critic cannot give (it races the very selection that reads it).

        Runs as the expansion task, i.e. outside ``state.lock``: the slow parts (rollout
        and reward call both) never hold the tree.

        A scoring failure does not throw the rollout away -- the chain comes back
        unscored and the node keeps its default score. The generation was expensive and
        is still valid; only the estimate is missing.
        """
        chain, metadata = await self._expander.expand(payload)
        if self._scorer is None:
            return Expansion(chain, metadata=metadata)
        try:
            scores = await self._scorer.score(chain)
        except Exception as exc:
            return Expansion(chain, score_error=exc, metadata=metadata)
        return Expansion(chain, scores=scores, metadata=metadata)

    async def run_on(self, state: SchedulerState[P]) -> SchedulerState[P]:
        """Search over an existing (possibly shared) tree until termination.

        Lets a second scheduler -- e.g. a ``CriticScheduler`` -- run concurrently over
        the same ``state`` under ``state.lock``. Gating comes from ``state.gating`` (set
        when the state was built), not this scheduler's own default.

        Returns the very ``state`` it was handed. The tree *is* the result: its
        terminals and its best node are questions asked of it
        (:meth:`SchedulerState.terminals`, :meth:`SchedulerState.best`), and a record
        that answered them once at return froze them at the moment the run stopped --
        which is exactly wrong for the caller that runs a second pass over the same
        tree, as every later search pass does.

        Termination ends the *launching*, not the run: once ``remaining()`` is 0 the
        loop drains -- it keeps awaiting and integrating what is already in flight until
        nothing is, and only then returns (see the module docstring for why, and for the
        overshoot that buys). The in-flight expansions are cancelled only when the
        caller cancels, or when something escapes the loop.
        """
        if self._running:
            raise RuntimeError("a scheduler cannot run concurrently with itself")
        allocation = (
            self._allocation if self._allocation is not None else FifoSession[P]()
        )
        view = SchedulerView(state)
        allocation.start(view)
        self._running = True
        running: dict[ExpansionTask[P], Reservation[P]] = {}
        # Latched, not re-asked: once termination has said stop, nothing this loop does
        # afterwards may start another expansion. Re-asking each round would let a
        # non-monotone predicate (a goal that a drained result makes false again)
        # re-open a pass that has already been declared over.
        draining = False
        try:
            while state.frontier or running:
                if not draining:
                    async with state.lock:
                        # `remaining` is the single termination signal: 0 means launch
                        # no more (budget spent / goal met / node cap), else it caps how
                        # many may launch now.
                        remaining = self._termination.remaining(view)
                        if remaining == 0:
                            draining = True
                        else:
                            self._fill_slots(
                                state, view, running, remaining, allocation
                            )
                if not running:
                    # Nothing in flight and nothing more coming: either the budget is
                    # spent and the drain is complete, or nothing was selectable and the
                    # search has settled. The same exit either way -- what separates
                    # them is only whether `draining` was ever set.
                    break
                done, _ = await asyncio.wait(
                    running, return_when=asyncio.FIRST_COMPLETED
                )
                async with state.lock:
                    # Walked in launch order, not in `done` order. `asyncio.wait`
                    # answers with a *set*, so iterating it folds the expansions that
                    # finished in the same round into the tree in hash -- that is,
                    # memory address -- order: node ids, and with them every id
                    # tie-break downstream, come out different on each process for the
                    # same seed and the same backend. `running` is insertion-ordered by
                    # launch, so this attaches them in the order selection started
                    # them, which is reproducible.
                    for task in [live for live in running if live in done]:
                        try:
                            self._integrate(state, running[task], task, allocation)
                        finally:
                            del running[task]
        finally:
            # Empty on the ordinary path, since the drain leaves nothing running: this
            # is the cancellation/exception path only.
            try:
                await self._abort(state, running, allocation)
            finally:
                try:
                    allocation.stop()
                finally:
                    self._running = False
        async with state.lock:
            # One line per run, at the end: the whole account of what the search spent
            # and what died, for an operator watching a training job rather than a
            # policy reading `view.stats` mid-run.
            logger.info("scheduler run finished stats=%s", dict(state.stats))
            return state

    def _fill_slots(
        self,
        state: SchedulerState[P],
        view: SchedulerView[P],
        running: dict[ExpansionTask[P], Reservation[P]],
        remaining: int | None,
        allocation: AllocationSession[P],
    ) -> None:
        """Launch expansions into free slots, bounded by concurrency and the budget.

        ``remaining`` (in-flight already charged against the budget by the caller) makes
        a rollout budget a *hard* cap: completed + in-flight never exceeds it -- no
        over-launch, and no overshoot when a batch finishes in one wait.
        """
        cap = self._max_concurrency
        if remaining is not None:
            cap = min(cap, remaining)
        while len(running) < cap:
            reservation = allocation.select(view)
            if reservation is None:
                break
            node = reservation.node
            gate_held = False
            try:
                node.gate.reserve()
                gate_held = True
                state.refresh(node.id)
                coroutine = self._expand(node.payload)
                try:
                    task = asyncio.create_task(coroutine)
                except BaseException:
                    coroutine.close()
                    raise
                running[task] = reservation
            except BaseException:
                try:
                    allocation.cancel(reservation)
                finally:
                    if gate_held:
                        node.gate.release()
                    state.refresh(node.id)
                raise

    def _integrate(
        self,
        state: SchedulerState[P],
        reservation: Reservation[P],
        task: ExpansionTask[P],
        allocation: AllocationSession[P],
    ) -> None:
        # Release before attaching: keeping the slot through attach would temporarily
        # evict the parent and change FIFO frontier order. The lock spans both steps.
        reservation.node.gate.release()
        try:
            self._attach(state, reservation, task, allocation)
        finally:
            try:
                if allocation.active(reservation):
                    allocation.cancel(reservation)
            finally:
                state.refresh(reservation.node.id)

    def _attach(
        self,
        state: SchedulerState[P],
        reservation: Reservation[P],
        task: ExpansionTask[P],
        allocation: AllocationSession[P],
    ) -> None:
        parent = reservation.node
        try:
            expansion = task.result()
        except Exception as exc:  # a dead expansion scores nothing, never sinks the run
            self._fail(state, reservation, exc, allocation)
            return
        if not expansion.chain:
            # A strategy may legitimately produce nothing -- a rollout that made no
            # progress at all. Treated as a failure so the budget still charges for the
            # attempt and the session releases its reservation.
            self._fail(state, reservation, _NoProgress(), allocation)
            return
        created = state.attach_chain(parent.id, expansion.chain)
        for node in created:
            node.metadata.update(expansion.metadata)
        if self._scorer is not None and expansion.scores is not None:
            # Completion hooks see the scores. The critic channel stays separate
            # from realized ancestor returns.
            self._scorer.write(created, expansion.scores)
        elif expansion.score_error is not None:
            # the chain was kept and only the estimate is missing (see `_expand`); the
            # reason is kept on the parent, or a dead reward server looks like silence.
            state.stats[STAT_SCORE_FAILURES] += 1.0
            parent.metadata["score_error"] = repr(expansion.score_error)
            # WARNING, not DEBUG: the run continues, so nothing else says a reward
            # server is down -- and a whole training run scored by a dead one is the
            # failure this line exists to make visible.
            logger.warning(
                "expansion scoring failed node=%d error=%r",
                parent.id,
                expansion.score_error,
            )
        state.stats[STAT_ROLLOUTS] += 1.0
        state.stats[STAT_NODES] += float(len(created))
        # What the expansion actually cost, in the unit compute is spent in. Read off
        # the tree rather than reported by the strategy: the turn log is append-only, so
        # the difference between the leaf's count and the parent's is exact -- and stays
        # exact however many folds the chain passed through.
        charged = _cost(created[-1].payload) - _cost(parent.payload)
        state.stats[STAT_TURNS] += float(charged)
        allocation.complete(reservation, state, created)
        logger.debug(
            "expansion integrated node=%d created=%d turns=%d",
            parent.id,
            len(created),
            charged,
        )

    async def _abort(
        self,
        state: SchedulerState[P],
        running: dict[ExpansionTask[P], Reservation[P]],
        allocation: AllocationSession[P],
    ) -> None:
        if not running:
            return
        for task in running:
            task.cancel()

        async def cleanup() -> None:
            hook_error: BaseException | None = None
            try:
                async with state.lock:
                    for reservation in running.values():
                        try:
                            if allocation.active(reservation):
                                allocation.cancel(reservation)
                        except BaseException as exc:
                            # One broken observer must not prevent other purchases
                            # from settling, including when it raises CancelledError.
                            if hook_error is None:
                                hook_error = exc
                        finally:
                            reservation.node.gate.release()
                            state.refresh(reservation.node.id)
            finally:
                await asyncio.gather(*running, return_exceptions=True)
            if hook_error is not None:
                raise hook_error

        # Keep ownership until cleanup finishes, even through repeated cancellation.
        # No background release may outlive run_on and race the next pass's regating.
        # Bypass user task factories here: a factory failure during admission is
        # itself a reason we reach cleanup with older purchases still outstanding.
        cleanup_task = asyncio.Task(cleanup(), name="allocation-cleanup")
        interrupted = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                interrupted = True
        cleanup_task.result()
        if interrupted:
            raise asyncio.CancelledError

    def _fail(
        self,
        state: SchedulerState[P],
        reservation: Reservation[P],
        exc: BaseException,
        allocation: AllocationSession[P],
    ) -> None:
        state.record_failure(reservation.node.id, exc)
        allocation.fail(reservation, exc)
        logger.warning(
            "expansion failed node=%d stats=%s",
            reservation.node.id,
            dict(state.stats),
            exc_info=exc,
        )


def _cost(payload: Any) -> int:
    """What a payload has spent, in the unit the turn budget counts.

    Read off the payload rather than reported by the expander, so the scheduler stays
    pure over the tree: it adds a number it does not interpret. ``turns_taken`` and not
    ``turns``: the latter is the whole log, and adding tuples here would be a type error
    a default of ``0`` would hide. A payload that keeps neither contributes nothing, and
    a turn budget over such a tree never fires.
    """
    return getattr(payload, "turns_taken", 0)


class _NoProgress(RuntimeError):
    """An expansion that returned no payloads at all."""

    def __init__(self) -> None:
        super().__init__("expansion produced no checkpoints")


__all__ = ["Scheduler"]
