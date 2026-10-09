"""The rollout checkpoint: the append-only turn log and the readers over it.

A :class:`RolloutState` is the whole continuation point of a rollout -- the
conversation, the env state, the workspace, and the log of
:class:`~focus.harness.turn.Turn` s taken so far -- and it is also the *final
result* of one, since a finished rollout is just a checkpoint nothing continues from.
The driver that produces it lives next door in
:mod:`focus.harness.runner`; this module is the data and the questions you
can ask of it, so a consumer (scheduler, exporter, preparation) can depend on the
checkpoint without dragging in the loop.

**The log is turns; regions are derived.** ``turns`` only ever grows by append, and
:func:`~focus.harness.packing.regions` derives the masked training
sequences from it on demand. There is no active region to extend, no record to replace
with a longer copy, and an edge is a plain tail slice -- which is what makes
:meth:`RolloutState.edge` a slice rather than an identity-and-offset walk.

**Forking.** ``messages`` is the source of truth (the token prefix is re-rendered from
it each turn), ``state`` is the env's, and ``workspace`` is durable scratch.
:meth:`RolloutState.fork` copies the workspace and shares the immutable rest, so a
caller can snapshot any turn for inspection and fork roots or folds for best-of-N,
tree search, or what-if continuations. Because the env and its tools take the workspace
as a per-turn argument (never captured), the branches stay isolated.
"""

from __future__ import annotations

from collections.abc import Collection
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any
from uuid import uuid4

from focus.codec import Message
from focus.execution.budget import ResourceBudget
from focus.harness.checkpoint import Effect
from focus.harness.env import StepResult
from focus.harness.turn import (
    Turn,
    TurnTag,
    logprob,
    trainable_logprobs,
)
from focus.harness.workspace import Workspace

#: The tag every task turn carries, and the default filter for anything asking about the
#: *work* rather than about the folds that punctuated it.
TASK: tuple[TurnTag, ...] = ("task",)


#: A checkpoint whose *task-state* type the reader does not care about.
#:
#: The scheduler, the exporters and the preparation estimators all read a rollout's
#: turns, rewards and logprobs and never touch :attr:`RolloutState.state`, so there is
#: nothing for them to be generic over. This is the deliberate ``Any`` that says so --
#: and it has to be ``Any`` rather than ``object``: ``RolloutState`` is invariant in
#: ``S`` (:meth:`RolloutState.edge` *takes* a checkpoint as well as yielding one), so
#: a ``Node[RolloutState[object]]`` would not accept a concrete task's node.
type AnyRollout = "RolloutState[Any]"


def _tagged[S](
    turns: tuple[Turn[S], ...], tags: Collection[TurnTag] | None
) -> tuple[Turn[S], ...]:
    """``turns`` restricted to those tags -- ``None`` means every turn, unfiltered."""
    if tags is None:
        return turns
    return tuple(turn for turn in turns if turn.tag in tags)


@dataclass(frozen=True)
class RolloutState[S]:
    """The complete, forkable checkpoint of a rollout -- and its final result.

    Holds everything needed to *continue* a rollout: ``messages`` (the live
    conversation, the source of truth -- the token prefix is re-rendered from it, not
    stored); ``seed`` (the system + first user turn, what a fold folds back to -- see
    :meth:`Compactor.compact` for the shape); the env ``state``; the ``workspace``
    (durable scratch); and ``turns`` -- the append-only log of
    :class:`~focus.harness.turn.Turn` s.

    The append-only generation trace supplies exact training spans. Explicit effects
    and checkpoint_at select restorable states independently of diagnostic fold tags.
    Snapshot copies mutable messages, task state, execution state and workspace;
    generation records and their branch-independent rescoring caches remain shared.
    """

    messages: tuple[Message, ...]
    seed: tuple[Message, ...]
    state: S
    workspace: Workspace
    #: The append-only turn log -- task turns and fold turns interleaved, in order.
    turns: tuple[Turn[S], ...] = field(default_factory=tuple)
    #: Did this episode end because the step budget ran out (:meth:`Runner.finish` drove
    #: one last answer turn) rather than on its own? ``done`` stays ``True`` either way
    #: -- the episode *is* over -- so this is the flag that tells the two endings apart,
    #: for a metric ("what fraction were forced?") or a filter over training data.
    truncated: bool = False
    #: Persisted across pickle and dataclass copies. Version 0 is analysis-only.
    checkpoint_version: int = field(default=2, kw_only=True)
    checkpoint_at: int = 0
    context_version: int = 0
    effects: tuple[Effect, ...] = ()
    invocation_id: str = "root"
    episode_id: str = field(default_factory=lambda: uuid4().hex)
    branch_id: str = field(default_factory=lambda: uuid4().hex)
    execution: dict[str, Any] = field(default_factory=dict)
    history: tuple[Message, ...] = ()
    context_refs: tuple[int | None, ...] = ()
    advances: int = 0
    steps_remaining: int | None = None
    snapshot_available: bool = True
    budget: ResourceBudget = field(default_factory=ResourceBudget)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Give unversioned legacy pickles a durable analysis-only marker."""
        self.__dict__.update(state)
        if "forkable" in state or "checkpoint_version" not in state:
            object.__setattr__(self, "checkpoint_version", 0)

    @property
    def forkable(self) -> bool:
        """Search may branch at a live, settled state selected by CheckpointPolicy."""
        return (
            not self.done
            and not self.truncated
            and self.snapshot_available
            and len(self.turns) == self.checkpoint_at
        )

    def tagged(self, tags: Collection[TurnTag] | None = TASK) -> tuple[Turn[S], ...]:
        """The turns carrying any of ``tags`` (``None`` means every turn)."""
        return _tagged(self.turns, tags)

    @cached_property
    def transitions(self) -> tuple[StepResult[S], ...]:
        """Every env transition realized so far, in order. Fold turns have none.

        Cached, and safe to cache: a checkpoint is frozen and its log only ever grows
        by producing a *new* checkpoint, so this tuple cannot go stale. It is worth
        caching because it is the base of ``done`` / ``reward`` / ``turns_taken``, which
        a search asks of every node it walks past -- recomputing an O(turns) scan for
        each of those made a subtree walk quadratic in the log length. (A frozen
        dataclass has no ``__slots__``, and ``cached_property`` writes straight into
        ``__dict__``, so this needs no exception to the freeze.)
        """
        return tuple(t.transition for t in self.turns if t.transition is not None)

    @cached_property
    def folds(self) -> int:
        """How many times the context was rewritten -- maximal runs of fold turns.

        Derived, not counted: every fold appends at least one fold-tagged turn (a
        compactor that generated nothing still appends a zero-token marker), so the log
        already says how often folding happened and a stored counter could only
        disagree with it.

        A *run*, so two folds in a row -- a compactor that did not relieve its own
        trigger on the first try -- count as one. Nothing happened between them, and
        what a reader of this number is after (where the context was rewritten, how
        many restorable checkpoints the line has) is a property of the rewrite, not of
        how many attempts it took. Cached for the same reason
        :attr:`transitions` is: a checkpoint is frozen and its log only grows by
        producing a new checkpoint.
        """
        return sum(
            turn.tag == "fold" and (i == 0 or self.turns[i - 1].tag != "fold")
            for i, turn in enumerate(self.turns)
        )

    def boundaries(self) -> tuple[int, ...]:
        """Indices of the turns the context was rewritten immediately before.

        A boundary is the first turn *after* a run of fold turns: the one place the
        prompt is not an extension of the previous turn's, and therefore where a branch
        restores a context rather than a suffix. Read off the tag rather than stored on
        the turn, so a log assembled by any means answers the same way the runner's
        does.
        """
        return tuple(
            i
            for i in range(1, len(self.turns))
            if self.turns[i - 1].tag == "fold" and self.turns[i].tag != "fold"
        )

    @property
    def reward(self) -> float:
        """Total *raw* reward over every env transition (outcome + step).

        The unweighted view, for tree ranking and backup. A training statistic uses
        :meth:`RewardConfig.configured_return` instead, so the weighting it applied is
        recorded rather than assumed.
        """
        return self.reward_outcome + self.reward_step

    @property
    def reward_outcome(self) -> float:
        """Task-success reward: nonzero where an episode ended."""
        return sum(t.reward_outcome for t in self.transitions)

    @property
    def reward_step(self) -> float:
        """Shaping reward summed over every transition."""
        return sum(t.reward_step for t in self.transitions) + sum(
            t.compaction_penalty + sum(t.reward_components.values()) for t in self.turns
        )

    @property
    def done(self) -> bool:
        """Whether the last env transition ended the episode.

        Reads the transitions, not the last turn: a fold appends turns that took no env
        step, so "the last turn" and "the last thing that happened in the world" are not
        the same question -- and only the second one can end an episode.
        """
        transitions = self.transitions
        return bool(transitions) and transitions[-1].done

    @property
    def turns_taken(self) -> int:
        """Env transitions realized so far -- one per task turn.

        The unit compute is actually spent in, and monotone because the log is
        append-only. So the difference between two checkpoints on one line is exactly
        what the later one cost for scheduler accounting. Fold turns are not counted
        here; a fold-only advance still consumes one step of the runner's budget.
        """
        return len(self.transitions)

    def edge(self, parent: RolloutState[S]) -> tuple[Turn[S], ...]:
        """The turns added since ``parent`` -- a tail slice of the log.

        The edge, not the rollout: ``turns`` accumulates the whole history, so "what
        happened on this branch" is a difference against the checkpoint it branched
        from. Because the log only ever grows by append and :meth:`fork` shares it by
        reference, that difference is exactly the tail past the parent's length -- which
        is why this is a slice rather than the identity-and-offset walk it replaced.
        """
        # `raise`, not `assert`: `python -O` strips an assert, and what it would be
        # stripping is the one check that this slice *is* the edge. Against a
        # non-prefix checkpoint the arithmetic still returns a tuple of turns -- the
        # wrong ones -- and they would be exported as an edge, scored, and trained on.
        if self.turns[: len(parent.turns)] != parent.turns:
            raise ValueError(
                "edge against a checkpoint that is not a prefix of this one: parent "
                + f"has {len(parent.turns)} turns, this checkpoint {len(self.turns)}"
            )
        return self.turns[len(parent.turns) :]

    def generated_since(
        self,
        parent: RolloutState[S],
        version: str,
        *,
        tags: Collection[TurnTag] | None = TASK,
    ) -> list[float]:
        """Logprobs of every token generated after ``parent``, in order.

        ``tags`` restricts the edge to turns of those
        :attr:`~focus.harness.turn.Turn.tag` s (``None`` means every turn). A
        criterion about the *task* work reads ``("task",)`` so the tokens of an agentic
        fold -- a summary written under a different prompt -- do not stand in for
        uncertainty about the task.

        Silent about an unscored turn (no ``version`` key yields nothing), because a
        criterion reading this treats "never scored" as "no evidence". Where a missing
        channel is a *bug* rather than an absence -- an anchor never re-scored, say --
        use :meth:`edge_logprob`, which says so.
        """
        return trainable_logprobs(_tagged(self.edge(parent), tags), version)

    def edge_logprob(
        self,
        parent: RolloutState[S],
        version: str,
        *,
        tags: Collection[TurnTag] | None = None,
    ) -> float:
        """This edge's sequence log-probability under ``version``, strictly.

        Raises if any turn on the edge was never scored under ``version``, like
        :func:`~focus.harness.turn.logprob`: an importance ratio built from a
        silently-absent anchor channel would read as ``exp(0) = 1``, i.e. "the two
        policies agree exactly" -- the one wrong answer that looks reasonable.

        Defaults to *every* turn, unlike :meth:`generated_since`: an importance
        correction is over the whole edge the policy produced, folds included.
        """
        return logprob(_tagged(self.edge(parent), tags), version)

    def check_continuation(self) -> None:
        """Reject legacy continuation while allowing historical analysis."""
        if self.checkpoint_version != 2 or "forkable" in self.__dict__:
            raise ValueError(
                "Legacy rollout checkpoints are analysis-only; generate fresh "
                "schema-2 rollouts before replay or continuation."
            )

    def snapshot(self) -> RolloutState[S]:
        """Copy branch-owned state while retaining immutable generation records.

        Unguarded (no :attr:`forkable` check, unlike :meth:`fork`): it freezes the
        mutable workspace so a later in-place :meth:`Runner.advance` cannot reach back
        and change an already-recorded checkpoint. Use it to store a checkpoint that
        will keep advancing (e.g. every fold of a long rollout).
        """
        # Also handle legacy instances already in memory before deserialization
        # acquired versioning. replace() would otherwise discard their old flag.
        version = 0 if "forkable" in self.__dict__ else self.checkpoint_version
        return replace(
            self,
            workspace=self.workspace.fork(),
            messages=deepcopy(self.messages),
            seed=deepcopy(self.seed),
            state=deepcopy(self.state),
            execution=deepcopy(self.execution),
            history=deepcopy(self.history),
            checkpoint_version=version,
        )

    def fork(self) -> RolloutState[S]:
        """An independent branch with inherited remaining resources and a new scope.

        A :meth:`snapshot` guarded by :attr:`forkable` -- raises :class:`ValueError`
        when branching is disallowed, so a scheduler that forgot to test ``forkable``
        first is caught rather than silently aliasing.
        """
        self.check_continuation()
        if not self.forkable:
            raise ValueError("checkpoint is not forkable; branching disallowed here")
        return replace(self.snapshot(), branch_id=uuid4().hex)


__all__ = ["TASK", "AnyRollout", "RolloutState"]
