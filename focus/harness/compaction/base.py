"""Compaction protocol: the input/result bags and the :class:`Compactor` ABC.

Concrete strategies live beside this module (``agentic``, …). The runner only depends
on this surface -- a trigger saying when, and a fold given a :class:`Compaction`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from focus.codec import Message
from focus.harness.compaction.triggers import Trigger
from focus.harness.turn import Turn
from focus.harness.workspace import Workspace

if TYPE_CHECKING:
    # Type-only: a fold *is* a rollout, so the runner imports this protocol module.
    # Naming its type here at runtime would close that loop.
    from focus.harness.runner import Runner


class IncompleteCompactionError(ValueError):
    """Rejected model output, including the exact generations that produced it.

    Callers may treat this as a terminal task failure. Transport and implementation
    errors remain ordinary exceptions and must not become training outcomes.
    """

    def __init__(
        self, *, turns: tuple[Turn[Any], ...], reasons: tuple[str, ...]
    ) -> None:
        super().__init__(
            "Incomplete compaction reply; refusing to replace task context"
            + (f" ({', '.join(reasons)})" if reasons else "")
        )
        self.turns = turns
        self.reasons = reasons


@dataclass(frozen=True)
class Compaction[S]:
    """Everything a compaction is handed -- the conversation to fold, and the runner.

    A compactor reads ``messages`` (the grown conversation), persists what matters into
    ``workspace``, and returns the messages to continue from -- ``seed`` (the system +
    first user message) for a plain reset, or ``seed`` with what it kept folded in; how
    the fold is shaped is the compactor's own policy.

    Its one collaborator is the ``runner`` being folded, not a bag of that runner's
    parts: a model-driven fold is a rollout under the *same* policy on a different
    world, so it asks for :meth:`~focus.harness.runner.Runner.derive` instead
    of reassembling a runner out of a generator, a codec, a parser and a version tag
    smuggled across this seam -- and cannot end up guessing a tool dialect the rollout
    never spoke. A compactor that generates nothing simply ignores it.
    """

    messages: tuple[Message, ...]
    seed: tuple[Message, ...]
    state: S
    workspace: Workspace
    #: The runner whose rollout is being folded: the model, the codec, the tool dialect
    #: and the policy-version tag in one piece. Loose in the action parameter because a
    #: fold's env speaks the parser's actions, which need not be the task env's.
    runner: Runner[S, Any]

    checkpoint: Any = None


@dataclass(frozen=True)
class CompactionResult[S]:
    """What one compaction produced: the messages to continue from, and the new state.

    Compaction is a state transition like ``env.step`` -- it reads ``state`` (via
    :class:`Compaction`) and returns a new one, so a compactor can reset the counters
    its :attr:`~Compactor.trigger` watched (e.g. a per-compaction search budget), which
    is what :meth:`~...compaction.triggers.Trigger.relieve` is for.

    ``turns`` are the turns a model-driven compactor generated getting here --
    transition-free, because a fold rewrites the working context rather than acting on
    the world. The runner appends them to its log like any others (stamping them
    ``"fold"`` on the way in, so the tag is its own statement rather than a plug-in's),
    so a fold's tool-call turns are first-class trainable data rather than a special
    kind of record. A plain compactor returns none, and the runner leaves one zero-token
    ``"fold"`` turn in their place -- the marker that records that anything happened.
    """

    messages: tuple[Message, ...]
    state: S
    #: ``Turn[Any]``, not ``Turn[S]``: a generative compactor's turns are
    #: generated in the *fold's* own world (``FoldEnv``), so their state
    #: parameter is not the task's -- and it is vacuous besides, since these
    #: turns are retagged transition-free before they get here.
    turns: tuple[Turn[Any], ...] = ()


class Compactor[S](ABC):
    """A trigger and a fold: *when* to compact, and *how*.

    Exactly two parts, and only one of them is this class's own. :attr:`trigger` is the
    cheap criterion the runner checks before every turn, and it is a
    :class:`~...compaction.triggers.Trigger` -- an object
    that composes (``AnyOf``) -- rather than a method here, because *when* is the half
    an author varies without varying the fold at all. When it fires, the runner calls
    :meth:`compact` with a :class:`Compaction`, whose :class:`CompactionResult` gives
    the messages and state to continue from.

    There is no "compacts never": a runner with nothing to fold takes ``compactor=None``
    and asks no question per turn, which is the same behaviour without an object that
    exists only to answer ``False``.
    """

    #: When to fold. Required -- every compactor sets it, because a fold with no
    #: criterion is one the runner could never call. An annotation rather than an
    #: abstract method, so forgetting it is not a construction error here; the
    #: :class:`~focus.harness.runner.Runner` refuses a compactor without one
    #: instead, which is the first place both halves are in hand. It also owns the other
    #: half of the pairing: the runner front-loads the check, so a trigger that fires
    #: and is never :meth:`~...compaction.triggers.Trigger.relieve` d re-fires on the
    #: very next turn and the rollout folds forever.
    trigger: Trigger[S]

    @abstractmethod
    async def compact(self, compaction: Compaction[S]) -> CompactionResult[S]:
        """The context to continue from, once :attr:`trigger` has fired.

        Awaited by the runner, so an override is ``async def`` even when it generates
        nothing and could have answered synchronously: the interesting fold is a nested
        rollout, and a seam that changed shape with the implementation would make the
        cheap compactor and the model-driven one two different plug-ins. A plain
        ``def`` here type-checks as an incompatible override and, unchecked, surfaces
        only as ``object CompactionResult can't be used in 'await' expression`` inside
        an expansion the scheduler then charges as a dead rollout.

        Return the messages to continue from -- ``compaction.seed``, alone or with what
        was kept folded in -- the state to continue with, and the turns generating that
        summary cost (``()`` for a fold that called no model; the runner tags whatever
        is returned ``"fold"`` and appends it to the log).
        """


__all__ = ["Compaction", "CompactionResult", "Compactor", "IncompleteCompactionError"]
