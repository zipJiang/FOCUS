"""A turn: one generate as the engine saw it -- the atom of the checkpoint's log.

A checkpoint's log is an append-only tuple of :class:`Turn` s. Each one records the
exact token prefix that was sent to the backend, the completion that came back, the
version-keyed logprobs over that completion, and -- for a task turn -- the env
transition the action produced. Nothing is ever rewritten: a rollout advances by
appending one more turn, so the difference between two checkpoints on one line is a
plain tail slice.

*Regions are derived, not stored.* What multi-turn RL trains on is a masked token string
-- base prefix (masked), generation (trainable), observation (masked), generation... --
and that is exactly what :func:`~focus.harness.packing.regions` builds from
a run of turns whose prefixes chain. A chat template that rewrites history (Qwen3+
stripping ``<think>``) breaks the chain and the same run packs into two sequences
instead of one; nothing about the log changes. Because packing is a function of the log,
a region is never a thing that can go stale.

*One marker, and it is the tag.* :attr:`Turn.tag` is ``"task"`` for the work and
``"fold"`` for the turns a context fold generated. A fold is a nested rollout over its
own env (see :mod:`focus.harness.compaction.fold_env`), so its turns are
ordinary, trainable turns; the tag is what keeps them tellable apart from task work.
Every fold leaves a run of fold-tagged turns -- a compactor that generated nothing
still appends one zero-token marker turn -- so *everything else about folding is
derived from the tag*: how many times the context was rewritten is the number of
maximal fold runs, and where a branch may restore a context is the turn after one (see
:attr:`~focus.harness.rollout.RolloutState.folds` and
:meth:`~focus.harness.rollout.RolloutState.boundaries`). A second field
saying either of those things could disagree with the log; a tag cannot disagree with
itself.

``logprobs`` is version-keyed exactly like
:attr:`~focus.scheduler.core.tree.Node.critic`: the behavior policy fills its
own key at generation time, and a scorer (:mod:`focus.harness.rescoring`)
fills
another policy's key later. Storing them per *turn* rather than per region is what makes
an edge exact -- a token's logprob belongs to the generate that produced it, and no
offset arithmetic is needed to say which tokens a fork added.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from focus.generation import TokenId
from focus.harness.env import StepResult

#: What kind of work a turn is: the task, or the agentic fold that rewrote its context.
#:
#: A closed set, and spelled as one: the two tags are read by name all over the engine
#: (a tag filter, a region flush, an exported span's metadata), and a misspelling used
#: to be a silent empty filter -- ``tags=("tsak",)`` selects no turns, and every reader
#: downstream simply sees a rollout that did nothing.
#:
#: Pass :data:`~focus.harness.rollout.TASK` rather than an inline
#: ``("task",)`` where one tag is wanted: mypy reads a *one-element* tuple literal as
#: ``tuple[str]`` against a ``Collection`` parameter however its member is spelled, so
#: the inline form is rejected even when it is right. A list (``["task"]``) or a
#: two-element tuple infers the literals fine.
type TurnTag = Literal["task", "fold"]


# ``S_co`` is covariant: a ``Turn`` is frozen and only ever *yields* the state its
# transition carried, so ``Turn[Concrete]`` is a ``Turn[object]`` -- which is what lets
# the state-agnostic readers below take any turn at all.
@dataclass(frozen=True)
class Turn[S_co]:
    """One generate: the prefix sent, the completion, and what the world did with it."""

    #: Exactly what the engine was conditioned on -- the ids the runner sent, never a
    #: reconstruction. Packing chains turns by testing this against the running pack.
    prefix: tuple[TokenId, ...]
    #: What the engine generated. The only trainable span this turn contributes.
    tokens: tuple[TokenId, ...] = ()
    #: Per-token logprobs over :attr:`tokens`, keyed by policy version. Mutable dict on
    #: a frozen record (like ``Node.critic``) -- an action's logprobs under a policy are
    #: branch-independent, so filling one in place is safe across forks sharing this
    #: turn.
    logprobs: dict[str, tuple[float, ...]] = field(default_factory=dict)
    #: Whether :attr:`tokens` faithfully record what the policy did. Two falsifiers,
    #: one claim: a backend that re-encoded text it could not perfectly invert, or a
    #: parser whose action arrived beside the tokens (a structured tool-call channel)
    #: rather than in them. Either way, training against these ids would train against
    #: something the policy did not do.
    exact: bool = True
    #: The *task* env transition this turn produced. ``None`` on a fold turn: a fold
    #: rewrites context, it does not act on the world, so it is not a time step.
    transition: StepResult[S_co] | None = None
    #: ``"task"`` for the work, ``"fold"`` for a context fold's own generations. The
    #: only mark a fold leaves, and the one every question about folding is answered
    #: from.
    tag: TurnTag = "task"
    #: Raw shaping on a compaction attempt; does not advance task time.
    compaction_penalty: float = 0.0
    #: Nonempty for malformed attempts, including after the episode penalty cap.
    compaction_issue: str = ""
    invocation_id: str = "root"
    policy_id: str = ""
    format_id: str = ""
    context_version: int = 0
    actor: bool = True


def trainable_logprobs(turns: Sequence[Turn[object]], version: str) -> list[float]:
    """Every generated token's logprob under ``version``, in generation order.

    Only the generated positions exist here at all: a turn stores logprobs over its
    completion, so there is no mask to apply and no ``0.0``-at-an-observation to confuse
    with a real logprob of ``p = 1``.

    Lenient about a turn that lacks ``version`` -- it contributes nothing rather than
    raising -- because a criterion reading this treats "never scored" as "no evidence".
    Where a missing channel is a *bug* rather than an absence (an anchor never
    re-scored), use :func:`logprob`, which says so.
    """
    return [lp for turn in turns for lp in turn.logprobs.get(version, ())]


def logprob(turns: Sequence[Turn[object]], version: str) -> float:
    """Sequence log-probability under ``version``, strictly.

    Raises :class:`KeyError` if any turn lacks the channel. An importance ratio built
    from a silently-absent anchor would read as ``exp(0) = 1`` -- "the two policies
    agree exactly" -- which is the one wrong answer that looks reasonable.
    """
    total = 0.0
    for turn in turns:
        if version not in turn.logprobs:
            raise KeyError(f"turn has no logprobs for version {version!r}")
        total += sum(turn.logprobs[version])
    return total


__all__ = [
    "Turn",
    "TurnTag",
    "logprob",
    "trainable_logprobs",
]
