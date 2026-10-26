"""A turn: one generate as the engine saw it -- the atom of the checkpoint's log.

A checkpoint's log is an append-only tuple of :class:`Turn` s. Each one records the
exact token prefix that was sent to the backend, the completion that came back, the
behavior policy's logprobs over that completion, and -- for a task turn -- the env
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

Tags label task versus summary work for diagnostics. They do not select checkpoints.
Effects record context changes independently; invocation and actor identity determine
training ownership. Exact generation prefixes survive all live-context edits.

``logprobs`` are the behavior policy's, recorded at generation time. Storing them per
*turn* rather than per region is what makes an edge exact -- a token's logprob belongs
to the generate that produced it, and no offset arithmetic is needed to say which
tokens a fork added.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from focus.generation import SamplingParams, TokenId
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
    #: The behavior policy's per-token logprobs over :attr:`tokens`, or ``None`` when
    #: the backend returned none (or a count that does not match the completion).
    logprobs: tuple[float, ...] | None = None
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
    #: diagnostic label; effects and CheckpointPolicy govern continuation.
    tag: TurnTag = "task"
    #: Raw shaping on a compaction attempt; does not advance task time.
    compaction_penalty: float = 0.0
    #: Nonempty for malformed attempts, including after the episode penalty cap.
    compaction_issue: str = ""
    invocation_id: str = "root"
    #: The generating policy's label, for diagnostics; see ``Policy.version``.
    policy_id: str = ""
    format_id: str = ""
    context_version: int = 0
    actor: bool = True
    sampling_params: SamplingParams | None = None
    reward_components: dict[str, float] = field(default_factory=dict)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickling old turns bypasses default factories; keep analysis usable."""
        self.__dict__.update(state)
        self.__dict__.setdefault("reward_components", {})


def trainable_logprobs(turns: Sequence[Turn[object]]) -> list[float]:
    """Every generated token's behavior logprob, in generation order.

    Only the generated positions exist here at all: a turn stores logprobs over its
    completion, so there is no mask to apply and no ``0.0``-at-an-observation to confuse
    with a real logprob of ``p = 1``. Lenient about a turn without logprobs -- it
    contributes nothing -- because a criterion reading this treats "never scored" as
    "no evidence". Training reads :func:`logprob`, which refuses one.
    """
    return [lp for turn in turns if turn.actor for lp in turn.logprobs or ()]


def logprob(turns: Sequence[Turn[object]]) -> float:
    """Sequence log-probability under the behavior policy, strictly.

    Raises :class:`ValueError` if an actor turn has no logprobs, or logprobs that are
    misaligned or non-finite: a trainer's importance ratio against the behavior policy
    needs every generated token's probability.
    """
    total = 0.0
    for turn in turns:
        if not turn.actor:
            continue
        values = turn.logprobs
        if (
            values is None
            or len(values) != len(turn.tokens)
            or any(not math.isfinite(v) for v in values)
        ):
            raise ValueError("actor logprobs must be present, aligned, and finite")
        total += sum(values)
    return total


__all__ = [
    "Turn",
    "TurnTag",
    "logprob",
    "trainable_logprobs",
]
