"""Regions, derived: pack a run of turns into prefix-stable training sequences.

A *region* is not a stored thing. It is what you get when consecutive
:class:`~focus.harness.turn.Turn` s chain -- each turn's ``prefix`` extends
the previous pack's tokens, i.e. the backend's KV cache hit -- and :func:`regions` is
the greedy merge that finds the fewest such chains. Extend the current pack while the
next turn's prefix starts with everything packed so far; otherwise flush and start a new
one.

Two things force a flush:

* a **prefix mismatch** -- the chat template rewrote history behind us (Qwen3+ stripping
  ``<think>`` from earlier assistant turns). That is a cache miss, not an inexact
  region: both packs are exact, and the later one restates the earlier generations as
  its own *masked* prefix, so no token is trainable twice.
* a **tag change** -- a fold's turns never share a pack with task turns, even in the
  freak case where their prefixes happen to chain. A pack is a training sample, and one
  that mixed a summary written under the compactor's instruction with task work would be
  labelled with whichever tag came first.

A turn is *in* a pack whatever it holds. A turn with an empty prefix and an empty
completion -- nothing conditioned on, nothing generated -- still gets a span, and if it
is alone in its pack that pack has no tokens at all. The alternative, dropping such a
turn, would make ``spans`` stop indexing the turn log. A zero-token pack costs nothing
to carry and nothing reads it (training preparation keeps only packs with tokens).
The empty pack preserves the position of its turn.

The agent produces exactly one such turn, and on purpose: the marker a silent fold
leaves (:meth:`~focus.harness.agent.Agent._fold`), which is how a fold that
generated nothing is still visible in the log. It packs alone -- its tag flushes the
task turns either side of it -- into a pack with spans and no tokens, and every reader
of a pack is stated over its spans for that reason. Beyond it, this is for a log
assembled by other means: a test, an import, a backend that reported an empty
``prefix_tokens`` against an empty ``cond``.

No render fallback lives here: the agent sends the backend a token list and records
exactly that list as the turn's ``prefix``, so packing never has to guess where a
completion sat inside a re-render.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from focus.generation import TokenId
from focus.harness.turn import Turn, TurnTag


@dataclass(frozen=True)
class PackedSequence:
    """One prefix-stable training string packed from consecutive turns.

    ``[base (mask 0)] + [gen1 (mask 1)] + [obs1 (mask 0)] + [gen2 (mask 1)] ...`` -- the
    shape a trainer consumes. Derived by
    :func:`~focus.harness.packing.regions`; never stored on the log.
    """

    tokens: tuple[TokenId, ...]
    #: Per-token, aligned to :attr:`tokens`: ``True`` where the model generated it.
    trainable: tuple[bool, ...]
    #: Behavior logprobs aligned to :attr:`tokens`, with ``0.0`` at masked positions.
    #: ``None`` unless *every* packed turn that generated anything carries them -- see
    #: ``regions``.
    logprobs: tuple[float, ...] | None = None
    #: ``False`` when any packed turn's tokens do not faithfully record what the policy
    #: did -- a re-encode of the model's text, or an action that arrived beside the
    #: tokens rather than in them (see :attr:`Turn.exact`). This *is* the record a
    #: trainer receives (an exported record's ``spans`` are these), so the caveat
    #: travels with the tokens it is about.
    exact: bool = True
    #: The tag shared by every turn in this pack -- a pack never mixes task and fold.
    tag: TurnTag = "task"
    #: ``(turn_index, start, end)`` per packed turn: where that turn's completion sits
    #: inside :attr:`tokens`.
    spans: tuple[tuple[int, int, int], ...] = ()
    #: Keep malformed compaction supervision even below an advantage cutoff.
    retain_for_training: bool = False
    actor: bool = True
    invocation_id: str = "root"
    format_id: str = ""


def _scored(turn: Turn[object]) -> bool:
    """Whether this turn carries logprobs aligned to its completion."""
    return turn.logprobs is not None and len(turn.logprobs) == len(turn.tokens)


def regions(turns: Sequence[Turn[object]]) -> tuple[PackedSequence, ...]:
    """The regions of a turn log: fewest packs such that each is a cache-hit chain.

    *The* region notion in this codebase: an edge is a slice of the turn log, and its
    regions are what a trainer, a scorer, or an export pass reads off that slice.

    A pack carries ``logprobs`` only when **every** turn in that pack that generated
    anything carries them. Zero-filling an unscored turn would write
    ``0.0`` -- a logprob meaning ``p = 1``, i.e. *certain* -- across tokens nothing
    scored, so a half-scored pack would read as fully scored and confidently wrong. A
    turn whose completion is *empty* is not such a turn: it has no tokens to be wrong
    about, and counting it would let one empty generate unscore everything packed with
    it. The masked positions (prefix and observations) do carry ``0.0``, but those are
    positions no reader counts:
    :attr:`PackedSequence.trainable` says which are real, and the strict reading goes
    through the turn-level helpers in :mod:`focus.harness.turn` instead.
    """
    packs: list[PackedSequence] = []
    # A *tuple*, not a list: every turn is tested against the whole running pack (the
    # chat template may have rewritten history anywhere in it), so this comparison is
    # O(turns x prefix) however it is written -- and `prefix[:n] == tokens` is one
    # tuple compare where the element-wise `all(map(eq, ...))` it replaced walked the
    # pack an item at a time, several times slower on a real conversation's prefix.
    tokens: tuple[TokenId, ...] = ()
    trainable: list[bool] = []
    spans: list[tuple[int, int, int]] = []
    exact = True
    tag: TurnTag = "task"

    def flush() -> None:
        nonlocal tokens, trainable, spans, exact, tag
        # `spans`, not `tokens`, is what says a pack has anything in it: a turn that
        # was conditioned on nothing and generated nothing packs to no tokens and is
        # still a turn, and a flush that read `tokens` here returned with its spans
        # still in hand -- so the next pack inherited them under its own tag.
        if not spans:
            return
        # Only the turns that actually generated something get a vote. A turn whose
        # completion is empty has no column to contribute and nothing to be wrong
        # about, so counting it as "unscored" would drop the pack's logprobs on account
        # of tokens that do not exist -- and one empty generate would silently unscore
        # every other turn packed with it.
        logprobs: tuple[float, ...] | None = None
        if all(_scored(turns[i]) for i, start, end in spans if end > start):
            column = [0.0] * len(tokens)
            for i, start, end in spans:
                if end > start:
                    column[start:end] = turns[i].logprobs or ()
            logprobs = tuple(column)
        packs.append(
            PackedSequence(
                tokens=tokens,
                actor=turns[spans[0][0]].actor,
                invocation_id=turns[spans[0][0]].invocation_id,
                format_id=turns[spans[0][0]].format_id,
                trainable=tuple(trainable),
                logprobs=logprobs,
                exact=exact,
                tag=tag,
                spans=tuple(spans),
                retain_for_training=any(turns[i].compaction_issue for i, _, _ in spans),
            )
        )
        tokens, trainable, spans = (), [], []
        exact = True
        tag = "task"

    previous: Turn[object] | None = None
    for index, turn in enumerate(turns):
        n = len(tokens)
        chains = len(turn.prefix) >= n and turn.prefix[:n] == tokens
        identity_changed = previous is not None and (
            turn.invocation_id,
            turn.format_id,
            turn.actor,
            turn.context_version,
        ) != (
            previous.invocation_id,
            previous.format_id,
            previous.actor,
            previous.context_version,
        )
        if spans and (not chains or turn.tag != tag or identity_changed):
            flush()
            n = 0  # the flush emptied the pack, so the whole prefix is the delta
        if not spans:
            tag = turn.tag
        # A packed run is exactly `prefix + tokens` of its last turn: what was packed
        # before is a prefix of this turn's prefix (that is what `chains` just
        # established), and after a flush there is nothing packed at all. So the pack
        # extends by one concatenation instead of an append per token, and `start` --
        # where this turn's completion lands -- is where its prefix ends.
        start = len(turn.prefix)
        tokens = turn.prefix + turn.tokens
        trainable += [False] * (start - n)
        trainable += [turn.actor] * len(turn.tokens)
        spans.append((index, start, start + len(turn.tokens)))
        exact = exact and turn.exact
        previous = turn
    flush()
    return tuple(packs)


__all__ = ["PackedSequence", "regions"]
