"""Re-scoring: recompute an action's log-probabilities under a *different* policy.

Usually the behavior policy's logprobs (captured at generation) are all that is needed.
When a later pass wants the same actions scored under another policy version -- e.g. a
frozen anchor for an importance ratio -- that is what :func:`reevaluate` is for. It only
fills :attr:`~focus.harness.turn.Turn.logprobs`; rewards, critics, and the env
transitions are untouched.

Scoring is per *region* but storage is per *turn*, and the two are bridged by
:attr:`~focus.harness.packing.PackedSequence.spans`. A region packed out of
the
log (:func:`~focus.harness.packing.regions`) is a self-contained token string
with a trainable mask, so re-scoring it is one teacher-forced pass -- no prefix
reconstruction -- and the returned column is then sliced back onto the turns that
produced it. Keeping the result on the turns rather than on the pack is what makes it
survive: packs are derived fresh on every call, so a logprob written to one would be
gone the next time anything asked for a region.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from focus.execution import ExecutionServices
from focus.generation import (
    GenerateResult,
    Policy,
    SamplingParams,
    TokenId,
)
from focus.generation.policy import PreparedPrompt
from focus.harness.packing import regions

if TYPE_CHECKING:
    from focus.harness.rollout import AnyRollout


@runtime_checkable
class PolicyScorer(Protocol):
    """Teacher-force a token sequence and return its per-token log-probabilities."""

    async def score(
        self, tokens: tuple[TokenId, ...], trainable: tuple[bool, ...]
    ) -> tuple[float, ...]:
        """Per-token logprobs of ``tokens`` under this policy, aligned to ``tokens``.

        ``trainable`` marks the positions that matter (the generated tokens); a scorer
        may skip the rest and return any value (e.g. ``0.0``) there. The returned tuple
        has the same length as ``tokens``.
        """
        ...


class TokenScorer:
    """A :class:`PolicyScorer` over any ``Policy`` that returns prompt logprobs.

    Teacher-forces the sequence by generating *from* it and reading back the backend's
    ``prompt_logprobs`` (the per-token logprobs of the prompt itself). vLLM and tinker
    support this; slime/openai-chat do not (they return no prompt logprobs) and raise.
    Point one at the target policy's weights and pass it to :func:`reevaluate`.

    Never handed an empty sequence: a pack that holds no tokens is answered by
    :func:`reevaluate` itself (its columns are all ``()``), so nothing here has to
    defend against a prompt a backend would reject.

    ``timeout`` bounds one teacher-forced pass in seconds (``None``, the default, waits
    as long as the backend does) -- the re-scoring counterpart of
    ``Runner.generate_timeout``, and needed for the same reason: nothing else bounds one
    request, and :func:`~focus.loop.rescore_anchor` runs after the last
    search pass, with the whole prompt's exported records waiting behind it. Unlike a
    generation timeout, a fire here is **not** survivable: it comes out of
    :meth:`score` as :class:`TimeoutError` and out of ``rescore_anchor`` unchanged. The
    anchor channel is read strictly at export (an absent anchor logprob would otherwise
    read as ``exp(0) = 1`` -- "the policies agree exactly"), so a pack left unscored
    does not degrade the batch, it silently corrects nothing. Failing the run loudly is
    the honest answer; a shorter deadline is not.
    """

    def __init__(
        self,
        policy: Policy[Any],
        *,
        params: SamplingParams | None = None,
        timeout: float | None = None,
        services: ExecutionServices | None = None,
    ) -> None:
        if not policy.provides_exact_tokens:
            raise ValueError("Token scoring requires an exact-token policy")
        self.services = services or ExecutionServices()
        self._policy = policy
        self._timeout = timeout
        # prompt_logprobs=0 -> just the actual token's logprob; max_tokens=1 as vLLM
        # rejects 0 (the one sampled token is discarded, it doesn't affect the prompt).
        self._params = params or SamplingParams(
            prompt_logprobs=0, max_tokens=1, temperature=0.0
        )

    async def score(
        self, tokens: tuple[TokenId, ...], trainable: tuple[bool, ...]
    ) -> tuple[float, ...]:
        if len(trainable) != len(tokens):
            raise ValueError("scoring mask must align with tokens")
        result = await self._generate(tokens)
        lps = result.prompt_logprobs
        if len(lps) != len(tokens):
            raise ValueError(
                "scorer backend returned no aligned prompt_logprobs "
                + f"({len(lps)} for {len(tokens)} tokens); it does not support scoring"
            )
        values: list[float] = []
        for value, keep in zip(lps, trainable, strict=True):
            if keep and (value is None or not math.isfinite(value)):
                raise ValueError("missing or non-finite actor prompt logprob")
            values.append(
                float(value) if value is not None and math.isfinite(value) else 0.0
            )
        return tuple(values)

    async def _generate(self, tokens: tuple[TokenId, ...]) -> GenerateResult:
        """The teacher-forced pass, bounded by ``timeout`` when one is configured.

        The unbounded path is the call itself rather than ``wait_for(..., None)``: one
        pass per unscored region of every node, and a caller that asked for no timeout
        should not pay a task wrapper per pack to be told so.
        """
        if self._timeout is None:
            return await self.services.generate(
                self._policy,
                PreparedPrompt(tokens, self._policy.format, (), ()),
                self._params,
                token_input=True,
            )
        try:
            return await asyncio.wait_for(
                self.services.generate(
                    self._policy,
                    PreparedPrompt(tokens, self._policy.format, (), ()),
                    self._params,
                    token_input=True,
                ),
                self._timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"anchor scoring exceeded timeout={self._timeout}s "
                + f"(prompt_tokens={len(tokens)}); the importance ratios this pass was "
                + "computing cannot be filled in, so the run fails rather than "
                + "exporting "
                + "edges whose correction silently reads as 1"
            ) from exc


async def reevaluate(
    rollout: AnyRollout, *, version: str, scorer: PolicyScorer
) -> None:
    """Fill ``logprobs[version]`` on every turn that lacks it, in place.

    Idempotent per turn: a region whose turns all carry ``version`` is not re-scored, so
    a second call with a new ``version`` adds a key without touching existing ones, and
    a repeat call with the same one is free. Mutates the turns' logprob dicts, which is
    safe across forks -- an action's logprobs under a policy are branch-independent.

    "Free" literally: a log that already carries ``version`` on every turn returns
    before packing anything. Turns are shared objects across forked checkpoints, so one
    leaf's pass fills every ancestor's turns, and a caller walking a whole tree
    (:func:`~focus.loop.rescore_anchor`) then pays one packing pass per
    *unscored* log instead of one per node.
    """
    turns = rollout.turns
    if all(not turn.actor or version in turn.logprobs for turn in turns):
        return  # every region would be skipped below; do not pay to pack them first
    for seq in regions(turns):
        if not seq.actor:
            continue
        if all(version in turns[i].logprobs for i, _, _ in seq.spans):
            continue
        if not seq.tokens:
            # A pack of turns that were conditioned on nothing and generated nothing
            # -- in a runner's log, the marker turn a silent fold leaves. There is no
            # sequence to teacher-force -- an empty prompt is what a backend rejects,
            # not what it scores -- and the answer is known: every span in such a pack
            # is zero-width, so the column each turn gets is `()`.
            for index, _, _ in seq.spans:
                turns[index].logprobs.setdefault(version, ())
            continue
        column = await scorer.score(seq.tokens, seq.trainable)
        if len(column) != len(seq.tokens) or any(
            keep and not math.isfinite(value)
            for value, keep in zip(column, seq.trainable, strict=True)
        ):
            raise ValueError("scorer returned misaligned or non-finite actor logprobs")
        for index, start, end in seq.spans:
            turns[index].logprobs.setdefault(version, tuple(column[start:end]))


__all__ = ["PolicyScorer", "TokenScorer", "reevaluate"]
