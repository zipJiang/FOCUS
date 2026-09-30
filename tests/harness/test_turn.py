from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from focus.codec import ChatCodec
from focus.generation import (
    GenerateResult,
    Policy,
    SamplingParams,
)
from focus.harness import (
    TokenScorer,
    Turn,
    logprob,
    reevaluate,
    regions,
    trainable_logprobs,
)
from tests.helpers import (
    ConstantScorer,
    FakeChatTokenizer,
    ScriptedGenerator,
    StalledGenerator,
    complete,
    rollout,
    toy_runner,
    trainable_text,
    turn,
)

# -- fakes ---------------------------------------------------------------------------


class PromptLogprobGenerator(Policy[Any]):
    """A Policy that teacher-forces: returns per-prompt-token logprobs.

    The first prompt token has no preceding context (``None``), mirroring vLLM.
    """

    def __init__(self, value: float = -0.3, *, empty: bool = False) -> None:
        self._value = value
        self._empty = empty  # emulate a backend with no prompt-logprob support

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        prefix = tuple(prefix_tokens)
        pls: list[float | None] = (
            [] if self._empty else [None, *([self._value] * (len(prefix) - 1))]
        )
        return GenerateResult(
            text="x", tokens=(ord("x"),), prefix_tokens=prefix, prompt_logprobs=pls
        )


def _runs(mask: tuple[bool, ...]) -> int:
    """Count maximal runs of True -- i.e. the number of generated turns in a region."""
    return sum(1 for i, m in enumerate(mask) if m and (i == 0 or not mask[i - 1]))


# -- the log is turns; the region is derived -----------------------------------------


@pytest.mark.asyncio
async def test_each_generate_is_one_turn_and_they_pack_into_one_region() -> None:
    result = await complete(toy_runner(ScriptedGenerator(["go", "go", "stop"])), "hi")

    assert len(result.turns) == 3  # one record per generate, not one per region
    assert [t.transition is not None for t in result.turns] == [True] * 3
    packed = regions(result.turns)
    assert len(packed) == 1  # ...and an append-only run is still one training sequence
    assert _runs(packed[0].trainable) == 3  # one generated span per turn
    assert len(packed[0].trainable) == len(packed[0].tokens)
    # exactly the generated content is trainable (role markers / observations are not)
    assert trainable_text(result) == "gogostop"


@pytest.mark.asyncio
async def test_a_region_interleaves_base_generation_and_observations() -> None:
    # a region is append-only: a masked-out base prefix, then alternating generated
    # (mask 1) and observation (mask 0) spans -- so any turn boundary is a valid branch.
    result = await complete(toy_runner(ScriptedGenerator(["go", "stop"])), "hi")

    mask = regions(result.turns)[0].trainable
    starts = [i for i, m in enumerate(mask) if m and (i == 0 or not mask[i - 1])]
    assert len(starts) == 2  # one generated span per turn ("go", "stop")
    assert not any(mask[: starts[0]])  # the base prefix (seed) is fully masked out
    assert any(mask[starts[0] : starts[1]])  # first turn's tokens precede the second
    # the terminal 'stop' turn appends no observation -> the region ends trainable
    assert mask[-1]


@pytest.mark.asyncio
async def test_behavior_logprobs_captured_per_turn_and_aligned() -> None:
    result = await complete(
        toy_runner(ScriptedGenerator(["go", "stop"], logprobs=-0.1)), "hi"
    )

    for t in result.turns:
        assert len(t.logprobs["policy"]) == len(t.tokens)  # aligned to the completion
        assert all(v == -0.1 for v in t.logprobs["policy"])
    # sequence logprob = sum over generated tokens ("go" + "stop" = 6 of them)
    assert logprob(result.turns, "policy") == pytest.approx(-0.1 * 6)
    assert trainable_logprobs(result.turns, "policy") == [-0.1] * 6


@pytest.mark.asyncio
async def test_the_packed_region_zeroes_the_mask_and_keeps_the_completions() -> None:
    """0.0 at an observation is `p = 1`; the mask says which positions are real."""
    result = await complete(
        toy_runner(ScriptedGenerator(["go", "stop"], logprobs=-0.1)), "hi"
    )

    seq = regions(result.turns)[0]
    column = seq.logprobs["policy"]
    assert len(column) == len(seq.tokens)
    assert all(
        (v == -0.1) == keep for v, keep in zip(column, seq.trainable, strict=True)
    )


@pytest.mark.asyncio
async def test_logprobs_absent_when_the_backend_returns_none() -> None:
    result = await complete(toy_runner(ScriptedGenerator(["go", "stop"])), "hi")

    assert all(t.logprobs == {} for t in result.turns)  # opt-in: nothing stored
    assert regions(result.turns)[0].logprobs == {}  # ...and nothing invented for a pack
    assert trainable_text(result)  # but the tokens/mask are still present


@pytest.mark.asyncio
async def test_an_empty_completion_is_scored_vacuously_not_left_unscored() -> None:
    """Nothing generated is a zero contribution, not a missing channel."""
    result = await complete(
        toy_runner(ScriptedGenerator(["", "stop"], logprobs=-0.5)), "hi"
    )

    assert result.turns[0].tokens == ()
    assert result.turns[0].logprobs["policy"] == ()
    # the strict reading is what an importance ratio uses: it must not raise here
    assert logprob(result.turns, "policy") == pytest.approx(-0.5 * 4)
    assert set(regions(result.turns)[0].logprobs) == {"policy"}


@pytest.mark.asyncio
async def test_reward_and_done_match_transition_semantics() -> None:
    result = await complete(toy_runner(ScriptedGenerator(["go", "go", "stop"])), "hi")

    assert result.done
    last = result.turns[-1].transition
    assert last is not None
    assert last.done
    assert result.reward == 1.0  # only the terminal 'stop' transition rewards
    assert result.reward_outcome == 1.0
    assert result.reward_step == 0.0


# -- the strict / lenient pair over a turn run ---------------------------------------


def _scored(*columns: tuple[float, ...]) -> tuple[Turn[None], ...]:
    """A chaining run of turns -- each prefix extends the last, so they pack as one."""
    turns: list[Turn[None]] = []
    prefix: tuple[int, ...] = (0,)
    for column in columns:
        tokens = tuple(range(len(prefix), len(prefix) + len(column)))
        turns.append(Turn(prefix=prefix, tokens=tokens, logprobs={"policy": column}))
        prefix = (*prefix, *tokens)
    return tuple(turns)


def test_trainable_logprobs_reads_every_turn_in_order() -> None:
    turns = _scored((-0.1, -0.2), (-0.5,))

    assert trainable_logprobs(turns, "policy") == [-0.1, -0.2, -0.5]
    assert logprob(turns, "policy") == pytest.approx(-0.8)
    assert trainable_logprobs(turns, "nobody") == []  # the lenient reading


def test_logprob_refuses_a_version_only_some_turns_carry() -> None:
    """Half a channel would sum to a plausible number missing a whole turn's tokens."""
    turns = _scored((-0.1, -0.2), (-0.5,))
    turns[0].logprobs["anchor"] = (-0.3, -0.3)

    with pytest.raises(KeyError, match="anchor"):
        logprob(turns, "anchor")
    assert trainable_logprobs(turns, "anchor") == [-0.3, -0.3]  # still lenient

    turns[1].logprobs["anchor"] = (-0.3,)
    assert logprob(turns, "anchor") == pytest.approx(-0.9)


def test_a_pack_carries_only_the_versions_every_one_of_its_turns_has() -> None:
    """A fabricated 0.0 over an unscored turn would read as `p = 1` -- certainty."""
    turns = _scored((-0.1,), (-0.5,))
    turns[0].logprobs["anchor"] = (-0.3,)

    seq = regions(turns)[0]
    assert set(seq.logprobs) == {"policy"}  # anchor is on one turn only: not the pack's

    turns[1].logprobs["anchor"] = (-0.4,)
    assert set(regions(turns)[0].logprobs) == {"anchor", "policy"}


# -- re-scoring: pack to score, turns to store ---------------------------------------


@pytest.mark.asyncio
async def test_reevaluate_fills_a_second_version_without_clobbering() -> None:
    result = await complete(
        toy_runner(ScriptedGenerator(["go", "stop"], logprobs=-0.1)), "hi"
    )
    scorer = ConstantScorer(-0.5)

    await reevaluate(result, version="anchor", scorer=scorer)

    assert all(set(t.logprobs) == {"policy", "anchor"} for t in result.turns)
    assert logprob(result.turns, "policy") == pytest.approx(-0.1 * 6)
    assert logprob(result.turns, "anchor") == pytest.approx(-0.5 * 6)

    # idempotent: a second pass for the same version does no new scoring
    calls = scorer.calls
    await reevaluate(result, version="anchor", scorer=scorer)
    assert scorer.calls == calls


@pytest.mark.asyncio
async def test_reevaluate_writes_a_pack_score_back_onto_its_own_turns() -> None:
    """Packs are derived fresh every call, so a score kept on one would not survive."""
    turns = _scored((-0.1, -0.2), (-0.5,))

    await reevaluate(rollout(*turns), version="anchor", scorer=ConstantScorer(-0.5))

    assert [t.logprobs["anchor"] for t in turns] == [(-0.5, -0.5), (-0.5,)]
    assert logprob(turns, "anchor") == pytest.approx(-1.5)
    # and the derived pack now agrees, because it is rebuilt from the turns
    assert regions(turns)[0].logprobs["anchor"].count(-0.5) == 3


@pytest.mark.asyncio
async def test_reevaluate_scores_a_split_run_pack_by_pack() -> None:
    """A template rewrite splits the run; each pack is its own teacher-forced pass."""
    turns: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3, 4)),
        Turn(prefix=(9, 8), tokens=(7,)),  # a miss: not an extension of the first pack
    )
    assert len(regions(turns)) == 2
    scorer = ConstantScorer(-0.5)

    await reevaluate(rollout(*turns), version="anchor", scorer=scorer)

    assert scorer.calls == 2
    assert logprob(turns, "anchor") == pytest.approx(-0.5 * 3)


@pytest.mark.asyncio
async def test_reevaluate_answers_a_tokenless_pack_without_a_backend_call() -> None:
    """A turn conditioned on nothing and generating nothing packs to no sequence.

    There is nothing to teacher-force -- an empty prompt is what a backend rejects, not
    what it scores -- and the answer is already known, so the channel is filled with the
    empty column the turn's own length asks for and the strict reader stops raising.
    """
    turns: tuple[Turn[None], ...] = (Turn(prefix=(), tokens=()),)
    scorer = ConstantScorer(-0.5)

    await reevaluate(rollout(*turns), version="anchor", scorer=scorer)

    assert scorer.calls == 0
    assert turns[0].logprobs["anchor"] == ()
    assert logprob(turns, "anchor") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_generator_scorer_aligns_prompt_logprobs() -> None:
    scorer = TokenScorer(PromptLogprobGenerator(-0.3))
    tokens = (10, 11, 12, 13)
    out = await scorer.score(tokens, (False, True, True, False))
    assert len(out) == len(tokens)  # aligned to the sequence
    assert out[0] == 0.0  # the first token's None mapped to 0.0
    assert out[1:] == (-0.3, -0.3, -0.3)


@pytest.mark.asyncio
async def test_generator_scorer_rejects_a_backend_without_prompt_logprobs() -> None:
    scorer = TokenScorer(PromptLogprobGenerator(empty=True))
    with pytest.raises(ValueError, match="does not support scoring"):
        await scorer.score((1, 2, 3), (True, True, True))


@pytest.mark.asyncio
async def test_a_stalled_rescoring_backend_is_bounded_by_the_scorer_timeout() -> None:
    """Nothing else bounds a teacher-forced pass, and the whole prompt waits behind it.

    ``rescore_anchor`` runs after the last search pass with every exported record of the
    prompt behind it, so one swallowed request holds the training step. The bound raises
    rather than degrading: an unfilled anchor channel is not a coarser correction but a
    silent ``exp(0) = 1``.
    """
    gen = StalledGenerator()
    scorer = TokenScorer(gen, timeout=0.01)

    with pytest.raises(TimeoutError, match=r"anchor scoring exceeded timeout=0\.01s"):
        await scorer.score((1, 2, 3), (True, True, True))

    assert gen.cancelled == 1  # the request was cancelled, not left running


@pytest.mark.asyncio
async def test_an_unset_scorer_timeout_waits_as_long_as_the_backend_does() -> None:
    """The default is the unbounded call itself, not `wait_for(..., None)`."""
    scorer = TokenScorer(PromptLogprobGenerator(-0.3))  # no timeout
    assert await scorer.score((10, 11), (False, True)) == (0.0, -0.3)


@pytest.mark.asyncio
async def test_reevaluate_with_generator_scorer_fills_a_version() -> None:
    result = await complete(
        toy_runner(ScriptedGenerator(["go", "stop"], logprobs=-0.1)), "hi"
    )
    await reevaluate(
        result, version="anchor", scorer=TokenScorer(PromptLogprobGenerator(-0.3))
    )
    assert all(set(t.logprobs) == {"policy", "anchor"} for t in result.turns)
    assert all(len(t.logprobs["anchor"]) == len(t.tokens) for t in result.turns)


def test_the_codec_round_trips_the_fake_tokenizer() -> None:
    codec = ChatCodec(FakeChatTokenizer())
    assert codec.decode(codec.encode("hi")) == "hi"


def test_a_misspelled_tag_is_a_type_error_not_an_empty_filter() -> None:
    """The tag vocabulary is closed (``TurnTag``), and the type checker is what says so.

    A typo used to cost nothing at runtime and everything downstream: ``tags=("tsak",)``
    matches no turn, so every reader sees a rollout that did no work -- an empty edge, a
    zero entropy score, a sample with no tokens. The ``type: ignore`` below *is* the
    assertion: ``warn_unused_ignores`` is on, so this line fails ``mypy`` the moment the
    call stops being an error.

    Spelled as a list because a one-element tuple literal is inferred as ``tuple[str]``
    against a ``Collection`` parameter however its members are spelled -- so
    ``("task",)`` does not type-check either, and :data:`~focus.harness.TASK`
    is the value to pass there. (A list literal reports the offending member itself,
    hence ``list-item`` rather than ``arg-type``.)
    """
    rs = rollout(turn([-1.0]), turn([-2.0], tag="fold"))

    assert len(rs.tagged(["task"])) == 1
    assert len(rs.tagged(["task", "fold"])) == 2
    assert rs.tagged(["tsak"]) == ()  # type: ignore[list-item]
