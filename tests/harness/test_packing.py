"""Packing: the fewest prefix-stable regions a run of turns merges into."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from focus.codec import ChatCodec, Message
from focus.generation import PolicyFormat
from focus.generation.parsing.base import TextActionParser
from focus.generation.policy import Policy
from focus.harness import Turn, regions
from focus.harness.runner import Runner
from tests.helpers import (
    FakeChatTokenizer,
    ScriptedGenerator,
    ToyEnv,
    complete,
    toy_runner,
)


class RewritingTokenizer(FakeChatTokenizer):
    """A template that drops ``</think>`` reasoning from *earlier* assistant turns.

    This is Qwen3.5's real behaviour: reasoning survives only on assistant turns after
    the last ``user`` message, so appending one rewrites the render behind us -- and
    makes it *shorter*, which is the case a mask can desync on silently. Packing is the
    only thing that cares, so it lives here rather than in ``tests.helpers``.
    """

    def apply_chat_template(
        self,
        conversation: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        return_tensors: str | None,
    ) -> list[int]:
        last_user = max(
            (i for i, m in enumerate(conversation) if m["role"] == "user"),
            default=-1,
        )
        ids: list[int] = []
        for i, m in enumerate(conversation):
            content = m["content"]
            if m["role"] == "assistant" and i < last_user and "</think>" in content:
                content = content.split("</think>")[-1]
            ids.append(ord(m["role"][0]))
            ids.extend(ord(c) for c in content)
        if add_generation_prompt:
            ids.append(ord("a"))
        return ids


def test_regions_merge_on_a_cache_hit() -> None:
    hit: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3,), logprobs=(-0.1,)),
        Turn(prefix=(1, 2, 3, 4), tokens=(5,), logprobs=(-0.2,)),
    )

    packed = regions(hit)

    assert len(packed) == 1
    assert packed[0].tokens == (1, 2, 3, 4, 5)
    assert packed[0].trainable == (False, False, True, False, True)
    assert packed[0].exact
    assert packed[0].tag == "task"
    assert packed[0].logprobs == (0.0, 0.0, -0.1, 0.0, -0.2)


def test_regions_split_on_a_miss_and_restate_the_prefix_masked() -> None:
    miss: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3,)),
        Turn(prefix=(9, 8), tokens=(7,)),
        Turn(prefix=(9, 8, 7, 6), tokens=(5,)),
    )

    split = regions(miss)

    assert len(split) == 2
    assert split[0].tokens == (1, 2, 3)
    assert split[1].tokens == (9, 8, 7, 6, 5)
    # the second pack restates the first's generation *masked*, so nothing trains twice
    assert split[1].trainable == (False, False, True, False, True)


def test_spans_locate_each_turns_completion_inside_its_pack() -> None:
    """The bridge between per-region scoring and per-turn storage."""
    turns: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3, 4)),
        Turn(prefix=(1, 2, 3, 4, 9), tokens=(5,)),
        Turn(prefix=(0,), tokens=(6,)),  # a miss: starts a second pack
    )

    packed = regions(turns)

    assert [seq.spans for seq in packed] == [((0, 2, 4), (1, 5, 6)), ((2, 1, 2),)]
    for seq in packed:
        for index, start, end in seq.spans:
            assert seq.tokens[start:end] == turns[index].tokens
            assert all(seq.trainable[start:end])


def test_a_pack_never_mixes_task_and_fold_turns() -> None:
    """Even when the prefixes chain: a sample is one kind of work or the other."""
    chaining: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3,)),
        Turn(prefix=(1, 2, 3), tokens=(4,), tag="fold"),
        Turn(prefix=(1, 2, 3, 4), tokens=(5,)),
    )

    packed = regions(chaining)

    assert [seq.tag for seq in packed] == ["task", "fold", "task"]
    assert [seq.tokens for seq in packed] == [(1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5)]


def test_a_version_missing_from_one_turn_is_missing_from_the_pack() -> None:
    """Padding it with 0.0 would claim `p = 1` over tokens nothing scored."""
    partial: tuple[Turn[None], ...] = (
        Turn(prefix=(1,), tokens=(2,), logprobs=(-1.0,)),
        Turn(prefix=(1, 2), tokens=(3,)),
    )

    assert regions(partial)[0].logprobs is None


def test_a_misaligned_logprob_column_is_not_packed() -> None:
    """A backend that returned the wrong count is an absence, not a shifted column."""
    bad: tuple[Turn[None], ...] = (Turn(prefix=(1,), tokens=(2, 3), logprobs=(-1.0,)),)

    assert regions(bad)[0].logprobs is None


def test_a_turn_that_generated_nothing_does_not_unscore_the_pack() -> None:
    """An empty completion carries no column, so it cannot be the missing one."""
    turns: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3, 4), logprobs=(-0.1, -0.2)),
        Turn(prefix=(1, 2, 3, 4, 5), tokens=()),  # the backend returned nothing
        Turn(prefix=(1, 2, 3, 4, 5, 6), tokens=(7,), logprobs=(-0.3,)),
    )

    packed = regions(turns)

    assert len(packed) == 1
    assert packed[0].tokens == (1, 2, 3, 4, 5, 6, 7)
    assert packed[0].logprobs == (0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3)


def test_a_turn_whose_prefix_is_exactly_the_pack_adds_no_masked_delta() -> None:
    """The zero-delta extension: the env said nothing between the two generations.

    A pack is built as its last turn's ``prefix + tokens``, and the masked run between
    two generations is what the prefix grew by. When it grew by nothing -- no
    observation, no template text -- that run is empty, and the second generation has
    to start exactly where the first ended rather than one position either side of it.
    """
    turns: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3,), logprobs=(-0.1,)),
        Turn(prefix=(1, 2, 3), tokens=(4,), logprobs=(-0.2,)),
    )

    packed = regions(turns)

    assert len(packed) == 1
    assert packed[0].tokens == (1, 2, 3, 4)
    assert packed[0].trainable == (False, False, True, True)
    assert packed[0].spans == ((0, 2, 3), (1, 3, 4))
    assert packed[0].logprobs == (0.0, 0.0, -0.1, -0.2)


def test_a_prefix_shorter_than_the_pack_is_a_miss_even_where_it_agrees() -> None:
    """A shortened re-render is a cache miss, not a partial hit.

    The template dropped everything after ``(1, 2)``, so the new prefix agrees with the
    pack as far as it goes and is still not a continuation of it -- packing on that
    agreement would drop the tokens the pack holds past it, and mask the tokens the
    first turn generated as if the second had been conditioned on them.
    """
    turns: tuple[Turn[None], ...] = (
        Turn(prefix=(1, 2), tokens=(3, 4)),
        Turn(prefix=(1, 2), tokens=(5,)),
    )

    packed = regions(turns)

    assert len(packed) == 2
    assert packed[0].tokens == (1, 2, 3, 4)
    assert packed[1].tokens == (1, 2, 5)
    assert packed[1].trainable == (False, False, True)


def test_packing_an_empty_log_yields_nothing() -> None:
    assert regions(()) == ()
    assert regions(()) == ()


def test_a_turn_that_holds_nothing_at_all_still_gets_a_pack() -> None:
    """Nothing conditioned on and nothing generated: a pack with no tokens.

    Not dropped, because ``spans`` is what indexes the turn log: a turn no pack mentions
    is a turn ``reevaluate`` can never reach, so the strict ``logprob`` would raise over
    a log that had been scored end to end.
    """
    packed = regions((Turn(prefix=(), tokens=(), tag="fold"),))

    assert len(packed) == 1
    assert packed[0].tokens == ()
    assert packed[0].trainable == ()
    assert packed[0].tag == "fold"
    assert packed[0].spans == ((0, 0, 0),)


def test_an_empty_pack_does_not_carry_its_spans_into_the_next_one() -> None:
    """The flush has to clear what it emitted, even when it emitted no tokens.

    Everything chains onto an empty prefix, so nothing but the tag can separate these
    two -- and a flush that returned early on an empty pack kept the fold turn's
    zero-width span, handing it to the task pack that followed under the task's tag.
    """
    turns: tuple[Turn[None], ...] = (
        Turn(prefix=(), tokens=(), tag="fold"),
        Turn(prefix=(1, 2), tokens=(3,), tag="task"),
    )

    packed = regions(turns)

    assert [seq.tag for seq in packed] == ["fold", "task"]
    assert [seq.spans for seq in packed] == [((0, 0, 0),), ((1, 2, 3),)]
    assert [seq.tokens for seq in packed] == [(), (1, 2, 3)]


def test_inexactness_taints_only_its_own_pack() -> None:
    turns: tuple[Turn[None], ...] = (
        Turn(prefix=(1,), tokens=(2,), exact=False),
        Turn(prefix=(9,), tokens=(8,)),  # a miss, so a fresh pack -- and still exact
    )

    assert [seq.exact for seq in regions(turns)] == [False, True]


# -- against the real runner, through a real(ish) template ----------------------------


@pytest.mark.asyncio
async def test_an_append_only_render_keeps_the_run_in_one_exact_region() -> None:
    result = await complete(toy_runner(ScriptedGenerator(["go", "stop"])), "hi")

    packed = regions(result.turns)
    assert len(packed) == 1
    assert packed[0].exact
    assert len(packed[0].trainable) == len(packed[0].tokens)
    assert sum(packed[0].trainable) == len("go") + len("stop")
    # a prefix is a whole render of a conversation that always has a system and a user
    # turn, so the runner cannot produce the tokenless turn packing has to allow for
    assert all(turn.prefix for turn in result.turns)


@pytest.mark.asyncio
async def test_a_template_that_strips_think_splits_the_run_into_exact_packs() -> None:
    """Qwen3.5 drops reasoning from assistant turns before the last user turn.

    That makes the render *shorter* behind us -- the case that used to desync a mask
    silently. Here it is an ordinary cache miss: two packs, both exact, and the second
    restates the first generation without its reasoning.
    """
    first, second = "reasoning</think>a", "more</think>b"
    runner: Runner[Any, str] = Runner(
        policy=Policy(
            format=PolicyFormat(
                "custom", ChatCodec(RewritingTokenizer()), TextActionParser()
            ),
            complete=(ScriptedGenerator([first, second])).agenerate_tokens,
            exact=(ScriptedGenerator([first, second])).provides_exact_tokens,
            native=(ScriptedGenerator([first, second])).provides_native_tool_calls,
            version="policy",
        ),
        env=ToyEnv(),
        system_prompt="s",
        max_steps=2,
        finish_prompt=None,
    )

    result = await complete(runner, "q", max_steps=2)

    packed = regions(result.turns)
    assert len(packed) == 2
    assert all(seq.exact for seq in packed)
    codec = ChatCodec(RewritingTokenizer())
    generated = [
        codec.decode(
            [t for t, keep in zip(seq.tokens, seq.trainable, strict=True) if keep]
        )
        for seq in packed
    ]
    assert generated == [first, second]
    # the second pack's masked prefix carries the *rewritten* first turn
    masked = zip(packed[1].tokens, packed[1].trainable, strict=True)
    conditioning = codec.decode([t for t, keep in masked if not keep])
    assert "reasoning" not in conditioning
    assert "a" in conditioning


def test_the_fake_template_chains_turns_the_way_a_real_one_does() -> None:
    """The fixtures' load-bearing property: a generation marker ends the prefix."""
    codec = ChatCodec(FakeChatTokenizer())
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert codec.render(messages)[-1] == ord("a")
    assert codec.render(messages, add_generation_prompt=False)[-1] != ord("a")
