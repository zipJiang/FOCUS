"""Seeded randomized invariants over the data pipeline: log -> regions -> records.

Every other test in this suite pins one hand-written shape. These pin the *relations*
that must hold for any shape at all -- a token is trained on exactly once, a span
indexes back to the turn that generated it, an edge is a tail slice, a record's spans
are the regions of its edge -- over a few hundred randomly generated turn logs each.

The generator (:func:`_random_log`) is the whole point: it produces the cases nobody
writes by hand -- a prefix that chains, one the chat template rewrote *shorter*, a
completion that came back empty, a policy that scored some turns and not others, folds
interleaved with task work. Everything is seeded through :class:`random.Random`, so a
failure names a seed and a case index and reproduces exactly; there is no hypothesis
dependency and nothing here is slow (the whole module is a fraction of a second).

Assertions are written against an *independent* statement of the invariant -- the
concatenation of the turns' own tokens, a hand-rolled discounted sum -- rather than by
calling the function under test a second way, so a wrong implementation cannot satisfy
them by being consistently wrong.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator, Sequence
from dataclasses import replace

import pytest

from focus.export import to_samples
from focus.harness import (
    RolloutState,
    StepResult,
    Turn,
    TurnTag,
    regions,
)
from focus.harness.packing import PackedSequence
from focus.harness.turn import logprob, trainable_logprobs
from focus.preparation import prepare_samples
from focus.preparation.records import _spans_of
from focus.scheduler import SchedulerState, SchedulerView
from focus.values import RewardConfig
from tests.estimators import FixedWeight, RootReturn
from tests.helpers import rollout, turn
from tests.trees import stamped_child
from tests.trees import training_root_state as root_state

#: One test item per seed, ``CASES`` random logs inside each -- ~200 cases per
#: invariant, at a fraction of the per-item cost of parametrizing 200 times.
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
CASES = 25

#: Ids a *prompt* is made of (a rendered message, an observation). Disjoint from the
#: generated ids below, so "did this masked position come from a completion?" is a
#: question about the token itself rather than about bookkeeping.
_FILLER = (1, 499)
#: Where generated ids start. Every generated token in a log is unique, which is what
#: makes "appears exactly once as a trainable position" checkable by identity.
_GENERATED = 1000


# -- the generator -------------------------------------------------------------------


def _random_prefix(
    rng: random.Random,
    packed: tuple[int, ...],
    marker: Iterator[int],
    p_chain: float,
) -> tuple[int, ...]:
    """The ids the engine was conditioned on: a cache hit, a rewrite, or a restart.

    ``packed`` is what the running pack holds -- which is always the previous turn's
    ``prefix + tokens``, since that is what ``regions`` carries forward whether or not
    it flushed. So the three branches here are exactly the three cases packing has to
    tell apart.

    The fourth is degenerate and belongs here for exactly that reason: *nothing* was
    conditioned on, so the pack's first trainable position is its first position. The
    runner cannot produce it (a prefix is a full render), but a log assembled by other
    means can -- and ``_random_log`` pairs it with an empty completion often enough to
    reach the turn that packs to no tokens at all.
    """
    if rng.random() < 0.08:
        return ()
    if packed and rng.random() < p_chain:
        # the ordinary case: everything so far, plus whatever the env said back
        return packed + tuple(rng.randint(*_FILLER) for _ in range(rng.randint(0, 3)))
    if len(packed) > 1 and rng.random() < 0.5:
        # the template rewrote history *shorter* (Qwen3+ stripping <think> from an
        # earlier assistant turn): a cache miss whose prefix restates generations
        return packed[: rng.randint(1, len(packed) - 1)]
    # a prompt the pack cannot possibly extend: the marker id appears nowhere else, so
    # no accident of the random filler can make this chain
    return (next(marker), *(rng.randint(*_FILLER) for _ in range(rng.randint(0, 3))))


def _random_column(
    rng: random.Random, width: int, p_misaligned: float
) -> tuple[float, ...] | None:
    """Behavior logprobs over a completion of ``width`` tokens -- sometimes absent.

    Missing from one turn in four, as when a backend was asked for none.
    ``p_misaligned`` emits a column that does *not* line up with the completion, which
    only a defensive reader can survive: ``regions`` must refuse to count such a turn as
    scored rather than packing a ragged row.
    """
    if rng.random() < 0.25:
        return None
    length = width
    if rng.random() < p_misaligned:
        length = width + 1 if width == 0 else width - 1
    return tuple(round(-rng.random(), 4) for _ in range(length))


def _random_log(
    rng: random.Random,
    *,
    max_turns: int = 6,
    p_chain: float = 0.65,
    p_misaligned: float = 0.0,
    terminal: bool = False,
) -> tuple[Turn[None], ...]:
    """A turn log: chaining and non-chaining prefixes, mixed tags, empty completions.

    ``terminal`` ends the log on a task turn whose transition is ``done`` -- what a
    finished trajectory looks like, and what export requires before it will score an
    edge. Fold turns carry no transition, which is the invariant the runner maintains
    and every reward reader depends on.

    One turn in eight is *degenerate*: an empty prefix and an empty completion, the
    only shape that packs to no tokens at all. Drawn as a shape of its own rather than
    left to the product of two rare draws, because it is the case every reader states
    over ``tokens`` gets wrong -- a pack whose tag, whose spans and whose very existence
    can only be read off its spans.
    """
    generated = itertools.count(_GENERATED)
    marker = itertools.count(_FILLER[1] + 1)
    turns: list[Turn[None]] = []
    packed: tuple[int, ...] = ()
    for _ in range(rng.randint(1, max_turns)):
        degenerate = rng.random() < 0.125
        prefix = () if degenerate else _random_prefix(rng, packed, marker, p_chain)
        tokens = (
            ()
            if degenerate
            else tuple(next(generated) for _ in range(rng.choice((0, 1, 1, 2, 3))))
        )
        tag: TurnTag = "fold" if rng.random() < 0.25 else "task"
        turns.append(
            Turn(
                prefix=prefix,
                tokens=tokens,
                logprobs=_random_column(rng, len(tokens), p_misaligned),
                exact=rng.random() < 0.9,
                transition=None
                if tag == "fold"
                else StepResult(
                    next_state=None,
                    reward_outcome=float(rng.randint(0, 2)),
                    reward_step=round(rng.random(), 3),
                ),
                tag=tag,
            )
        )
        packed = prefix + tokens
    if terminal:
        turns[-1] = replace(
            turns[-1],
            tag="task",
            transition=StepResult(next_state=None, reward_outcome=1.0, done=True),
        )
    return tuple(turns)


def _generated(turns: Sequence[Turn[object]]) -> list[int]:
    """Every token the model produced, in generation order -- the independent truth."""
    return [token for t in turns for token in t.tokens]


def _trainable(seq: PackedSequence) -> list[int]:
    """The tokens a trainer takes a gradient on in one pack, in order."""
    return [t for t, keep in zip(seq.tokens, seq.trainable, strict=True) if keep]


# -- 1. regions ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_every_generated_token_is_trainable_exactly_once_and_in_order(
    seed: int,
) -> None:
    """Packing is a re-arrangement of the log, not a rewrite of it.

    The concatenated trainable positions of the packs are the concatenated completions
    of the turns -- same tokens, same order, no duplicates. A merge that trained one
    generation twice (both packs claiming it after a rewrite) or dropped one (a flush
    losing its spans) would show up here and nowhere else, because both mistakes leave
    every individual pack looking perfectly well-formed.
    """
    rng = random.Random(seed)
    for case in range(CASES):
        turns = _random_log(rng, p_misaligned=0.2)
        packs = regions(turns)

        packed = [t for seq in packs for t in _trainable(seq)]
        assert packed == _generated(turns), case
        assert len(set(packed)) == len(packed), case  # ids are unique by construction
        # a pack restates earlier work as its own masked prefix, so the packed total is
        # never smaller than what was generated -- and usually much larger
        assert sum(len(seq.tokens) for seq in packs) >= len(packed), case


@pytest.mark.parametrize("seed", SEEDS)
def test_spans_tile_the_trainable_positions_and_index_back_to_their_turn(
    seed: int,
) -> None:
    """``spans`` is the bridge from a scored region back to per-turn storage.

    Every turn appears in exactly one pack; its span is where its completion sits in
    that pack's tokens; the spans tile the trainable mask exactly (no gap, no overlap).
    ``reevaluate`` writes a scorer's column back through these offsets, so an off-by-one
    here would silently attribute one turn's logprobs to its neighbour.

    *Every* turn -- including the degenerate one that packs to no tokens. A packer that
    dropped it (a flush that returns early on an empty pack, keeping its spans) leaves
    a turn no span indexes: nothing can score it, and the strict readers then raise over
    a log that was scored end to end. So the exhaustiveness is the assertion, and the
    counter below is what keeps the generator honest about producing the case.
    """
    rng = random.Random(seed)
    hollow = 0
    for case in range(CASES):
        turns = _random_log(rng)
        packs = regions(turns)
        hollow += sum(1 for seq in packs if not seq.tokens)

        seen: list[int] = []
        for seq in packs:
            tiled: list[int] = []
            for index, start, end in seq.spans:
                assert seq.tokens[start:end] == turns[index].tokens, case
                assert all(seq.trainable[start:end]), case
                tiled.extend(range(start, end))
                seen.append(index)
            marked = [i for i, keep in enumerate(seq.trainable) if keep]
            assert tiled == marked, case  # ordered, disjoint, and exhaustive
        assert seen == list(range(len(turns))), case
    assert hollow, "no pack ever came out empty"


@pytest.mark.parametrize("seed", SEEDS)
def test_a_masked_position_never_holds_a_completion_of_its_own_pack(seed: int) -> None:
    """Inside one pack, the mask is off exactly where the model did not write.

    Not a global statement, and deliberately so: the pack *after* a cache miss restates
    earlier generations as its own masked prefix -- that is what keeps them from being
    trained on twice. What must never happen is a pack masking a generation it also
    claims, which is how a desynced mask silently drops a turn from the loss.
    """
    rng = random.Random(seed)
    for case in range(CASES):
        turns = _random_log(rng)
        for seq in regions(turns):
            mine = {t for index, _, _ in seq.spans for t in turns[index].tokens}
            masked = {
                t for t, keep in zip(seq.tokens, seq.trainable, strict=True) if not keep
            }
            assert not (masked & mine), case


@pytest.mark.parametrize("seed", SEEDS)
def test_a_pack_breaks_exactly_where_the_prefix_or_the_tag_breaks(seed: int) -> None:
    """Two turns share a pack iff the later one extends the earlier and shares its tag.

    Stated against the two facts packing is allowed to look at -- so the greedy merge is
    checked to be neither too eager (a template rewrite quietly packed as a cache hit,
    which desyncs the mask) nor too shy (an extra flush, which restates a whole prompt
    per turn and costs a trainer real tokens). Fold work never shares a sample with task
    work even where the prefixes do chain: a sample is one kind of work or the other.
    """
    rng = random.Random(seed)
    merges = splits = 0
    for case in range(CASES):
        turns = _random_log(rng)
        packs = regions(turns)
        pack_of = {index: p for p, seq in enumerate(packs) for index, _, _ in seq.spans}

        for seq in packs:
            assert {turns[i].tag for i, _, _ in seq.spans} == {seq.tag}, case
            assert seq.exact == all(turns[i].exact for i, _, _ in seq.spans), case

        for i in range(len(turns) - 1):
            held = turns[i].prefix + turns[i].tokens
            chains = turns[i + 1].prefix[: len(held)] == held
            together = turns[i + 1].tag == turns[i].tag and chains
            assert (pack_of[i + 1] == pack_of[i]) is together, (case, i)
            merges += together
            splits += not together
    assert merges, "no two turns ever shared a pack"
    assert splits, "no turn ever forced a flush"


@pytest.mark.parametrize("seed", SEEDS)
def test_a_pack_carries_logprobs_iff_every_generating_turn_in_it_did(
    seed: int,
) -> None:
    """Zero-filling an unscored turn would read as ``p = 1``: confidently wrong.

    So a pack carries a column only when every turn that generated anything carries
    one *aligned to its completion*, and the column is then exactly those turns' own
    logprobs at their own positions. A turn with an empty completion has no column to
    contribute and no vote: counting it would let one empty generate unscore
    everything packed with it.
    """
    rng = random.Random(seed)
    dropped = kept = 0
    for case in range(CASES):
        turns = _random_log(rng, p_misaligned=0.2)
        for seq in regions(turns):
            expected = all(
                turns[i].logprobs is not None
                and len(turns[i].logprobs or ()) == end - start
                for i, start, end in seq.spans
                if end > start
            )
            assert (seq.logprobs is not None) == expected, case
            kept += expected
            dropped += not expected

            if seq.logprobs is not None:
                column = seq.logprobs
                assert len(column) == len(seq.tokens), case
                flat = [column[i] for i, keep in enumerate(seq.trainable) if keep]
                assert flat == [
                    lp
                    for i, start, end in seq.spans
                    if end > start
                    for lp in turns[i].logprobs or ()
                ], case
                # the masked positions carry 0.0 -- a value no reader counts
                assert all(
                    column[i] == 0.0 for i, keep in enumerate(seq.trainable) if not keep
                ), case
    assert kept, "no pack ever carried a column"
    assert dropped, "no pack ever had to drop one"


# -- 2. RolloutState: the edge and the readers over it -------------------------------


def _cut(
    rng: random.Random, turns: tuple[Turn[None], ...]
) -> tuple[RolloutState[None], RolloutState[None], tuple[Turn[None], ...]]:
    """A parent checkpoint, the child that continued it, and the edge between them."""
    k = rng.randint(0, len(turns))
    return rollout(*turns[:k]), rollout(*turns), turns[k:]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_edge_is_the_tail_slice_and_its_logprobs_are_the_edges(seed: int) -> None:
    """What happened on this branch is a difference against where it branched.

    The log only grows by append and a fork shares it by reference, so the edge is the
    tail past the parent's length -- and ``generated_since`` is nothing but that slice's
    logprobs, filtered by tag. The tag filter is why an agentic fold's summary tokens
    cannot stand in for uncertainty about the task.
    """
    rng = random.Random(seed)
    for case in range(CASES):
        parent, child, edge = _cut(rng, _random_log(rng))

        assert child.edge(parent) == edge, case
        assert child.generated_since(parent, tags=None) == [
            lp for t in edge for lp in t.logprobs or ()
        ], case
        assert child.generated_since(parent) == [
            lp for t in edge if t.tag == "task" for lp in t.logprobs or ()
        ], case
        assert child.tagged(None) == child.turns, case
        assert child.tagged(["fold"]) == tuple(
            t for t in child.turns if t.tag == "fold"
        ), case


@pytest.mark.parametrize("seed", SEEDS)
def test_an_edge_against_a_checkpoint_that_is_not_a_prefix_is_refused(
    seed: int,
) -> None:
    """Against a foreign checkpoint the arithmetic still returns turns, wrongly.

    They would be exported as an edge, scored and trained on, so this is a raise rather
    than an assert (``python -O`` strips the latter, and what it would strip is the one
    check that this slice *is* the edge).
    """
    rng = random.Random(seed)
    for _case in range(CASES):
        turns = _random_log(rng)
        child = rollout(*turns)
        k = rng.randint(0, len(turns))
        foreign = rollout(*turns[:k], turn([-9.0]))

        with pytest.raises(ValueError, match="not a prefix"):
            child.edge(foreign)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_checkpoint_counters_read_the_transitions_not_the_turns(seed: int) -> None:
    """A fold appends turns that took no env step -- and must cost nothing for it.

    ``turns_taken``, ``done`` and the reward sums are statements about the world, so
    they count transitions. ``folds`` and ``boundaries`` are statements about the
    *context*, and both are read off the one mark a fold leaves -- the tag -- so they
    are checked here against an independent reading of the same tags rather than
    against a counter the builder was told. And a discounted return is identical to the
    one the same log would have had if it had never folded, which is what keeps a fold
    from shrinking the credit of the task work after it.
    """
    rng = random.Random(seed)
    for case in range(CASES):
        turns = _random_log(rng, terminal=rng.random() < 0.5)
        rs = rollout(*turns)
        steps = [t.transition for t in turns if t.transition is not None]
        tags = [t.tag for t in turns]

        assert rs.turns_taken == len(steps) == len(rs.tagged())
        assert rs.done is bool(steps and steps[-1].done)
        assert rs.reward == pytest.approx(sum(s.reward for s in steps))
        assert rs.reward_outcome == pytest.approx(sum(s.reward_outcome for s in steps))
        assert rs.reward_step == pytest.approx(sum(s.reward_step for s in steps))
        # a fold is a *run* of fold turns: `"".join` of the tags' initials, and the
        # count is how many times "f" starts one
        assert rs.folds == len(
            [g for g in "".join(t[0] for t in tags).split("t") if g]
        ), case
        assert rs.boundaries() == tuple(
            i
            for i in range(1, len(tags))
            if tags[i - 1] == "fold" and tags[i] == "task"
        ), case

        config = RewardConfig(gamma=rng.choice((1.0, 0.9)), beta_step=0.5)
        assert config.configured_return(turns) == pytest.approx(
            config.configured_return([t for t in turns if t.tag != "fold"])
        ), case


# -- 3. turn-level readers -----------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_the_turn_readers_are_the_per_turn_concatenation_and_sum(seed: int) -> None:
    """Storage is per turn, so both readings are stated over the turns themselves.

    ``trainable_logprobs`` has no mask to apply -- a turn stores logprobs over its
    completion only -- so there is no ``0.0``-at-an-observation to confuse with a real
    logprob of ``p = 1``; and ``logprob`` is its sum wherever it does not raise.
    """
    rng = random.Random(seed)
    for case in range(CASES):
        turns = _random_log(rng)

        flat = trainable_logprobs(turns)
        assert flat == [lp for t in turns for lp in t.logprobs or ()]
        if all(t.logprobs is not None for t in turns):
            assert logprob(turns) == pytest.approx(sum(flat)), case
        else:
            with pytest.raises(ValueError, match="logprobs"):
                logprob(turns)


# -- 4. export -----------------------------------------------------------------------


def _random_tree(rng: random.Random, config=None) -> SchedulerState[RolloutState[None]]:
    """A root plus a few stamped children and chains, each extending its parent's log.

    What successive search passes leave behind: branches off the root and
    step lines under them, every edge stamped with the provenance export reads.
    """
    root_log = tuple(
        replace(
            t,
            exact=True,
            logprobs=(-0.5,) * len(t.tokens) if t.logprobs is None else t.logprobs,
        )
        for t in _random_log(rng, max_turns=2)
    )
    state: SchedulerState[RolloutState[None]] = root_state(
        rollout(*root_log), width=8, reward_config=config
    )
    for _ in range(rng.randint(1, 3)):
        parent_id, log = 0, root_log
        for _ in range(rng.randint(1, 2)):
            log = log + tuple(
                replace(
                    t,
                    exact=True,
                    logprobs=(
                        (-0.5,) * len(t.tokens) if t.logprobs is None else t.logprobs
                    ),
                )
                for t in _random_log(rng, max_turns=3, terminal=rng.random() < 0.75)
            )
            child = stamped_child(state, parent_id, rollout(*log))
            if child.payload.done:
                break  # a finished episode has nothing to branch from
            parent_id = child.id
    return state


def _finished_ids(state: SchedulerState[RolloutState[None]]) -> set[int]:
    """Ids some finished path runs through -- the edges export is allowed to score.

    A statistic about a prefix is measured over a path that *ended*, because the outcome
    reward lands on the episode-ending transition alone.
    """
    finished: set[int] = set()
    for node in state.nodes.values():
        stack = [node.id]
        while stack:
            current = state.nodes[stack.pop()]
            if current.payload.done:
                finished.add(node.id)
                break
            stack.extend(current.children)
    return finished


def _exportable_ids(state: SchedulerState[RolloutState[None]]) -> set[int]:
    """The finished ids whose edge holds a token at all -- what export may write.

    An edge every one of whose turns was conditioned on nothing and generated nothing
    packs to no tokens, and a training sample with no tokens is not a coarse record but
    an empty one, so export drops it. Stated here over the turns themselves (any prefix
    or any completion) rather than by packing them a second way.
    """
    return {
        node_id
        for node_id in set(state.nodes) - {state.root_id}
        if state.nodes[node_id].payload.done or state.nodes[node_id].payload.forkable
        if any(
            turn.prefix or turn.tokens
            for turn in state.nodes[node_id].payload.edge(
                state.nodes[_parent_of(state, node_id)].payload
            )
        )
    }


def _parent_of(state: SchedulerState[RolloutState[None]], node_id: int) -> int:
    parent_id = state.nodes[node_id].parent_id
    assert parent_id is not None
    return parent_id


def _refined(
    state: SchedulerState[RolloutState[None]], node_id: int, config: RewardConfig
) -> tuple[float, float | None]:
    """``(V~, G-bar)`` at one checkpoint, spelled out from the paper's recursion.

    Every checkpoint in these trees carries a network prior of zero.
    """
    node = state.nodes[node_id]
    if node.payload.done:
        return 0.0, None
    prior = node.critic
    assert prior is not None
    if not node.children:
        return prior, None
    backups = []
    for child_id in node.children:
        child = state.nodes[child_id]
        start = len(node.payload.turns)
        duration = sum(t.transition is not None for t in child.payload.turns[start:])
        successor, _ = _refined(state, child_id, config)
        backups.append(
            _discounted(child.payload.turns, start, config)
            + config.gamma**duration * successor
        )
    n = len(backups)
    mean = sum(backups) / n
    kappa = config.value_prior_strength
    return (kappa * prior + n * mean) / (kappa + n), mean


def _discounted(turns: Sequence[Turn[None]], start: int, config: RewardConfig) -> float:
    """The return of the suffix from ``start``, spelled out rather than called."""
    total, discount = 0.0, 1.0
    for t in turns[start:]:
        if t.transition is None:
            continue  # a fold acts on no world: no reward, and no discount step
        total += discount * (
            t.transition.reward_outcome + config.beta_step * t.transition.reward_step
        )
        discount *= config.gamma
    return total


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("estimator", [FixedWeight(0.25), RootReturn()])
def test_a_record_is_the_regions_of_its_edge_and_the_return_over_it(
    seed: int, estimator
) -> None:
    """One record per finished, stamped edge; its spans are that edge's regions.

    The fork's shared history is excluded structurally -- the edge is a slice, and its
    regions are derived from the slice -- rather than by comparing what the parent
    already had. ``edge_tokens`` sums the mask over every span because how many regions
    an edge breaks into is a fact about the chat template, not about the work: a trainer
    normalising per sample would otherwise pay a four-way split four times over.
    """
    rng = random.Random(seed)
    for case in range(CASES):
        config = RewardConfig(
            gamma=rng.choice((1.0, 0.9)), beta_step=rng.choice((1.0, 0.5))
        )
        state = _random_tree(rng, config)
        view = SchedulerView(state)

        result = prepare_samples(
            view._state,
            estimator=estimator,
            reward_config=config,
        )

        expected = sorted(_exportable_ids(state))
        assert len({r.node_id for r in result.critic}) == len(result.critic), case
        assert sorted(r.node_id for r in result.actor) == sorted(expected), case

        every = result.actor
        for record in every:
            node = state.nodes[record.node_id]
            assert node.parent_id is not None
            parent = state.nodes[node.parent_id]
            edge = node.payload.edge(parent.payload)
            assert record.spans == _spans_of(edge), case
            assert record.edge_tokens == len(_generated(edge)), case

        expanded = {
            node.id
            for node in state.nodes.values()
            if node.children and not node.payload.done
        }
        assert {r.node_id for r in result.critic} == expanded, case
        for critic in result.critic:
            node = state.nodes[critic.node_id]
            _, branch_mean = _refined(state, node.id, config)
            assert critic.target == pytest.approx(branch_mean), case
            assert critic.context == node.value_context

        for record in result.actor:
            assert record.group_id == state.nodes[record.node_id].parent_id, case
            if record.estimator == "test_fixed":
                assert record.weight == pytest.approx(0.25), case
                assert dict(record.diagnostics) == {}, case
            else:
                # the return from the *root's* prefix, against a root nothing scored:
                # these roots carry turns of their own, and rewards banked before the
                # search began belong to no edge in it
                assert record.weight == pytest.approx(
                    _discounted(
                        state.nodes[record.node_id].payload.turns,
                        len(state.nodes[0].payload.turns),
                        config,
                    )
                ), case
                assert record.diagnostics["advantage"] == pytest.approx(
                    record.weight
                ), case


@pytest.mark.parametrize("seed", SEEDS)
def test_flattening_gives_one_sample_per_span_that_says_where_it_sits(
    seed: int,
) -> None:
    """A trainer sees a flat list, so every sample repeats its edge's identity and size.

    ``span_index`` / ``span_count`` say which piece of the edge this is, the three
    columns stay aligned to each other (a loss mask shorter than its tokens is a silent
    misweighting, not a crash), and the lane says which head the scalar trains: the
    actor's is the weight its estimator wrote, folded with importance; the critic's is
    the return.
    """
    rng = random.Random(seed)
    for case in range(CASES):
        state = _random_tree(rng)
        result = prepare_samples(
            state,
            estimator=RootReturn(),
            reward_config=RewardConfig(),
        )

        samples = to_samples(result, group_index=3)

        every = result.actor
        assert len(samples) == sum(len(r.spans) for r in every) + len(result.critic), (
            case
        )
        lanes = [s["metadata"]["lane"] for s in samples]
        assert lanes == ["actor"] * sum(len(r.spans) for r in result.actor) + [
            "critic"
        ] * len(result.critic), case

        for sample in samples:
            meta = sample["metadata"]
            if meta["lane"] == "critic":
                assert "context" in sample
                assert "target" in sample
                continue
            assert len(sample["loss_mask"]) == len(sample["tokens"]), case
            if sample["logprobs"]:
                assert len(sample["logprobs"]) == len(sample["tokens"]), case
            assert 0 <= meta["span_index"] < meta["span_count"], case
            assert sample["group_index"] == 3
            assert meta["tag"] in ("task", "fold")

        for record in result.actor:
            mine = [
                s
                for s in samples
                if s["metadata"]["node_id"] == record.node_id
                and s["metadata"]["lane"] == "actor"
            ]
            assert [s["metadata"]["span_index"] for s in mine] == list(
                range(len(record.spans))
            ), case
            assert {s["metadata"]["span_count"] for s in mine} == {len(record.spans)}
            assert {s["metadata"]["edge_tokens"] for s in mine} == {record.edge_tokens}
            assert all(s["reward"] == pytest.approx(record.weight) for s in mine), case
            assert all(s["metadata"]["estimator"] == "test_root_return" for s in mine)

        for critic in result.critic:
            mine = [
                s
                for s in samples
                if s["metadata"]["node_id"] == critic.node_id
                and s["metadata"]["lane"] == "critic"
            ]
            assert all(s["target"] == pytest.approx(critic.target) for s in mine), case
