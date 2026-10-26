"""What an edge hands a trainer: regions of the turn slice, tagged, plus the return.

Two halves, and the second is what the first is *for*: building the records off a tree,
then flattening them into the one span-per-sample list a trainer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from focus.export import format_edges, format_tree, to_samples
from focus.harness import PackedSequence, TurnTag
from focus.preparation import Advantage, prepare_samples
from focus.preparation.records import (
    ESTIMATOR_KEY,
    LANE_KEY,
    ActorSample,
    Batch,
    CriticSample,
    edge_spans,
)
from focus.scheduler import SchedulerView
from focus.values import RewardConfig
from tests.estimators import FixedWeight, RootReturn
from tests.helpers import rollout, turn
from tests.trees import stamped_child
from tests.trees import training_root_state as root_state


def _tree(parent, child, config=None):
    """One stamped edge: the view over it, the fork it leaves, and the child."""
    state = root_state(parent, width=4, reward_config=config)
    node = stamped_child(state, 0, child)
    return SchedulerView(state), state.nodes[0], node


def test_edge_spans_are_the_regions_of_the_edge_and_carry_their_tags() -> None:
    """A fork's shared history is excluded structurally -- the edge is a tail slice."""
    shared = turn([-1.0])
    parent_state = rollout(shared)
    child_state = rollout(
        shared, turn([-2.0], tag="fold"), turn([-3.0], done=True, reward=1.0)
    )
    _, parent, child = _tree(parent_state, child_state)

    spans = edge_spans(child, parent)

    assert [s.tag for s in spans] == ["fold", "task"]  # the shared turn is not on it
    assert [s.logprobs for s in spans] == [(-2.0,), (-3.0,)]


def test_the_critic_target_counts_task_transitions_only() -> None:
    """A fold takes no env step, so it neither rewards nor advances the discount."""
    shared = turn([-1.0], reward=0.5)
    parent_state = rollout(shared)
    child_state = rollout(
        shared, turn([-2.0], tag="fold"), turn([-3.0], done=True, reward=1.0)
    )
    view, _, _ = _tree(parent_state, child_state, RewardConfig(gamma=0.5))

    result = prepare_samples(
        view._state,
        estimator=None,
        reward_config=RewardConfig(gamma=0.5),
    )

    # only the terminal task turn is on the edge: the parent's 0.5 and the fold are
    # out, and the fold takes no discount step; a terminal has no target of its own
    assert [r.target for r in result.critic] == pytest.approx([1.0])
    assert result.critic[0].context == "root context"


def test_an_estimator_fills_actor_records_over_the_same_spans() -> None:
    shared = turn([-1.0])
    parent_state = rollout(shared)
    child_state = rollout(shared, turn([-3.0], done=True, reward=1.0))
    view, _, child = _tree(parent_state, child_state)

    result = prepare_samples(
        view._state,
        estimator=RootReturn(),
        reward_config=RewardConfig(),
    )

    assert [r.node_id for r in result.actor] == [child.id]
    assert result.actor[0].spans == edge_spans(child, view.root)
    # what the estimator wrote is what export carries: the whole return, against a
    # root with no baseline
    assert result.actor[0].weight == pytest.approx(1.0)
    assert result.actor[0].diagnostics == {"advantage": 1.0, "baseline": 0.0}


def test_an_estimators_diagnostics_are_frozen_on_the_way_out() -> None:
    """A record's account of itself is settled once the export has written it.

    ``assign`` writes an ordinary dict -- often one it built for a whole group of
    siblings -- and a record that kept a reference to it could be rewritten afterwards
    by whatever still held the other end.
    """
    shared = turn([-1.0])
    view, _, _ = _tree(rollout(shared), rollout(shared, turn([-3.0], done=True)))

    class Leaky(RootReturn):
        name = "leaky"
        shared_dict: dict[str, float] = {"advantage": 1.0}

        def advantages(self, edges, values):
            return [Advantage(0.0, self.shared_dict) for _ in edges]

    leaky = Leaky()
    result = prepare_samples(
        view._state,
        estimator=leaky,
        reward_config=RewardConfig(),
    )

    leaky.shared_dict["advantage"] = 99.0
    assert result.actor[0].diagnostics == {"advantage": 1.0}
    with pytest.raises(TypeError):
        result.actor[0].diagnostics["advantage"] = 2.0  # type: ignore[index]


def test_an_unnamed_estimator_is_labelled_by_its_class() -> None:
    """The name says which recipe scored the edge; a subclass need not spell one."""
    shared = turn([-1.0])
    view, _, _ = _tree(rollout(shared), rollout(shared, turn([-3.0], done=True)))

    class Unnamed(RootReturn):
        pass

    result = prepare_samples(
        view._state,
        estimator=Unnamed(),
        reward_config=RewardConfig(),
    )
    assert {r.estimator for r in result.actor} == {"Unnamed"}


def test_repreparing_an_estimator_creates_independent_records() -> None:
    shared = turn([-1.0])
    view, _, child = _tree(rollout(shared), rollout(shared, turn([-3.0], done=True)))
    estimator = FixedWeight(0.5)
    first = prepare_samples(
        view._state,
        estimator=estimator,
        reward_config=RewardConfig(),
    )
    second = prepare_samples(
        view._state,
        estimator=estimator,
        reward_config=RewardConfig(),
    )
    assert [r.node_id for r in first.actor] == [child.id]
    assert first.actor == second.actor
    assert first.actor[0] is not second.actor[0]


def test_a_record_reports_the_trainable_size_of_the_whole_edge() -> None:
    """Regions split an edge for the template's sake; credit is still per edge."""
    shared = turn([-1.0])
    parent_state = rollout(shared)
    child_state = rollout(
        shared, turn([-2.0], tag="fold"), turn([-3.0], done=True, reward=1.0)
    )
    view, _, _ = _tree(parent_state, child_state)

    result = prepare_samples(
        view._state,
        estimator=RootReturn(),
        reward_config=RewardConfig(),
    )
    record = result.actor[0]

    assert len(record.spans) == 2  # the split is a template artifact ...
    assert record.edge_tokens == sum(sum(s.trainable) for s in record.spans)
    # ... so the two spans still describe one unit of credit, sized once
    assert not hasattr(result.critic[0], "edge_tokens")


def test_the_result_carries_the_search_stats_and_its_failures() -> None:
    """Copied off the view at export, not referenced: a result outlives its tree, and a
    trainer that had to keep the tree alive to read five numbers would hold every
    rollout's tokens for a metric."""
    shared = turn([-1.0])
    state = root_state(rollout(shared), width=4)
    stamped_child(state, 0, rollout(shared, turn([-3.0], done=True, reward=1.0)))
    state.stats["rollouts"] = 3.0
    state.record_failure(1, RuntimeError("dead endpoint"))

    result = prepare_samples(state, estimator=None, reward_config=RewardConfig())

    assert result.stats["rollouts"] == 3.0
    assert result.stats["failures"] == 1.0
    assert [node_id for node_id, _ in result.failures] == [1]
    assert "dead endpoint" in result.failures[0][1]

    state.stats["rollouts"] += 1.0  # the tree may keep growing; the export did not
    assert result.stats["rollouts"] == 3.0


# -- flattening a result into trainer samples ----------------------------------------


def _span(n: int = 3, *, tag: TurnTag = "task") -> PackedSequence:
    return PackedSequence(
        tokens=tuple(range(n)),
        trainable=tuple(i % 2 == 1 for i in range(n)),
        logprobs=tuple(-0.1 * i for i in range(n)),
        tag=tag,
    )


def _result() -> Batch:
    """One GAE edge, a child-relative sibling pair, and a critic target."""
    return Batch(
        actor=(
            ActorSample(
                node_id=1,
                spans=(_span(), _span(2)),
                group_id=0,
                estimator="direct_branch_td",
                weight=-1.5,
                diagnostics={"advantage": -1.5, "baseline": 0.25},
                search_pass=0,
            ),
            ActorSample(
                node_id=5,
                spans=(_span(),),
                group_id=7,
                estimator="test_custom",
                weight=0.4,
                diagnostics={"advantage": 0.4, "confidence": 1.0},
                search_pass=1,
            ),
            ActorSample(
                node_id=6,
                spans=(_span(),),
                group_id=7,
                estimator="test_custom",
                weight=-0.4,
                diagnostics={"advantage": -0.4, "confidence": 1.0},
                search_pass=1,
            ),
        ),
        critic=(
            CriticSample(
                node_id=3,
                context="critic context",
                target=0.25,
            ),
        ),
    )


def test_every_sample_says_which_head_it_trains_and_what_scored_it() -> None:
    samples = to_samples(_result(), group_index=4)

    assert {s["metadata"][LANE_KEY] for s in samples} == {"actor", "critic"}
    actor = [s for s in samples if s["metadata"][LANE_KEY] == "actor"]
    assert {s["metadata"][ESTIMATOR_KEY] for s in actor} == {
        "direct_branch_td",
        "test_custom",
    }
    assert all(s["group_index"] == 4 for s in samples)


def test_child_relative_siblings_become_one_sample_each() -> None:
    samples = [
        s
        for s in to_samples(_result(), 0)
        if s["metadata"].get(ESTIMATOR_KEY) == "test_custom"
    ]

    assert len(samples) == 2
    assert [s["reward"] for s in samples] == [0.4, -0.4]
    assert all(s["metadata"]["group_id"] == 7 for s in samples)

    # the recipe's own account of each number, under one metadata key rather than as
    # columns a trainer might mistake for coefficients
    assert [s["metadata"]["diagnostics"]["advantage"] for s in samples] == [0.4, -0.4]
    assert all(s["metadata"]["diagnostics"]["confidence"] == 1.0 for s in samples)
    assert all("confidence" not in s and "advantage" not in s for s in samples)


def test_a_multi_region_edge_becomes_one_sample_per_region() -> None:
    samples = [s for s in to_samples(_result(), 0) if s["metadata"].get("node_id") == 1]

    assert len(samples) == 2
    assert all(s["reward"] == pytest.approx(-1.5) for s in samples)


def test_the_loss_mask_stays_aligned_to_the_tokens() -> None:
    for sample in to_samples(_result(), 0):
        if sample["metadata"]["lane"] == "critic":
            assert sample["context"] == "critic context"
            continue
        assert len(sample["loss_mask"]) == len(sample["tokens"])
        assert not sample["logprobs"] or len(sample["logprobs"]) == len(
            sample["tokens"]
        )


def test_a_custom_sample_factory_builds_the_trainer_s_own_type() -> None:
    @dataclass
    class FakeSample:
        tokens: list[int]
        lane: str

    def factory(*, span, reward, lane, group_index, **extra):
        del reward, group_index, extra
        return FakeSample(tokens=list(span.tokens), lane=lane)

    samples = to_samples(
        _result(),
        0,
        sample_factory=factory,
        critic_sample_factory=lambda **kwargs: FakeSample(tokens=[], lane="critic"),
    )

    assert samples
    assert all(isinstance(s, FakeSample) for s in samples)


def test_a_spans_tag_reaches_the_trainer_sample() -> None:
    """Fold tokens and task tokens are the same shape; only the tag tells them apart."""
    result = Batch(
        actor=(
            ActorSample(
                node_id=1,
                group_id=0,
                spans=(_span(), _span(2, tag="fold")),
                estimator="test_fixed",
            ),
        )
    )

    samples = to_samples(result, 0)

    assert [s["metadata"]["tag"] for s in samples] == ["task", "fold"]


def test_every_sample_carries_the_edge_it_came_from_and_that_edge_s_size() -> None:
    """Span count is a template artifact; the edge is the unit of credit."""
    samples = [s for s in to_samples(_result(), 0) if s["metadata"].get("node_id") == 1]

    # _span(3) has one trainable token, _span(2) has one: two over the whole edge
    assert [s["metadata"]["edge_tokens"] for s in samples] == [2, 2]
    assert [s["metadata"]["span_index"] for s in samples] == [0, 1]
    assert [s["metadata"]["span_count"] for s in samples] == [2, 2]


def test_the_edge_shape_lets_a_trainer_normalise_per_edge_not_per_span() -> None:
    """Two spans or one, an edge's samples must sum to the same share of the batch."""
    samples = to_samples(_result(), 0)
    split = [s for s in samples if s["metadata"].get("node_id") == 1]
    whole = [s for s in samples if s["metadata"].get("node_id") == 5]

    def share(sample: dict[str, Any]) -> float:
        return float(sum(sample["loss_mask"]) / sample["metadata"]["edge_tokens"])

    assert sum(share(s) for s in split) == pytest.approx(1.0)
    assert sum(share(s) for s in whole) == pytest.approx(1.0)
    assert len(split) > len(whole)  # ... even though it yields more samples


def test_critic_samples_export_checkpoint_inputs_once() -> None:
    samples = [s for s in to_samples(_result(), 0) if s["metadata"]["lane"] == "critic"]
    assert len(samples) == 1
    assert samples[0]["context"] == "critic context"
    assert samples[0]["target"] == 0.25
    assert "span_count" not in samples[0]["metadata"]
    assert "tokens" not in samples[0]


def test_advantage_cutoff_filters_whole_edges_and_preserves_critic_and_batch() -> None:
    result = _result()
    # Both signs and the exact boundary are excluded, across all of their spans.
    result.actor[0].weight = 0.04
    result.actor[1].weight = -0.04
    result.actor[2].weight = -0.041
    before = to_samples(result, 4)
    filtered = to_samples(result, 4, min_abs_advantage=0.04)
    assert filtered == [
        s
        for s in before
        if s["metadata"]["lane"] == "critic" or s["metadata"]["node_id"] == 6
    ]
    assert to_samples(result, 4) == before
    assert len(result.actor) == 3


def test_advantage_cutoff_ignores_diagnostics() -> None:
    result = _result()
    result.actor[0].weight = 0.03
    result.actor[0].diagnostics = {"advantage": 10.0}
    result.actor[1].weight = 0.05
    samples = to_samples(result, 0, min_abs_advantage=0.04)
    assert {
        s["metadata"]["node_id"] for s in samples if s["metadata"]["lane"] == "actor"
    } == {5, 6}


def test_filtered_edge_keeps_original_prompt_normalization() -> None:
    result = _result()
    samples = to_samples(result, 0, min_abs_advantage=0.5)
    actor = [s for s in samples if s["metadata"]["lane"] == "actor"]
    assert len(actor) == 2  # Both regions of the single retained edge.
    contribution = sum(
        s["reward"]
        * sum(s["loss_mask"])
        / (s["metadata"]["edge_tokens"] * s["metadata"]["group_edge_count"])
        for s in actor
    )
    assert contribution == pytest.approx(-1.5 / 3)
    assert all(s["metadata"]["group_edge_count"] == 3 for s in actor)


def test_zero_cutoff_and_all_filtered_actor_groups() -> None:
    result = replace(_result(), actor=(_result().actor[0],))
    result.actor[0].weight = 0.0
    assert len(to_samples(result, 0)) == 3
    samples = to_samples(result, 0, min_abs_advantage=0.0)
    assert len(samples) == 1
    assert samples[0]["metadata"]["lane"] == "critic"
    assert to_samples(replace(result, critic=()), 0, min_abs_advantage=0.0) == []
    assert to_samples(
        result,
        0,
        min_abs_advantage=0.0,
        critic_sample_factory=lambda **fields: fields["target"],
    ) == [0.25]


def test_filtered_custom_factory_receives_original_edge_count() -> None:
    received = to_samples(
        _result(),
        7,
        min_abs_advantage=0.5,
        sample_factory=lambda **fields: fields,
        critic_sample_factory=lambda **fields: fields,
    )
    actor = [s for s in received if "span" in s]
    assert len(actor) == 2
    assert [s["span_index"] for s in actor] == [0, 1]
    assert all(s["group_index"] == 7 and s["group_edge_count"] == 3 for s in actor)


@pytest.mark.parametrize("cutoff", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_advantage_cutoff_is_rejected(cutoff: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        to_samples(Batch(), 0, min_abs_advantage=cutoff)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -float("inf")])
def test_filter_does_not_hide_nonfinite_actor_weights(weight: float) -> None:
    result = _result()
    result.actor[0].weight = weight
    with pytest.raises(ValueError, match="non-finite actor advantages"):
        to_samples(result, 0, min_abs_advantage=0.04)


def test_a_result_survives_the_trip_back_from_a_worker() -> None:
    """Pickled whole, ``stats`` included -- and still unwritable on the far side.

    A ``mappingproxy`` has no pickle support at all, so a proxy on this field made the
    *whole* record unshippable at every value of ``stats``, the empty default included:
    a result could not come back from a worker process, be written to disk to replay a
    batch, or be `deepcopy`d.
    """
    import pickle

    shared = turn([-1.0])
    state = root_state(rollout(shared), width=4)
    stamped_child(state, 0, rollout(shared, turn([-3.0], done=True, reward=1.0)))
    state.stats["rollouts"] = 3.0

    for result in (
        Batch(),
        prepare_samples(
            state,
            estimator=None,
            reward_config=RewardConfig(),
        ),
    ):
        restored = pickle.loads(pickle.dumps(result))

        assert dict(restored.stats) == dict(result.stats)
        with pytest.raises(TypeError):
            restored.stats["rollouts"] = 1.0


def test_every_exported_sample_is_strict_json() -> None:
    """What a worker hands a trainer has to survive a JSON encoder that means it.

    ``json.dumps`` writes a bare ``NaN`` / ``Infinity`` by default -- syntax no other
    reader accepts -- so a batch can look serialisable right up to the moment something
    else parses it.
    """
    import json

    shared = turn([-1.0])
    state = root_state(rollout(shared), width=4)
    stamped_child(state, 0, rollout(shared, turn([-3.0], done=True, reward=1.0)))
    result = prepare_samples(
        state,
        estimator=RootReturn(),
        reward_config=RewardConfig(),
    )

    for sample in to_samples(result, 0):
        json.dumps(sample, allow_nan=False)


def test_a_non_finite_scalar_is_refused_rather_than_exported() -> None:
    """Named where it was made, not discovered as a ``nan`` loss three steps later."""
    shared = turn([-1.0])
    state = root_state(rollout(shared), width=4)
    stamped_child(state, 0, rollout(shared, turn([-3.0], done=True, reward=1.0)))
    view = SchedulerView(state)

    with pytest.raises(ValueError, match="non-finite weight"):
        prepare_samples(
            view._state,
            estimator=FixedWeight(float("inf")),
            reward_config=RewardConfig(),
        )
    with pytest.raises(ValueError, match="'test_fixed'"):
        prepare_samples(
            view._state,
            estimator=FixedWeight(float("nan")),
            reward_config=RewardConfig(),
        )


def test_a_non_finite_diagnostic_is_refused_too() -> None:
    """Nothing trains on it, and it is still refused.

    A diagnostic goes out over the same JSON encoder as the weight -- where a bare
    ``NaN`` is syntax no strict reader accepts -- and it is also the thing a run is
    debugged by: a broken baseline that reached only the diagnostics is still an
    estimator whose arithmetic came apart.
    """
    shared = turn([-1.0])
    state = root_state(rollout(shared), width=4)
    stamped_child(state, 0, rollout(shared, turn([-3.0], done=True, reward=1.0)))

    class Unexplainable(RootReturn):
        name = "unexplainable"

        def advantages(self, edges, values):
            return [Advantage(1.0, {"baseline": float("nan")}) for _ in edges]

    with pytest.raises(ValueError, match="non-finite diagnostic 'baseline'"):
        prepare_samples(
            state,
            estimator=Unexplainable(),
            reward_config=RewardConfig(),
        )


def test_an_actor_record_ships_with_its_diagnostics() -> None:
    """The frozen proxy is the live record's, not the wire format's -- as ``stats``."""
    import pickle

    shared = turn([-1.0])
    state = root_state(rollout(shared), width=4)
    stamped_child(state, 0, rollout(shared, turn([-3.0], done=True, reward=1.0)))
    result = prepare_samples(
        state,
        estimator=RootReturn(),
        reward_config=RewardConfig(),
    )

    restored = pickle.loads(pickle.dumps(result))

    assert dict(restored.actor[0].diagnostics) == dict(result.actor[0].diagnostics)
    with pytest.raises(TypeError):
        restored.actor[0].diagnostics["advantage"] = 1.0


# -- the edge report -------------------------------------------------------------------


def test_the_edge_report_names_every_tag_an_edge_actually_carries() -> None:
    """A pack never mixes tags; an *edge* is several packs and may.

    The report read ``spans[0].tag`` and called it the edge's, so an edge whose fold and
    task turns landed in different packs printed as pure ``task`` -- the one thing the
    column exists to tell you apart.
    """
    result = Batch(
        actor=(
            ActorSample(
                node_id=1,
                group_id=0,
                spans=(_span(tag="task"), _span(tag="fold")),
                estimator="test_fixed",
            ),
        ),
    )

    assert "tag=task+fold" in format_edges(result)


def test_the_edge_report_survives_a_record_with_no_spans() -> None:
    """``build_result`` never makes one, but a hand-built record can.

    A reporter that raises on the record someone is trying to look at is the reporter
    failing at its only job, so an empty span tuple prints ``-`` like an absent stamp.
    """
    result = Batch(
        actor=(
            ActorSample(
                node_id=2,
                group_id=0,
                spans=(),
                estimator="test_fixed",
            ),
        ),
    )

    line = format_edges(result)

    assert "tag=-" in line
    assert "tokens=   0" in line


# -- the tree report ------------------------------------------------------------------


def _searched_tree():
    """Two siblings off the root, one of them carried a turn deeper by a later pass.

    The shape the report has to say something about: a node that ended the episode, a
    node that did not but has a finished path through it, and one scored by the value
    head while its sibling was not.
    """
    shared = turn([-1.0])
    state = root_state(rollout(shared), width=4)
    first = stamped_child(
        state, 0, rollout(shared, turn([-2.0], done=True, reward=1.0))
    )
    second = stamped_child(state, 0, rollout(shared, turn([-3.0], reward=0.25)))
    stamped_child(
        state,
        second.id,
        rollout(shared, turn([-3.0], reward=0.25), turn([-4.0], done=True, reward=0.5)),
        search_pass=1,
    )
    for node in state.nodes.values():
        node.critic = None
        node.value_context = None
    state.nodes[0].critic = 0.25
    first.critic = 0.75
    return SchedulerView(state)


def test_the_tree_report_prints_the_search_and_joins_the_weights_onto_it() -> None:
    """Pinned in full: the columns are a claim about the *tree*, not about the records.

    An edge report can only speak for the edges that became records; this is where a
    node the search created and nothing scored -- the root, an interior node, a dead
    expansion's parent -- is still visible, with its flags and its critic estimate.
    """
    view = _searched_tree()
    result = prepare_samples(
        view._state,
        estimator=RootReturn(),
        reward_config=RewardConfig(),
    )

    assert format_tree(view, result) == "\n".join(
        [
            "node   0  parent=  -  --  reward= 0.000  critic[value=+0.250]",
            (
                "  node   1  parent=  0  -d  reward= 1.000  critic[value=+0.750]  "
                + "test_root_return=+0.7500"
            ),
            "  node   2  parent=  0  --  reward= 0.250",
            "    node   3  parent=  2  -d  reward= 0.750  test_root_return=+0.5000",
        ]
    )


def test_the_tree_report_reads_a_tree_nobody_has_scored() -> None:
    """The tree is a deliverable before any estimator has an opinion about it."""
    lines = format_tree(_searched_tree()).splitlines()

    assert [line.strip()[:8] for line in lines] == [
        "node   0",
        "node   1",
        "node   2",
        "node   3",
    ]
    assert not any("trajectory_return" in line for line in lines)


def test_mixed_custom_export_requires_both_factories():
    with pytest.raises(ValueError, match="critic_sample_factory"):
        to_samples(_result(), 0, sample_factory=lambda **kwargs: kwargs)
    with pytest.raises(ValueError, match="sample_factory"):
        to_samples(_result(), 0, critic_sample_factory=lambda **kwargs: kwargs)
    samples = to_samples(
        _result(),
        2,
        sample_factory=lambda **kwargs: ("actor", kwargs["reward"]),
        critic_sample_factory=lambda **kwargs: (
            "critic",
            kwargs["context"],
            kwargs["target"],
        ),
    )
    assert samples[-1] == ("critic", "critic context", 0.25)


def test_malformed_compaction_retained_below_cutoff_with_exact_weights():
    from focus.harness.packing import regions

    bad = replace(
        turn([-0.5], tag="fold"),
        compaction_penalty=-0.002,
        compaction_issue="recovered_wrapper",
    )
    result = _result()
    result.actor[0].spans = regions((bad,))
    result.actor[0].weight = -0.002
    result.actor[1].weight = result.actor[2].weight = 0
    samples = to_samples(result, 0, min_abs_advantage=0.02)
    actors = [s for s in samples if s["metadata"]["lane"] == "actor"]
    assert len(actors) == 1
    assert actors[0]["reward"] == -0.002
    assert result.actor[0].spans[0].tokens == bad.prefix + bad.tokens


def test_compaction_shaping_enters_actor_and_critic_without_discount_step():
    shared = turn([-1.0], reward=0.5)
    bad = replace(
        turn([-2.0], tag="fold"), compaction_penalty=-0.01, compaction_issue="malformed"
    )
    child_state = rollout(shared, bad, turn([-3.0], done=True, reward=1))
    config = RewardConfig(gamma=0.5, beta_step=2)
    view, parent, child = _tree(rollout(shared), child_state, config)
    from focus.scheduler.core.returns import edge_return

    assert edge_return(child, parent, config) == pytest.approx(0.98)
    assert config.configured_return(child_state.turns) == pytest.approx(0.99)
    result = prepare_samples(
        view._state,
        estimator=RootReturn(),
        reward_config=config,
    )
    assert result.actor[0].weight == pytest.approx(0.98)
    assert result.critic[0].target == pytest.approx(0.98)
