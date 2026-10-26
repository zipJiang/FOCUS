"""Refined TD attribution, checkpoint supervision, and supported recipes."""

from __future__ import annotations

import pickle

import pytest

from focus import DirectBranchTdEstimator, GrpoEstimator, prepare_samples
from focus.allocation.evidence import RolloutLedger, evidence_of
from focus.preparation.records import ActorSample
from focus.reward.config import RewardConfig
from focus.scheduler import SchedulerView
from focus.scheduler.core.critic import VALUE_CONTEXTS_KEY
from tests.helpers import VERSION, DemoValue, turn
from tests.trees import search_payload, stamped_child
from tests.trees import training_root_state as root_state


def setup(config=None):
    config = config or RewardConfig()
    state = root_state(search_payload(), reward_config=config)
    view = SchedulerView(state)
    ledger = RolloutLedger(config)
    ledger.bind(view)
    return state, view, ledger, config


def score(node, value):
    node.critic["v-base"] = value
    node.metadata[VALUE_CONTEXTS_KEY] = {"v-base": f"checkpoint {node.id}"}


def slot(node):
    return ActorSample(
        node_id=node.id,
        group_id=node.parent_id,
        spans=(),
        reward_config_id="rc-0",
        estimator="direct_branch_td",
    )


def assign(view, config, *nodes):
    records = [slot(n) for n in nodes]
    DirectBranchTdEstimator().assign(records, view=view, reward_config=config)
    return records


def test_refined_terminal_weights_and_critic_contexts():
    state, view, _ledger, config = setup()
    score(view.root, 0.4)
    a = stamped_child(state, 0, search_payload(turn([-1.0], reward=0, done=True)))
    b = stamped_child(state, 0, search_payload(turn([-1.0], reward=1, done=True)))
    score(a, 999)
    score(b, 999)
    batch = prepare_samples(
        state,
        estimator=DirectBranchTdEstimator(),
        reward_config=config,
        behavior_version=VERSION,
    )
    assert [x.weight for x in batch.actor] == pytest.approx([-1.4 / 3, 1 - 1.4 / 3])
    # the critic regresses on the branch mean; terminals have no remaining value
    assert [x.target for x in batch.critic] == pytest.approx([0.5])
    assert [x.context for x in batch.critic] == ["checkpoint 0"]
    assert batch.actor[0].diagnostics["parent_direct_branches"] == 2
    restored = pickle.loads(pickle.dumps(state))
    again = prepare_samples(
        restored,
        estimator=DirectBranchTdEstimator(),
        reward_config=config,
        behavior_version=VERSION,
    )
    assert batch == again
    assert pickle.loads(pickle.dumps(batch)) == batch


def test_descendant_evidence_reaches_ancestors_through_recursive_backups():
    state, view, _ledger, config = setup()
    n = stamped_child(state, 0, search_payload(turn([-1.0])))
    c = stamped_child(state, n.id, search_payload(*n.payload.turns, turn([-1.0])))
    score(view.root, 0.2)
    score(n, 0.6)
    score(c, 0.5)
    # V~(n) = (0.6 + 0.5) / 2 and V~(root) = (0.2 + V~(n)) / 2
    assert assign(view, config, n)[0].weight == pytest.approx(0.55 - 0.375)
    for reward in [1, 1, 0]:
        stamped_child(
            state,
            c.id,
            search_payload(*c.payload.turns, turn([-1.0], reward=reward, done=True)),
        )
    # V~(c) = (0.5 + 2) / 4 now backs up into V~(n) and V~(root)
    assert assign(view, config, n)[0].weight == pytest.approx(0.6125 - 0.40625)
    stamped_child(
        state, n.id, search_payload(*n.payload.turns, turn([-1.0], reward=0, done=True))
    )
    refined_n = (0.6 + 0.625 + 0.0) / 3
    refined_root = (0.2 + refined_n) / 2
    assert assign(view, config, n)[0].weight == pytest.approx(refined_n - refined_root)
    assert evidence_of(view, config).observations.get(0) is None


def test_original_chain_credits_only_selected_and_new_checkpoints():
    state, _view, ledger, _config = setup()
    n = stamped_child(state, 0, search_payload(turn([-1.0])))
    score(n, 0.3)
    c = state.add_child(n.id, search_payload(*n.payload.turns, turn([-1.0])))
    leaf = state.add_child(
        c.id, search_payload(*c.payload.turns, turn([-1.0], reward=1, done=True))
    )
    from tests.trees import stamp

    for node in (c, leaf):
        stamp(node)
    ledger.record(n, [c, leaf])
    assert {k: v.returns.n for k, v in ledger.observations.items()} == {
        n.id: 1,
        c.id: 1,
        leaf.id: 1,
    }
    assert ledger.counts(leaf).returns.mean == 0


def test_duration_discount_and_fold_only_turns():
    config = RewardConfig(gamma=0.5, value_prior_strength=2)
    state, view, _ledger, config = setup(config)
    score(view.root, 0.2)
    n = stamped_child(
        state,
        0,
        search_payload(
            turn([-1.0], reward=1), turn([-1.0], tag="fold"), turn([-1.0], reward=2)
        ),
    )
    score(n, 0.8)
    record = assign(view, config, n)[0]
    edge = 1 + 0.5 * 2 + 0.5**2 * 0.8
    assert record.weight == pytest.approx(edge - (2 * 0.2 + edge) / 3)
    assert record.diagnostics["duration"] == 2


@pytest.mark.parametrize("prior", [None, float("nan"), float("inf")])
def test_missing_and_nonfinite_prior_rejected(prior):
    state, view, _ledger, config = setup()
    n = stamped_child(state, 0, search_payload(turn([-1.0], done=True)))
    view.root.critic.clear()
    if prior is not None:
        view.root.critic["v-base"] = prior
    with pytest.raises(ValueError, match="finite network value"):
        assign(view, config, n)


def test_incomplete_suffixes_are_not_evidence():
    state, view, ledger, config = setup()
    unfinished = stamped_child(state, 0, search_payload(turn([-1.0]), truncated=True))
    score(unfinished, 0.3)
    ledger.record(view.root, [unfinished])
    assert not ledger.observations
    # the critic target still backs up through the unfinished branch's prior
    critic = prepare_samples(
        state, estimator=None, reward_config=config, behavior_version=VERSION
    ).critic
    assert [r.node_id for r in critic] == [0]
    assert critic[0].target == pytest.approx(0.3)


def test_evidence_config_and_legacy_rejection():
    state, view, _ledger, config = setup()
    with pytest.raises(ValueError, match="reward configuration"):
        prepare_samples(
            state,
            estimator=None,
            reward_config=RewardConfig(gamma=0.5),
            behavior_version=VERSION,
        )
    view.root.metadata.pop("rollout_evidence")
    with pytest.raises(ValueError, match="analysis-only"):
        prepare_samples(
            state, estimator=None, reward_config=config, behavior_version=VERSION
        )


def test_grpo_allows_linear_folds_but_rejects_internal_branching():
    state, view, _ledger, config = setup()
    n = stamped_child(state, 0, search_payload(turn([-1.0])))
    a = stamped_child(
        state, n.id, search_payload(*n.payload.turns, turn([-1.0], done=True, reward=1))
    )
    b = stamped_child(state, 0, search_payload(turn([-1.0], done=True, reward=0)))
    records = [slot(x) for x in [n, a, b]]
    GrpoEstimator().assign(records, view=view, reward_config=config)
    assert records[0].weight == records[1].weight > 0 > records[2].weight
    stamped_child(
        state, n.id, search_payload(*n.payload.turns, turn([-1.0], done=True, reward=0))
    )
    with pytest.raises(ValueError, match="branching below the root"):
        GrpoEstimator().assign(records, view=view, reward_config=config)


@pytest.mark.parametrize("strength", [0.0, -1.0, float("nan"), float("inf")])
def test_prior_strength_must_be_positive_and_finite(strength):
    with pytest.raises(ValueError, match="value_prior_strength"):
        RewardConfig(value_prior_strength=strength)


async def test_critic_export_reuses_exact_custom_scoring_inputs():
    from focus.scheduler.core.critic import NodeScorer

    state, view, _ledger, config = setup()
    leaf = stamped_child(state, 0, search_payload(turn([-1.0], reward=1, done=True)))
    seen: list[str] = []

    def serialize(payload):
        context = f"custom context call {len(seen)}"
        seen.append(context)
        return context

    scorer = NodeScorer(model=DemoValue(), version="v-base", serialize=serialize)
    nodes = [view.root, leaf]
    scores = await scorer.score([n.payload for n in nodes])
    scorer.write(nodes, scores)
    batch = prepare_samples(
        state,
        estimator=DirectBranchTdEstimator(),
        reward_config=config,
        behavior_version=VERSION,
    )
    # the terminal leaf has no target; the root's is the context it was scored on
    assert [r.context for r in batch.critic] == seen[:1]
    assert len(seen) == 2  # preparation never calls the serializer again


def test_inexact_generation_still_provides_checkpoint_critic_supervision():
    from dataclasses import replace

    state, _view, _ledger, config = setup()
    generated = replace(turn([-1.0], reward=1, done=True), exact=False)
    stamped_child(state, 0, search_payload(generated))
    batch = prepare_samples(
        state, estimator=None, reward_config=config, behavior_version=VERSION
    )
    assert [r.node_id for r in batch.critic] == [0]
    with pytest.raises(ValueError, match="inexact"):
        prepare_samples(
            state,
            estimator=DirectBranchTdEstimator(),
            reward_config=config,
            behavior_version=VERSION,
        )
