"""Direct branch counts, recursive credit, and distinct critic targets."""

import pickle

import pytest

from focus import (
    DirectBranchTd,
)
from focus.preparation import prepare_samples
from focus.values import RewardConfig, tree_values
from tests.helpers import turn
from tests.test_preparation_estimators import score, setup
from tests.trees import search_payload, stamped_child


def example(kappa=1):
    state, view, _, config = setup(RewardConfig(value_prior_strength=kappa))
    score(view.root, 0.5)
    b = stamped_child(state, 0, search_payload(turn([-1.0])))
    score(b, 0.5)
    for reward in (0, 1):
        stamped_child(
            state,
            b.id,
            search_payload(*b.payload.turns, turn([-1.0], reward=reward, done=True)),
        )
    stamped_child(state, 0, search_payload(turn([-1.0], reward=1, done=True)))
    return state, view, config, b


def test_recursive_example_and_descendant_multiplicity():
    state, view, config, b = example()
    values = tree_values(view, config)
    assert values[b.id].refined == 0.5
    assert values[0].backup == 0.75
    assert values[0].refined == pytest.approx(2 / 3)
    assert values[0].branches == values[b.id].branches == 2
    for reward in (0, 1):
        stamped_child(
            state,
            b.id,
            search_payload(*b.payload.turns, turn([-1.0], reward=reward, done=True)),
        )
    values = tree_values(view, config)
    assert values[0].backup == 0.75
    assert values[0].refined == pytest.approx(2 / 3)
    assert values[0].branches == 2
    assert values[b.id].branches == 4


def test_critic_mean_is_not_actor_posterior_and_tree_is_unchanged():
    state, _view, config, b = example(kappa=1)
    before = pickle.dumps(state)
    estimator = DirectBranchTd()
    batch = prepare_samples(state, estimator=estimator, reward_config=config)
    assert pickle.dumps(state) == before
    assert estimator.name == "direct_branch_td"
    targets = {r.node_id: r for r in batch.critic}
    assert set(targets) == {0, b.id}  # No terminal remaining-value supervision.
    assert targets[0].target == 0.75
    assert targets[0].diagnostics["refined_value"] == pytest.approx(2 / 3)
    assert targets[b.id].target == 0.5
    edge = next(r for r in batch.actor if r.node_id == b.id)
    assert edge.weight == pytest.approx(0.5 - 2 / 3)
    assert pickle.loads(pickle.dumps(batch)) == batch


def test_bootstrap_leaf_without_target_and_fold_duration_discount():
    state, view, _, config = setup(RewardConfig(gamma=0.5, value_prior_strength=1))
    score(view.root, 0.2)
    child = stamped_child(
        state,
        0,
        search_payload(
            turn([-1.0], reward=1), turn([-1.0], tag="fold"), turn([-1.0], reward=2)
        ),
    )
    score(child, 0.8)
    values = tree_values(view, config)
    assert values[child.id].refined == 0.8
    assert values[child.id].backup is None
    assert values[0].backup == pytest.approx(1 + 0.5 * 2 + 0.5**2 * 0.8)
    batch = prepare_samples(
        state,
        estimator=DirectBranchTd(),
        reward_config=config,
    )
    assert [r.node_id for r in batch.critic] == [0]
    assert batch.actor[0].diagnostics["duration"] == 2


@pytest.mark.parametrize("prior", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_successor_prior_rejected(prior):
    state, view, _, config = setup()
    child = stamped_child(state, 0, search_payload(turn([-1.0])))
    child.critic = None
    if prior is not None:
        score(child, prior)
    values = tree_values(view, config)
    # the gap reaches the root's backup, so neither has a refined value...
    assert values[child.id].refined is None
    assert values[0].refined is None
    assert values[0].backup is None
    # ...the actor recipe refuses the tree, and the critic skips what depends on it
    with pytest.raises(ValueError, match="finite network value"):
        prepare_samples(state, reward_config=config)
    assert prepare_samples(state, estimator=None, reward_config=config).critic == ()
