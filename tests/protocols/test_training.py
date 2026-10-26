import pytest

from focus.generation.interfaces import finite_logprobs
from focus.harness import Turn
from focus.harness.packing import regions
from focus.harness.turn import logprob
from tests.helpers import rollout


def test_helper_turns_are_never_trainable():
    actor: Turn[None] = Turn(prefix=(1,), tokens=(2,), logprobs=(-0.2,))
    helper: Turn[None] = Turn(
        prefix=(1, 2),
        tokens=(3,),
        actor=False,
        invocation_id="helper",
        exact=False,
    )
    rs = rollout(actor, helper)
    assert logprob(rs.turns) == pytest.approx(-0.2)
    assert not any(regions(rs.turns)[1].trainable)


@pytest.mark.parametrize(
    "values", [[None], [float("nan")], [float("inf")], [-0.1, -0.2]]
)
def test_completion_likelihoods_are_never_fabricated(values):
    with pytest.raises(ValueError, match="logprobs"):
        finite_logprobs(values, 1)


@pytest.mark.parametrize("columns", [None, (), (float("nan"),)])
def test_actor_likelihoods_are_strict(columns):
    turn: Turn[None] = Turn(prefix=(1,), tokens=(2,), logprobs=columns)
    with pytest.raises((KeyError, ValueError), match="logprobs"):
        logprob([turn])
