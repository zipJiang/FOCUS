import pytest

from focus import FifoAllocator, SearchPass, WidthGating
from focus.config import RolloutConfig, RolloutSpec
from focus.generation import GenerateResult, Policy
from focus.generation.interfaces import finite_logprobs
from focus.harness import Runner, Turn
from focus.harness.packing import regions
from focus.harness.rescoring import reevaluate
from focus.harness.turn import logprob
from focus.integrations.slime import make_generate
from focus.loop import Runtime
from focus.protocols import RLM, Memex
from tests.estimators import FixedWeight
from tests.helpers import ConstantScorer, SubmitEnv, rollout, tool_call
from tests.protocols.test_rlm import recursive_runner


async def test_helper_likelihood_is_fixed_and_never_rescored():
    actor: Turn[None] = Turn(prefix=(1,), tokens=(2,), logprobs={"policy": (-0.2,)})
    helper: Turn[None] = Turn(
        prefix=(1, 2),
        tokens=(3,),
        actor=False,
        invocation_id="helper",
        policy_id="fixed",
        exact=False,
    )
    rs = rollout(actor, helper)
    scorer = ConstantScorer(-0.1)
    await reevaluate(rs, version="anchor", scorer=scorer)
    assert scorer.calls == 1
    assert helper.logprobs == {}
    assert logprob(rs.turns, "anchor") - logprob(rs.turns, "policy") == pytest.approx(
        0.1
    )
    assert not any(regions(rs.turns)[1].trainable)


@pytest.mark.parametrize(
    "values", [[None], [float("nan")], [float("inf")], [-0.1, -0.2]]
)
def test_completion_likelihoods_are_never_fabricated(values):
    with pytest.raises(ValueError, match="logprobs"):
        finite_logprobs(values, 1)


@pytest.mark.parametrize("columns", [{}, {"policy": ()}, {"policy": (float("nan"),)}])
def test_actor_likelihoods_are_strict(columns):
    turn: Turn[None] = Turn(prefix=(1,), tokens=(2,), logprobs=columns)
    with pytest.raises((KeyError, ValueError), match="logprobs"):
        logprob([turn], "policy")


async def test_slime_exports_rlm_memory_and_actor_children_once():
    outputs = [
        tool_call("memory_compress", indices=[1], summary="external task"),
        tool_call(
            "rlm_python",
            code=(
                "print(llm_query('actor part'))\n"
                "print(llm_query('helper part', model='fixed'))"
            ),
        ),
        "actor-child-answer",
        tool_call("submit", answer="root answer"),
    ]
    base, _ = recursive_runner(outputs)

    async def help_complete(tokens, params):
        # Even an inexact helper with no actor logprobs must remain usable.
        return GenerateResult(
            text="fixed-helper-answer",
            tokens=tuple(map(ord, "fixed-helper-answer")),
            exact_generation=False,
        )

    helper = Policy(format=base.policy.format, version="fixed", complete=help_complete)
    runner = Runner(
        policy=base.policy,
        env=SubmitEnv([]),
        system_prompt="solve",
        services=base.services,
        protocol=RLM(context=Memex(), helpers={"fixed": helper}),
    )
    generate = make_generate(
        RolloutSpec(
            config=RolloutConfig(
                max_concurrency=1,
                passes=(SearchPass(FifoAllocator(), budget=1, gating=WidthGating(1)),),
            )
        ),
        lambda _: Runtime(runner),
        estimator=FixedWeight(),
    )
    samples = await generate(object(), "external evidence")
    actor_samples = [s for s in samples if s["metadata"]["lane"] == "actor"]
    trained = "".join(
        runner.policy.decode(
            [
                token
                for token, keep in zip(s["tokens"], s["loss_mask"], strict=True)
                if keep
            ]
        )
        for s in actor_samples
    )
    assert trained == "".join(outputs)
    assert "fixed-helper-answer" not in trained
    assert any(s["metadata"]["invocation_id"] != "root" for s in actor_samples)
    assert sum(u.calls for u in runner.services.usage.values()) == 5
    await runner.services.aclose()
