import pickle
from dataclasses import replace

import pytest

from focus.codec import ChatCodec
from focus.execution import AdmissionPool, ExecutionServices, LocalInferenceExecutor
from focus.execution.budget import ResourceBudget
from focus.execution.monty import MontyExecutor
from focus.generation import GenerateResult, Policy, PolicyFormat, SamplingParams
from focus.generation.parsing.hermes import HermesToolCallParser
from focus.harness import Runner
from focus.harness.packing import regions
from focus.protocols import RLM
from focus.protocols.rlm import InvocationEnv
from focus.scheduler.core.execution import RolloutExpander
from tests.helpers import FakeChatTokenizer, tool_call
from tests.protocols.test_rlm import recursive_runner


class ScoredInvocation(InvocationEnv):
    async def step(self, state, action, *, workspace=None):
        result = await super().step(state, action, workspace=workspace)
        return replace(
            result, reward_outcome=float(result.done and result.next_state == "root A")
        )


async def test_fork_grandchild_restores_ancestors_budget_and_training_trace():
    generated: list[str] = []

    async def complete(tokens, params):
        prompt = "".join(map(chr, tokens))
        if not generated:
            text = tool_call(
                "rlm_python",
                code="items=[]\nitems.append(1)\na=rlm_query('child')\n"
                "print('ROOT_RESULT',len(items),a)",
            )
        elif len(generated) == 1:
            text = tool_call(
                "rlm_python",
                code="items=[]\nitems.append(1)\na=rlm_query('grandchild')\n"
                "print('CHILD_RESULT',len(items),a)",
            )
        else:
            answer = "A" if params.temperature == 0.1 else "B"
            if "ROOT_RESULT" in prompt:
                assert f"ROOT_RESULT 1 child {answer}\n" in prompt
                text = "root " + answer
            elif "CHILD_RESULT" in prompt:
                assert f"CHILD_RESULT 1 {answer}\n" in prompt
                text = "child " + answer
            else:
                text = answer
        generated.append(text)
        return GenerateResult(
            text=text, tokens=tuple(map(ord, text)), logprobs=[-0.1] * len(text)
        )

    policy = Policy(
        format=PolicyFormat(
            "test-tools", ChatCodec(FakeChatTokenizer()), HermesToolCallParser()
        ),
        complete=complete,
    )

    def make_runner():
        return Runner(
            policy=policy,
            env=ScoredInvocation(),
            system_prompt="solve",
            protocol=RLM(),
            services=ExecutionServices(
                inference=LocalInferenceExecutor(AdmissionPool(1)),
                monty=MontyExecutor(AdmissionPool(1)),
            ),
            resource_budget=ResourceBudget(model_calls=10, process_cells=4),
        )

    runner = make_runner()
    restarted = make_runner()
    try:
        checkpoint = await runner.advance(await runner.start("input"))
        checkpoint = await runner.advance(checkpoint)
        assert checkpoint.forkable
        assert len(checkpoint.turns) == 2
        assert checkpoint.budget.model_calls == 8
        child = checkpoint.execution["rlm_pending"].children[0]
        assert child.execution["rlm_pending"].children[0].turns == ()
        assert runner.services.monty.pool.stats.active == 0
        prefix = checkpoint.turns
        left = await runner.run(
            checkpoint.fork(), sampling_params=SamplingParams(temperature=0.1)
        )
        # Restore in fresh services: no live interpreter or coroutine is needed.
        restored = pickle.loads(pickle.dumps(checkpoint)).fork()
        right = await restarted.run(
            restored, sampling_params=SamplingParams(temperature=0.9)
        )
        assert (left.state, right.state) == ("root A", "root B")
        assert (left.reward, right.reward) == (1, 0)
        assert len(generated) == 8
        assert checkpoint.turns == prefix
        for endpoint in (left, right):
            assert len(endpoint.edge(checkpoint)) == 3
            assert endpoint.turns[:2] == prefix
            assert endpoint.budget.model_calls == 5
            assert endpoint.budget.process_cells == 2
            assert len(endpoint.transitions) == 1
            child = endpoint.execution["invocations"][0]
            grandchild = child.execution["invocations"][0]
            assert child.branch_id == grandchild.branch_id == endpoint.branch_id
            packed = regions(endpoint.turns)
            trained = sum(sum(p.trainable) for p in packed)
            assert trained == sum(len(t.tokens) for t in endpoint.turns)
        assert left.branch_id != right.branch_id != checkpoint.branch_id
    finally:
        await runner.services.aclose()
        await restarted.services.aclose()


async def test_search_expander_exposes_child_decisions_and_final_root_reward():
    runner, _ = recursive_runner(
        [
            tool_call("rlm_python", code="print(rlm_query('child'))"),
            tool_call("rlm_python", code="print(llm_query('leaf'))"),
            "leaf",
            "child",
            "root",
        ]
    )
    try:
        root = await runner.start("input")
        chain, _ = await RolloutExpander(runner).expand(root)
        assert [len(c.turns) for c in chain] == [1, 2, 3, 4, 5]
        assert all(c.forkable for c in chain[:-1])
        assert chain[-1].done
        previous = root
        for checkpoint in chain:
            assert len(checkpoint.edge(previous)) == 1
            previous = checkpoint
    finally:
        await runner.services.aclose()


@pytest.mark.parametrize("finish_directly", [False, True])
async def test_pending_children_drain_after_parent_step_limit(finish_directly):
    runner, _ = recursive_runner(
        [
            tool_call("rlm_python", code="print(llm_query('child'))"),
            "child-result",
            "root-result",
        ]
    )
    try:
        root = replace(await runner.start("input"), steps_remaining=1)
        if finish_directly:
            result = await runner.finish(await runner.advance(root))
        else:
            result = await runner.run(root)
        assert result.done
        assert result.truncated
        assert result.state == "root-result"
        assert "child-result" in str(result.history)
        assert len(result.turns) == 3
    finally:
        await runner.services.aclose()


async def test_parallel_children_fork_without_replaying_completed_sibling():
    runner, _ = recursive_runner(
        [
            tool_call("rlm_python", code="print(rlm_query_batched(['one','two']))"),
            "one-result",
            tool_call("rlm_python", code="x=1; print('two started')"),
            "two-A",
            "root-A",
            "two-B",
            "root-B",
        ],
        capacity=2,
    )
    try:
        pending = await runner.advance(await runner.start("input"))
        checkpoint = await runner.advance(pending)
        assert checkpoint.execution["rlm_pending"].children[0].done
        left = await runner.run(checkpoint.fork())
        right = await runner.run(checkpoint.fork())
        assert "['one-result', 'two-A']" in str(left.history)
        assert "['one-result', 'two-B']" in str(right.history)
        for result in (left, right):
            assert len(result.turns) == 5
            for effect in result.effects:
                assert (
                    result.turns[effect.turn_index].invocation_id
                    == effect.invocation_id
                )
        assert runner.services.usage[checkpoint.episode_id].calls == 7
    finally:
        await runner.services.aclose()


async def test_invalid_query_can_be_caught_inside_python():
    runner, _ = recursive_runner(
        [
            tool_call(
                "rlm_python",
                code="try:\n    rlm_query('no')\n"
                "except ValueError:\n    print('caught')",
            ),
            "root",
        ],
        protocol=RLM(max_depth=0),
    )
    try:
        result = await runner.run_from("input")
        assert "caught" in str(result.history)
        assert len(result.turns) == 2
    finally:
        await runner.services.aclose()


async def test_exhausted_cell_budget_is_model_visible_and_releases_capacity():
    runner, _ = recursive_runner(
        [
            tool_call("rlm_python", code="print(llm_query('child'))"),
            "child",
        ],
        protocol=RLM(timeout=5),
    )
    try:
        pending = await runner.advance(await runner.start("input"))
        # The budget travels in the checkpoint rather than resetting each advance.
        session = pending.execution["rlm_session"]
        assert session.suspended
        assert 0 < session.remaining < 5
        pending = replace(
            pending,
            execution={
                **pending.execution,
                "rlm_session": replace(session, remaining=0),
            },
        )
        result = await runner.advance(pending)
        assert result.truncated
        assert not result.snapshot_available
        assert "rlm_pending" not in result.execution
        assert "TimeoutError" in str(result.history)
        assert runner.services.monty.pool.stats.active == 0
        inference = runner.services.inference
        assert isinstance(inference, LocalInferenceExecutor)
        assert inference.pool.stats.active == 0
    finally:
        await runner.services.aclose()


async def test_child_generation_time_does_not_consume_cell_budget():
    runner, _ = recursive_runner(
        [
            tool_call("rlm_python", code="print(llm_query_batched(['a', 'b']))"),
            "child-a",
            "child-b",
            "root",
        ],
        capacity=2,
        # Each child generation alone outlasts the parent's Python budget.
        delay=lambda i: 1.5 if i in (1, 2) else 0.001,
        protocol=RLM(timeout=1),
    )
    try:
        result = await runner.run_from("input")
        assert not result.truncated
        assert "['child-a', 'child-b']" in str(result.history)
        assert "TimeoutError" not in str(result.history)
    finally:
        await runner.services.aclose()


async def test_delayed_effect_indices_survive_interleaved_sibling_generations():
    runner, _ = recursive_runner(
        [
            tool_call("rlm_python", code="print(rlm_query_batched(['one','two']))"),
            tool_call("rlm_python", code="print(llm_query('leaf'))"),
            tool_call("rlm_python", code="print('sibling')"),
            "leaf-answer",
            "sibling-answer",
            "child-answer",
            "root-answer",
        ],
        capacity=2,
    )
    try:
        result = await runner.run_from("input")
        assert len(result.turns) == 7
        assert [e.turn_index for e in result.effects if e.kind == "invoke"] == [2, 1, 0]
        for effect in result.effects:
            assert result.turns[effect.turn_index].invocation_id == effect.invocation_id
    finally:
        await runner.services.aclose()


async def test_legacy_python_state_rejected_before_generation():
    from focus.execution.process import ProcessSnapshot

    runner, _ = recursive_runner(["must not generate"])
    try:
        root = await runner.start("input")
        legacy = replace(
            root,
            execution={
                **root.execution,
                "rlm_session": ProcessSnapshot.variables(context="old"),
            },
        )
        with pytest.raises(ValueError, match="analysis-only"):
            await runner.advance(legacy)
        assert not runner.services.usage
    finally:
        await runner.services.aclose()
