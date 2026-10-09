import asyncio

from focus.codec import ChatCodec
from focus.execution import AdmissionPool, ExecutionServices, LocalInferenceExecutor
from focus.execution.budget import ResourceBudget
from focus.generation import GenerateResult, Policy, PolicyFormat
from focus.generation.parsing.hermes import HermesToolCallParser
from focus.harness import Runner
from focus.protocols import RLM, Memex
from focus.protocols.rlm import InvocationEnv
from tests.helpers import FakeChatTokenizer, tool_call


def recursive_runner(outputs, *, capacity=1, protocol=None, budget=None):
    counter, active, peak = 0, 0, 0

    async def complete(tokens, params):
        nonlocal counter, active, peak
        i = counter
        counter += 1
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02 if i == 1 else 0.001)
            text = outputs[i]
            return GenerateResult(
                text=text, tokens=tuple(map(ord, text)), logprobs=[-0.1] * len(text)
            )
        finally:
            active -= 1

    services = ExecutionServices(
        inference=LocalInferenceExecutor(AdmissionPool(capacity))
    )
    policy = Policy(
        format=PolicyFormat(
            "test-tools", ChatCodec(FakeChatTokenizer()), HermesToolCallParser()
        ),
        complete=complete,
    )
    return Runner(
        policy=policy,
        env=InvocationEnv(),
        system_prompt="solve",
        services=services,
        protocol=protocol or RLM(),
        resource_budget=budget,
    ), lambda: peak


async def test_nested_parents_complete_with_capacity_one():
    runner, peak = recursive_runner(
        [
            tool_call("rlm_python", code="print(rlm_query('nested'))"),
            tool_call("rlm_python", code="print(llm_query('leaf'))"),
            "leaf-result",
            "nested-result",
            "root-result",
        ]
    )
    root = await runner.start("secret external input")
    assert "secret external input" not in str(root.messages)
    result = await asyncio.wait_for(runner.run(root), 5)
    assert result.state == "root-result"
    assert len(result.turns) == 5
    assert sum(t.transition is not None for t in result.turns) == 1
    assert len({t.invocation_id for t in result.turns}) == 3
    assert runner.services.usage[root.episode_id].calls == 5
    assert peak() == 1
    await runner.services.aclose()


async def test_parallel_children_overlap_and_keep_input_order():
    runner, peak = recursive_runner(
        [
            tool_call("rlm_python", code="print(llm_query_batched(['one','two']))"),
            "one-result",
            "two-result",
            "root-result",
        ],
        capacity=2,
    )
    result = await runner.run_from("external")
    assert peak() == 2
    assert "['one-result', 'two-result']" in str(result.history)
    assert result.turns[1].invocation_id != result.turns[2].invocation_id
    await runner.services.aclose()


async def test_limits_and_snapshot_budget_are_explicit():
    runner, _ = recursive_runner(
        [
            tool_call("rlm_python", code="print(rlm_query('nested'))"),
            "root-result",
        ],
        protocol=RLM(max_depth=0),
        budget=ResourceBudget(model_calls=3),
    )
    root = await runner.start("external")
    result = await runner.advance(root)
    assert "depth limit" in result.messages[-1]["content"]
    assert result.budget.model_calls == 2
    assert result.fork().budget.model_calls == 2
    await runner.services.aclose()


async def test_rlm_composes_memory_and_preserves_task_completion():
    runner, _ = recursive_runner(
        [
            tool_call("memory_compress", indices=[1], summary="input is external"),
            tool_call("rlm_python", code="print(llm_query('part'))"),
            "child-result",
            "root-result",
        ],
        protocol=RLM(context=Memex()),
    )
    result = await runner.run_from("external input")
    assert result.state == "root-result"
    assert len(result.execution["memex"]) == 1
    assert len(result.turns) == 4
    assert result.transitions[-1].done
    await runner.services.aclose()
