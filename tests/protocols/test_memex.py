import json

from focus.generation.parsing.hermes import HermesToolCallParser
from focus.protocols import Memex
from focus.reward.config import RewardConfig
from tests.helpers import ScriptedGenerator, tool_call, toy_runner


def memory_runner(outputs, **kwargs):
    return toy_runner(
        ScriptedGenerator(outputs),
        parser=HermesToolCallParser(),
        protocol=Memex(**kwargs),
    )


async def test_archive_retrieves_exact_records_and_indices_survive_forks():
    runner = memory_runner(
        [
            tool_call("memory_compress", indices=[1], summary="short"),
            tool_call("memory_retrieve", index=0),
        ]
    )
    root = await runner.start("Exact evidence: 3.14159 — do not reconstruct me.")
    compact = await runner.advance(root)
    assert compact.context_version == 1
    assert compact.execution["memex"][0].references == (1,)
    assert root.messages[1] not in compact.messages
    assert compact.forkable
    retrieved = await runner.advance(compact.fork())
    assert json.loads(retrieved.messages[-1]["content"]) == [root.history[1]]
    assert retrieved.turns[0] == compact.turns[0]
    sibling = compact.fork()
    other, _, _ = await runner.protocol.operation(
        sibling, "memory_compress", {"indices": [0], "summary": "system"}, runner
    )
    assert len(other.execution["memex"]) == 2
    assert len(compact.execution["memex"]) == 1


async def test_invalid_archive_is_atomic():
    runner = memory_runner([tool_call("memory_compress", indices=[100], summary="bad")])
    root = await runner.start("evidence")
    result = await runner.advance(root)
    assert "memex" not in result.execution
    assert result.messages[:2] == root.messages
    assert result.context_version == 0
    assert "Invalid memory_compress" in result.messages[-1]["content"]


async def test_memory_reward_is_configurable_without_task_time():
    runner = memory_runner(
        [tool_call("memory_compress", indices=[1], summary="short")],
        operation_cost=0.25,
    )
    result = await runner.advance(await runner.start("evidence"))
    assert result.transitions == ()
    assert RewardConfig(gamma=0.1).configured_return(result.turns) == -0.25
    assert (
        RewardConfig(component_weights={"memory": 0}).configured_return(result.turns)
        == 0
    )
