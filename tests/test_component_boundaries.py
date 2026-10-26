from __future__ import annotations

import pytest

from focus import (
    ExecutionServices,
    InMemoryWorkspace,
    NullWorkspace,
    Runtime,
)
from focus.config import RolloutSpec
from focus.generation.parsing import HermesToolCallParser
from focus.harness import BaseTool, Env, StepResult
from focus.harness.workspace import StorageTool
from focus.integrations.slime import make_generate
from focus.preparation import PreparedBatch
from tests.helpers import ScriptedGenerator, SubmitEnv, tool_call, toy_runner


async def test_custom_environment_checks_workspace_before_reset_or_generation():
    class Task(Env[None, str]):
        resets = 0

        def validate_workspace(self, workspace):
            if not isinstance(workspace, InMemoryWorkspace):
                raise ValueError("Task needs writable notes")

        async def reset(self):
            self.resets += 1
            return None

        async def step(self, state, action, *, workspace=None):
            return StepResult(None, done=True)

    policy = ScriptedGenerator(["answer"])
    env = Task()
    runner = toy_runner(policy, env=env)
    with pytest.raises(ValueError, match="writable notes"):
        await runner.start("question")
    assert env.resets == 0
    assert policy.prefixes == []
    await runner.run_from("question", workspace=InMemoryWorkspace())
    assert env.resets == 1
    assert len(policy.prefixes) == 1


async def test_tool_owned_validation_and_structural_tools_compose():
    seen = []

    class NeedsNotes(BaseTool):
        name = "notes"

        def validate_workspace(self, workspace):
            seen.append(workspace)
            if not isinstance(workspace, InMemoryWorkspace):
                raise ValueError("notes needs a store")

        def call(self, params, **kwargs):
            return "ok"

    class ExternalTool:
        name = "external"
        description = "A third-party structural tool with no validation hook"
        parameters = {"type": "object"}

        def call(self, params, **kwargs):
            return "ok"

    policy = ScriptedGenerator([tool_call("submit", answer="42")])
    runner = toy_runner(
        policy,
        env=SubmitEnv([NeedsNotes(), ExternalTool()]),
        parser=HermesToolCallParser(),
    )
    with pytest.raises(ValueError, match="notes needs"):
        await runner.run_from("question")
    assert policy.prefixes == []
    workspace = InMemoryWorkspace()
    endpoint = await runner.run_from("question", workspace=workspace)
    assert endpoint.done
    assert seen[-1] is workspace
    # A structural tool can also work with the default workspace by itself.
    SubmitEnv([ExternalTool()]).validate_workspace(NullWorkspace())


def test_storage_validation_accepts_capabilities_without_concrete_type():
    class Store(NullWorkspace):
        def read(self, key):
            return "value"

        def write(self, key, value):
            pass

        def delete(self, key):
            pass

        def paths(self):
            return ()

    StorageTool().validate_workspace(Store())
    with pytest.raises(ValueError, match="read, write, delete, and paths"):
        StorageTool().validate_workspace(NullWorkspace())


@pytest.mark.parametrize("own_services", [False, True])
async def test_slime_preserves_factory_services_and_respects_ownership(
    monkeypatch, own_services
):
    closed = []

    class Services(ExecutionServices):
        async def aclose(self):
            closed.append(self)
            await super().aclose()

    services = Services()
    policy = ScriptedGenerator(["answer"])
    seen = []

    def factory(args):
        runner = toy_runner(policy, services=services)
        return Runtime(runner=runner)

    async def collect(prompt, runtime, **kwargs):
        seen.append(runtime)
        return PreparedBatch()

    monkeypatch.setattr("focus.integrations.slime.collect_samples", collect)
    hook = make_generate(
        RolloutSpec(), factory, estimator=None, own_services=own_services
    )
    # Distinct actor args create two runtimes that deliberately share services.
    await hook(object(), "first")
    await hook(object(), "second")
    assert all(runtime.runner.services is services for runtime in seen)
    await hook.aclose()
    await hook.aclose()
    assert closed == ([services] if own_services else [])
    if not own_services:
        await services.aclose()


async def test_slime_closes_other_owned_services_after_one_close_fails(monkeypatch):
    closed = []

    class Services(ExecutionServices):
        def __init__(self, name):
            super().__init__()
            self.name = name

        async def aclose(self):
            closed.append(self.name)
            await super().aclose()
            if self.name == "bad":
                raise RuntimeError("close failed")

    def factory(name):
        return Runtime(
            runner=toy_runner(ScriptedGenerator(["answer"]), services=Services(name))
        )

    async def collect(*args, **kwargs):
        return PreparedBatch()

    monkeypatch.setattr("focus.integrations.slime.collect_samples", collect)
    hook = make_generate(RolloutSpec(), factory, estimator=None)
    await hook("bad", "first")
    await hook("good", "second")
    with pytest.raises(ExceptionGroup, match="service cleanup failed"):
        await hook.aclose()
    assert closed == ["bad", "good"]
    await hook.aclose()
    assert closed == ["bad", "good"]
