"""The Slime integration: hooks Slime calls, built from the user's collector factory.

Slime itself is not installed here; its two modules the hooks import are stand-ins
with the shape Slime's customization interfaces define.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from focus import Agent, Collector, Fifo, Phase, Search
from focus.backends import HermesToolCallParser
from focus.execution import ExecutionServices
from focus.extend import InMemoryWorkspace, PackedSequence, StorageTool
from focus.integrations import Problem
from focus.integrations import slime as slime_integration
from focus.integrations.slime import (
    make_evaluate,
    make_generate,
    policy_for,
    preserve_advantages,
    problem_of,
    to_sample,
)
from focus.preparation.records import ActorSample, Batch, CriticSample
from tests.helpers import (
    DemoValue,
    FakeChatTokenizer,
    ScriptedGenerator,
    SubmitEnv,
    tool_call,
    toy_policy,
)


@dataclass
class Sample:
    """Slime's ``Sample``, reduced to the fields the hooks read and write."""

    index: int | None = 9
    prompt: str = "solve it"
    label: str | None = "42"
    metadata: dict[str, Any] = field(default_factory=dict)
    rollout_id: int | None = None

    class Status:
        COMPLETED = "completed"
        TRUNCATED = "truncated"


@pytest.fixture
def slime_modules(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Install the Slime modules the hooks import; return posted generation requests."""
    posted: list[Any] = []

    async def post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        posted.append((url, payload))
        raise AssertionError("tests that generate replace policy_for")

    monkeypatch.setitem(sys.modules, "slime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "slime.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "slime.utils.http_utils", SimpleNamespace(post=post)
    )
    monkeypatch.setitem(
        sys.modules, "slime.utils.types", SimpleNamespace(Sample=Sample)
    )
    return posted


def _scripted(*replies: str) -> Any:
    """The policy a generate call gets: scripted Hermes tool calls."""
    gen = ScriptedGenerator(list(replies), logprobs=-0.1)
    return toy_policy(gen, parser=HermesToolCallParser())


def _collector(policy: Any, *, tools: list[Any] | None = None, **kwargs: Any) -> Any:
    agent = Agent(policy=policy, env=SubmitEnv(tools or []), system_prompt="sys")
    return Collector(
        agent,
        critic=DemoValue(),
        search=Search([Phase(Fifo(), 2, max_children=3)], max_concurrency=1),
        **kwargs,
    )


def _submit() -> str:
    return "r</think>" + tool_call("submit", answer="42")


async def test_generate_turns_one_prompt_into_slime_samples(
    slime_modules: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _scripted(*[_submit()] * 4)
    monkeypatch.setattr(slime_integration, "policy_for", lambda args, **_: policy)
    problems: list[Problem] = []
    critic: list[Any] = []

    def collector_for(given: Any, problem: Problem) -> Any:
        assert given is policy
        problems.append(problem)
        return _collector(given)

    generate = make_generate(collector_for, critic_sink=critic.extend)
    samples = await generate(SimpleNamespace(), Sample(metadata={"task": "t"}))

    assert problems == [Problem("solve it", "42", 9, {"task": "t"})]
    assert samples
    assert {s.rollout_id for s in samples} == {9}
    assert all(s.metadata["focus"]["lane"] == "actor" for s in samples)
    assert all(s.metadata["task"] == "t" for s in samples)
    assert critic
    assert all(r["metadata"]["lane"] == "critic" for r in critic)


async def test_concurrent_prompts_each_get_their_own_collector(
    slime_modules: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each sample's problem reaches its own collector, so reference answers never
    cross between concurrent prompts."""
    monkeypatch.setattr(
        slime_integration, "policy_for", lambda args, **_: _scripted(*[_submit()] * 4)
    )
    labels: list[str] = []
    first_ready, release = asyncio.Event(), asyncio.Event()

    def collector_for(policy: Any, problem: Problem) -> Any:
        labels.append(problem.label)
        return _collector(policy)

    search = Collector.search

    async def paused(self: Collector, prompt: Any, *, workspace: Any = None) -> Any:
        tree = await search(self, prompt, workspace=workspace)
        if prompt == "first":
            first_ready.set()
            await release.wait()
        return tree

    monkeypatch.setattr(Collector, "search", paused)
    generate = make_generate(collector_for)
    async with asyncio.TaskGroup() as tasks:
        first = tasks.create_task(
            generate(None, Sample(index=1, prompt="first", label="a"))
        )
        await asyncio.wait_for(first_ready.wait(), timeout=5)
        try:
            second = await generate(None, Sample(index=2, prompt="second", label="b"))
        finally:
            release.set()

    assert labels == ["a", "b"]
    assert {s.rollout_id for s in first.result()} == {1}
    assert {s.rollout_id for s in second} == {2}


async def test_workspace_for_gives_each_prompt_its_own_store(
    slime_modules: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    put = "r</think>" + tool_call("storage", operate="put", key="k", value="v")
    monkeypatch.setattr(
        slime_integration, "policy_for", lambda args, **_: _scripted(put, _submit())
    )
    seen: list[Problem] = []

    def workspace_for(problem: Problem) -> InMemoryWorkspace:
        seen.append(problem)
        return InMemoryWorkspace()

    generate = make_generate(
        lambda policy, problem: _collector(policy, tools=[StorageTool()]),
        workspace_for=workspace_for,
    )
    samples = await generate(None, Sample())

    assert [problem.prompt for problem in seen] == ["solve it"]
    tokenizer = FakeChatTokenizer()
    transcripts = "".join(tokenizer.decode(s.tokens) for s in samples)
    assert "Successfully saved k." in transcripts


def _batch(*, failures: float = 0.0) -> Batch:
    span = PackedSequence(tokens=(0, 1), trainable=(False, True), logprobs=(0.0, -0.1))
    return Batch(
        actor=(
            ActorSample(node_id=1, spans=(span,), group_id=0, estimator="x", weight=2),
        ),
        critic=(CriticSample(node_id=1, context="checkpoint", target=1.0),),
        stats={"failures": failures},
    )


async def test_partial_collections_and_filters_are_explicit(
    slime_modules: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slime_integration, "policy_for", lambda args, **_: _scripted())
    failures = 1.0

    async def collect(self: Collector, prompt: Any, *, workspace: Any = None) -> Batch:
        return _batch(failures=failures)

    monkeypatch.setattr(Collector, "collect", collect)

    def collector_for(policy: Any, problem: Problem) -> Any:
        return _collector(policy)

    with pytest.raises(RuntimeError, match="collection failed"):
        await make_generate(collector_for)(None, Sample())
    kept = await make_generate(collector_for, allow_partial=True)(None, Sample())
    assert len(kept) == 1

    failures = 0.0
    critic: list[Any] = []
    filtered = make_generate(
        collector_for, critic_sink=critic.extend, min_abs_advantage=100.0
    )
    with pytest.raises(ValueError, match="no actor segments"):
        await filtered(None, Sample())
    assert len(critic) == 1  # filtering actor edges leaves critic targets alone


@pytest.mark.parametrize("own", [True, False])
async def test_the_collector_lifecycle_is_the_factory_s(
    slime_modules: list[Any], monkeypatch: pytest.MonkeyPatch, own: bool
) -> None:
    closed: list[object] = []

    class Services(ExecutionServices):
        async def aclose(self) -> None:
            closed.append(self)
            await super().aclose()

    services = Services()
    monkeypatch.setattr(slime_integration, "policy_for", lambda args, **_: _scripted())

    async def collect(self: Collector, prompt: Any, *, workspace: Any = None) -> Batch:
        return _batch()

    monkeypatch.setattr(Collector, "collect", collect)

    def collector_for(policy: Any, problem: Problem) -> Any:
        collector = _collector(policy, own=own)
        collector.agent.services = services
        return collector

    generate = make_generate(collector_for)
    await generate(None, Sample(index=1))
    await generate(None, Sample(index=2))
    assert closed == ([services, services] if own else [])
    await services.aclose()


def test_to_sample_aligns_the_response_and_keeps_tool_output_masked() -> None:
    record = {
        "tokens": [1, 2, 3, 4, 5],
        "loss_mask": [False, True, False, True, True],
        "logprobs": [0.0, -0.1, 0.0, -0.2, -0.3],
        "reward": 0.4,
        "metadata": {"lane": "actor"},
    }
    source = Sample(index=None, metadata={"task": "t"}, rollout_id=3)
    sample = to_sample(record, source, FakeChatTokenizer(), Sample)

    assert sample.tokens == [1, 2, 3, 4, 5]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 0, 1, 1]
    assert sample.rollout_log_probs == [-0.1, 0.0, -0.2, -0.3]
    assert sample.reward == 0.4
    assert sample.rollout_id == 3
    assert sample.status == Sample.Status.COMPLETED
    assert source.metadata == {"task": "t"}  # the prompt's sample is not mutated

    for bad, reason in (
        ({**record, "metadata": {"lane": "critic"}}, "actor records"),
        ({**record, "loss_mask": [True] * 5}, "conditioning context"),
        ({**record, "loss_mask": [False] * 5}, "trainable segment"),
        ({**record, "logprobs": [0.0]}, "Misaligned"),
    ):
        with pytest.raises(ValueError, match=reason):
            to_sample(bad, Sample(), FakeChatTokenizer(), Sample)
    with pytest.raises(ValueError, match="index or rollout_id"):
        to_sample(record, Sample(index=None), FakeChatTokenizer(), Sample)


async def test_evaluate_scores_one_unbranched_rollout(
    slime_modules: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        slime_integration, "policy_for", lambda args, **_: _scripted(_submit())
    )
    built: list[Problem] = []

    def collector_for(policy: Any, problem: Problem) -> Any:
        built.append(problem)
        return _collector(policy)

    result = await make_evaluate(collector_for)(None, Sample())

    assert [problem.label for problem in built] == ["42"]
    assert result.reward == 1.0
    assert result.status == Sample.Status.COMPLETED
    assert result.response_length == len(result.loss_mask) > 0
    assert len(result.rollout_log_probs) == result.response_length
    assert "submit" in result.response


def test_policy_for_serves_slime_s_router(
    slime_modules: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slime_integration, "_tokenizer", lambda _: FakeChatTokenizer())
    args = SimpleNamespace(
        hf_checkpoint="org/model", sglang_router_ip="router", sglang_router_port=9000
    )
    policy = policy_for(args, profile="hermes")
    assert policy.model == "org/model"
    assert policy._url == "http://router:9000/generate"
    assert isinstance(policy.format.parser, HermesToolCallParser)


def test_problem_of_reads_slime_s_sample() -> None:
    problem = problem_of(Sample(index=4, metadata={"k": "v"}))
    assert problem == Problem("solve it", "42", 4, {"k": "v"})


def test_reward_post_processing_keeps_advantages() -> None:
    samples = [SimpleNamespace(reward=0.3), SimpleNamespace(reward=-0.2)]
    assert preserve_advantages(None, samples) == ([0.3, -0.2], [0.3, -0.2])


def test_the_policy_loss_keeps_each_segment_s_advantage() -> None:
    torch = pytest.importorskip("torch")
    from focus.integrations.slime import advantages, sum_policy_tokens

    # At ratio 1, the PPO gradient for each selected token is -advantage.
    lp = [torch.tensor([-0.5, -0.5]), torch.tensor([-0.5])]
    masks = [torch.tensor([1, 0]), torch.tensor([1])]
    data: dict[str, Any] = {"rollout_log_probs": lp, "rewards": [0.3, -0.2]}
    args = SimpleNamespace(use_rollout_logprobs=True, normalize_advantages=False)
    advantages(args, data)
    current = torch.cat(lp).clone().requires_grad_()
    ratio = (current - torch.cat(lp)).exp()
    adv = torch.cat(data["advantages"])
    per_token = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
    sum_policy_tokens([4, 3], [2, 1], masks)(per_token).backward()
    torch.testing.assert_close(current.grad, torch.tensor([-0.3, 0.0, 0.2]))
    assert all(r.abs().sum() == 0 for r in data["returns"])
    with pytest.raises(ValueError, match="unnormalized"):
        advantages(SimpleNamespace(normalize_advantages=True), data)
    with pytest.raises(ValueError, match="rollout-based"):
        sum_policy_tokens([4], [2], masks, calculate_per_token_loss=True)
