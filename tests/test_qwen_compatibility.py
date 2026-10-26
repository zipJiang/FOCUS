"""Release contract for Qwen thinking and exact token continuation.

Run directly: uv run pytest -q tests/test_qwen_compatibility.py
The golden records come from workspace-harness-robust at be3d85a.
Do not regenerate them from the implementation being tested.
"""

import json
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from focus.codec import ChatCodec
from focus.generation import GenerateResult, Policy, PolicyFormat
from focus.generation.parsing import TextActionParser
from focus.harness import Agent, Env, StepResult, regions
from tests.helpers import FakeChatTokenizer, ScriptedGenerator

FIXTURES = Path(__file__).parent / "fixtures"


class TemplateTokenizer(FakeChatTokenizer):
    def __init__(self, filename):
        self.chat_template = (FIXTURES / "templates" / filename).read_text()
        self.template = ImmutableSandboxedEnvironment().from_string(self.chat_template)

    def apply_chat_template(self, conversation, **kwargs):
        return self.encode(
            self.template.render(
                messages=conversation,
                tools=kwargs.get("tools"),
                add_generation_prompt=kwargs.get("add_generation_prompt", True),
            )
        )


@pytest.mark.parametrize("role", ["user", "tool", "wrapped"])
@pytest.mark.parametrize("forced", [False, True])
async def test_qwen_matches_frozen_harness_contract(role, forced):
    """Compare complete prompts and training packs with independently saved output."""
    codec = ChatCodec(TemplateTokenizer("qwen35.jinja"))
    generated = ["REASON_A\n</think>\n\nDRAFT_A", "REASON_B\n</think>\n\nDRAFT_B"]
    calls: list[tuple[int, ...]] = []

    async def complete(prefix, params):
        text = generated[len(calls)]
        tokens = tuple(codec.encode(text))
        calls.append(tuple(prefix))
        return GenerateResult(
            prefix_tokens=tuple(prefix),
            tokens=tokens,
            text=text,
            logprobs=[-0.125] * len(tokens),
            stop_reason="stop",
        )

    class Task(Env[int, str]):
        async def reset(self):
            return 0

        async def step(self, state, action, *, workspace=None):
            text = (
                "<tool_response>observation</tool_response>"
                if role == "wrapped"
                else "observation"
            )
            return StepResult(
                next_state=state + 1,
                done=state + 1 == 2,
                messages=(
                    {"role": "user" if role == "wrapped" else role, "content": text},
                ),
                reward_outcome=float(state + 1 == 2),
            )

    runner = Agent(
        policy=Policy(
            format=PolicyFormat("qwen-test", codec, TextActionParser()),
            complete=complete,
        ),
        env=Task(),
        system_prompt="Solve the task.",
        max_steps=1 if forced else 2,
        finish_prompt="Finish now." if forced else None,
    )
    try:
        state = await runner.run_from("TASK_SENTINEL")
        actual = {
            "prefixes": [list(t.prefix) for t in state.turns],
            "tokens": [list(t.tokens) for t in state.turns],
            "logprobs": [t.logprobs for t in state.turns],
            "messages": state.messages,
            "packs": [
                {
                    k: getattr(p, k)
                    for k in (
                        "tokens",
                        "trainable",
                        "logprobs",
                        "exact",
                        "tag",
                        "spans",
                    )
                }
                for p in regions(state.turns)
            ],
            "done": state.done,
            "truncated": state.truncated,
        }
        expected = json.loads(
            (FIXTURES / "qwen_compatibility" / "reference.json").read_text()
        )
        # JSON normalizes tuple/list representation without changing token values.
        assert json.loads(json.dumps(actual)) == expected[f"{role}-{forced}"]
    finally:
        await runner.services.aclose()


@pytest.mark.parametrize("feedback", ["user", "tool", "wrapped"])
@pytest.mark.parametrize("forced", [False, True])
async def test_qwen_thinking_rewrites_preserve_original_training_tokens(
    feedback, forced
):
    """A rendered prompt may drop reasoning; recorded actions must remain exact."""
    from dataclasses import replace

    from focus.generation.parsing.base import TextActionParser
    from focus.harness import regions
    from tests.helpers import ToyEnv

    class FeedbackEnv(ToyEnv):
        async def step(self, state, action, *, workspace=None):
            result = await super().step(state, action, workspace=workspace)
            content = "observation"
            if feedback == "wrapped":
                content = "<tool_response>observation</tool_response>"
            return replace(
                result,
                messages=(
                    {
                        "role": "user" if feedback == "wrapped" else feedback,
                        "content": content,
                    },
                ),
            )

    codec = ChatCodec(TemplateTokenizer("qwen35.jinja"))

    generated = ["REASON_A\n</think>\n\nDRAFT_A", "REASON_B\n</think>\n\nDRAFT_B"]
    gen = ScriptedGenerator(generated, logprobs=-0.125)
    policy = Policy(
        format=PolicyFormat("qwen-test", codec, TextActionParser()),
        complete=gen.agenerate_tokens,
    )
    runner = Agent(
        policy=policy,
        env=FeedbackEnv(done_at=2),
        system_prompt="task",
        max_steps=1 if forced else 2,
        finish_prompt="Finish now." if forced else None,
    )
    try:
        state = await runner.run_from("TASK_SENTINEL")
        prompts = [codec.decode(t.prefix) for t in state.turns]
        assert ("REASON_A" in prompts[1]) == (feedback != "user")
        assert "DRAFT_A" in prompts[1]
        assert "REASON_A" in state.messages[2]["content"]
        assert [t.prefix for t in state.turns] == gen.prefixes
        assert [codec.decode(t.tokens) for t in state.turns] == generated
        packs = regions(state.turns)
        assert all(p.exact for p in packs)
        learned = [
            token
            for p in packs
            for token, keep in zip(p.tokens, p.trainable, strict=True)
            if keep
        ]
        assert learned == [token for t in state.turns for token in t.tokens]
        probabilities = [
            value
            for p in packs
            for value, keep in zip(p.logprobs or (), p.trainable, strict=True)
            if keep
        ]
        assert probabilities == [
            value for t in state.turns for value in t.logprobs or ()
        ]
        if feedback == "user":
            assert len(packs) == 2
            prefix = codec.decode(packs[1].tokens[: packs[1].spans[0][1]])
            assert "REASON_A" not in prefix
            assert not any(packs[1].trainable[: packs[1].spans[0][1]])
    finally:
        await runner.services.aclose()


async def test_qwen_continuation_keeps_unfinished_thinking_and_exact_fork_prefixes():
    from focus import AppendOnly
    from focus.backends import (
        GenerateResult,
        SamplingParams,
        TokenChunk,
        TokenChunkParser,
    )
    from focus.extend import EveryDecision
    from focus.harness import Env, StepResult, regions

    class Segments(Env[int, TokenChunk]):
        async def reset(self):
            return 0

        async def step(self, state, action, *, workspace=None):
            return StepResult(next_state=state + 1, done=action.finished)

    codec = ChatCodec(TemplateTokenizer("qwen35.jinja"))
    first = tuple(codec.encode("REASON_A".ljust(100) + "\n  "))
    second = tuple(codec.encode("REASON_B\n</think>\n\nANSWER"))
    calls: list[tuple[int, ...]] = []

    async def complete(prefix, params):
        calls.append(tuple(prefix))
        tokens = first if len(calls) == 1 else second
        return GenerateResult(
            tokens=tokens,
            prefix_tokens=tuple(prefix),
            stop_reason="length" if len(calls) == 1 else "stop",
            logprobs=[-0.1] * len(tokens),
        )

    policy = Policy(
        format=PolicyFormat("qwen-segments", codec, TokenChunkParser()),
        complete=complete,
    )
    runner = Agent(
        policy=policy,
        env=Segments(),
        system_prompt="task",
        context=AppendOnly(continuation=True),
        branch_points=EveryDecision(),
        sampling_params=SamplingParams(max_tokens=len(first)),
        finish_prompt=None,
    )
    try:
        checkpoint = await runner.advance(await runner.start("question"))
        for _ in range(2):
            endpoint = await runner.advance(checkpoint.fork())
            assert endpoint.done
            assert endpoint.turns[-1].prefix == calls[0] + first
            assert "REASON_A" in codec.decode(endpoint.turns[-1].prefix)
            packs = regions(endpoint.turns)
            assert len(packs) == 1
            assert packs[0].tokens == calls[0] + first + second
        assert checkpoint.turns[-1].tokens == first
        assert not checkpoint.done
    finally:
        await runner.services.aclose()
