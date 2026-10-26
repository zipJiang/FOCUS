"""Real template rendering around stateless reminders and the forced final turn."""

from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from focus.codec import ChatCodec
from focus.generation import Policy, PolicyFormat
from focus.generation.parsing import (
    HermesToolCallParser,
    QwenXMLToolCallParser,
)
from focus.harness import Runner
from tests.helpers import FakeChatTokenizer, ScriptedGenerator, SubmitEnv, tool_call

FIXTURES = Path(__file__).parents[1] / "fixtures" / "templates"


class TemplateTokenizer(FakeChatTokenizer):
    def __init__(self, filename):
        self.chat_template = (FIXTURES / filename).read_text()
        self.template = ImmutableSandboxedEnvironment().from_string(self.chat_template)

    def apply_chat_template(self, conversation, **kwargs):
        text = self.template.render(
            messages=conversation,
            tools=kwargs.get("tools"),
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
        )
        return self.encode(text)


@pytest.mark.parametrize("filename", ["qwen35.jinja", "qwen25_hermes.jinja"])
@pytest.mark.parametrize("max_steps", [2, 3])
async def test_feedback_preserves_task_history_and_final_submit(filename, max_steps):
    tokenizer = TemplateTokenizer(filename)
    codec = ChatCodec(tokenizer)
    parser: HermesToolCallParser | QwenXMLToolCallParser
    if filename == "qwen35.jinja":
        parser = QwenXMLToolCallParser()
        submit = (
            "<tool_call>\n<function=submit>\n"
            "<parameter=answer>42</parameter>\n</function>\n</tool_call>"
        )
    else:
        parser = HermesToolCallParser()
        submit = tool_call("submit", answer="42")
    gen = ScriptedGenerator(
        [
            "<think>REASON_A</think>DRAFT_A",
            "<think>REASON_B</think>DRAFT_B",
            "<think>REASON_C</think>" + submit,
        ]
    )
    policy = Policy(
        format=PolicyFormat(filename, codec, parser), complete=gen.agenerate_tokens
    )
    runner = Runner(
        policy=policy,
        env=SubmitEnv([]),
        system_prompt="Solve the task.",
        max_steps=max_steps,
    )
    result = await runner.run_from("TASK_SENTINEL")
    assert result.state.answer == "42"
    assert result.reward == 1
    assert result.done
    assert result.truncated == (max_steps == 2)
    assert len(gen.prefixes) == 3
    prompts = [codec.decode(prefix) for prefix in gen.prefixes]
    assert "DRAFT_A" in prompts[1]
    assert "DRAFT_A" in prompts[2]
    assert "DRAFT_B" in prompts[2]
    assert "<tool_response>" in prompts[1]
    assert "<tool_response>" in prompts[2]
    if filename == "qwen35.jinja":
        assert "REASON_A" in prompts[1]
        assert "REASON_A" in prompts[2]
        assert "REASON_B" in prompts[2]
    assert all("TASK_SENTINEL" in prompt for prompt in prompts)
    assert [t.prefix for t in result.turns] == gen.prefixes
    assert not any(m.get("tool_calls") for m in result.messages)


def test_qwen_ordinary_user_feedback_would_drop_reasoning():
    codec = ChatCodec(TemplateTokenizer("qwen35.jinja"))
    history = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "<think>REASON_A</think>draft"},
    ]
    text = codec.decode(
        codec.render([*history, {"role": "user", "content": "continue"}])
    )
    assert "REASON_A" not in text


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
    runner = Runner(
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
            for value, keep in zip(p.logprobs["policy"], p.trainable, strict=True)
            if keep
        ]
        assert probabilities == [
            value for t in state.turns for value in t.logprobs["policy"]
        ]
        if feedback == "user":
            assert len(packs) == 2
            prefix = codec.decode(packs[1].tokens[: packs[1].spans[0][1]])
            assert "REASON_A" not in prefix
            assert not any(packs[1].trainable[: packs[1].spans[0][1]])
    finally:
        await runner.services.aclose()


async def test_qwen_continuation_keeps_unfinished_thinking_and_exact_fork_prefixes():
    from focus import (
        AppendOnly,
        EveryDecision,
        GenerateResult,
        SamplingParams,
        TokenChunk,
        TokenChunkParser,
    )
    from focus.harness import Env, StepResult, regions

    class Segments(Env[int, TokenChunk]):
        async def reset(self):
            return 0

        async def step(self, state, action, *, workspace=None):
            return StepResult(next_state=state + 1, done=action.finished)

    codec = ChatCodec(TemplateTokenizer("qwen35.jinja"))
    first = tuple(codec.encode("REASON_A".ljust(100) + "\n  "))
    second = tuple(codec.encode("REASON_B\n</think>\n\nANSWER"))
    calls = []

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
    runner = Runner(
        policy=policy,
        env=Segments(),
        system_prompt="task",
        protocol=AppendOnly(continuation=True),
        checkpoints=EveryDecision(),
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
