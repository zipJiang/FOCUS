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
from focus.harness import Agent
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
    runner = Agent(
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
