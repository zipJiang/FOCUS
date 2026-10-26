"""Context protocols must preserve the native adapters' tool-history contract."""

import pytest

from focus import (
    OpenAIChatPolicy,
    OpenAIResponsesPolicy,
    Runner,
)
from focus.generation import NativeToolCall
from tests.generation.test_openai_chat import (
    _Completions,
    _tool_call,
)
from tests.generation.test_openai_chat import (
    _response as chat_response,
)
from tests.generation.test_openai_chat import (
    _wire as chat_wire,
)
from tests.generation.test_openai_responses import (
    _call_item,
    _Responses,
    _text_item,
)
from tests.generation.test_openai_responses import (
    _response as responses_response,
)
from tests.generation.test_openai_responses import (
    _wire as responses_wire,
)
from tests.helpers import FakeChatTokenizer, SubmitEnv, native_format
from tests.protocols.helpers import KeepRecent


@pytest.fixture(params=["chat", "responses"])
def native_backend(request, monkeypatch):
    format = native_format(FakeChatTokenizer())
    policy: OpenAIChatPolicy | OpenAIResponsesPolicy
    wire: _Completions | _Responses
    if request.param == "chat":
        policy = OpenAIChatPolicy(model="fake", format=format)
        wire = chat_wire(monkeypatch, chat_response("done"))
    else:
        policy = OpenAIResponsesPolicy(model="fake", format=format)
        wire = responses_wire(monkeypatch, responses_response(_text_item("done")))

    def respond(*calls: NativeToolCall):
        if request.param == "chat":
            wire.response = chat_response(
                None if calls else "done",
                tool_calls=[_tool_call(c.name, c.arguments, c.call_id) for c in calls],
            )
        else:
            wire.response = responses_response(
                *[_call_item(c.name, c.arguments, c.call_id) for c in calls]
                if calls
                else [_text_item("done")]
            )

    return policy, respond


async def test_custom_protocol_preserves_native_batch_and_history(native_backend):
    policy, respond = native_backend
    respond(
        NativeToolCall("keep_recent", '{"n": 1}', "c1"),
        NativeToolCall("keep_recent", '{"n": 1}', "c2"),
    )
    runner = Runner(
        policy=policy, env=SubmitEnv([]), system_prompt="sys", protocol=KeepRecent()
    )
    try:
        root = await runner.start("original evidence")
        changed = await runner.advance(root)
        assert changed.context_version == 2
        assert changed.messages[0] == root.messages[1]
        assert [m.get("tool_call_id") for m in changed.messages[-2:]] == ["c1", "c2"]
        assert changed.transitions == ()
        assert [e.turn_index for e in changed.effects] == [0, 0]
        assert [(e.context_before, e.context_after) for e in changed.effects] == [
            (0, 1),
            (1, 2),
        ]
        assert all(
            ref is None or message == changed.history[ref]
            for message, ref in zip(changed.messages, changed.context_refs, strict=True)
        )
        policy.validate_context(changed.messages)
        respond(NativeToolCall("submit", '{"answer": "done"}', "finish"))
        assert (await runner.advance(changed)).done
    finally:
        await runner.services.aclose()


async def test_invalid_native_context_edit_rolls_back(native_backend):
    policy, respond = native_backend
    respond(NativeToolCall("unknown", "{}", "old"))
    runner = Runner(
        policy=policy, env=SubmitEnv([]), system_prompt="sys", protocol=KeepRecent()
    )
    try:
        previous = await runner.advance(await runner.start("evidence"))
        # Retaining only the tool reply would orphan its assistant call.
        respond(NativeToolCall("keep_recent", '{"n": 1}', "new"))
        rejected = await runner.advance(previous)
        assert rejected.context_version == 0
        assert rejected.execution == previous.execution
        assert rejected.messages[: len(previous.messages)] == previous.messages
        assert "Invalid" in rejected.messages[-1]["content"]
        policy.validate_context(rejected.messages)
        respond(NativeToolCall("submit", '{"answer": "done"}', "finish"))
        assert (await runner.advance(rejected)).done
    finally:
        await runner.services.aclose()


async def test_mixed_native_batch_rejects_context_and_task_operations(native_backend):
    policy, respond = native_backend
    respond(
        NativeToolCall("keep_recent", '{"n": 1}', "edit"),
        NativeToolCall("submit", '{"answer": "premature"}', "finish"),
    )
    runner = Runner(
        policy=policy, env=SubmitEnv([]), system_prompt="sys", protocol=KeepRecent()
    )
    try:
        root = await runner.start("evidence")
        rejected = await runner.advance(root)
        assert rejected.context_version == 0
        assert rejected.state == root.state
        assert rejected.transitions == ()
        assert not rejected.done
        assert [m.get("tool_call_id") for m in rejected.messages[-2:]] == [
            "edit",
            "finish",
        ]
        assert all("separately" in m["content"] for m in rejected.messages[-2:])
        policy.validate_context(rejected.messages)
    finally:
        await runner.services.aclose()
