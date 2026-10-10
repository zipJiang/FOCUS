"""Context protocols must preserve the native adapters' tool-history contract."""

import json

import pytest

from focus import OpenAIChatPolicy, OpenAIResponsesPolicy, Runner
from focus.generation import NativeToolCall
from focus.protocols import CLM, Memex
from focus.protocols.rlm import InvocationEnv
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
from tests.helpers import FakeChatTokenizer, native_format


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


async def test_memory_compression_preserves_native_exchange(native_backend):
    policy, respond = native_backend
    respond(
        NativeToolCall(
            "memory_compress", json.dumps({"indices": [1], "summary": "short"}), "c1"
        ),
        NativeToolCall("memory_retrieve", '{"index": 0}', "c2"),
    )
    runner = Runner(
        policy=policy, env=InvocationEnv(), system_prompt="sys", protocol=Memex()
    )
    try:
        root = await runner.start("original evidence")
        changed = await runner.advance(root)
        assert changed.context_version == 1
        assert root.messages[1] not in changed.messages
        assert [m.get("tool_call_id") for m in changed.messages[-2:]] == ["c1", "c2"]
        assert json.loads(changed.messages[-1]["content"]) == [root.history[1]]
        assert all(
            ref is None or message == changed.history[ref]
            for message, ref in zip(changed.messages, changed.context_refs, strict=True)
        )
        respond()
        assert (await runner.advance(changed)).done
    finally:
        await runner.services.aclose()


async def test_clm_replacement_preserves_native_exchange(native_backend):
    policy, respond = native_backend
    replacement = [{"role": "user", "content": "replacement"}]
    code = (
        "from pathlib import Path\n"
        f"Path('context.json').write_text({json.dumps(replacement)!r})"
    )
    respond(
        NativeToolCall("context_edit", json.dumps({"code": code}), "c1"),
        NativeToolCall("context_edit", '{"code": "n = 2"}', "c2"),
    )
    runner = Runner(
        policy=policy, env=InvocationEnv(), system_prompt="sys", protocol=CLM()
    )
    try:
        changed = await runner.advance(await runner.start("original evidence"))
        assert changed.context_version == 2
        assert changed.messages[0] == replacement[0]
        assert "original evidence" not in str(changed.messages)
        assert [m.get("tool_call_id") for m in changed.messages[-2:]] == ["c1", "c2"]
        assert len(changed.messages) == len(changed.context_refs)
        respond()
        assert (await runner.advance(changed)).done
    finally:
        await runner.services.aclose()


@pytest.mark.parametrize("protocol", [CLM(), Memex()])
async def test_invalid_native_context_edit_rolls_back(native_backend, protocol):
    policy, respond = native_backend
    respond(NativeToolCall("unknown", "{}", "old"))
    runner = Runner(
        policy=policy, env=InvocationEnv(), system_prompt="sys", protocol=protocol
    )
    try:
        previous = await runner.advance(await runner.start("evidence"))
        # Drop only the assistant half of a completed native tool exchange.
        if isinstance(protocol, CLM):
            code = (
                "import json\nfrom pathlib import Path\n"
                "p = Path('context.json')\nmessages = json.loads(p.read_text())\n"
                "messages = [m for m in messages if m['role'] != 'assistant']\n"
                "p.write_text(json.dumps(messages))\n"
                "leaked = True"
            )
            call = NativeToolCall("context_edit", json.dumps({"code": code}), "new")
        else:
            call = NativeToolCall(
                "memory_compress", '{"indices": [2], "summary": "short"}', "new"
            )
        respond(call)
        rejected = await runner.advance(previous)
        assert rejected.context_version == 0
        assert rejected.execution == previous.execution
        assert rejected.messages[: len(previous.messages)] == previous.messages
        assert "Invalid" in rejected.messages[-1]["content"]
        respond()
        assert (await runner.advance(rejected)).done
    finally:
        await runner.services.aclose()


@pytest.mark.parametrize("invalid_return", [False, True])
async def test_child_tool_feedback_allows_native_retry(native_backend, invalid_return):
    policy, respond = native_backend
    calls = [NativeToolCall("unknown", "{}", "c1")]
    if invalid_return:
        calls.insert(0, NativeToolCall("rlm_return", '{"answer": 42}', "c0"))
    respond(*calls)
    runner = Runner(policy=policy, env=InvocationEnv(), system_prompt="sys")
    try:
        rejected = await runner.advance(await runner.start("question"))
        assert not rejected.done
        assert [
            m.get("tool_call_id") for m in rejected.messages if m["role"] == "tool"
        ] == [c.call_id for c in calls]
        respond(NativeToolCall("rlm_return", '{"answer": "corrected"}', "next"))
        result = await runner.advance(rejected)
        assert result.done
        assert result.state == "corrected"
    finally:
        await runner.services.aclose()
