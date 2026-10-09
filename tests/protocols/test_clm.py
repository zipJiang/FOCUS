import json

from focus.generation.parsing.hermes import HermesToolCallParser
from focus.protocols import CLM
from tests.helpers import ScriptedGenerator, tool_call, toy_runner


def editor(code):
    return toy_runner(
        ScriptedGenerator([tool_call("context_edit", code=code), "done"]),
        parser=HermesToolCallParser(),
        protocol=CLM(),
    )


async def test_edit_changes_next_prompt_and_keeps_deleted_generation():
    messages = [
        {"role": "system", "content": "new instructions"},
        {"role": "user", "content": "replacement"},
    ]
    runner = editor(
        "import json\nwith open('context.json','w') as f:\n    json.dump("
        + repr(messages)
        + ", f)\ndel f"
    )
    root = await runner.start("original secret")
    changed = await runner.advance(root)
    assert changed.context_version == 1
    assert changed.forkable
    assert changed.turns[0].tokens
    assert "original secret" not in json.dumps(changed.messages)
    final = await runner.advance(changed)
    assert final.turns[0] == changed.turns[0]
    next_prompt = runner.policy.decode(final.turns[1].prefix)
    assert "replacement" in next_prompt
    assert "original secret" not in next_prompt
    assert "original secret" in runner.policy.decode(final.turns[0].prefix)
    await runner.services.aclose()


async def test_invalid_edit_preserves_document_and_session():
    runner = editor("with open('context.json','w') as f: f.write('invalid')")
    root = await runner.start("keep this")
    result = await runner.advance(root)
    assert result.context_version == 0
    assert result.messages[:2] == root.messages
    assert "Invalid context_edit" in result.messages[-1]["content"]
    assert "clm_session" not in result.execution
    await runner.services.aclose()


async def test_branch_edits_have_independent_python_globals():
    code = "n = globals().get('n', 0) + 1\nprint(n)"
    runner = editor(code)
    root = await runner.start("question")
    one = await runner.advance(root)
    assert one.execution["clm_session"].namespace
    a, b = one.fork(), one.fork()
    # Direct protocol operation avoids the scripted backend's terminal next output.
    a, output, _ = await runner.protocol.operation(
        a, "context_edit", {"code": code}, runner
    )
    b, other, _ = await runner.protocol.operation(
        b, "context_edit", {"code": code}, runner
    )
    assert output == other == "Context committed. 2\n"
    assert a.execution is not b.execution
    await runner.services.aclose()
