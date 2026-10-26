import pytest

from focus.harness import NullWorkspace, Summarize
from focus.harness.compaction.summary import parse_summary
from focus.harness.context import IncompleteCompactionError
from tests.harness.test_compaction import _compaction
from tests.helpers import ScriptedGenerator

NOTES = (
    "## State\nSearching.\n## Evidence\nDoc 7: Alice.\n"
    "## Open questions\nNone\n## Next steps\nSubmit."
)
VALID = f"<summary>\n{NOTES}\n</summary>"


@pytest.mark.parametrize(
    "text", [VALID, f"```xml\n{VALID}\n```", f"Comment\n{VALID}\nDone", VALID.upper()]
)
def test_closed_summary(text):
    parsed = parse_summary(text, at_cap=True)
    assert not parsed.recovered
    assert parsed.notes.lower() == NOTES.lower()


def test_recover_only_visible_populated_notes():
    assert parse_summary(NOTES).recovered
    assert parse_summary("<summary>" + NOTES).recovered
    assert parse_summary("<think>secret</think>" + VALID).notes == NOTES
    assert parse_summary("reasoning</think>" + VALID, prefix="<think>").notes == NOTES
    assert parse_summary(VALID + "<think>truncated", at_cap=True).notes == NOTES


@pytest.mark.parametrize(
    "text",
    [
        "<think>" + VALID,
        "<think>" + VALID + "</think>",
        VALID + VALID,
        "<summary>partial</summary>",
        VALID.replace("Submit.", ""),
        VALID.replace("## Evidence", "## State"),
    ],
)
def test_reject_ambiguous_or_incomplete_notes(text):
    with pytest.raises(ValueError, match=r"sections|summary|section"):
        parse_summary(text)


def test_reject_missing_close_at_cap_and_open_prefix_reasoning():
    with pytest.raises(ValueError, match="truncated"):
        parse_summary("<summary>" + NOTES, at_cap=True)
    with pytest.raises(ValueError, match="unclosed"):
        parse_summary(VALID, prefix="<think>")


async def test_retry_keeps_original_context_and_exact_failed_generations():
    gen = ScriptedGenerator(
        ["<think>unfinished", "reasoning</think>" + VALID], logprobs=-0.25
    )
    request = _compaction(
        [{"role": "user", "content": "original evidence"}], NullWorkspace(), gen
    )
    compactor = Summarize[None](structured_summary=True, max_reply_tokens=4096)
    result = await compactor.compact(request)
    assert len(result.turns) == 2
    failed, good = result.turns
    assert failed.compaction_penalty == -0.01
    assert failed.compaction_issue
    assert good.compaction_penalty == 0
    assert failed.tokens == tuple(map(ord, "<think>unfinished"))
    assert all(
        t.exact and t.transition is None and t.tag == "fold" for t in result.turns
    )
    assert failed.logprobs == (-0.25,) * len(failed.tokens)
    retry_prompt = request.runner.policy.decode(gen.prefixes[-1])
    assert "original evidence" in retry_prompt
    assert "unfinished" not in retry_prompt
    assert "previous compaction was unusable" in retry_prompt
    assert NOTES in result.messages[-1]["content"]
    assert "reasoning" not in result.messages[-1]["content"]


async def test_retry_is_bounded_and_rejects_oversize_summary():
    gen = ScriptedGenerator([VALID, VALID])
    request = _compaction([], NullWorkspace(), gen)
    compactor = Summarize[None](
        structured_summary=True, max_reply_tokens=4096, summary_target_tokens=10
    )
    with pytest.raises(IncompleteCompactionError) as caught:
        await compactor.compact(request)
    assert len(gen.prefixes) == 2
    assert len(caught.value.turns) == 2
    assert sum(t.compaction_penalty for t in caught.value.turns) == -0.02
    assert caught.value.reasons == ("summary_too_long",)
    assert "summary_too_long" in str(caught.value)


async def test_wrapper_recovery_is_penalized_without_retry():
    gen = ScriptedGenerator([NOTES])
    result = await Summarize[None](structured_summary=True).compact(
        _compaction([], NullWorkspace(), gen)
    )
    assert len(result.turns) == 1
    assert result.turns[0].compaction_penalty == -0.002
    assert result.turns[0].compaction_issue == "recovered_wrapper"


async def test_backend_failure_not_retried_or_relabelled():
    class Broken(ScriptedGenerator):
        async def agenerate_tokens(self, tokens, sampling_params=None):
            raise RuntimeError("backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        await Summarize[None](structured_summary=True).compact(
            _compaction([], NullWorkspace(), Broken([]))
        )


async def test_complete_notes_at_generation_cap_are_accepted():
    gen = ScriptedGenerator([VALID])
    result = await Summarize[None](
        structured_summary=True, max_reply_tokens=len(VALID)
    ).compact(_compaction([], NullWorkspace(), gen))
    assert result.turns[0].compaction_penalty == 0
    assert len(gen.prefixes) == 1


async def test_required_complete_reply_repairs_even_a_closed_reply_at_cap():
    gen = ScriptedGenerator([VALID + "x", VALID])
    result = await Summarize[None](
        structured_summary=True,
        require_complete_reply=True,
        max_reply_tokens=len(VALID) + 1,
    ).compact(_compaction([], NullWorkspace(), gen))
    assert len(gen.prefixes) == 2
    assert result.turns[0].compaction_issue == "reply_cap"
    assert result.turns[0].compaction_penalty == -0.01
