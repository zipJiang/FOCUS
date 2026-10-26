from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest

from focus.codec import ChatCodec, Message
from focus.generation import GenerateResult
from focus.generation.parsing import (
    ActionParser,
    HermesToolCallParser,
    NativeToolCallParser,
    QwenXMLToolCallParser,
)
from focus.harness import (
    Agent,
    AnyOf,
    BaseTool,
    Compaction,
    CompactionResult,
    ContextManager,
    FoldEnv,
    FoldState,
    InMemoryWorkspace,
    NullWorkspace,
    PromptTokens,
    RoundBudget,
    StateCounter,
    StorageTool,
    SubmitFields,
    Summarize,
    ToolEnv,
    Workspace,
    build_budget,
    regions,
)
from focus.harness.compaction.agentic import _EMPTY_KEPT
from focus.harness.compaction.fold_env import STORE_FIRST
from focus.harness.workspace.null import _FOLDED_GUIDANCE
from tests.helpers import (
    FakeChatTokenizer,
    ScriptedGenerator,
    complete,
    tool_call,
    toy_runner,
)


def _turn(*calls: str) -> str:
    """A Hermes assistant turn: reasoning, then </think>, then the tool call(s)."""
    return "reasoning\n</think>\n\n" + "\n".join(calls)


def _store(key: str, value: str) -> str:
    return tool_call("storage", operate="put", key=key, value=value)


def _reply(text: str) -> str:
    """A no-tool assistant turn: reasoning, then the visible text (ends the fold)."""
    return f"reasoning\n</think>\n\n{text}"


def _folding_runner(
    gen: ScriptedGenerator, parser: ActionParser[Any] | None = None
) -> Agent[Any, Any]:
    """The rollout a compaction is folding.

    Only what :meth:`Agent.derive` carries matters here -- the generator, the codec,
    the tool dialect and the policy object. The fold runs on its own ``FoldEnv``,
    so this runner's env is never stepped.
    """
    return toy_runner(gen, parser=parser or HermesToolCallParser())


def _compaction(
    messages: list[Message], ws: Workspace, gen: ScriptedGenerator
) -> Compaction[None]:
    seed: tuple[Message, ...] = (
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Question: q"},
    )
    return Compaction(
        messages=tuple(messages),
        seed=seed,
        state=None,
        workspace=ws,
        agent=_folding_runner(gen),
    )


def _parse(text: str):
    return HermesToolCallParser().parse(GenerateResult(text=text, tokens=()))


def _decode(prefix: tuple[int, ...]) -> str:
    return ChatCodec(FakeChatTokenizer()).decode(prefix)


# -- FoldEnv: the environment a fold runs in ------------------------------------------


@pytest.mark.asyncio
async def test_fold_env_keeps_the_text_of_a_no_tool_reply() -> None:
    env = FoldEnv(InMemoryWorkspace().compaction_tools())
    state = replace(await env.reset(), used_tools=True)  # already stored: no nudge

    result = await env.step(state, _parse(_reply("keep this")))

    assert result.done
    assert result.next_state.kept == "keep this"
    assert not result.messages  # a terminal transition carries no observation


@pytest.mark.asyncio
async def test_fold_env_nudges_a_first_reply_that_stored_nothing_exactly_once() -> None:
    env = FoldEnv(InMemoryWorkspace().compaction_tools())
    state = await env.reset()

    nudge = await env.step(state, _parse(_reply("kept, keys: author")))
    assert not nudge.done
    assert nudge.messages == ({"role": "user", "content": STORE_FIRST},)
    assert nudge.next_state.nudged
    assert not nudge.next_state.kept

    # asked once, then taken at its word however often it replies
    again = await env.step(nudge.next_state, _parse(_reply("kept anyway")))
    assert again.done
    assert again.next_state.kept == "kept anyway"


@pytest.mark.asyncio
async def test_fold_env_does_not_nudge_a_fold_that_touched_memory() -> None:
    ws = InMemoryWorkspace()
    env = FoldEnv(ws.compaction_tools())

    stored = await env.step(await env.reset(), _parse(_store("k", "v")), workspace=ws)
    assert stored.next_state.used_tools
    assert not stored.done
    assert ws.read("k") == "v"

    ended = await env.step(stored.next_state, _parse(_reply("kept")))
    assert ended.done
    assert ended.next_state.kept == "kept"


@pytest.mark.asyncio
async def test_fold_env_without_tools_never_nudges() -> None:
    """Nothing durable behind it: the fold summary *is* the memory, so just take it."""
    env = FoldEnv(NullWorkspace().compaction_tools())

    result = await env.step(await env.reset(), _parse(_reply("kept")))

    assert result.done
    assert result.next_state.kept == "kept"


@pytest.mark.asyncio
async def test_fold_env_finish_keeps_the_text_too() -> None:
    """The forced ending and the voluntary one write the kept text the same way."""
    env = FoldEnv(InMemoryWorkspace().compaction_tools())

    result = await env.finish(await env.reset(), _parse(_reply("last words")))

    assert result.done
    assert result.next_state.kept == "last words"


@pytest.mark.asyncio
async def test_fold_state_carries_no_submit_fields_and_needs_none() -> None:
    """The fold environment needs no task-submission fields."""
    env = FoldEnv(InMemoryWorkspace().compaction_tools())

    assert not hasattr(FoldState(), "submissions")
    assert env.on_tool_calls(FoldState(), _parse(_turn())) == FoldState(used_tools=True)


# -- Triggers: when to fold, and what relieves it -------------------------------------


@dataclass(frozen=True)
class BudgetState:
    remaining: int = 5
    maximum: int = 5
    other_counter: int = 7


def test_prompt_tokens_fires_past_its_budget_and_relieves_itself() -> None:
    trigger = PromptTokens(5)

    assert not trigger.fires((1, 2, 3), None)
    assert trigger.fires(tuple(range(6)), None)
    assert trigger.relieve(None) is None  # a shorter context is the whole relief


def test_state_counter_fires_when_spent_and_refills_on_relief() -> None:
    trigger = StateCounter("remaining", "maximum")
    prompt = (1,)

    assert trigger.fires(prompt, BudgetState(remaining=0))
    assert not trigger.fires(prompt, BudgetState(remaining=3))
    # Relief refills only the configured counter.
    relieved = trigger.relieve(BudgetState(remaining=0, other_counter=9))
    assert relieved == BudgetState(remaining=5, maximum=5, other_counter=9)
    assert not trigger.fires(prompt, relieved)


def test_state_counter_preserves_unrelated_counters() -> None:
    trigger = StateCounter("remaining", "maximum")
    assert trigger.relieve(BudgetState(remaining=0, other_counter=9)).other_counter == 9


def test_any_of_fires_on_either_and_relieves_through_both() -> None:
    trigger = AnyOf(PromptTokens(10), StateCounter("remaining", "maximum"))

    assert trigger.fires(tuple(range(11)), BudgetState(remaining=3))
    assert trigger.fires((1,), BudgetState(remaining=0))
    assert not trigger.fires((1,), BudgetState(remaining=3))
    assert trigger.relieve(BudgetState(remaining=0)).remaining == 5


# -- Summarize: the fold as a nested rollout ------------------------------------


@pytest.mark.asyncio
async def test_compact_stores_then_replies_with_what_to_keep() -> None:
    ws = InMemoryWorkspace()
    # offload via storage, then a no-tool reply ends it (the fold's own convention)
    gen = ScriptedGenerator([_turn(_store("log", "detail")), _reply("kept summary")])
    compactor = Summarize[None](max_interactions=8)
    c = _compaction([{"role": "user", "content": "grown context"}], ws, gen)

    result = await compactor.compact(c)

    assert ws.read("log") == "detail"  # the storage put landed in the workspace
    assert gen.calls == 2  # store, then reply
    # the fold is the seed's system turn + ONE user turn: the original prompt with the
    # compacted state attached (not the summary appended as a second user message).
    assert len(result.messages) == 2
    assert result.messages[0] == c.seed[0]
    folded = result.messages[1]
    assert folded["role"] == "user"
    assert folded["content"].startswith("Question: q")
    assert "kept summary" in folded["content"]
    assert result.state is None  # default trigger: state carried through unchanged


@pytest.mark.asyncio
async def test_memory_is_not_inlined_but_the_model_is_told_to_read_it() -> None:
    """The fix for near-duplicate keys is a habit, not a dump.

    Inlining the workspace into either prompt would grow with the workspace and undo the
    fold; both sides instead name storage and how to reach it.
    """
    ws = InMemoryWorkspace()
    ws.write("author", "Agatha Christie: 66 novels")
    gen = ScriptedGenerator([_reply("kept summary")])
    c = _compaction([{"role": "user", "content": "grown context"}], ws, gen)

    result = await Summarize[None](max_interactions=8).compact(c)

    compacting = _decode(gen.prefixes[0])
    folded = result.messages[-1]["content"]
    for prompt in (compacting, folded):
        assert "Agatha Christie: 66 novels" not in prompt  # no stored value inlined
        # ...just a pointer at where to look; the how lives on the tool's own schema
        assert "durable memory" in prompt.lower()
        assert "storage tool" in prompt


@pytest.mark.asyncio
async def test_the_workspace_supplies_what_both_prompts_say_about_it() -> None:
    """Where-to-put-what is the backend's prose, quoted by the compactor verbatim."""
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_reply("kept")])
    c = _compaction([{"role": "user", "content": "grown"}], ws, gen)

    result = await Summarize[None]().compact(c)

    assert ws.compaction_guidance() in _decode(gen.prefixes[0])
    assert ws.guidance() in result.messages[-1]["content"]
    # a KV store's standing advice still holds after a fold, so the post-fold hook is
    # the same prose -- `folded_guidance` exists for the backend where it is not.
    assert ws.folded_guidance() == ws.guidance()


@pytest.mark.asyncio
async def test_the_folded_context_claims_its_own_authorship() -> None:
    """The summary re-enters in a *user* turn, where unattributed text reads as someone
    else's claim -- so the framing says whose notes these are and how far to trust them.

    Both directions matter. Without the first the model re-derives what it already
    settled; without the second it reads the notes as licence to answer from whatever it
    recalls, which is the failure the fold is supposed to prevent.
    """
    gen = ScriptedGenerator([_reply("kept summary")])
    c = _compaction([{"role": "user", "content": "grown"}], InMemoryWorkspace(), gen)

    result = await Summarize[None]().compact(c)

    folded = result.messages[-1]["content"]
    assert "You wrote the notes below yourself" in folded  # self-authored, moments ago
    assert "do not re-derive or second-guess them" in folded  # ...and therefore trusted
    assert "general knowledge" in folded  # the converse: gaps are gaps, not recall
    notes = folded.split("<your_notes>")[1].split("</your_notes>")[0]
    assert "kept summary" in notes  # the summary rides inside the tag it is framed by


@pytest.mark.asyncio
async def test_the_default_instruction_asks_for_notes_a_future_self_will_trust() -> (
    None
):
    """The instruction is the other end of that contract: notes trusted as one's own are
    only worth trusting if they were written to be -- figures copied, sources named."""
    gen = ScriptedGenerator([_reply("kept")])

    await Summarize[None]().compact(
        _compaction([{"role": "user", "content": "grown"}], InMemoryWorkspace(), gen)
    )

    instruction = _decode(gen.prefixes[0])
    assert "writing notes to your future self" in instruction
    assert "copy figures, quotes and names exactly" in instruction


@pytest.mark.asyncio
async def test_a_first_reply_that_stored_nothing_is_asked_once_to_store() -> None:
    """Writing the fold in one turn is cheaper than storing, and reads identical.

    The model narrates the keys it would have made ("Keys: author, count") and the
    workspace stays empty -- a fold whose summary advertises a memory nothing holds.
    """
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator(
        [
            _reply("kept, keys: author"),
            _turn(_store("author", "Christie")),
            _reply("kept"),
        ]
    )

    result = await Summarize[None]().compact(
        _compaction([{"role": "user", "content": "grown"}], ws, gen)
    )

    assert ws.read("author") == "Christie"  # the nudge turned a claim into a put
    assert gen.calls == 3
    assert "kept" in result.messages[-1]["content"]


@pytest.mark.asyncio
async def test_the_store_first_nudge_is_sent_at_most_once() -> None:
    """One nudge, not a loop -- a model that will not store still has to fold."""
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_reply("kept")])  # never calls a tool, however often asked

    result = await Summarize[None](max_interactions=8).compact(
        _compaction([{"role": "user", "content": "grown"}], ws, gen)
    )

    assert gen.calls == 2  # first reply, then the nudged one -- taken at its word
    assert "kept" in result.messages[-1]["content"]
    # the nudge reached the model as a user turn, not a tool result
    assert STORE_FIRST in _decode(gen.prefixes[1])


@pytest.mark.asyncio
async def test_a_compactor_that_read_memory_back_is_not_nudged() -> None:
    """Nothing new to store is a legitimate outcome; the nudge is for doing nothing."""
    ws = InMemoryWorkspace()
    ws.write("author", "Christie")
    gen = ScriptedGenerator(
        [_turn(tool_call("storage", operate="scan")), _reply("kept: see key author")]
    )

    await Summarize[None]().compact(
        _compaction([{"role": "user", "content": "grown"}], ws, gen)
    )

    assert gen.calls == 2  # scan, reply -- no nudge in between


@pytest.mark.asyncio
async def test_a_null_workspace_gets_no_storage_tool_and_no_memory_claim() -> None:
    """With nothing durable behind it, the fold summary *is* the memory -- say so.

    Offering the tool would invite a put that is silently dropped, and the fold's usual
    "read it back from durable memory" would point the next round at a store that is
    empty by construction -- which is how a summary comes to assert facts nothing holds.
    """
    ws = NullWorkspace()
    gen = ScriptedGenerator([_reply("kept summary")])
    c = _compaction([{"role": "user", "content": "grown"}], ws, gen)

    result = await Summarize[None](max_interactions=8).compact(c)

    compacting = _decode(gen.prefixes[0])
    assert "storage" not in compacting.lower()  # no tool schema, no tool budget
    assert "tool calls first" not in compacting  # no call budget without a tool
    assert ws.compaction_guidance() in compacting  # ...its own advice instead
    folded = result.messages[-1]["content"]
    assert "kept summary" in folded
    assert "durable memory" not in folded.lower()  # no pointer at a memory that isn't
    assert "storage" not in folded.lower()
    # ...but the fold *did* change what is true here, and only after it: the notes are
    # now the whole record, so their gaps are gaps rather than things to recall.
    assert _FOLDED_GUIDANCE in folded
    assert "only record of your earlier work" in folded


def test_a_null_workspace_exports_no_tools() -> None:
    assert NullWorkspace().tools() == ()


@pytest.mark.asyncio
async def test_null_fold_workspace_keeps_actor_workspace_unchanged() -> None:
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_reply("kept summary")])
    c = _compaction([{"role": "user", "content": "grown"}], ws, gen)
    result = await Summarize[None](fold_workspace=NullWorkspace()).compact(c)
    assert gen.calls == 1
    compacting = _decode(gen.prefixes[0])
    assert NullWorkspace().compaction_guidance() in compacting
    assert "storage" not in compacting.lower()
    assert "tool calls first" not in compacting
    assert ws.snapshot() == {}
    assert ws.folded_guidance() in result.messages[-1]["content"]


@pytest.mark.asyncio
async def test_custom_transcript_serialization_bounds_historical_text() -> None:
    gen = ScriptedGenerator([_reply("kept")])
    c = _compaction([{"role": "user", "content": "old task"}], NullWorkspace(), gen)
    await Summarize[None](
        transcript_template=(
            "HISTORY_START\n{transcript}\nHISTORY_END\nSummarize only."
        )
    ).compact(c)
    prompt = _decode(gen.prefixes[0])
    assert "HISTORY_START\nuser: old task\nHISTORY_END\nSummarize only." in prompt


class _ChannelScripted(ScriptedGenerator):
    """A chat-backend-shaped script: every result reports an *open* native channel.

    ``()``, not ``None`` -- the channel is the backend's property, not one call's.
    What this pins is the toolless-fold path: a ``NullWorkspace`` fold derives a runner
    with the rollout's own native-parser policy and a ``FoldEnv`` that has no tools,
    and its prose reply must read as "called nothing", never as the misconfiguration
    ``None`` names.
    """

    provides_native_tool_calls = True

    async def agenerate_tokens(
        self, prefix_tokens: Any, sampling_params: Any = None
    ) -> GenerateResult:
        result = await super().agenerate_tokens(prefix_tokens, sampling_params)
        result.native_tool_calls = ()
        return result


@pytest.mark.asyncio
async def test_a_toolless_fold_under_a_native_policy_keeps_its_reply() -> None:
    """The regression the first live NullWorkspace fold hit: the fold itself died.

    The fold's final act is a prose reply; under the native parser that reply carries
    no structured call, and the channel must still read as open, or the parser's
    loud misconfiguration error fires mid-fold and takes the whole rollout with it.
    """
    ws = NullWorkspace()
    gen = _ChannelScripted(["reasoning\n</think>\n\nkept summary"])
    c = replace(
        _compaction([{"role": "user", "content": "grown"}], ws, gen),
        agent=_folding_runner(gen, parser=NativeToolCallParser()),
    )

    result = await Summarize[None](max_interactions=8).compact(c)

    assert "kept summary" in result.messages[-1]["content"]


@pytest.mark.asyncio
async def test_transcript_tool_calls_are_shown_inert() -> None:
    """Verbatim <tool_call> blocks in the transcript are examples the model imitates."""
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_reply("kept")])
    grown = [{"role": "assistant", "content": _turn(_store("author", "Christie"))}]
    c = _compaction(grown, ws, gen)

    await Summarize[None]().compact(c)

    prompt = _decode(gen.prefixes[0])
    assert "<tool_call>" not in prompt.split("</think>")[0]  # the transcript is inert
    assert "[tool call:" in prompt
    assert "author" in prompt  # content still readable


@pytest.mark.asyncio
async def test_empty_reply_does_not_fold_to_an_empty_context() -> None:
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_reply("")])  # e.g. a thinking model truncated mid-reason
    c = _compaction([{"role": "user", "content": "grown"}], ws, gen)

    result = await Summarize[None]().compact(c)

    assert _EMPTY_KEPT in result.messages[-1]["content"]  # no blank hole in the context


_SYSTEM_ONLY: tuple[Message, ...] = ({"role": "system", "content": "s"},)
_FEW_SHOT: tuple[Message, ...] = (
    {"role": "system", "content": "s"},
    {"role": "user", "content": "example q"},
    {"role": "assistant", "content": "example a"},
    {"role": "user", "content": "Question: q"},
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("seed", "carried"),
    [
        (_SYSTEM_ONLY, 1),  # no request turn to rewrite, so the fold appends one
        (_FEW_SHOT, 3),  # the example pair is kept; only the last user turn folds
    ],
)
async def test_fold_rewrites_the_request_turn_whatever_shape_the_seed_has(
    seed: tuple[Message, ...], carried: int
) -> None:
    gen = ScriptedGenerator([_reply("kept")])
    grown = [{"role": "user", "content": "grown"}]
    c = replace(_compaction(grown, InMemoryWorkspace(), gen), seed=seed)

    result = await Summarize[None]().compact(c)

    # every turn before the request is untouched; the request turn carries the state
    assert result.messages[:carried] == seed[:carried]
    folded = result.messages[-1]
    assert folded["role"] == "user"
    assert "kept" in folded["content"]
    assert not folded["content"].startswith("\n")  # no blank slot where a prompt wasn't


# -- what the fold contributes to the rollout's log ------------------------------------


@pytest.mark.asyncio
async def test_a_fold_returns_one_turn_per_generation() -> None:
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_turn(_store("log", "detail")), _reply("kept summary")])
    c = _compaction([{"role": "user", "content": "grown context"}], ws, gen)

    result = await Summarize[None](max_interactions=8).compact(c)

    assert len(result.turns) == 2  # the agent retags them as fold turns


@pytest.mark.asyncio
async def test_a_folds_generations_are_trainable_and_its_transcript_is_masked() -> None:
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_turn(_store("log", "detail")), _reply("kept summary")])
    c = _compaction([{"role": "user", "content": "grown context"}], ws, gen)

    result = await Summarize[None](max_interactions=8).compact(c)

    codec = ChatCodec(FakeChatTokenizer())
    packed = regions(result.turns)
    decoded = "".join(
        codec.decode(
            [t for t, keep in zip(seq.tokens, seq.trainable, strict=True) if keep]
        )
        for seq in packed
    )
    # exactly the model's two assistant turns; the transcript and the tool results
    # around them are masked out.
    assert decoded == _turn(_store("log", "detail")) + _reply("kept summary")
    assert "grown context" not in decoded  # the transcript observation is not trainable
    assert all(t.logprobs is None for t in result.turns)  # backend returned none


@pytest.mark.asyncio
async def test_budget_forces_a_reply() -> None:
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_turn(_store("p", "c"))])  # only ever calls a tool
    compactor = Summarize[None](max_interactions=3)

    result = await compactor.compact(
        _compaction([{"role": "user", "content": "c"}], ws, gen)
    )

    assert gen.calls == 3 + 1  # k interactions + one forced no-tool reply
    assert len(result.messages) >= 1
    assert len(result.turns) == 4  # ...and every one of them is on the log


@pytest.mark.asyncio
async def test_a_custom_instruction_may_name_the_interaction_budget() -> None:
    """``instruction`` is an author-supplied template, and the call budget is one of the
    things it can talk about -- ``{k}`` is the number `_BUDGET` renders, offered to a
    template that wants to phrase it differently. Dropping it from the substitution
    turns such a template into a ``KeyError`` raised mid-rollout, at the first fold."""
    gen = ScriptedGenerator([_reply("kept summary")])
    compactor = Summarize[None](
        max_interactions=3,
        instruction="Compact.{workspace}{budget} Budget: {k}.",
    )

    await compactor.compact(
        _compaction([{"role": "user", "content": "grown"}], InMemoryWorkspace(), gen)
    )

    assert "Budget: 3." in _decode(gen.prefixes[0])


@pytest.mark.asyncio
async def test_compactor_reads_state_to_trigger_and_resets_it_on_compact() -> None:
    gen = ScriptedGenerator([_reply("kept")])
    compactor = Summarize[BudgetState](
        trigger=AnyOf(PromptTokens(10_000), StateCounter("remaining", "maximum"))
    )

    # the trigger reads the state field, not just the prompt length:
    assert compactor.trigger.fires((1,), BudgetState(remaining=0))
    assert not compactor.trigger.fires((1,), BudgetState(remaining=3))

    c = Compaction(
        messages=({"role": "user", "content": "x"},),
        seed=({"role": "system", "content": "s"},),
        state=BudgetState(remaining=0),
        workspace=InMemoryWorkspace(),
        agent=_folding_runner(gen),
    )
    result = await compactor.compact(c)
    assert result.state == BudgetState(remaining=5)  # the trigger relieved itself


# -- the fold is the rollout's own runner, derived -------------------------------------


def _xml_store(key: str, value: str) -> str:
    """The same storage call in Qwen3.5's XML dialect, which Hermes cannot read."""
    return (
        "<tool_call>\n<function=storage>\n"
        + "<parameter=operate>\nput\n</parameter>\n"
        + f"<parameter=key>\n{key}\n</parameter>\n"
        + f"<parameter=value>\n{value}\n</parameter>\n"
        + "</function>\n</tool_call>"
    )


@pytest.mark.asyncio
async def test_the_fold_speaks_the_rollout_s_tool_dialect() -> None:
    """The rollout's parser *is* the dialect: a Qwen rollout folds in Qwen XML.

    The compactor holds no parser of its own to fall back to, so this only works if the
    runner carried its own across ``derive`` -- under the Hermes parser the same XML is
    a nameless call that dispatch reports back, and nothing reaches the workspace.
    """
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator([_xml_store("author", "Christie"), _reply("kept")])
    c = replace(
        _compaction([{"role": "user", "content": "grown"}], ws, gen),
        agent=_folding_runner(gen, QwenXMLToolCallParser()),
    )

    await Summarize[None](max_interactions=4).compact(c)

    assert ws.read("author") == "Christie"


@pytest.mark.asyncio
async def test_a_rollout_folds_by_deriving_the_runner_it_is_already_using() -> None:
    """End to end through ``Agent``: no bag of parts crosses the compaction seam."""
    ws = InMemoryWorkspace()
    gen = ScriptedGenerator(
        [_turn(_store("author", "Christie")), _reply("kept summary")], logprobs=-0.5
    )
    runner: Agent[Any, Any] = toy_runner(
        gen,
        parser=HermesToolCallParser(),
        compactor=Summarize[None](max_prompt_tokens=1, max_interactions=4),
    )

    rs = await runner.start("Question: q", workspace=ws)
    folded = await runner.advance(rs)  # the prompt is already over budget: fold now

    assert folded.folds == 1
    assert ws.read("author") == "Christie"  # the fold ran the workspace's own tools
    assert [t.tag for t in folded.turns] == ["fold", "fold"]
    assert "kept summary" in folded.messages[-1]["content"]
    assert all(t.format_id == runner.policy.format.name for t in folded.turns)


def test_the_default_trigger_only_fires_past_the_token_budget() -> None:
    compactor = Summarize[None](max_prompt_tokens=5)
    assert not compactor.trigger.fires((1, 2, 3), None)
    assert compactor.trigger.fires(tuple(range(6)), None)


def test_workspace_exports_tools_and_null_exports_none() -> None:
    ws = InMemoryWorkspace()
    assert [t.name for t in ws.tools()] == ["storage"]
    assert isinstance(ws.tools()[0], StorageTool)
    assert [t.name for t in ws.compaction_tools()] == ["storage"]
    assert NullWorkspace().tools() == ()
    assert NullWorkspace().compaction_tools() == ()


class _TriggerlessCompactor(ContextManager[Any]):
    """Compacts on request only: no trigger, so the agent never folds by itself."""

    async def compact(self, compaction: Compaction[Any]) -> CompactionResult[Any]:
        return CompactionResult(compaction.seed, compaction.state)


async def test_a_manager_without_a_trigger_folds_only_when_asked() -> None:
    runner = toy_runner(ScriptedGenerator(["go"]), compactor=_TriggerlessCompactor())
    root = await runner.start("hi")
    stepped = await runner.advance(root)
    assert stepped.folds == 0  # never on its own

    folded = await runner.compact(stepped)
    assert folded.folds == 1
    assert folded.messages == root.seed
    assert folded.context_version == stepped.context_version + 1


@dataclass(frozen=True)
class RoundState(SubmitFields, RoundBudget):
    """The state shape `budget` is written for: both harness mixins, nothing else."""


class _Probe(BaseTool):
    name = "probe"
    description = "look at something"
    parameters = {"type": "object", "properties": {}}

    def call(self, params: str | dict[str, Any], **kwargs: object) -> str:
        del params, kwargs
        return "nothing here"


class _RoundEnv(ToolEnv[RoundState]):
    """A two-calls-per-round task. Neither half of the budget is written by hand."""

    def __init__(self, budget: int) -> None:
        super().__init__([_Probe()])
        self._budget = budget

    async def initial_state(self) -> RoundState:
        return RoundState(calls_remaining=self._budget, calls_max=self._budget)


@pytest.mark.asyncio
async def test_the_default_budget_compactor_folds_a_spent_round() -> None:
    """`build_budget()` with no arguments, end to end, on a state that just mixes in.

    The point of the defaults is that they line up: `RoundBudget` supplies the field
    names `StateCounter` looks for, `ToolEnv.on_tool_calls` spends them, and the fold
    refills them. Until both mixin and decrement existed, `{"name": "budget"}` folded
    only on tokens -- the counter half read a field nothing in the library defined, so
    it was dead unless a script wrote its own countdown.
    """
    gen = ScriptedGenerator(
        [
            _turn(tool_call("probe")),  # 2 -> 1
            _turn(tool_call("probe")),  # 1 -> 0: the round is spent
            _reply("kept: probed twice, nothing found"),  # the fold's one turn
            _turn(tool_call("submit", answer="42")),
        ]
    )
    runner = toy_runner(
        gen,
        env=_RoundEnv(2),
        parser=HermesToolCallParser(),
        compactor=build_budget(),  # every knob at its default
        max_steps=4,
    )

    rs = await complete(runner, "Question: q")

    assert rs.folds == 1  # the counter fired, not the 8192-token default
    assert [t.tag for t in rs.turns] == ["task", "task", "fold", "task"]
    assert rs.state.calls_remaining == 2  # refilled from `calls_max` by the relief
    assert rs.state.answer == "42"
    assert rs.done


def test_a_round_budget_that_refills_to_nothing_is_refused() -> None:
    """A fold takes no env step, so a trigger that survives its own relief is a hang.

    `RoundBudget` defaults to zero -- unconfigured, not unlimited -- and a state left
    there would fire, fold, refill to zero and fire again forever, advancing nothing.
    """
    with pytest.raises(ValueError, match="calls_max"):
        StateCounter().relieve(RoundState())


# -- Trigger.check: the state contract, settled at the root checkpoint ----------------


@dataclass(frozen=True)
class _BudgetlessState(SubmitFields):
    """A perfectly good task state that simply never opted into `RoundBudget`."""


class _BudgetlessEnv(ToolEnv[_BudgetlessState]):
    async def initial_state(self) -> _BudgetlessState:
        return _BudgetlessState()


class _UnconfiguredEnv(ToolEnv[RoundState]):
    """Mixes `RoundBudget` in but never sets it, so `calls_max` is still its zero."""

    async def initial_state(self) -> RoundState:
        return RoundState()


@pytest.mark.asyncio
async def test_a_state_without_a_round_budget_is_refused_at_the_root() -> None:
    """The missing-field failure moves from inside an expansion to before the run.

    `StateCounter.fires` reads `getattr(state, remaining)`, and it is asked before the
    very first turn -- so a state with no such field used to raise a bare
    `AttributeError` that the scheduler logged as one dead rollout among many.
    """
    runner = toy_runner(
        parser=HermesToolCallParser(),
        gen=ScriptedGenerator(["x"]),
        env=_BudgetlessEnv([]),
        compactor=build_budget(),
    )

    with pytest.raises(TypeError, match="RoundBudget"):
        await runner.start("q")


@pytest.mark.asyncio
async def test_a_round_budget_left_at_its_default_is_refused_at_the_root() -> None:
    """`calls_max=0` means unconfigured, and the fold that would hang never starts."""
    runner = toy_runner(
        parser=HermesToolCallParser(),
        gen=ScriptedGenerator(["x"]),
        env=_UnconfiguredEnv([]),
        compactor=build_budget(),
    )

    with pytest.raises(ValueError, match="calls_max"):
        await runner.start("q")


@pytest.mark.asyncio
async def test_a_state_that_carries_the_budget_passes_the_check() -> None:
    runner = toy_runner(
        parser=HermesToolCallParser(),
        gen=ScriptedGenerator(["x"]),
        env=_RoundEnv(2),
        compactor=build_budget(),
    )

    rs = await runner.start("q")

    assert rs.state == RoundState(calls_remaining=2, calls_max=2)


@pytest.mark.asyncio
async def test_a_runner_without_a_compactor_asks_no_trigger_at_all() -> None:
    """`compactor=None` is a single unbounded context: there is no contract to check."""
    runner = toy_runner(
        ScriptedGenerator(["x"]), env=_BudgetlessEnv([]), parser=HermesToolCallParser()
    )

    assert await runner.start("q") is not None


def test_any_of_checks_every_member() -> None:
    """The composite's contract is the union of its members', not the first one's."""
    trigger = AnyOf(PromptTokens(10), StateCounter())

    trigger.check(RoundState(calls_remaining=2, calls_max=2))  # both satisfied
    with pytest.raises(TypeError, match="RoundBudget"):
        trigger.check(_BudgetlessState())


def test_a_prompt_only_trigger_has_no_state_contract() -> None:
    PromptTokens(10).check(object())  # no contract to state, so nothing to refuse


@pytest.mark.asyncio
async def test_the_fold_states_its_length_budget_and_is_held_to_it() -> None:
    """The kept context is bounded twice: said in the prompt, capped at the sampler.

    Saying it is what makes the fold worth doing -- a summary as long as the transcript
    leaves the prompt as full as it was, so `PromptTokens` fires again on the next turn
    and the rollout folds until its steps run out. Capping it is what keeps the fold
    *runnable*: it is handed the grown transcript whole, so its prompt is the very one
    that just overran the budget, and asking the task's own generation budget on top of
    that is a request the backend refuses -- which kills the rollout at the fold.
    """
    seen: dict[str, Any] = {}
    compaction = _compaction(
        [{"role": "user", "content": "hi"}],
        NullWorkspace(),
        ScriptedGenerator(["kept context"]),
    )
    real_derive = compaction.agent.derive

    def spy(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return real_derive(**kwargs)

    compaction.agent.derive = spy  # type: ignore[method-assign]
    compactor: Summarize[Any] = Summarize(max_reply_tokens=777)

    await compactor.compact(compaction)

    assert seen["sampling_params"].max_tokens == 777
    assert "777 tokens" in str(seen["system_prompt"])
    assert "259 words" in str(seen["system_prompt"])  # 777 // 3
