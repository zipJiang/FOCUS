from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from focus.codec import ChatCodec
from focus.generation import (
    GenerateResult,
    PolicyFormat,
    SamplingParams,
    TokenId,
)
from focus.generation.parsing import HermesToolCallParser, TextActionParser
from focus.generation.policy import Policy
from focus.harness import (
    TASK,
    Compaction,
    CompactionResult,
    Compactor,
    Env,
    InMemoryWorkspace,
    NullWorkspace,
    RolloutState,
    Runner,
    StepResult,
    StorageTool,
    Trigger,
    Workspace,
    regions,
)
from tests.helpers import (
    FakeChatTokenizer,
    ScriptedGenerator,
    SubmitEnv,
    ToyEnv,
    ToyState,
    WhenTrigger,
    complete,
    rollout,
    toy_runner,
    trainable_text,
    turn,
)


def memory(rs: RolloutState[Any]) -> InMemoryWorkspace:
    """The KV surface of a rollout's workspace.

    ``RolloutState.workspace`` is the base :class:`Workspace`, which mandates no store
    at all; every rollout in this module wires the in-memory one, so the reads below go
    through the type that actually has ``read``/``write``.
    """
    assert isinstance(rs.workspace, InMemoryWorkspace)
    return rs.workspace


class QuietMemory(InMemoryWorkspace):
    """Store without guidance prose -- keeps length-based compactors honest."""

    def guidance(self) -> str:
        return ""

    def compaction_guidance(self) -> str:
        return ""


@pytest.mark.asyncio
async def test_start_composes_workspace_guidance_into_the_seed() -> None:
    runner = toy_runner(ScriptedGenerator(["stop"]))
    rs = await runner.start("hi", workspace=InMemoryWorkspace())
    assert "durable memory" in rs.messages[0]["content"]
    bare = await runner.start("hi", workspace=NullWorkspace())
    assert bare.messages[0]["content"] == "sys"


@pytest.mark.asyncio
async def test_root_takes_a_seed_verbatim() -> None:
    """The fold's entry point: an opening already built, not composed from a prompt."""
    runner = toy_runner(ScriptedGenerator(["stop"]))
    seed = ({"role": "system", "content": "verbatim"},)
    rs = await runner.root(seed, workspace=InMemoryWorkspace())
    assert rs.messages == seed == rs.seed  # no guidance folded in
    assert rs.state == ToyState()
    assert not rs.turns


@pytest.mark.asyncio
async def test_run_from_is_start_then_run() -> None:
    gen = ScriptedGenerator(["stop"])
    runner = toy_runner(gen)
    result = await runner.run_from("q", workspace=NullWorkspace())
    assert result.done
    assert result.state == ToyState(1)


@pytest.mark.asyncio
async def test_run_loops_until_the_env_says_done() -> None:
    gen = ScriptedGenerator(["go", "go", "stop"])
    result = await complete(toy_runner(gen), "hello")
    assert len(result.turns) == 3  # one turn per generate, appended in order
    assert result.turns_taken == 3
    assert all(t.tag == "task" and t.transition is not None for t in result.turns)
    assert result.done
    last = result.turns[-1].transition
    assert last is not None
    assert last.done
    assert result.state == ToyState(3)
    assert result.reward == 1.0
    assert result.folds == 0
    assert len(gen.prefixes) == 3  # one generation per turn


@pytest.mark.asyncio
async def test_an_append_only_run_of_turns_packs_into_one_region() -> None:
    """Nothing rewrote the render, so every turn's prefix extends the last: one pack."""
    result = await complete(toy_runner(ScriptedGenerator(["go", "go", "stop"])), "hi")

    packed = regions(result.turns)
    assert len(packed) == 1
    assert packed[0].tag == "task"
    assert len(packed[0].spans) == 3  # ...but each turn is still locatable inside it
    assert trainable_text(result) == "gogostop"


@pytest.mark.asyncio
async def test_the_prompt_grows_with_each_assistant_and_observation_turn() -> None:
    gen = ScriptedGenerator(["go", "stop"])
    await complete(toy_runner(gen), "hi")
    # the second generation ran from a longer prefix: seed + assistant + user obs
    assert len(gen.prefixes[1]) > len(gen.prefixes[0])
    codec = ChatCodec(FakeChatTokenizer())
    assert gen.prefixes[0] == codec.render(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    )


@pytest.mark.asyncio
async def test_a_turn_records_the_prefix_that_was_actually_sent() -> None:
    """No reconstruction: the runner has the id list, so the turn stores that list."""
    gen = ScriptedGenerator(["go", "stop"])
    result = await complete(toy_runner(gen), "hi")

    assert [t.prefix for t in result.turns] == gen.prefixes


#: Past CPython's cached small ints (up to 256), so every rendered id is a *fresh*
#: object -- which is what a real tokenizer returns, and the only regime in which the
#: sharing below is measurable at all. The ord-based fakes elsewhere in these tests sit
#: entirely inside the cached range, where every equal id is the same object for free.
BIG_ID = 1000


class BigIdTokenizer(FakeChatTokenizer):
    """The fake template with every id shifted past the cached-small-int range."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return [
            BIG_ID + i
            for i in super().encode(text, add_special_tokens=add_special_tokens)
        ]

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return super().decode(
            [t - BIG_ID for t in token_ids], skip_special_tokens=skip_special_tokens
        )

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        return_tensors: str | None,
    ) -> list[int]:
        return [
            BIG_ID + i
            for i in super().apply_chat_template(
                conversation,
                tools=tools,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                return_dict=return_dict,
                return_tensors=return_tensors,
            )
        ]


class BigIdGenerator(ScriptedGenerator):
    """A scripted generator speaking :class:`BigIdTokenizer`'s ids.

    The completion's ids have to be the template's own, or the next turn's render does
    not extend ``prefix + tokens`` and the log stops chaining -- which is exactly the
    condition the sharing is keyed on.
    """

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        result = await super().agenerate_tokens(prefix_tokens, sampling_params)
        return replace(result, tokens=tuple(BIG_ID + t for t in result.tokens))


@pytest.mark.asyncio
async def test_nested_prefixes_share_the_ids_they_agree_on() -> None:
    """The log's prefixes cost one id object per position, not one per slot.

    Each turn stores the whole prefix it was conditioned on and those prefixes are
    nested, so an N-turn log holds O(N x context) id *slots* -- that part is the price
    of a turn that can name the ids it saw. What must not also be quadratic is the
    number of id *objects*: a real tokenizer allocates a fresh ``int`` per id it
    returns, so re-rendering the conversation every turn once cost 28 bytes per slot on
    top of the 8-byte pointer -- ~13 MiB for one 32-turn agentic rollout, over 95% of
    everything it weighs, and ~160 MiB for a tree of 300 checkpoints.

    Counted in objects rather than in bytes: the bound is then exact arithmetic, true
    on any machine and any build, where a ``tracemalloc`` threshold would be a guess
    that a debug interpreter or a future ``int`` layout could break.
    """
    runner = Runner(
        policy=Policy(
            format=PolicyFormat(
                "custom", ChatCodec(BigIdTokenizer()), TextActionParser()
            ),
            complete=(BigIdGenerator(["go"] * 12)).agenerate_tokens,
            version="policy",
        ),
        env=ToyEnv(done_at=12),
        system_prompt="s" * 200,  # a system prompt worth re-rendering
        max_steps=12,
    )

    result = await complete(runner, "hi")

    slots = sum(len(t.prefix) for t in result.turns)
    context = len(result.turns[-1].prefix)
    distinct = {id(token) for t in result.turns for token in t.prefix}
    assert slots > 5 * context  # the prefixes really are nested, i.e. quadratic
    assert len(distinct) == context  # ...and still hold one object per position


@pytest.mark.asyncio
async def test_a_negative_step_budget_runs_until_the_env_ends() -> None:
    """``max_steps=-1`` is unlimited: the loop only stops when the env says done.

    ``range(-1)`` is empty, so handing the sentinel to ``range`` as-is would take
    *zero* turns -- the opposite of unlimited -- and then ``finish`` would force an
    ending on a rollout that never ran.
    """
    gen = ScriptedGenerator(["go", "go", "go", "stop"])
    result = await complete(toy_runner(gen, max_steps=-1), "x")

    assert result.turns_taken == 4
    assert result.done
    assert not result.truncated
    assert len(gen.prefixes) == 4  # no forced extra turn


@pytest.mark.asyncio
async def test_a_per_call_negative_budget_overrides_the_runners() -> None:
    """``run(..., max_steps=-1)`` is unlimited even when the runner is bounded."""
    gen = ScriptedGenerator(["go", "go", "stop"])
    result = await complete(toy_runner(gen, max_steps=1), "x", max_steps=-1)

    assert result.turns_taken == 3
    assert result.done
    assert not result.truncated


@pytest.mark.asyncio
async def test_run_truncates_at_max_steps_without_a_finish_prompt() -> None:
    gen = ScriptedGenerator(["go"])  # never stops
    result = await complete(toy_runner(gen, max_steps=4, finish_prompt=None), "x")
    assert result.turns_taken == 4
    assert not result.done
    assert result.truncated
    assert not result.forkable
    assert len(gen.prefixes) == 4  # no extra generation


class AnsweringEnv(ToyEnv):
    """Records the forced answer into the state, as a real env's answer path would."""

    async def finish(self, state: ToyState, action: str) -> StepResult[ToyState]:
        return StepResult(ToyState(state.n + 1), reward_outcome=0.5, done=True)


@pytest.mark.asyncio
async def test_spending_the_budget_forces_one_last_answer_turn() -> None:
    """A rollout out of steps must still answer -- in-band, not bolted on outside."""
    gen = ScriptedGenerator(["go", "go", "go", "my best answer"])
    runner = toy_runner(
        gen, env=AnsweringEnv(), max_steps=3, finish_prompt="out of steps, answer now"
    )

    result = await complete(runner, "x")

    assert len(gen.prefixes) == 4  # three budgeted turns, then the forced one
    assert result.done
    assert result.truncated  # over, and tellably not on its own terms
    assert result.state == ToyState(4)  # env.finish ran, so the answer reached the task
    assert result.reward == 0.5
    # the finish prompt is in the conversation, and the answer is the turn after it
    assert result.messages[-2]["content"] == "out of steps, answer now"
    assert result.messages[-1] == {"role": "assistant", "content": "my best answer"}


@pytest.mark.asyncio
async def test_the_forced_answer_is_trainable_and_its_prompt_is_masked() -> None:
    """The forced turn belongs to the log, not to a hand-built prefix elsewhere."""
    gen = ScriptedGenerator(["go", "my best answer"])
    runner = toy_runner(
        gen, env=AnsweringEnv(), max_steps=1, finish_prompt="answer now"
    )

    result = await complete(runner, "x")

    trainable = trainable_text(result)
    assert trainable == "go" + "my best answer"  # both generations, nothing else
    assert "answer now" not in trainable  # the finish prompt is the world's, not ours


@pytest.mark.asyncio
async def test_the_forced_answer_is_drawn_under_the_runs_own_params() -> None:
    """`finish` takes the per-call params like every other turn, so a rollout sampled
    from a wider proposal answers under that law rather than reverting to the runner
    default -- which would leave the edge's provenance naming a law its last turn was
    not sampled under."""
    gen = ScriptedGenerator(["go", "my best answer"])
    runner = toy_runner(
        gen,
        env=AnsweringEnv(),
        max_steps=1,
        finish_prompt="answer now",
        sampling_params=SamplingParams(temperature=0.2),
    )

    explore = SamplingParams(temperature=1.5)

    result = await complete(runner, "x", sampling_params=explore)

    assert result.truncated  # the budget, not the env, ended it
    assert len(gen.params) == 2  # the budgeted turn, then the forced one
    assert gen.params[-1] == gen.params[0]  # same law across the whole line
    assert gen.params[-1] is not None
    assert gen.params[-1].temperature == 1.5


@pytest.mark.asyncio
async def test_finish_is_a_no_op_once_the_episode_ended() -> None:
    gen = ScriptedGenerator(["stop"])
    runner = toy_runner(gen, max_steps=4)
    result = await complete(runner, "x")
    assert result.done
    assert not result.truncated

    assert await runner.finish(result) is result  # nothing to force
    assert len(gen.prefixes) == 1


class ToyCompactor(Compactor[ToyState]):
    """Compact when the context grows past a threshold; persist and reset to seed.

    Length-based so it *relieves its own trigger*: resetting to the (short) seed drops
    the prompt back under the threshold, as the front-loaded check requires. Generates
    nothing, so it contributes no turns of its own -- the runner's marker records it.
    """

    # the seed is short; one step grows it past 10
    trigger: Trigger[ToyState] = WhenTrigger(lambda prompt, state: len(prompt) > 10)

    async def compact(
        self, compaction: Compaction[ToyState]
    ) -> CompactionResult[ToyState]:
        # a real compactor reads compaction.messages; we just persist and reset to seed
        assert isinstance(compaction.workspace, InMemoryWorkspace)
        compaction.workspace.write("summary", f"compacted at step {compaction.state.n}")
        return CompactionResult(compaction.seed, compaction.state)


@pytest.mark.asyncio
async def test_compaction_persists_to_workspace_and_resets_the_messages() -> None:
    gen = ScriptedGenerator(["go", "stop"])
    ws = QuietMemory()
    codec = ChatCodec(FakeChatTokenizer())
    runner = toy_runner(gen, compactor=ToyCompactor())
    result = await complete(runner, "hi", workspace=ws)

    assert result.folds == 1
    assert ws.read("summary") == "compacted at step 1"  # compaction wrote to workspace
    # a silent fold generates nothing and still leaves its one marker turn -- which is
    # the whole record of it, and what makes the task turn after it the boundary
    assert result.turns_taken == 2
    assert [t.tag for t in result.turns] == ["task", "fold", "task"]
    assert result.boundaries() == (2,)

    # compaction was its own turn: the run generated twice (go, then stop), not thrice
    seed = codec.render(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    )
    assert len(gen.prefixes) == 2  # the compaction turn did not generate
    assert gen.prefixes[1] == seed  # post-compaction generation ran from the reset seed
    assert gen.prefixes[0] == seed  # (the first did too)


@pytest.mark.asyncio
async def test_compaction_is_its_own_turn_without_generation() -> None:
    gen = ScriptedGenerator(["go", "stop"])
    runner = toy_runner(gen, compactor=ToyCompactor())
    rs = await runner.start("hi", workspace=QuietMemory())
    rs = await runner.advance(rs)  # a step: generates "go", grows the context
    assert len(gen.prefixes) == 1
    assert len(rs.turns) == 1

    compacted = await runner.advance(rs)  # compaction alone: no generation
    assert len(gen.prefixes) == 1  # the generator was not called this turn
    assert compacted.folds == 1
    # the log is append-only: the fold adds its marker and rewrites nothing
    assert compacted.turns[:1] == rs.turns
    marker = compacted.turns[1]
    assert (marker.tag, marker.prefix, marker.tokens) == ("fold", (), ())
    assert marker.transition is None  # a fold takes no env step
    assert compacted.messages == rs.seed  # messages replaced by the compactor's output


class StubbornCompactor(ToyCompactor):
    """Fires twice in a row: the first fold does not relieve its own trigger."""

    def __init__(self, folds: int = 2) -> None:
        self._folds = folds
        self.fired = 0
        self.trigger = WhenTrigger(lambda prompt, state: self.fired < self._folds)

    async def compact(
        self, compaction: Compaction[ToyState]
    ) -> CompactionResult[ToyState]:
        self.fired += 1
        return CompactionResult(compaction.seed, compaction.state)


@pytest.mark.asyncio
async def test_two_folds_in_a_row_are_one_rewrite_with_one_boundary() -> None:
    """The log says the context was rewritten, not how many attempts it took.

    Both folds are silent, so both leave a marker -- but they are one *run* of fold
    turns with no work between them, and what a reader of ``folds`` is after (how many
    restorable contexts this line has) is the run, not the attempt. The boundary is
    likewise one: the first task turn after the run.
    """
    gen = ScriptedGenerator(["go", "go", "stop"])
    runner = toy_runner(gen, compactor=StubbornCompactor(), max_steps=8)

    result = await complete(runner, "hi", workspace=QuietMemory())

    assert [t.tag for t in result.turns][:3] == ["fold", "fold", "task"]
    assert result.folds == 1
    assert result.boundaries() == (2,)


class CountingTrigger(Trigger[Any]):
    """Counts the times the runner asked it anything, and always says no."""

    def __init__(self) -> None:
        self.asked = 0

    def fires(self, prompt: Sequence[TokenId], state: Any) -> bool:
        del prompt, state
        self.asked += 1
        return False


class WatchedCompactor(ToyCompactor):
    """:class:`ToyCompactor` under a trigger the test holds a reference to."""

    def __init__(self, trigger: Trigger[ToyState]) -> None:
        self.trigger = trigger


@pytest.mark.asyncio
async def test_a_runner_with_no_compactor_asks_no_fold_question() -> None:
    """``compactor=None`` is not a compactor answering "no" -- it is no question at all.

    The difference is a render and a call per turn of every rollout in the search, which
    is why the no-op strategy that existed to answer ``False`` is gone: a runner with
    nothing to fold now skips the check structurally, and the counter here is what says
    so rather than a reading of the branch.
    """
    trigger = CountingTrigger()

    watched = await complete(
        toy_runner(
            ScriptedGenerator(["go", "go", "stop"]),
            compactor=WatchedCompactor(trigger),
            max_steps=8,
        ),
        "hi",
    )
    assert trigger.asked == len(watched.turns) > 1  # once per turn taken

    asked_before = trigger.asked
    plain = await complete(
        toy_runner(ScriptedGenerator(["go", "go", "stop"]), max_steps=8), "hi"
    )

    assert len(plain.turns) == len(watched.turns)  # the same rollout, unfolded
    assert trigger.asked == asked_before  # and nothing was asked to run it


class TalkativeCompactor(ToyCompactor):
    """A fold that generated -- the shape an :class:`AgenticCompactor` returns."""

    async def compact(
        self, compaction: Compaction[ToyState]
    ) -> CompactionResult[ToyState]:
        from focus.harness import Turn

        return CompactionResult(
            compaction.seed,
            compaction.state,
            turns=(Turn(prefix=(9, 9), tokens=(1, 2), tag="fold"),),
        )


@pytest.mark.asyncio
async def test_a_fold_that_generated_joins_the_log_tagged_and_bounded() -> None:
    """task…, fold…, task(boundary) -- one log, and the fold is tellable apart in it."""
    gen = ScriptedGenerator(["go", "stop"])
    runner = toy_runner(gen, compactor=TalkativeCompactor())

    result = await complete(runner, "hi", workspace=QuietMemory())

    assert [t.tag for t in result.turns] == ["task", "fold", "task"]
    assert result.boundaries() == (2,)
    assert result.folds == 1
    assert result.turns_taken == 2  # the fold took no env step
    assert result.done  # ...and `done` reads the last *transition*, not the last turn
    assert result.turns[1].transition is None


class UntaggedCompactor(ToyCompactor):
    """A generative fold that left its turns at ``Turn``'s default tag."""

    async def compact(
        self, compaction: Compaction[ToyState]
    ) -> CompactionResult[ToyState]:
        from focus.harness import Turn

        return CompactionResult(
            compaction.seed,
            compaction.state,
            turns=(Turn(prefix=(9, 9), tokens=(1, 2)),),  # tag defaults to "task"
        )


@pytest.mark.asyncio
async def test_the_runner_tags_a_folds_turns_itself() -> None:
    """The tag is the fold's only record, so the runner writes it rather than hoping.

    ``Compactor.compact`` already promises this ("the runner tags whatever is returned
    ``fold``"), and four readings now rest on it: ``folds``, ``boundaries()``, where
    ``iter_run`` yields, and which checkpoints are forkable. A
    compactor that left ``Turn``'s default tag used to cost a mislabelled pack; it must
    not cost the fold itself.
    """
    gen = ScriptedGenerator(["go", "stop"])
    runner = toy_runner(gen, compactor=UntaggedCompactor(), max_steps=8)

    rs = await runner.advance(await runner.start("hi", workspace=QuietMemory()))
    region = await anext(runner.iter_run(rs))  # the fold, and the region ends there

    assert [t.tag for t in region.turns] == ["task", "fold"]
    assert region.folds == 1
    assert region.turns_taken == 1  # the fold's turns are not time steps
    assert len(gen.prefixes) == 1  # the region stopped at the fold, it did not go on


@pytest.mark.asyncio
async def test_regions_never_mix_task_and_fold_turns() -> None:
    gen = ScriptedGenerator(["go", "stop"])
    runner = toy_runner(gen, compactor=TalkativeCompactor())

    result = await complete(runner, "hi", workspace=QuietMemory())

    assert [seq.tag for seq in regions(result.turns)] == ["task", "fold", "task"]


class WritingEnv(Env[ToyState, str]):
    """Records the action into the per-step workspace (via StorageTool); 'stop' ends.

    Proves the runner threads the branch's workspace into ``env.step`` -- and thus into
    a tool that honors the call-time ``workspace`` -- rather than a captured one.
    """

    async def reset(self) -> ToyState:
        return ToyState()

    async def step(
        self, state: ToyState, action: str, *, workspace: Workspace | None = None
    ) -> StepResult[ToyState]:
        assert isinstance(workspace, InMemoryWorkspace)
        StorageTool().call(
            json.dumps({"operate": "put", "key": "last", "value": action}),
            workspace=workspace,
        )
        nxt = ToyState(state.n + 1)
        done = action == "stop"
        return StepResult(
            next_state=nxt,
            messages=({"role": "user", "content": f"obs{nxt.n}"},),
            done=done,
        )


@pytest.mark.asyncio
async def test_fork_isolates_workspace_and_diverges() -> None:
    gen = ScriptedGenerator(["go", "stop"])
    runner = toy_runner(gen, env=WritingEnv())
    # one shared prefix turn: the env writes last="go" into the base workspace
    base = await runner.start("hi", workspace=InMemoryWorkspace())
    base = await runner.advance(base)
    assert memory(base).read("last") == "go"

    fork_a, fork_b = base.snapshot(), base.snapshot()
    # sentinel writes on each branch are invisible to the base and to each other
    memory(fork_a).write("id", "A")
    memory(fork_b).write("id", "B")
    assert "id" not in memory(base)
    assert memory(fork_a).read("id") == "A"
    assert memory(fork_b).read("id") == "B"
    # the shared prefix (messages/state) is identical across branches
    assert fork_a.messages == fork_b.messages == base.messages
    assert isinstance(base, RolloutState)

    # advancing one branch touches only its own workspace and its own trace
    fork_a = await runner.advance(fork_a)  # generates "stop": writes last="stop", ends
    assert fork_a.done
    assert memory(fork_a).read("last") == "stop"
    assert memory(base).read("last") == "go"  # base untouched by the branch's step
    assert memory(fork_b).read("last") == "go"  # sibling untouched too
    assert len(fork_a.messages) > len(fork_b.messages)  # A diverged, B did not


@pytest.mark.asyncio
async def test_a_storage_tool_with_no_store_behind_it_is_refused_at_the_root() -> None:
    """A `storage` tool executes against the *rollout's* workspace, so pairing one with
    a `NullWorkspace` is a rollout whose memory cannot work -- and nothing downstream
    would say so: `dispatch` catches the tool's `AttributeError` and hands
    `'NullWorkspace' object has no attribute 'write'` back as the tool result the model
    reads, unlogged and uncounted, while the rollout generates on around a memory it
    never had. It is per-run configuration (a caller that built the env from a workspace
    and then forgot to pass it), so `root` refuses it before a token is generated."""
    runner = toy_runner(
        ScriptedGenerator(["go"]),
        env=SubmitEnv([StorageTool()]),
        parser=HermesToolCallParser(),
    )

    # the message has to name the way *out*, in each of the two vocabularies a caller
    # arrives with -- a harness run, and a slime actor's per-prompt factory
    with pytest.raises(ValueError, match="workspace=") as caught:
        await runner.start("hi", workspace=NullWorkspace())
    assert "workspace_factory" in str(caught.value)
    # the default workspace is that same NullWorkspace: an omission is refused too
    with pytest.raises(ValueError, match="NullWorkspace"):
        await runner.root(({"role": "user", "content": "hi"},))

    # a real store is the whole fix; nothing else about the env changes
    assert await runner.start("hi", workspace=InMemoryWorkspace())


# -- the edge: what a checkpoint generated since the one it branched from -------------


def test_the_edge_is_the_tail_past_the_parents_log() -> None:
    """fork() shares the tuple, so the edge is a slice -- no walk, no offsets."""
    shared = turn([-1.0, -2.0])
    parent = rollout(shared)
    added = turn([-3.0])
    child = rollout(shared, added)

    assert child.edge(parent) == (added,)
    assert child.generated_since(parent, "policy") == [-3.0]
    assert parent.edge(parent) == ()


def test_an_edge_against_a_non_prefix_checkpoint_is_refused() -> None:
    """A `ValueError`, not an `assert`: `python -O` strips the assert, and the slice it
    was guarding still returns turns against an unrelated checkpoint -- the wrong ones,
    which would then be exported as an edge, scored and trained on."""
    child = rollout(turn([-1.0]), turn([-2.0]))
    stranger = rollout(turn([-9.0]))

    with pytest.raises(ValueError, match="not a prefix"):
        child.edge(stranger)
    # and a longer "parent" is refused too, rather than yielding an empty edge
    with pytest.raises(ValueError, match="not a prefix"):
        rollout(turn([-1.0])).edge(child)


def test_generated_since_is_silent_about_an_unscored_channel() -> None:
    parent = rollout()
    child = rollout(turn([-1.0], masked=4))  # 4 conditioning tokens, 1 generated

    assert child.generated_since(parent, "policy") == [-1.0]
    assert child.generated_since(parent, "other") == []


def test_edge_logprob_sums_the_edge_and_refuses_an_unscored_channel() -> None:
    """A missing anchor channel must not read as `exp(0) = 1`, i.e. "policies agree"."""
    first = turn([-1.0])
    parent = rollout(first)
    child = rollout(first, turn([-2.0, -0.5]))

    assert child.edge_logprob(parent, "policy") == pytest.approx(-2.5)
    with pytest.raises(KeyError, match="anchor"):
        child.edge_logprob(parent, "anchor")
    assert child.generated_since(parent, "anchor") == []  # the lenient reading


def test_edge_logprob_refuses_an_edge_only_half_scored() -> None:
    """A scored turn beside an unscored one is no scored edge: the sum would short."""
    scored, unscored = turn([-1.0]), turn()
    child = rollout(scored, unscored)

    with pytest.raises(KeyError, match="anchor"):
        child.edge_logprob(rollout(), "anchor")
    with pytest.raises(KeyError, match="policy"):
        child.edge_logprob(rollout(), "policy")  # the second turn carries nothing


def test_the_edge_filters_by_tag_and_counts_folds_by_default() -> None:
    """`generated_since` is about task work; `edge_logprob` is about the whole edge."""
    parent = rollout()
    child = rollout(turn([-1.0]), turn([-2.0], tag="fold"))

    assert child.generated_since(parent, "policy") == [-1.0]  # task only, by default
    assert child.generated_since(parent, "policy", tags=None) == [-1.0, -2.0]
    assert child.edge_logprob(parent, "policy") == pytest.approx(-3.0)
    assert child.edge_logprob(parent, "policy", tags=TASK) == pytest.approx(-1.0)


def test_done_and_rewards_read_the_transitions_not_the_last_turn() -> None:
    """A fold appended after the terminal turn must not un-finish the episode."""
    ended = rollout(turn([-1.0], done=True, reward=1.0), turn([-2.0], tag="fold"))

    assert ended.done
    assert ended.reward == 1.0
    assert ended.turns_taken == 1
    assert len(ended.turns) == 2


# -- a backend that never answers -----------------------------------------------------


class HangingGenerator(ScriptedGenerator):
    """Never returns from the ``nth`` call -- a request the backend swallowed."""

    def __init__(self, completions: Sequence[str], *, nth: int = 1) -> None:
        super().__init__(completions)
        self._nth = nth
        self.cancelled = 0

    async def agenerate_tokens(
        self,
        prefix_tokens: Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        if self.calls + 1 == self._nth:
            self.prefixes.append(tuple(prefix_tokens))
            self.params.append(sampling_params)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return await super().agenerate_tokens(prefix_tokens, sampling_params)


@pytest.mark.asyncio
async def test_a_generation_that_never_returns_is_bounded_by_generate_timeout() -> None:
    """A hung request must cost one turn, not the whole prompt.

    Nothing else bounds it: a scheduler stops on a budget, and a turn that never
    returns never spends any -- so without this the expansion, and every rollout of
    that prompt behind it, waits forever. The message names the bound that fired
    because the bare ``TimeoutError`` says nothing about which of the run's waits it
    was.
    """
    gen = HangingGenerator(["go"])
    runner = toy_runner(gen, generate_timeout=0.01)

    with pytest.raises(TimeoutError, match=r"generate_timeout=0\.01s"):
        await complete(runner, "x")

    assert gen.cancelled == 1  # the request is cancelled, not left running behind us


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, -1])
async def test_an_unset_generate_timeout_waits_as_long_as_the_backend_does(
    timeout: float | None,
) -> None:
    """The default -- and ``-1`` -- is today's behaviour: a slow backend is not a
    failed one. ``-1`` is the same sentinel ``max_steps`` uses for unlimited."""
    gen = ScriptedGenerator(["stop"])
    runner = toy_runner(gen, generate_timeout=timeout)

    result = await asyncio.wait_for(complete(runner, "x"), timeout=5)

    assert result.done


class AlwaysFoldCompactor(ToyCompactor):
    """A fold whose trigger is always hungry, so any generation must fold first."""

    trigger: Trigger[ToyState] = WhenTrigger(lambda prompt, state: True)


@pytest.mark.asyncio
async def test_finish_folds_before_forcing_an_ending() -> None:
    """The forced last turn asks the fold trigger, like every other generation.

    It used to be the one that did not, and the asymmetry was not academic: `advance`
    keeps the prompt under the trigger's threshold, then generates up to `max_tokens`
    *on top of it* and appends the tool results. A rollout that then ran out of steps
    handed all of that to `finish`, which rendered it and asked the backend for another
    `max_tokens` -- so a context the runner had been keeping in budget all along broke
    it at the last turn, and a live rollout died there with its answer already earned.
    """
    runner = toy_runner(
        ScriptedGenerator(["stop"]),
        compactor=AlwaysFoldCompactor(),
        max_steps=4,
    )
    rs = await runner.start("hi", workspace=InMemoryWorkspace())

    forced = await runner.finish(rs)

    assert forced.folds == 1  # the fold happened, and inside `finish`
    assert forced.done  # and it still forced the ending
    assert forced.truncated


@pytest.mark.asyncio
async def test_finish_does_not_fold_when_the_context_is_within_budget() -> None:
    """The fold is conditional, not a ritual: a short context is forced as it stands."""
    runner = toy_runner(
        ScriptedGenerator(["stop"]),
        compactor=WatchedCompactor(CountingTrigger()),  # always answers "no"
        max_steps=4,
    )
    rs = await runner.start("hi", workspace=InMemoryWorkspace())

    forced = await runner.finish(rs)

    assert forced.folds == 0
    assert forced.done
    assert forced.truncated


async def test_branching_is_explicit_at_selected_boundaries():
    runner = toy_runner(ScriptedGenerator(["go", "stop"]))
    root = await runner.start("hi")
    assert root.forkable
    assert root.fork().forkable
    middle = await runner.advance(root)
    assert not middle.forkable
    with pytest.raises(ValueError, match="not forkable"):
        middle.fork()
    # Explicit snapshots still support offline inspection and prefix compaction.
    assert middle.snapshot().turns == middle.turns
    marker = turn([], tag="fold", transition=False)
    folded = replace(
        middle, turns=(*middle.turns, marker), checkpoint_at=len(middle.turns) + 1
    )
    assert folded.forkable
    assert not replace(folded, truncated=True).forkable
    terminal = await runner.advance(folded)
    assert terminal.done
    assert not terminal.forkable
    assert await runner.advance(terminal) is terminal


async def test_bare_truncation_at_fold_emits_one_unselectable_endpoint():
    runner = toy_runner(
        ScriptedGenerator(["go"]), compactor=UntaggedCompactor(), finish_prompt=None
    )
    first = await runner.advance(await runner.start("hi", workspace=QuietMemory()))
    chain = [s async for s in runner.iter_run(first, max_steps=1)]
    assert len(chain) == 1
    assert chain[0].turns[-1].tag == "fold"
    assert chain[0].truncated
    assert not chain[0].forkable
    assert not chain[0].done
