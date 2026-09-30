"""LLM-driven compaction: a nested rollout over :class:`FoldEnv`."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Literal, TypeVar

from focus.generation import SamplingParams
from focus.harness.compaction.base import (
    Compaction,
    CompactionResult,
    Compactor,
    IncompleteCompactionError,
)
from focus.harness.compaction.fold_env import FoldEnv, FoldState
from focus.harness.compaction.summary import (
    REPAIR_INSTRUCTION,
    SUMMARY_INSTRUCTION,
    parse_summary,
)
from focus.harness.compaction.triggers import PromptTokens, Trigger
from focus.harness.transcript import format_transcript
from focus.harness.turn import Turn
from focus.harness.workspace import Workspace
from focus.registry import register

logger = logging.getLogger(__name__)

S = TypeVar("S")

#: The compactor's own system prompt. ``{workspace}`` is filled from the workspace's
#: :meth:`~...workspace.base.Workspace.compaction_guidance` -- where to put what, which
#: is the workspace's business, not this module's: a store worth offloading into and one
#: that keeps nothing need opposite advice, and only the backend knows which it is.
DEFAULT_INSTRUCTION = """\
You are compacting the working context of an agent that is part-way through a task. \
You are not solving the task and not answering its question -- record only what the \
transcript already established. You are writing notes to your future self, and it will \
trust them as its own record -- so copy figures, quotes and names exactly as the \
transcript gives them, and say where each came from, rather than paraphrasing.

{workspace}

When you are done, reply with no tool call: that reply replaces the whole transcript \
as the agent's working context.{budget}{limit}"""

#: Appended to the instruction only when there is a workspace to call tools against.
_BUDGET = " Use at most {k} tool calls first."

#: The length the kept context must come in under, stated to the model *and* enforced as
#: the fold's ``max_tokens``. Both halves are needed. Stated, because a fold is only
#: worth doing if the summary is smaller than what it replaces -- one as long as the
#: transcript leaves the prompt as full as it was, the trigger fires again on the very
#: next turn, and the rollout folds until its steps run out. Enforced, because the model
#: is being asked to respect a budget it cannot count, and the backend rejects the whole
#: request if the prompt plus this budget overruns the window.
_LIMIT = (
    " Keep it under about {words} words -- generation stops at {tokens} tokens and a "
    + "reply cut off mid-sentence loses its tail. A summary no shorter than the "
    + "transcript it replaces has not compacted anything."
)

_FINALIZE = "Tool-call budget spent. Reply now with the context to keep (no tool call)."

#: What the agent's context becomes after a fold: the original prompt and the compacted
#: state as ONE user turn, then (from the workspace's own
#: :meth:`~...workspace.base.Workspace.folded_guidance`) how to reach durable memory.
#: Memory is named, not inlined -- an inlined index would grow with the workspace and
#: undo the fold; a workspace with nothing to offer contributes nothing here.
#:
#: The framing is the other half of the fold's value. The summary re-enters inside a
#: *user* turn, so text of unstated provenance arriving there reads as someone else's
#: claim, and a model rightly distrusts it: it re-derives what it already settled, or
#: falls back on what it happens to remember. So the notes say whose they are and when
#: they were written, and state the trust rule in both directions -- what they record is
#: established and is not to be second-guessed; what they omit is *not* established and
#: is to be re-read from the task's own sources, never from general knowledge.
_CONTINUE = """\
{prompt}

<your_notes>
You wrote the notes below yourself, moments ago: your context grew past its budget, so \
you read the whole transcript and recorded what it had established before it was \
discarded. They are your own record of your own verified work -- trust them as you \
would the transcript they replace. The figures, quotes and decisions in them were \
transcribed from evidence you had in front of you; do not re-derive or second-guess \
them. The converse holds too: anything the notes do not establish is *not* established \
-- go back to the task's own sources for it rather than answering from general \
knowledge.

{kept}
</your_notes>{workspace}

Continue the task from here."""

_EMPTY_KEPT = "(compaction produced no summary.)"

_ASSISTANT_MEMORY = """\
<assistant_memory>
These are your own trusted notes from immediately before context compaction. Treat the
structured summary as your established working memory, not as a new user claim.

{kept}
</assistant_memory>{environment}"""

_ASSISTANT_RESUME = """\
Continue the task from your assistant-authored memory. Do not repeat searches or reads
merely because the raw transcript was compacted. Treat State and Evidence as settled.
Reread a source only when the specific missing or uncertain item is explicitly listed
under Open questions; the compactor was required to put every necessary reread there.
Do not reinterpret omitted transcript detail as an instruction to start over."""


@register(Compactor, "agentic")
class AgenticCompactor(Compactor[S]):
    """An LLM that compacts via the workspace's tools within a bounded call budget.

    A fold *is* a rollout, so this owns no loop. It asks the rollout's own
    :class:`~focus.harness.runner.Runner` to
    :meth:`~focus.harness.runner.Runner.derive` one over a
    :class:`~...compaction.fold_env.FoldEnv` -- same generate/parse/dispatch/observe
    machinery, same turn log, same budget-forces-an-ending path -- and its whole job is
    the three things that are genuinely strategy: *when* (a composable
    :class:`~...compaction.triggers.Trigger`), *what to say* (the instruction and the
    folded continuation), and *how long* (``max_interactions``). Everything it generates
    comes back as ordinary turns, retagged ``"fold"``, and the rollout's log absorbs
    them; there is no compaction-shaped record anywhere.

    It owns only its strategy in the other sense too -- the machine comes from the
    :class:`Compaction` (the rollout's own runner), not this object, so a fold speaks
    with the same weights, through the same codec, in the same tool dialect as the
    rollout it is folding, and none of that has to be passed along field by field.
    ``sampling_params`` is the one exception it keeps: a summarizer may legitimately
    want a colder temperature than the task.

    Both prompts it writes -- its own instruction and the folded context -- point at
    durable memory without inlining it (see ``_CONTINUE``): a model that does not know
    the workspace is there re-saves what it cannot see, and one handed it inline pays
    for the whole workspace on every turn, which is the fold undoing itself. *What* they
    say about the workspace is the workspace's own
    (:meth:`~...workspace.base.Workspace.compaction_guidance` and
    :meth:`~...workspace.base.Workspace.folded_guidance`), not this module's -- a store
    worth offloading long, precise detail into and a
    :class:`~...workspace.null.NullWorkspace`, whose fold summary *is* the only memory,
    need opposite advice, and only the backend knows which one it is.

    To fold on something other than prompt length, pass a ``trigger`` -- there is
    nothing left to subclass.
    """

    def __init__(
        self,
        *,
        trigger: Trigger[S] | None = None,
        max_interactions: int = 8,
        max_prompt_tokens: int = 8192,
        max_reply_tokens: int = 2048,
        summary_target_tokens: int | None = None,
        require_complete_reply: bool = False,
        structured_summary: bool = False,
        sampling_params: SamplingParams | None = None,
        instruction: str = DEFAULT_INSTRUCTION,
        fold_workspace: Workspace | None = None,
        transcript_template: str = "{transcript}",
        memory_role: Literal["user", "assistant"] = "user",
        state_renderer: Callable[[S], str] | None = None,
        resume_instruction: str = _ASSISTANT_RESUME,
        include_folded_guidance: bool = True,
    ) -> None:
        # `max_prompt_tokens` stays as the sugar for the overwhelmingly common trigger,
        # so the simple case never has to name a Trigger class at all.
        self.trigger: Trigger[S] = trigger or PromptTokens(max_prompt_tokens)
        self._max_interactions = max_interactions
        #: Ceiling on any one of the fold's generations -- its reasoning, a tool call,
        #: or the kept reply. It is what keeps a fold inside the model's window: the
        #: fold is handed the grown transcript *whole*, so its prompt is the one that
        #: just overran the budget, and asking the task's own (large) generation budget
        #: on top of it is a request the backend refuses -- killing the rollout at the
        #: fold rather than merely failing to compact.
        self._max_reply_tokens = max_reply_tokens
        if (
            summary_target_tokens is not None
            and not 0 < summary_target_tokens <= max_reply_tokens
        ):
            raise ValueError(
                "Summary target must be positive and within generation allowance"
            )
        self._summary_target_tokens = summary_target_tokens
        self._require_complete_reply = require_complete_reply
        self._structured_summary = structured_summary
        self._sampling_params = sampling_params
        self._instruction = instruction
        # An explicit workspace isolates the fold's tools from the actor's memory.
        # In particular NullWorkspace makes summarization tool-free. The actor's
        # workspace and post-fold guidance remain those of the original rollout.
        self._fold_workspace = fold_workspace
        self._transcript_template = transcript_template
        if memory_role not in ("user", "assistant"):
            raise ValueError("memory_role must be 'user' or 'assistant'")
        self._memory_role = memory_role
        self._state_renderer = state_renderer
        self._resume_instruction = resume_instruction
        self._include_folded_guidance = include_folded_guidance

    async def compact(self, compaction: Compaction[S]) -> CompactionResult[S]:
        if not self._structured_summary:
            return await self._compact_once(compaction)
        failed: tuple[Turn[object], ...] = ()
        for attempt in range(2):
            try:
                result = await self._compact_once(compaction, repair=bool(attempt))
            except IncompleteCompactionError as exc:
                failed += exc.turns
                if attempt:
                    raise IncompleteCompactionError(
                        turns=failed, reasons=exc.reasons
                    ) from exc
            else:
                return replace(result, turns=failed + result.turns)
        raise AssertionError("unreachable")

    async def _compact_once(
        self, compaction: Compaction[S], *, repair: bool = False
    ) -> CompactionResult[S]:
        workspace = (
            compaction.workspace
            if self._fold_workspace is None
            else self._fold_workspace.fork()
        )
        # Tools come from the workspace: a KV store exports StorageTool; NullWorkspace
        # (or any backend with nothing to offer) exports none; a custom workspace can
        # export a different surface. Empty tools means no call budget to advertise.
        env: FoldEnv = FoldEnv(tuple(workspace.compaction_tools()))
        instruction = self._instruction.format(
            workspace=workspace.compaction_guidance(),
            budget=_BUDGET.format(k=self._max_interactions) if env.schemas else "",
            # Words rather than tokens for the guidance, and only a third of the
            # budget: a model cannot count its own tokens, and on a thinking model the
            # same ceiling has to cover the reasoning that precedes the reply.
            limit=_LIMIT.format(
                words=(self._summary_target_tokens or self._max_reply_tokens) // 3,
                tokens=self._max_reply_tokens,
            ),
            # the raw number too: `instruction` is an author's template, and one that
            # phrases the budget itself needs `{k}` rather than the sentence `{budget}`
            k=self._max_interactions,
        )
        if self._structured_summary:
            instruction += SUMMARY_INSTRUCTION
            if repair:
                instruction += REPAIR_INSTRUCTION
        # `derive`, not a fresh Runner: same model, same codec, same tool dialect,
        # same policy-version tag -- only the world, the prompts and the budget differ.
        # The same call-time-workspace mechanism as the rollout, too: the tools take no
        # captured store; the runner forwards this fold's workspace per call.
        fold = compaction.runner.derive(
            env=env,
            system_prompt=instruction,
            max_steps=self._max_interactions,
            # Budget spent: `finish` drives the one forced no-tool reply, in-band.
            finish_prompt=_FINALIZE,
            # `replace` onto whatever the caller passed, so `max_tokens` is the one
            # field this class insists on. Note that any per-call params at all pin
            # `temperature`/`top_p` to their defaults, since `merge_sampling_params`
            # overlays every non-`None` field -- a fold now samples at those rather
            # than inheriting the task's.
            sampling_params=replace(
                self._sampling_params or SamplingParams(),
                max_tokens=self._max_reply_tokens,
            ),
        )
        # `root`, not `start`: the seed is the instruction plus the transcript verbatim,
        # and composing workspace guidance into it would repeat what the instruction
        # already quotes.
        rs = await fold.root(
            (
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": self._transcript_template.format(
                        transcript=format_transcript(compaction.messages, sep="\n\n")
                    ),
                },
            ),
            workspace=workspace,
        )
        rs = await fold.run(rs)
        if self._structured_summary:
            turns = tuple(replace(t, tag="fold", transition=None) for t in rs.turns)
            last = turns[-1] if turns else None
            try:
                if last is None:
                    raise ValueError("no_turn")
                if (
                    self._require_complete_reply
                    and len(last.tokens) >= self._max_reply_tokens
                ):
                    raise ValueError("reply_cap")
                parsed = parse_summary(
                    fold.policy.decode(last.tokens),
                    prefix=fold.policy.decode(last.prefix[-32:]),
                    at_cap=len(last.tokens) >= self._max_reply_tokens,
                )
                if len(fold.policy.format.codec.encode(parsed.notes)) > (
                    self._summary_target_tokens or self._max_reply_tokens
                ):
                    raise ValueError("summary_too_long")
            except ValueError as exc:
                if last is not None:
                    turns = (
                        *turns[:-1],
                        replace(
                            last, compaction_penalty=-0.01, compaction_issue=str(exc)
                        ),
                    )
                raise IncompleteCompactionError(
                    turns=turns, reasons=(str(exc),)
                ) from exc
            if parsed.recovered and last is not None:
                turns = (
                    *turns[:-1],
                    replace(
                        last,
                        compaction_penalty=-0.002,
                        compaction_issue="recovered_wrapper",
                    ),
                )
            return CompactionResult(
                self._continue_from(compaction, parsed.notes),
                self.trigger.relieve(compaction.state),
                turns=turns,
            )
        if self._require_complete_reply:
            last = rs.turns[-1] if rs.turns else None
            raw = fold.policy.decode(last.tokens) if last is not None else ""
            prefix = fold.policy.decode(last.prefix[-32:]) if last is not None else ""
            reasons = tuple(
                name
                for name, failed in (
                    ("no_turn", last is None),
                    (
                        "reply_cap",
                        last is not None and len(last.tokens) >= self._max_reply_tokens,
                    ),
                    ("empty_notes", not rs.state.kept.strip()),
                    (
                        "unclosed_prefix_reasoning",
                        prefix.rstrip().endswith("<think>") and "</think>" not in raw,
                    ),
                    (
                        "unclosed_reply_reasoning",
                        "<think>" in raw and "</think>" not in raw,
                    ),
                )
                if failed
            )
            if reasons:
                raise IncompleteCompactionError(
                    turns=tuple(
                        replace(turn, tag="fold", transition=None) for turn in rs.turns
                    ),
                    reasons=reasons,
                )
        # What the fold cost and what it produced, in sizes only: the kept text is the
        # rollout's own data and never goes to a log. A fold that spends its whole
        # interaction budget, or one that stored nothing, shows up here.
        # Guarded: `_tool_calls` walks every message of every turn, and an argument to
        # `logger.debug` is computed whether or not DEBUG is enabled.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "fold complete interactions=%d tools_called=%d kept_chars=%d",
                len(rs.turns),
                _tool_calls(rs.turns),
                len(rs.state.kept),
            )
        # Retag on the way out: inside the fold these were the task, and the FoldEnv's
        # transitions are about the fold's own bookkeeping, not the rollout's world.
        turns = tuple(replace(turn, tag="fold", transition=None) for turn in rs.turns)
        return CompactionResult(
            self._continue_from(compaction, rs.state.kept),
            self.trigger.relieve(compaction.state),
            turns=turns,
        )

    def _continue_from(
        self, compaction: Compaction[S], kept: str
    ) -> tuple[dict[str, str], ...]:
        """The seed with its request turn rewritten to carry the compacted state.

        Not ``seed + kept`` as an *extra* user turn -- that reads as the user asserting
        the summary back at the agent (and, when the summary is answer-shaped, as the
        answer being handed to it). The seed's last user turn is the request, so the
        state rides on that one message; every turn before it (system prompts, few-shot
        examples) is carried through untouched, whatever shape the seed has.
        """
        seed = compaction.seed
        if self._memory_role == "assistant":
            environment = ""
            if self._state_renderer is not None:
                rendered = self._state_renderer(compaction.state).strip()
                if rendered:
                    environment = (
                        '\n\n<environment_state source="harness">\n'
                        + rendered
                        + "\n</environment_state>"
                    )
            guidance = (
                compaction.workspace.folded_guidance().strip()
                if self._include_folded_guidance
                else ""
            )
            resume = self._resume_instruction.strip()
            if guidance:
                resume += "\n\n" + guidance
            return (
                *seed,
                {
                    "role": "assistant",
                    "content": _ASSISTANT_MEMORY.format(
                        kept=kept.strip() or _EMPTY_KEPT,
                        environment=environment,
                    ),
                },
                {"role": "user", "content": resume},
            )
        users = [i for i, m in enumerate(seed) if m["role"] == "user"]
        task = users[-1] if users else len(seed)  # no request turn: append one
        prompt = seed[task]["content"] if task < len(seed) else ""
        guidance = compaction.workspace.folded_guidance()
        content = _CONTINUE.format(
            prompt=prompt,
            kept=kept.strip() or _EMPTY_KEPT,
            workspace=f"\n\n{guidance}" if guidance else "",
        )
        # lstrip: with no request turn the template's leading slot renders empty.
        return (*seed[:task], {"role": "user", "content": content.lstrip()})


def _tool_calls(turns: Sequence[Turn[FoldState]]) -> int:
    """How many tool results the fold's turns came back with -- one per call made.

    Counted off the recorded transitions rather than tracked on :class:`FoldState`,
    which only knows *whether* a tool ran: this is a log line's business, not the
    env's, and nothing else needs the number.
    """
    return sum(
        1
        for turn in turns
        for message in (turn.transition.messages if turn.transition else ())
        if message["role"] == "tool"
    )


__all__ = ["DEFAULT_INSTRUCTION", "AgenticCompactor"]
