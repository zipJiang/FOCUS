"""The DeonticBench world: the statutes as a filesystem, the tools, the env.

What one case is *run in*, with nothing about datasets, scoring or the CLI -- those are
the benchmark adapter, which imports this. The split is the seam the benchmark
already had: two modes drive the same world (one trajectory per case, or a whole tree),
and both build it here through :func:`build_world` rather than each assembling a runner
of its own.

Three things make this a statute harness rather than a general terminal one, and all
three live here:

* **The statutes are a filesystem, addressed by provision.** ``statutes/section63.md``
  is indexed at its ``(a)``/``(c)(1)`` boundaries, so ``list`` says what exists,
  ``grep`` returns ``path:line`` hits the agent can cite, and ``read`` pages a whole
  provision rather than the one line that matched. Locating the governing rule is the
  task's real difficulty, so it gets primitives instead of ``cat``.
* **Arithmetic goes through ``python``.** SARA is bracket arithmetic on a filing status
  that turns on dates; Airline is a per-bag fee table plus a sum. Those are exactly the
  steps where an LLM's silent slips live, and where DAR's open models collapse.
* **Stopping is a decision.** The first valid ``submit`` ends the task.

Nothing here knows which of the five domains it is running. A :class:`Case` arrives
already normalized by the DeonticBench data adapter -- an answer spelled the way it
must be submitted, and statutes already split into markdown files -- so the only thing
this module reads off a case is whether its answer is a number or one of a few words.

Durable notes (when enabled) come from ``workspace.tools()`` -- typically ``storage``
-- not by remounting workspace keys under ``grep``/``read``. Corpus tools stay corpus
tools; the workspace decides how its own tool calls execute.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from focus.environments.toolkit import (
    Corpus,
    GrepTool,
    ListTool,
    PythonTool,
    ReadTool,
    Section,
    SourceFile,
)
from focus.generation import SamplingParams
from focus.generation.parsing import ToolTurn
from focus.generation.policy import Policy
from focus.harness import (
    Agent,
    InMemoryWorkspace,
    NullWorkspace,
    PromptTokens,
    RoundBudget,
    SubmitFields,
    Summarize,
    Tool,
    ToolEnv,
    Workspace,
)
from focus.harness.compaction import DEFAULT_INSTRUCTION
from focus.harness.context import AppendOnly, IncompleteCompactionError
from focus.harness.env import StepResult
from focus.harness.tools.submission import submission_payload, submitted_call
from focus.harness.turn import Turn

# -- the case ------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One DeonticBench instance, in the shape every domain shares.

    The five domains differ in what they ask and how they answer, and none of that is a
    field: ``answer_type`` and ``choices`` say everything the world and the scorer need,
    so neither has a table keyed by domain name. ``domain`` survives only for reporting.
    """

    id: str
    domain: str
    facts: str
    question: str
    #: The gold answer, spelled exactly as the agent is asked to submit it.
    answer: str
    #: ``"integer"`` or ``"choice"`` -- how to read an answer, not what it says.
    answer_type: str
    choices: tuple[str, ...] = ()
    statutes: tuple[SourceFile, ...] = ()


def answer_format(case: Case) -> str:
    """How the answer must look, as one phrase the prompts can drop into a sentence."""
    if case.answer_type == "integer":
        return "a whole number, digits only"
    return "exactly one word: " + " or ".join(case.choices)


# -- the statute corpus --------------------------------------------------------------

#: A markdown heading. Statutes reach the corpus already marked up -- the IRC's
#: indentation and the housing excerpts' ``[1] CITE`` lines were turned into headings by
#: the DeonticBench data adapter -- so one rule indexes every domain, and the
#: heuristics that read one source's whitespace convention are not in the hot path of an
#: env any more.
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def provisions(text: str) -> tuple[Section, ...]:
    """Index a statute by its headings, each spanning to the next one as shallow.

    A provision runs until the next heading at the same or shallower depth, so reading
    ``(c)`` gives all of (c) *including* its subclauses -- which is what "read the whole
    provision" has to mean for a statute whose exceptions live in its subclauses.
    """
    heads = [
        (len(m.group(1)), m.group(2).strip(), i)
        for i, line in enumerate(text.splitlines(), start=1)
        if (m := _HEADING.match(line))
    ]
    last = len(text.splitlines())
    out = []
    for k, (depth, label, line) in enumerate(heads):
        end = next(
            (other - 1 for d, _, other in heads[k + 1 :] if d <= depth),
            last,
        )
        if label:
            out.append(Section(label, line, end))
    return tuple(out)


def build_corpus(case: Case) -> Corpus:
    """The case's own files plus the statutes it cites.

    A statute file stays *one* file -- line numbers are absolute within it, so a
    ``path:line`` citation grep produces and a ``read`` of the same path agree. The
    provisions are an index over it, which is what lets ``read`` fetch ``(c)(1)`` alone
    instead of 109 lines, and what lets grep say which provision a hit landed in.
    """
    files = []
    if case.facts:
        files.append(SourceFile("case/facts.txt", case.facts))
    files.append(SourceFile("case/question.txt", case.question))
    return Corpus((*files, *case.statutes))


# -- tools: the toolkit's, in this task's vocabulary ----------------------------------

#: A tool's description is rendered into every prompt, and it is the only place the
#: *strategy* for using it can be taught -- so the toolkit's generic defaults are
#: replaced here by what this task's evidence actually looks like: provisions,
#: thresholds, defined terms, and the exception a line below the hit.

LIST_DESCRIPTION = (
    "List the statute and case files you can read, with the provisions each is divided "
    + "into. Call it first: which sections exist is itself a fact about the law, and "
    + "it is cheaper to look than to guess a path."
)

GREP_DESCRIPTION = (
    "Search the statutes and case files with a regular expression. Returns "
    + "matching lines as `path:line: text`, naming the provision each hit fell in, "
    + "which is how you cite what you relied on. Search for the operative words a "
    + "provision would use -- a threshold, a defined term, an exception -- not a "
    + "sentence copied from the question."
)

READ_DESCRIPTION = (
    "Read a file by path, with 1-based line numbers you can cite. Pass "
    + "`provision` to read just one subsection (the file list names them); otherwise "
    + "you get the whole file from `offset`. Read the whole provision, not just the "
    + "line grep matched -- the sentence that changes the answer is usually the "
    + "exception a line or two below it."
)

PROVISION_HELP = "one subsection of that file, e.g. '(c)(1)'"

PYTHON_DESCRIPTION = (
    "Run Python to compute something -- tax brackets, per-bag fees, date "
    + "differences. Print what you want back (a trailing bare expression is echoed). "
    + "Each call is a fresh process, so nothing carries over between calls. Imports "
    + "available: math, decimal, fractions, itertools, datetime, statistics, json, "
    + "re, collections."
)

#: Constructor per tool name, so ``--tools list,grep,read`` names them and nothing else
#: has to know the set. ``build_world`` supplies the corpus and the run's knobs.
TOOLS = {
    "list": lambda corpus, cfg: ListTool(
        corpus, description=LIST_DESCRIPTION, section_arg="provision"
    ),
    "grep": lambda corpus, cfg: GrepTool(corpus, description=GREP_DESCRIPTION),
    "read": lambda corpus, cfg: ReadTool(
        corpus,
        description=READ_DESCRIPTION,
        section_arg="provision",
        section_help=PROVISION_HELP,
        max_chars=cfg.read_max_chars,
    ),
    "python": lambda corpus, cfg: PythonTool(
        timeout=cfg.python_timeout, description=PYTHON_DESCRIPTION
    ),
}

# -- prompts -------------------------------------------------------------------------

#: Always: what the task is, and that the rules decide it rather than memory.
_TASK = """\
You are applying written rules to the facts of a case. The answer is whatever the \
rules say it is, so your job is to find the governing provisions and apply them \
exactly -- not to recall what the law usually says."""

#: Only when a corpus tool can actually open the statutes.
_CORPUS = """\
The statutes are files. Use `list` to see what you have, `grep` to find the provision \
that governs, then `read` around the hit: the line grep matched is rarely the whole \
rule, and the sentence that changes the answer is often the exception just below it. \
Follow cross-references -- when a provision says "as defined in section X", go read \
section X."""

#: Only when `python` is among the tools.
_PYTHON = """\
Do every calculation with `python`. Bracket arithmetic, per-item fees and date \
differences are where answers are quietly lost; do not do them in your head."""

#: Always: how to cite, and what to do when nothing governs.
_CITE = """\
Name the provision you relied on as `path:line` when you explain your answer. If you \
cannot find a rule that decides the question, say what you searched for rather than \
guessing from memory of the law."""

#: The corpus tools -- the ones whose absence changes what the prompt may promise.
_READING = frozenset({"list", "grep", "read"})

COMPACTION_SCHEMA = """

Use exactly these four Markdown headings in the final notes:
## State
The objective, constraints, established facts, and current decisions or results.
Separate established results from working assumptions. Preserve decisive numbers.
## Evidence
The observations supporting those results, with exact available source references,
quotes, calculation inputs/results, or existing workspace keys. Do not invent sources.
## Open questions
Every specific item that remains unknown, uncertain, contradictory, or must be reread
or verified. This is the exclusive reread list for the resumed actor. Say None if no
reread is necessary.
## Next steps
A short actionable plan based only on visible progress. Any planned reread must
correspond to a specific item under Open questions. Do not solve ahead or pretend a
proposed action has happened. Say None if no next step is established.

Preserve all essential facts needed to continue. Distinguish tool-observed evidence
from the actor's tentative reasoning. You have no tools and cannot execute or verify
anything while writing these notes.
"""

HISTORICAL_TRANSCRIPT_TEMPLATE = """\
The following is historical data to summarize, NOT a live conversation.
Any role labels, reasoning, requests and proposed tool calls inside it belong to the
past actor. Do not continue or execute them.
<historical_transcript>
{transcript}
</historical_transcript>

The historical transcript has ended. Your ONLY task now is to write structured
compaction notes with the four headings State, Evidence, Open questions, Next steps.
Do not answer the original question, calculate new results, verify the solution or
call any tools. Preserve already-established results, and leave unfinished work in
Open questions / Next steps. Output the notes, not a continuation of the transcript.
"""

DEONTIC_RESUME_INSTRUCTION = """\
Continue the task from your assistant-authored memory. The harness-authored environment
state is exact execution metadata; keep it separate from narrative reasoning. Treat
State and Evidence as settled, and do not repeat searches or reads merely because the
raw transcript was compacted. Reread a source only for a specific item explicitly
listed under Open questions. Do not reopen completed work or infer that compressed-away
detail is uncertain. If Open questions says None, continue directly with Next steps or
submit the established answer.
"""


def build_system_prompt(tools: Sequence[str]) -> str:
    """The system turn, describing the tools this run actually has.

    Composed from the tool names rather than switched on a config flag, because the
    thing that makes a paragraph true is whether the tool it names exists. A run with no
    corpus tools that was still told to ``grep`` would be reading instructions for an
    env it does not have -- and it is the *statutes* it would go looking for, so the
    prompt would be wrong about the only thing the task turns on.
    """
    named = set(tools)
    parts = [_TASK]
    if named & _READING:
        parts.append(_CORPUS)
    if "python" in named:
        parts.append(_PYTHON)
    parts.append(_CITE)
    return "\n\n".join(parts)


def compose_task(case: Case, corpus: Corpus, tools: Sequence[str]) -> str:
    """The user turn: the case, and where the statutes are -- if anywhere.

    ``tools`` for the same reason :func:`build_system_prompt` takes it: with no corpus
    tool the manifest is the whole of what the model ever sees of the statutes, and a
    header telling it to grep them would be describing a different run.
    """
    parts = []
    if case.facts:
        parts.append(f"## Case\n{case.facts}")
    heading = (
        "## Files\nThe statutes are files; grep and read them."
        if set(tools) & _READING
        else "## Provisions\nThis listing is all you have; no tool opens the full text."
    )
    parts.append(f"{heading}\n{corpus.manifest(label='provisions')}")
    parts.append(f"## Question\n{case.question}")
    parts.append(f"Submit {answer_format(case)}.")
    return "\n\n".join(parts)


def finish_prompt(case: Case) -> str:
    return (
        "You are out of steps: this is your last turn and no tool will run on it. "
        + "Call submit with your best-supported answer now -- "
        + f"{answer_format(case)} -- from "
        + "what you have already established. Do not answer 'unknown'."
    )


# -- env -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DeonticState(SubmitFields, RoundBudget):
    """Task state: what the agent looked at, and the per-round call budget.

    `SubmitFields` stores the answer. `RoundBudget` supplies `calls_remaining` and
    `calls_max`: the env charges the former and the compaction trigger refills it.
    """

    #: Every tool called, in order -- the per-tool counts fall out of `.count(...)`.
    calls: tuple[str, ...] = ()
    #: Why a model-authored fold ended this branch, empty on ordinary task endings.
    compaction_failure: tuple[str, ...] = ()


class DeonticRunner(Agent[DeonticState, ToolTurn]):
    """Optionally turn rejected model folds into honest terminal training branches.

    Only :class:`IncompleteCompactionError` is an authored outcome: the compactor has
    already retried, and the exception carries the exact generated turns. Backend,
    transport, and implementation errors remain exceptions and still fail collection.
    """

    def __init__(self, *, compaction_failure: str = "raise", **kwargs) -> None:
        if compaction_failure not in ("raise", "terminal_zero"):
            raise ValueError("Unknown compaction failure policy")
        super().__init__(**kwargs)
        self.compaction_failure = compaction_failure

    async def _compact(self, rs):
        try:
            return await super()._compact(rs)
        except IncompleteCompactionError as exc:
            if self.compaction_failure != "terminal_zero" or not exc.turns:
                raise
            state = replace(
                rs.state,
                answer=None,
                compaction_failure=exc.reasons,
            )
            # The rejected folds remain exact trainable generations but never replace
            # task context. A token-free task transition records the terminal outcome.
            terminal = Turn(
                prefix=(),
                logprobs=(),
                transition=StepResult(
                    next_state=state,
                    done=True,
                    reward_outcome=0.0,
                ),
            )
            return replace(rs, state=state, turns=rs.turns + exc.turns + (terminal,))


class DeonticEnv(ToolEnv[DeonticState]):
    """Statute world: tools spend the round's budget, ``submit`` answers."""

    def __init__(
        self,
        *,
        tools,
        budget: int,
        scorer=None,
        roll_call_budget: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(tools, **kwargs)
        self._budget = budget
        # Production Deontic uses this as independent execution bookkeeping. Keep the
        # opt-in local to build_world: synthetic compaction generation deliberately
        # constructs this env with a StateCounter trigger to manufacture fold targets.
        self._roll_call_budget = roll_call_budget
        #: ``answer -> float``, or ``None`` to score out of band. RL needs the number
        #: *in* band: a tree whose edges all return 0.0 has no return spread, so every
        #: sibling comparison is uniform and no estimator can say anything. Injected
        #: rather than subclassed so the trained agent and the evaluated agent are the
        #: same object built by the same `build_world`.
        self._scorer = scorer

    async def initial_state(self) -> DeonticState:
        return DeonticState(calls_remaining=self._budget, calls_max=self._budget)

    def _final_answer(self, turn: ToolTurn) -> str:
        """The answer this turn ends on: the submitted field, else the visible text.

        Deliberately the *same* chain `ToolEnv.on_submit` uses to fill `state.answer`,
        so the reward scored in band and the answer the bench scores out of band cannot
        disagree about what was answered.

        It is tempting to add `turn.reasoning` as a third fallback, because a thinking
        model that spends its whole token budget inside `<think>` ends with no visible
        text and scores 0 for work it actually did. Do not: it buys those points only
        for the in-band reward, so a tree would train on answers the benchmark records
        as abstentions. The budget is the thing to fix -- see `--max-tokens`.
        """
        call = submitted_call(turn, self._submit)
        payload = submission_payload(call, self._submit)
        return str(payload.get(self.answer_field, ""))

    def reward(self, state: DeonticState, turn: ToolTurn) -> float:
        if self._scorer is None:
            return 0.0
        return float(self._scorer(self._final_answer(turn)))

    def on_tool_calls(self, state: DeonticState, turn: ToolTurn) -> DeonticState:
        calls = (*state.calls, *(c.name for c in turn.tool_calls))
        state = super().on_tool_calls(state, turn)
        if not self._roll_call_budget:
            return replace(state, calls=calls)
        # This counter is execution bookkeeping, not a memory-pressure signal. Roll it
        # over independently when a call tranche is spent; PromptTokens alone decides
        # when context is compacted. Deriving it from total calls also handles a turn
        # dispatching several calls across a tranche boundary without going negative.
        if state.calls_max <= 0:
            raise ValueError("Deontic call counter size must be positive")
        remaining = state.calls_max - (len(calls) % state.calls_max)
        return replace(state, calls=calls, calls_remaining=remaining)


def render_environment_state(state: DeonticState) -> str:
    """Exact executed state kept outside the model-authored narrative summary."""
    counts = Counter(state.calls)
    return json.dumps(
        {
            "tool_calls_executed": list(state.calls),
            "tool_call_counts": {name: counts[name] for name in sorted(counts)},
            "calls_until_counter_refill": state.calls_remaining,
            "call_counter_size": state.calls_max,
            "answer_recorded": state.answer is not None,
            "compaction_failure": list(state.compaction_failure),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_deontic_compactor(cfg: RunConfig) -> Summarize[DeonticState]:
    """Production Deontic fold: token-triggered, validated and trusted on resume."""
    return Summarize(
        trigger=PromptTokens(cfg.max_prompt_tokens),
        max_reply_tokens=cfg.fold_reply_tokens,
        summary_target_tokens=min(2048, cfg.fold_reply_tokens),
        require_complete_reply=True,
        structured_summary=True,
        sampling_params=SamplingParams(temperature=0.2, top_p=0.95),
        instruction=DEFAULT_INSTRUCTION + COMPACTION_SCHEMA,
        fold_workspace=NullWorkspace(),
        transcript_template=HISTORICAL_TRANSCRIPT_TEMPLATE,
        memory_role="assistant",
        state_renderer=render_environment_state,
        resume_instruction=DEONTIC_RESUME_INSTRUCTION,
        include_folded_guidance=False,
    )


# -- how one run is configured -------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Everything a run varies, in one record -- the CLI's flags, resolved.

    ``policy`` is the model the run is *of*: whichever backend answers, the tokenizer
    that renders and decodes for it, and the parser for the dialect its tool calls come
    home in. One field rather than three, because the three have to agree and nothing
    but their being one object was ever checking that -- see
    :class:`~focus.generation.policy.Policy`, whose docstring is where the
    parser-mismatch warning this record used to carry now lives. A scripted generator
    inside one is how this env is tested with no server at all.
    """

    policy: Policy[ToolTurn]
    tools: tuple[str, ...]
    max_steps: int
    call_budget: int
    max_prompt_tokens: int
    python_timeout: float
    no_workspace: bool
    no_compact: bool
    #: ``answer -> float`` scored in band, or ``None`` for out-of-band scoring.
    #: Required for RL: see :class:`DeonticEnv`.
    scorer: Callable[[str], float] | None = None
    #: Ceiling on one ``read``'s whole response; 0 disables. See ReadTool.max_chars.
    read_max_chars: int = 0
    #: Ceiling on one of the fold's own generations. The fold is handed the grown
    #: transcript whole, so this and ``max_prompt_tokens`` share the model's window
    #: with the task's generation budget -- see the CLI's ``--max-prompt-tokens``.
    fold_reply_tokens: int = 4096
    #: What to do after both validated structured-fold attempts are rejected.
    compaction_failure: str = "raise"


# -- the world one case is run in ----------------------------------------------------


@dataclass(frozen=True)
class World:
    """One case, assembled and ready to run: the world, its scratch, and the ask.

    Three things rather than a runner alone, because a caller has to hand the other two
    back to whatever drives the rollout -- ``runner.run_from(prompt, workspace=...)``
    for one trajectory, ``collector.search(prompt, workspace=...)`` for a tree --
    and assembling them twice is how the two modes would quietly drift apart.
    """

    runner: Agent[DeonticState, ToolTurn]
    workspace: Workspace
    prompt: str


def build_world(case: Case, cfg: RunConfig) -> World:
    """Corpus, tools, env, compactor, runner, workspace and user turn, for one case."""
    corpus = build_corpus(case)
    # Workspace exports storage (or nothing); the corpus tools own the evidence only.
    workspace: Workspace = NullWorkspace() if cfg.no_workspace else InMemoryWorkspace()
    tools: list[Tool] = [
        *workspace.tools(),
        *(TOOLS[name](corpus, cfg) for name in cfg.tools),
    ]

    env = DeonticEnv(
        tools=tools,
        budget=cfg.call_budget,
        scorer=cfg.scorer,
        roll_call_budget=True,
    )
    runner: Agent[DeonticState, ToolTurn] = DeonticRunner(
        policy=cfg.policy,
        env=env,
        system_prompt=build_system_prompt(cfg.tools),
        # Fold on the token budget, or once the round's tool budget is spent: exactly
        # the composition `build_budget` names, over the `RoundBudget` counters
        # `DeonticState` mixes in and `ToolEnv` charges.
        context=AppendOnly() if cfg.no_compact else build_deontic_compactor(cfg),
        max_steps=cfg.max_steps,
        finish_prompt=finish_prompt(case),
        compaction_failure=cfg.compaction_failure,
    )
    return World(
        runner=runner,
        workspace=workspace,
        prompt=compose_task(case, corpus, cfg.tools),
    )
