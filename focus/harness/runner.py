"""The harness main loop: drive one rollout checkpoint to completion.

The :class:`Runner` binds the stateless pieces the harness needs -- one
:class:`~focus.generation.policy.Policy` (the model: its generator, its codec,
its parser) and the env it acts on -- and drives a
:class:`~focus.harness.rollout.RolloutState`: the complete, forkable
checkpoint of a rollout (conversation, env state, workspace, turn log), which lives in
:mod:`focus.harness.rollout` because a consumer of the log has no business
importing the loop. Each ``advance`` renders the messages to token ids with our own
tokenizer, generates from those ids, decodes into an assistant message, parses it into
an action, applies it in the env, appends the env's reply turns -- and records the whole
thing as one :class:`~...turn.Turn`. ``run`` loops ``advance`` until the env says done
(or a step budget is hit).

Rendering locally and sending the backend the *ids* -- never the strings -- is what
keeps the generation-side prefix identical to what a later training pass would
reconstruct from the same messages. It is also why a turn can store the exact prefix it
was conditioned on: the runner *has* that list, so nothing downstream has to reconstruct
one. It is the seam where the env's and the policy's shared action ``A`` must agree --
the runner pairs a compatible ``(policy, env)``.

**Start vs run.** :meth:`root` builds a checkpoint from an explicit seed; :meth:`start`
is the prompt-shaped front door that composes ``workspace.guidance()`` into the seed
system message first. Every continuation method -- ``advance``, ``iter_run``, ``run``,
``finish`` -- takes a :class:`~focus.harness.rollout.RolloutState`. A fresh
episode is ``await runner.run_from(prompt, workspace=...)``; a branch is ``await
runner.run(rs.fork())``.

**Running out of steps.** ``max_steps`` bounds a rollout, and hitting it would end one
mid-conversation with no answer -- which is the ending every caller then patches with a
"just answer now" turn of its own, outside the harness. :meth:`Runner.finish` is that
turn, in-band: the finish prompt, one generation against the same prefix, and
:meth:`Env.finish` ending the episode on it. The
checkpoint comes back ``done`` *and* ``truncated``, so the two endings stay tellable
apart. ``max_steps = -1`` is unlimited -- the loop then stops only when the env says
done (or a stride makes no progress), which also means a turn or node budget above it
bounds nothing; see :class:`~focus.scheduler.core.policies.Budget`.

**A backend that never answers.** ``max_steps`` bounds how many turns a rollout takes;
nothing bounds how long one of them waits. A request that hangs blocks its expansion --
and with it the whole prompt -- forever, because a scheduler stops on a *budget* and a
turn that never returns never spends any. ``generate_timeout`` is that bound, in
seconds, around each :meth:`~...generation.Policy.agenerate`: when it fires the
generation is cancelled and the turn raises :class:`TimeoutError`, which a scheduler
records as one dead expansion and re-selects the parent from -- a degraded rollout
rather than a stalled job. ``None`` (the default) or ``-1`` is the unbounded wait.

**Folding.** A runner with a :class:`Compactor` asks that compactor's ``trigger`` at
the *start* of each turn whether to fold; if it fires, that turn is the fold alone (no
task generation). :meth:`Compactor.compact` gets a :class:`Compaction` -- the grown
messages and seed, the state, the rollout's own workspace, and the runner itself -- and
returns the messages to continue from (the seed, folded to carry whatever the compactor
kept) plus whatever turns it generated getting there, which are appended to the log
tagged ``"fold"``. A compactor that generated nothing still leaves one zero-token marker
turn, so *every* fold is visible in the log as a run of fold-tagged turns and nothing
else has to be recorded about it (see :meth:`_fold`). Because the check is front-loaded,
a fold must *relieve its own trigger*, or the trigger fires again on the very next turn.
``compactor=None`` -- the default -- is a single unbounded context, and costs no
per-turn question at all.

Handing the compactor the *runner* rather than a bag of its parts is what makes
:meth:`Runner.derive` the whole of that seam: a model-driven fold is a rollout under the
same policy on a different world, so it asks for the same runner over its own env
instead of reassembling one from smuggled fields.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from copy import deepcopy
from dataclasses import replace
from itertools import count
from typing import Any

from focus.codec import Message
from focus.execution import ExecutionServices, Owner
from focus.execution.budget import ResourceBudget
from focus.execution.pool import DEFAULT_OWNER
from focus.generation import (
    GenerateResult,
    SamplingParams,
    TokenId,
)
from focus.generation.interfaces import (
    merge_sampling_params,
)
from focus.generation.policy import Policy, PreparedPrompt
from focus.harness.checkpoint import CheckpointPolicy, Effect, SparseCheckpoints
from focus.harness.compaction.base import Compaction, Compactor
from focus.harness.env import Env, StepResult
from focus.harness.rollout import RolloutState
from focus.harness.turn import Turn
from focus.harness.workspace import (
    NullWorkspace,
    StorageTool,
    Workspace,
)
from focus.protocols import AppendOnly, ContextProtocol

logger = logging.getLogger(__name__)


def _chained_prefix[S](
    previous: Turn[S] | None, prefix: tuple[TokenId, ...]
) -> tuple[TokenId, ...]:
    """``prefix``, holding the *previous turn's id objects* wherever the two agree.

    Same ids, same order, same equality -- all that changes is which Python objects the
    tuple points at, and that is most of what a long rollout weighs. Every turn stores
    the exact prefix it was conditioned on and those prefixes are nested, so an N-turn
    log holds O(N x context) id slots. A re-render hands back *fresh* ``int`` objects
    each turn (a real tokenizer allocates them; only ids up to 256 are CPython's cached
    singletons), so a slot stored exactly as rendered costs a pointer **plus** a 28-byte
    int -- 36 bytes a slot, ~13 MiB for a 32-turn agentic rollout and ~160 MiB for a
    300-node tree, over 95% of everything such a log holds.

    Rebuilding the head out of the previous turn's ids leaves one pointer per slot and
    one int per *distinct* position in the final context: ~4x less, for one tuple
    compare per turn against work that just tokenized the whole conversation. The
    remaining 8 bytes a slot are inherent to storing N nested tuples -- the price of
    letting every turn retain the exact ids needed for replay and rollout budgeting.

    The test is the packer's own (:func:`~focus.harness.packing.regions`): a
    prefix that extends ``previous.prefix + previous.tokens`` is the backend's cache-hit
    chain, which is the usual case and the only one worth sharing. A chat template that
    rewrote history behind us fails it and the prefix is stored exactly as rendered --
    sharing is an allocation detail, never a claim about what the ids are.
    """
    if previous is None:
        return prefix
    packed = previous.prefix + previous.tokens
    n = len(packed)
    if len(prefix) < n or prefix[:n] != packed:
        return prefix
    return packed + prefix[n:]


def _step_range(budget: int) -> range | Iterator[int]:
    """``budget`` iterations, or forever when ``budget < 0`` (``-1`` means unlimited).

    ``range(-1)`` is empty -- the opposite of unlimited -- which is why the sentinel
    cannot be handed to ``range`` as-is. ``0`` still means no steps.
    """
    return count() if budget < 0 else range(budget)


def _check_workspace(env: object, workspace: Workspace) -> None:
    """Refuse a rollout whose ``storage`` tool would have no store to write to.

    A :class:`~...workspace.storage.StorageTool` executes against the workspace the
    *rollout* carries -- the env forwards it per call, overriding whatever the tool was
    constructed with -- so an env that offers one under a
    :class:`~...workspace.null.NullWorkspace` is a rollout whose memory tool cannot
    work. Nothing downstream says so, which is why it is said here: a tool fault is
    caught by ``dispatch`` and handed back as the result string the model reads
    (``'NullWorkspace' object has no attribute 'write'``), where no log records it and
    no metric counts it, and the rollout generates on around a memory it never has. The
    mistake is per-run configuration -- a caller that built its env from a workspace and
    then forgot to pass that workspace to the run -- so it belongs beside the other two
    contracts :meth:`Runner.root` settles, before a token is generated.

    Narrow on purpose, in both directions. Only ``NullWorkspace`` is refused: any other
    backend is presumed to be the store its own tools were made for, and a workspace
    that duck-types the KV methods is a legitimate one. And only a tool the *library*
    knows needs a store is looked for, by type: a custom tool's requirements are its
    own, and matching on the name ``"storage"`` would fire on any tool that happened to
    be called that.
    """
    if not isinstance(workspace, NullWorkspace):
        return
    tools = getattr(env, "tools", ())
    if not isinstance(tools, Sequence) or not any(
        isinstance(tool, StorageTool) for tool in tools
    ):
        return
    raise ValueError(
        " ".join(
            (
                f"{type(env).__name__} offers the `storage` tool but this rollout's",
                "workspace is a NullWorkspace, which has no store: every storage call",
                "would come back to the model as an error string. Pass the workspace",
                "the env's tools were built from --",
                "`run_search(prompt, runtime, workspace=...)`,",
                "`runner.run_from(prompt, workspace=...)`, or, in a slime actor,",
                "`make_generate(..., workspace_factory=...)`, which builds one per",
                "prompt.",
            )
        )
    )


#: Logical feedback for one final answer attempt when the step budget expires. Task
#: specific in its wording only -- pass ``finish_prompt=`` to say it in the task's own
#: terms (a required answer format, say), or ``None`` to truncate bare.
DEFAULT_FINISH_PROMPT = (
    "You are out of steps: this is your last turn, and no further work tools will run. "
    "Give your best-supported final answer from what you have already gathered, "
    "in the form the task requires. If the task provides a final-answer tool, "
    "call that tool; otherwise answer directly in text."
)


class Runner[S, A]:
    """Drive a :class:`RolloutState`: start, advance one turn, or run to the end.

    Holds only stateless collaborators (the policy, env, compactor), so a single runner
    can drive many forked checkpoints concurrently. The workspace is not a runner field
    -- it lives in the :class:`RolloutState` and is supplied per run.

    One :class:`~focus.generation.policy.Policy`, not a generator plus a codec
    plus a parser: those three are facets of one model and the runner has never had a
    use for them apart. It is the seam where the policy's and env's shared action ``A``
    must agree -- the runner pairs a compatible ``(policy, env)``.
    """

    def __init__(
        self,
        *,
        policy: Policy[A],
        env: Env[S, A],
        system_prompt: str,
        protocol: ContextProtocol | None = None,
        sampling_params: SamplingParams | None = None,
        max_steps: int = 32,
        finish_prompt: str | None = DEFAULT_FINISH_PROMPT,
        tools: Sequence[dict[str, Any]] | None = None,
        generate_timeout: float | None = None,
        checkpoints: CheckpointPolicy | None = None,
        services: ExecutionServices | None = None,
        resource_budget: ResourceBudget | None = None,
        actor: bool = True,
    ) -> None:
        self.services = services or ExecutionServices()
        self._resource_budget = resource_budget or ResourceBudget()
        self.actor = actor
        self._checkpoints = checkpoints or SparseCheckpoints()
        self._policy: Policy[A] = policy
        self._env: Env[S, A] = env
        self._system_prompt: str = system_prompt
        self.protocol = protocol or AppendOnly()
        self._sampling_params: SamplingParams | None = sampling_params
        self._max_steps: int = max_steps
        self._finish_prompt: str | None = finish_prompt
        # Tool schemas rendered into every prompt. Explicit `tools` wins; otherwise a
        # ToolEnv (or any env exposing `.schemas`) supplies them automatically.
        task_tools = tools if tools is not None else getattr(env, "schemas", ())
        self._tools = (*tuple(task_tools or ()), *self.protocol.schemas())
        names = [tool["function"]["name"] for tool in self._tools]
        if len(names) != len(set(names)):
            raise ValueError("protocol and task tool names collide")
        # Seconds one generate may take before the turn is abandoned. `None` -- the
        # default -- waits forever, which is what a backend that always answers wants.
        # `-1` normalises to `None` here rather than staying in-band as it does for the
        # step budget, so a CLI passing -1 for "no limit" needs no special case and no
        # reader downstream has to know the sentinel at all.
        self._generate_timeout: float | None = (
            None
            if generate_timeout is not None and generate_timeout < 0
            else generate_timeout
        )

    @property
    def generate_timeout(self) -> float | None:
        return self._generate_timeout

    @property
    def env(self) -> Env[S, A]:
        return self._env

    @property
    def policy(self) -> Policy[A]:
        """The model this runner drives, whole.

        Read rather than reassembled: :func:`~focus.loop.run_search` asks it
        whether a rollout from here could train the model that produced it before
        spending a search on the answer, and a compactor deriving a nested runner
        carries the same object rather than four fields that could drift apart.
        """
        return self._policy

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        """The tool schemas rendered into this runner's prompts.

        A data collector needs the same schemas to render an exact critic context.
        Return a copy so an auditor cannot mutate the runner's generation contract.
        """
        return deepcopy(tuple(self._tools or ()))

    # `T` is scoped to this method: a derived runner's task state is a fold's own
    # world, unrelated to the `S` this runner drives.
    def derive[T](
        self,
        *,
        env: Env[T, A],
        system_prompt: str,
        max_steps: int,
        finish_prompt: str | None,
        sampling_params: SamplingParams | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Runner[T, A]:
        """The same runner over a different world: same policy, new env and prompt.

        A nested rollout -- a model-driven fold, and anything else shaped like it -- is
        not a second agent: it is *this* agent, generating with the same weights through
        the same codec and speaking the same tool dialect, on a world of its own. So it
        is built by deriving from this runner rather than by assembling a fresh one from
        the pieces, which is what forced a caller to smuggle a generator, a codec, a
        parser and a version string across the seam and then guess a parser when one was
        missing. It is now one field: the dialect (Hermes JSON vs Qwen XML vs a
        backend's own structure) is a property of the :class:`Policy`, decided once at
        the top and inherited whole.

        The derived runner never folds (``compactor=None``): the fold is the thing
        being run, and a fold that folds is a loop with no floor. ``tools``
        defaults to the new env's own schemas, and ``sampling_params`` to this runner's
        -- pass either to override, e.g. a colder temperature for a summarizer.
        """
        return Runner(
            policy=self._policy,
            env=env,
            system_prompt=system_prompt,
            protocol=AppendOnly(),
            sampling_params=(
                self._sampling_params if sampling_params is None else sampling_params
            ),
            max_steps=max_steps,
            finish_prompt=finish_prompt,
            tools=tools,
            # a nested rollout talks to the same backend, so it inherits the same
            # patience: a fold that hung forever would stall the turn that folded
            generate_timeout=self._generate_timeout,
            services=self.services,
            actor=self.actor,
        )

    async def root(
        self, seed: Sequence[Message], *, workspace: Workspace | None = None
    ) -> RolloutState[S]:
        """The root checkpoint for an explicit seed conversation.

        The seed is taken as given -- no workspace guidance is composed into it -- so a
        caller that has already built the exact opening (a fold driving a nested
        rollout, a few-shot prompt) is not second-guessed. :meth:`start` is the
        prompt-shaped front door on top of this.

        Both state contracts are settled here, on the state ``reset`` just returned and
        before a token is generated: a :class:`ToolEnv` checks the counters it writes
        (inside ``reset``, via its subclass wrapper) and the compactor's trigger checks
        the fields it reads. Kept adjacent on purpose -- they are the same kind of
        misconfiguration, and a run should not survive one to fail on the other.
        """
        # Default NullWorkspace: no store / tools / guidance unless the caller opts in.
        # `is not None`, not `or`: an empty Workspace is falsy.
        ws = workspace if workspace is not None else NullWorkspace()
        _check_workspace(self._env, ws)
        messages = tuple(seed)
        state = await self._env.reset()
        rs = RolloutState(
            messages=messages,
            seed=messages,
            state=state,
            workspace=ws,
            budget=self._resource_budget,
            history=deepcopy(messages),
            context_refs=tuple(range(len(messages))),
            steps_remaining=None if self._max_steps < 0 else self._max_steps,
        )
        return await self.protocol.initialize(rs, self)

    async def start(
        self, user_prompt: str, *, workspace: Workspace | None = None
    ) -> RolloutState[S]:
        """Build the root checkpoint from a prompt (and optional workspace).

        The prompt-shaped entry -- continue with :meth:`run`, :meth:`iter_run`,
        :meth:`run_from`, or :meth:`advance`. The workspace's own account of itself
        (``guidance()``, empty for a workspace with nothing to say) is composed into the
        task ``system_prompt`` here, so authors pass the bare task text and the
        workspace owns its memory advice.
        """
        ws = workspace if workspace is not None else NullWorkspace()
        system = self._system_prompt
        guidance = ws.guidance()  # empty when the workspace has nothing to say
        if guidance:
            system = f"{system}\n\n{guidance}"
        return await self.root(
            (
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ),
            workspace=ws,
        )

    async def advance(
        self, rs: RolloutState[S], *, sampling_params: SamplingParams | None = None
    ) -> RolloutState[S]:
        """One turn -- generate, apply it, append it to the log.

        The fold check runs first, on the current context. If it fires, this turn is the
        fold alone (no task generation): see :meth:`_fold`. Otherwise: render ->
        generate -> parse -> ``env.step`` -> append the world's reply turns, and record
        the whole thing as one :class:`~focus.harness.turn.Turn` carrying the
        exact prefix that was sent.

        Returns a new checkpoint. The workspace is threaded to ``env.step`` (and thus
        its tools) and is mutated **in place** -- ``fork`` the checkpoint before
        branching.

        ``sampling_params`` overlays the runner's defaults for this turn only, so a
        proposal can vary temperature per expansion without a second ``Runner`` per
        proposal -- the runner is shared across concurrent branches and must stay
        stateless for that to be safe.
        """
        rs.check_continuation()
        if rs.done or rs.truncated:
            return rs
        prepared = await self.protocol.before(rs, self)
        if len(prepared.turns) != len(rs.turns):
            return prepared
        rs = prepared
        cond = self._policy.prepare(rs.messages, tools=self._tools)
        return await self._generate_turn(
            rs, list(rs.messages), cond, sampling_params=sampling_params, ending=False
        )

    async def _generate_turn(
        self,
        rs: RolloutState[S],
        messages: list[Message],
        cond: PreparedPrompt,
        *,
        sampling_params: SamplingParams | None,
        ending: bool,
    ) -> RolloutState[S]:
        """The body both :meth:`advance` and :meth:`finish` are: generate against
        ``cond``, apply the decoded action, and append the whole thing as one turn.

        ``ending`` is the one difference, and it is the step budget's: the action goes
        through :meth:`Env.finish` (which executes nothing -- there is no budget left to
        act on a result), the world's reply turns are not shown (nothing follows them),
        and the checkpoint comes back ``truncated``. ``messages`` is the caller's own
        list because only it knows whether a finish prompt was appended first; it is
        mutated here (the assistant turn, then the world's reply) and frozen on the way
        into the new checkpoint.
        """
        owner = Owner(rs.episode_id, rs.branch_id, rs.invocation_id)
        self.services.attach_budget(owner, rs.budget)
        result = await self._generate(
            cond,
            self._params(sampling_params),
            owner=Owner(rs.episode_id, rs.branch_id, rs.invocation_id),
        )
        self._policy.record_reply(result, messages)
        action = self._policy.parse(result)
        origin = len(rs.turns)
        draft = replace(
            self._turn(rs, cond.tokens, result, None),
            sampling_params=deepcopy(
                merge_sampling_params(
                    self.policy.default_params, self._params(sampling_params)
                )
            ),
        )
        current = replace(
            rs,
            messages=tuple(messages),
            turns=(*rs.turns, draft),
            history=(*rs.history, *deepcopy(messages[len(rs.messages) :])),
        )
        current = replace(
            current,
            context_refs=(
                *rs.context_refs,
                *range(len(rs.history), len(current.history)),
            ),
        )
        applied = await self.protocol.apply(current, action, self, ending=ending)
        current = applied.rollout
        turns = list(current.turns)
        turns[origin] = replace(turns[origin], transition=applied.transition)
        effects = current.effects[len(rs.effects) :]
        return replace(
            current,
            budget=self.services.remaining(owner),
            turns=tuple(turns),
            truncated=rs.truncated or ending,
            checkpoint_at=len(turns) if self._checkpoints.selects(effects) else -1,
            advances=rs.advances + 1,
            steps_remaining=None
            if rs.steps_remaining is None
            else max(0, rs.steps_remaining - 1),
        )

    async def _generate(
        self,
        cond: PreparedPrompt,
        params: SamplingParams | None,
        *,
        owner: Owner = DEFAULT_OWNER,
    ) -> GenerateResult:
        """One generation, bounded by ``generate_timeout`` when one is configured.

        The unbounded path is the call itself, not ``wait_for(..., None)``: every turn
        of every rollout goes through here, and a run that asked for no timeout should
        not pay a task wrapper per turn to be told so.

        The timeout is re-raised with what a reader of a failed expansion needs and
        the bare :class:`TimeoutError` does not carry: how long was waited, and how big
        the prompt was. ``wait_for`` has already cancelled the generation by then, so
        the backend request is not left running behind the turn that gave up on it.
        """
        if self._generate_timeout is None:
            return await self.services.generate(self._policy, cond, params, owner=owner)
        try:
            return await asyncio.wait_for(
                self.services.generate(self._policy, cond, params, owner=owner),
                self._generate_timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"generation exceeded generate_timeout={self._generate_timeout}s "
                + f"(prompt_tokens={len(cond.tokens)}); the turn is abandoned and the "
                + "expansion "
                + "counts as one dead rollout"
            ) from exc

    def _turn(
        self,
        rs: RolloutState[S],
        cond: tuple[int, ...],
        result: GenerateResult,
        transition: StepResult[S] | None,
    ) -> Turn[S]:
        """Record one generate as a task turn.

        The prefix is ``cond`` -- literally the ids handed to the backend -- unless the
        backend reported its own (``prefix_tokens``, e.g. a server that re-templated).
        Nothing is reconstructed: a turn that had to locate its own completion inside a
        re-render is a turn whose mask can silently desync from what was generated. It
        is stored through :func:`_chained_prefix`, which changes no id and only lets the
        nested prefixes share the id *objects* they agree on.

        ``exact`` is the backend's own claim *and* the parser's: a turn whose action was
        never in its tokens is not a turn those tokens can be trained on, whatever the
        backend says about the ids. The caveat then travels with the tokens down the
        chain the packer already reads (``Turn.exact`` ->
        :class:`~focus.harness.packing.PackedSequence`), so no consumer has to
        learn about parsers to avoid training on one.

        Logprobs are opt-in and length-checked: a backend asked for none, or one that
        returned a count that does not match the completion, stores ``{}`` rather than a
        misaligned column. A completion that came back *empty* is the one case where an
        empty column is the right answer rather than an absence -- it is scored, and its
        contribution is zero -- and recording it as unscored is what would make
        :meth:`~...rollout.RolloutState.edge_logprob` raise over a turn that generated
        nothing.
        """
        tokens = tuple(result.tokens)
        lps = tuple(result.logprobs or ())
        return Turn(
            prefix=_chained_prefix(
                rs.turns[-1] if rs.turns else None,
                tuple(result.prefix_tokens) or cond,
            ),
            tokens=tokens,
            logprobs={self._policy.version: lps} if len(lps) == len(tokens) else {},
            exact=self._policy.is_exact(result),
            transition=transition,
            tag="task",
            invocation_id=rs.invocation_id,
            policy_id=self._policy.version,
            format_id=self._policy.format.name,
            context_version=rs.context_version,
            actor=self.actor,
        )

    def _params(self, sampling_params: SamplingParams | None) -> SamplingParams | None:
        """Overlay per-turn sampling controls; schemas belong to the prompt."""
        if sampling_params is None:
            merged = self._sampling_params
        elif self._sampling_params is None:
            merged = sampling_params
        else:
            merged = merge_sampling_params(self._sampling_params, sampling_params)
        return merged

    async def finish(
        self, rs: RolloutState[S], *, sampling_params: SamplingParams | None = None
    ) -> RolloutState[S]:
        """End a rollout that ran out of steps: one last turn that must answer.

        The step budget is the loop's concern, not the env's, so a rollout that spends
        it mid-episode would otherwise stop with no answer at all -- which is why every
        caller grew its own "just answer now" turn *outside* the harness, generated from
        a hand-built prefix and absent from the recorded log. This is that turn, done
        once and in-band: :data:`DEFAULT_FINISH_PROMPT` (or the task's own
        ``finish_prompt``) is appended, one turn is generated against the same tools and
        prefix as every other, and :meth:`Env.finish` ends the episode on it -- so the
        answer reaches the task state, the tokens are trainable, and the turn log is the
        whole rollout. No tool the model calls here is *run* -- there is no budget left
        to act on a result -- but
        :class:`~focus.harness.tools.environment.ToolEnv` still *reads* a
        ``submit`` call, since by now that is how the model ends every turn.

        ``sampling_params`` overlays the runner's defaults exactly as in
        :meth:`advance`, and for the same reason one step further on: the forced turn is
        part of the line it ends, so an expansion drawn from a non-anchor proposal must
        answer under that law too. Sampled under the runner defaults instead, the last
        turn of the edge would be the one turn its recorded provenance mis-names.

        An ended episode is unchanged. With ``finish_prompt=None``, mark bare
        truncation without generating.
        """
        if rs.done:
            return rs
        rs.check_continuation()
        if self._finish_prompt is None:
            return replace(rs, truncated=True)
        # Fold first if the context has outgrown its budget. :meth:`advance` asks the
        # trigger before every generation and this method used to be the one that did
        # not -- so a rollout that spent its steps on a full context asked the backend
        # for ``max_tokens`` on top of all of it, and the request was refused. The
        # rollout then died *at the last turn*, discarding the whole trajectory over an
        # ending it had already earned. Folding here costs one summarization and makes
        # the budget arithmetic the same everywhere: prompt <= max_prompt_tokens at
        # every generate, this one included.
        rs = await self.protocol.before(rs, self)
        if rs.done:
            return rs
        logger.debug(
            "forcing an ending truncated=True turns_taken=%d folds=%d",
            rs.turns_taken,
            rs.folds,
        )
        messages: list[Message] = [
            *rs.messages,
            {"role": "tool", "content": self._finish_prompt},
        ]
        return await self._generate_turn(
            rs,
            messages,
            self._policy.prepare(messages, tools=self._tools or ()),
            sampling_params=sampling_params,
            ending=True,
        )

    async def compact(self, rs: RolloutState[S]) -> RolloutState[S]:
        """Force one configured fold, even if its trigger has not fired.

        Uses the same compactor and bookkeeping as :meth:`advance`. No task step is
        generated. Like ``advance``, this may mutate the checkpoint's workspace;
        pass ``rs.snapshot()`` when retaining the original for comparison.
        """
        rs.check_continuation()
        if rs.done or rs.truncated:
            raise ValueError("Cannot compact a finished rollout for continuation")
        return await self.protocol.compact(rs, self)

    async def _fold(
        self, rs: RolloutState[S], compactor: Compactor[S]
    ) -> RolloutState[S]:
        """Fold the context: append whatever the compaction generated, then reseed.

        A model-driven compactor is itself a rollout (over its own env), so it returns
        turns -- transition-free, and stamped ``"fold"`` here whatever tag they arrive
        under -- and they join this log like any others.

        A plain compactor generates nothing, and the log would then have no trace of a
        rewrite that really happened: the next turn's prefix simply stops extending the
        previous one, which is also what a chat template that rewrote history looks
        like. So a silent fold appends a *marker*: one fold-tagged turn that was
        conditioned on nothing and generated nothing. It costs a pointer, it exports
        nothing (:func:`~focus.export.spans_of` keeps only packs holding
        tokens) and it is answered for free by a re-scoring pass, and in exchange every
        fold is one thing in one place: a run of fold-tagged turns.

        A post-fold checkpoint is eligible for branching while it remains live.
        """
        owner = Owner(rs.episode_id, rs.branch_id, rs.invocation_id)
        self.services.attach_budget(owner, rs.budget)
        folded = await compactor.compact(
            Compaction(
                messages=rs.messages,
                seed=rs.seed,
                state=rs.state,
                workspace=rs.workspace,
                runner=self,
                checkpoint=rs,
            )
        )
        # Built per fold, not shared: a `Turn`'s `logprobs` is a mutable dict, and one
        # marker object handed to every rollout would carry another rollout's re-scored
        # version keys into logs that were never scored.
        #
        # It carries an *empty* behavior column, exactly as `_turn` records a generate
        # that came back empty: a turn with no completion is scored, and its
        # contribution is zero. Left unscored it would instead be an absence, and
        # `edge_logprob` -- strict by design, and read over the whole edge including
        # folds -- would raise over an edge whose every generation really was scored.
        marker: Turn[S] = Turn(
            prefix=(), tokens=(), logprobs={self._policy.version: ()}, tag="fold"
        )
        # Tagged here, not taken on trust. The tag *is* the fold -- `folds`,
        # `boundaries`, the snapshots exposed by `iter_run`, and forkability
        # are all read off it -- so a compactor that left `Turn`'s
        # default would not merely mislabel a pack, it would erase the fold from a log
        # that really did rewrite its context. `compact` already promises the runner
        # does this; a plug-in's tagging discipline is not what the runner's own
        # bookkeeping should rest on. Copied only where the tag is wrong, so the
        # conforming compactor pays nothing.
        turns = tuple(
            t if t.tag == "fold" else replace(t, tag="fold") for t in folded.turns
        )
        return replace(
            rs,
            budget=self.services.remaining(owner),
            messages=tuple(folded.messages),
            context_refs=(None,) * len(folded.messages),
            state=folded.state,
            turns=(*rs.turns, *(turns or (marker,))),
            context_version=rs.context_version + 1,
            effects=(
                *rs.effects,
                Effect(
                    "compaction",
                    len(rs.turns),
                    rs.context_version,
                    rs.context_version + 1,
                ),
            ),
            checkpoint_at=len(rs.turns) + len(turns or (marker,)),
            advances=rs.advances + 1,
            steps_remaining=None
            if rs.steps_remaining is None
            else max(0, rs.steps_remaining - 1),
        )

    @property
    def max_steps(self) -> int:
        """Advances per expansion before the final answer attempt; -1 is unlimited."""
        return self._max_steps

    async def iter_run(
        self,
        rs: RolloutState[S],
        *,
        max_steps: int | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncIterator[RolloutState[S]]:
        """Run to completion, exposing isolated folds and the endpoint.

        One advance consumes one step, including a fold-only advance. Finalization
        is outside that budget and happens once. The live workspace stays owned by
        this loop; every yielded checkpoint has an independent snapshot.
        """
        rs.check_continuation()
        if rs.done or rs.truncated:
            yield rs.snapshot()
            return
        budget = self._max_steps if max_steps is None else max_steps
        if not rs.turns and max_steps is not None:
            rs = replace(rs, steps_remaining=None if max_steps < 0 else max_steps)
        if rs.steps_remaining is not None:
            budget = (
                rs.steps_remaining if budget < 0 else min(budget, rs.steps_remaining)
            )
        for step in _step_range(budget):
            before = len(rs.turns)
            rs = await self.advance(rs, sampling_params=sampling_params)
            if rs.done or rs.truncated:
                break
            if len(rs.turns) == before:
                break
            if rs.forkable:
                # Bare truncation at this boundary is one endpoint, not two nodes
                # with identical logs and an empty edge between them.
                if step == budget - 1 and self._finish_prompt is None:
                    break
                yield rs.snapshot()
        if not rs.done and not rs.truncated:
            rs = await self.finish(rs, sampling_params=sampling_params)
        yield rs.snapshot()

    async def run(
        self,
        rs: RolloutState[S],
        *,
        max_steps: int | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> RolloutState[S]:
        """Consume the shared loop and return its terminal or truncated endpoint."""
        async for checkpoint in self.iter_run(
            rs, max_steps=max_steps, sampling_params=sampling_params
        ):
            rs = checkpoint
        return rs

    async def run_from(
        self,
        prompt: str,
        *,
        workspace: Workspace | None = None,
        max_steps: int | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> RolloutState[S]:
        """Fresh episode: :meth:`start` then :meth:`run`. To branch, ``run(rs)``."""
        return await self.run(
            await self.start(prompt, workspace=workspace),
            max_steps=max_steps,
            sampling_params=sampling_params,
        )


__all__ = ["DEFAULT_FINISH_PROMPT", "Runner"]
