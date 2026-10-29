"""Drive exact model generations through context managers and model-free tasks.

Agent owns trace ordering, continuation limits, fold bookkeeping, and checkpoint
notifications. A context manager owns prompt preparation, compaction, and action
effects. ExecutionServices owns shared inference resources; derived agents inherit
it. Search sees settled snapshots.

start/root initialize an episode; advance makes one decision; iter_run exposes selected
checkpoints and the endpoint. finish makes the optional final answer attempt. Use fork
before pursuing another continuation from a saved state.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import EllipsisType
from typing import Any, Literal

from focus.branching import BranchPoints
from focus.codec import Message
from focus.context.base import (
    AppendOnly,
    Compaction,
    ContextManager,
    IncompleteCompactionError,
)
from focus.execution import ExecutionServices, Owner
from focus.execution.budget import ResourceBudget
from focus.execution.pool import DEFAULT_OWNER
from focus.generation import (
    GenerateResult,
    SamplingParams,
    TokenId,
)
from focus.generation._common import (
    merge_sampling_params,
)
from focus.generation.continuation import TokenChunkParser
from focus.generation.parsing.base import ActionParser, TextActionParser
from focus.generation.policy import Policy, PreparedPrompt
from focus.harness.env import Env, StepResult
from focus.harness.persistence import Checkpointer
from focus.harness.rollout import RolloutState
from focus.harness.turn import Effect, Turn
from focus.harness.workspace import (
    NullWorkspace,
    Workspace,
)

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


#: Tokens kept free for the final-answer turn when a reply could fill the context.
_FINISH_MARGIN = 1024

#: Logical feedback for one final answer attempt when the step budget expires. Task
#: specific in its wording only -- pass ``finish_prompt=`` to say it in the task's own
#: terms (a required answer format, say), or ``None`` to truncate bare.
DEFAULT_FINISH_PROMPT = (
    "You are out of steps: this is your last turn, and no further work tools will run. "
    "Give your best-supported final answer from what you have already gathered, "
    "in the form the task requires. If the task provides a final-answer tool, "
    "call that tool; otherwise answer directly in text."
)


@dataclass
class Execution:
    """How an agent's model calls run, persist, and recover.

    ``services`` shares inference capacity and usage accounting; pass one object to
    several agents (or collectors) to share it. An optional ``checkpointer`` persists
    every complete state; ``retries`` bounds retries of ``retry_on`` exceptions from
    the last committed state. ``resource_budget`` caps model calls and tokens per
    branch, and ``generate_timeout`` bounds one generation in seconds.
    """

    services: ExecutionServices = field(default_factory=ExecutionServices)
    checkpointer: Checkpointer | None = None
    retries: int = 2
    retry_on: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    generate_timeout: float | None = None

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("retries must be nonnegative")


def _action_parser(
    policy: Policy[Any],
    env: Env[Any, Any],
    context: ContextManager[Any],
    parser: ActionParser[Any] | None,
) -> ActionParser[Any]:
    """The parser that turns ``policy``'s replies into actions.

    An explicit ``parser`` is used as given. Otherwise the policy format's own parser
    is checked against the action kind the rollout needs -- the context manager's
    (``Segments`` decides in token chunks) or else the environment's. Text and token
    chunks can be read from any reply; a tool-call dialect cannot be invented, so a
    format without one is refused where tool calls are needed. A parser or a rollout
    that declares no kind is trusted.
    """
    chosen = parser
    if chosen is None:
        chosen = policy.format.parser
        wanted = context.action or env.action
        if wanted is not None and chosen.action not in (None, wanted):
            if wanted == "text":
                chosen = TextActionParser()
            elif wanted == "tokens":
                chosen = TokenChunkParser()
            else:
                owner = context if context.action else env
                raise ValueError(
                    f"{type(owner).__name__} needs {wanted} actions, but the policy "
                    + f"format parses {chosen.action} actions with "
                    + f"{type(chosen).__name__}; pass parser=... or a format with a "
                    + "tool-call dialect"
                )
    if chosen.requires_native_channel and not policy.provides_native_tool_calls:
        raise ValueError(
            f"{type(chosen).__name__} needs a backend with a structured tool-call "
            + "channel"
        )
    return chosen


def _as_fold(turn: Turn[Any]) -> Turn[Any]:
    """A turn generated while compacting: tagged ``"fold"``, with no task transition."""
    return replace(turn, tag="fold", transition=None)


def _spend_step[R: RolloutState[Any]](rs: R, **changes: Any) -> R:
    """``rs`` with ``changes`` applied, one decision later."""
    remaining = None if rs.steps_remaining is None else max(0, rs.steps_remaining - 1)
    return replace(rs, advances=rs.advances + 1, steps_remaining=remaining, **changes)


class Agent[S, A]:
    """Generate actions, apply task transitions, and record a rollout.

    The policy produces actions accepted by the environment. The context manager
    controls what the policy sees and when it is compacted; the branch-point policy
    selects checkpoints search may continue from. These collaborators are shared,
    while task state and workspace live in each RolloutState. The environment
    validates workspace requirements at initialization.

    ``parser`` overrides how replies become actions; by default the agent adapts the
    policy format's own parser to the action kind the context manager or
    environment needs. ``finish_prompt`` defaults (``...``) to
    :data:`DEFAULT_FINISH_PROMPT`, or to no final-answer turn when the context manager
    ends without one, as ``Segments`` does.
    ``execution`` groups services, persistence, retries, resource budgets, and the
    generation timeout. Subclass _advance, _compact, or _finish to keep custom behavior
    within the commit/retry boundary.
    """

    def __init__(
        self,
        *,
        policy: Policy[A],
        env: Env[S, A],
        system_prompt: str,
        context: ContextManager[S] | None = None,
        sampling_params: SamplingParams | None = None,
        max_steps: int = 32,
        finish_prompt: str | Message | EllipsisType | None = ...,
        tools: Sequence[dict[str, Any]] | None = None,
        branch_points: BranchPoints | None = None,
        context_limit: int | None = None,
        on_compaction_failure: Literal["raise", "end"] = "raise",
        execution: Execution | None = None,
        actor: bool = True,
        parser: ActionParser[A] | None = None,
    ) -> None:
        if on_compaction_failure not in ("raise", "end"):
            raise ValueError("on_compaction_failure must be 'raise' or 'end'")
        if context_limit is not None and context_limit < 1:
            raise ValueError("context_limit must be positive")
        self.context_limit = context_limit
        self.on_compaction_failure = on_compaction_failure
        self.execution = execution if execution is not None else Execution()
        self.actor = actor
        self.context: ContextManager[Any] = context or AppendOnly()
        self.context.check_policy(policy)
        self._parser: ActionParser[A] = _action_parser(
            policy, env, self.context, parser
        )
        if self.context.max_tokens is not None:
            sampling_params = merge_sampling_params(
                sampling_params or SamplingParams(),
                SamplingParams(max_tokens=self.context.max_tokens),
            )
        if finish_prompt is ...:
            finish_prompt = DEFAULT_FINISH_PROMPT if self.context.final_answer else None
        if branch_points is None:
            branch_points = self.context.branch_points()
        self._checkpoints = branch_points
        self._policy: Policy[A] = policy
        self._env: Env[S, A] = env
        self._system_prompt: str = system_prompt
        self._sampling_params: SamplingParams | None = sampling_params
        self._max_steps: int = max_steps
        self._finish_prompt: str | Message | None = finish_prompt
        # Tool schemas rendered into every prompt. Explicit `tools` wins; otherwise a
        # ToolEnv (or any env exposing `.schemas`) supplies them automatically.
        task_tools = tools if tools is not None else getattr(env, "schemas", ())
        self._tools = (*tuple(task_tools or ()), *self.context.schemas())
        names = [tool["function"]["name"] for tool in self._tools]
        if len(names) != len(set(names)):
            raise ValueError("context and task tool names collide")

    @property
    def services(self) -> ExecutionServices:
        return self.execution.services

    @services.setter
    def services(self, services: ExecutionServices) -> None:
        self.execution = replace(self.execution, services=services)

    @property
    def checkpointer(self) -> Checkpointer | None:
        return self.execution.checkpointer

    @property
    def generate_timeout(self) -> float | None:
        """Seconds one generation may take; ``None`` (or a negative value) waits."""
        timeout = self.execution.generate_timeout
        return None if timeout is not None and timeout < 0 else timeout

    def save_checkpoint(self, rs: RolloutState[S]) -> RolloutState[S]:
        """Commit a complete state, independently of search checkpoint selection.

        Explicitly call this after manually editing a saved state. Unsupported
        executor snapshots cannot become recovery points; their last intact parent
        remains in the store. Persistence is opt-in to avoid storing every rollout
        twice in large searches that need only in-memory branching.
        """
        if self.checkpointer is None or not rs.snapshot_available:
            return rs
        return self.checkpointer.put(rs)

    def restore(self, checkpoint_id: str) -> RolloutState[S]:
        """Recover the same branch using compatible agent collaborators.

        Do not restore a branch while it is still running. Logical resource budgets
        roll back with execution state; physical Usage counters retain failed work.
        Call fork() on the restored state to create an independent search branch.
        """
        if self.checkpointer is None:
            raise ValueError("Recovery requires a checkpointer")
        rs = self.checkpointer.get(checkpoint_id)
        self.services.budgets[rs.episode_id, rs.branch_id] = rs.budget
        return rs

    def restore_latest(
        self, episode_id: str, branch_id: str, invocation_id: str = "root"
    ) -> RolloutState[S]:
        """Recover the last intact commit for one branch and invocation."""
        if self.checkpointer is None:
            raise ValueError("Recovery requires a checkpointer")
        rs = self.checkpointer.latest(episode_id, branch_id, invocation_id)
        self.services.budgets[rs.episode_id, rs.branch_id] = rs.budget
        return rs

    async def _checkpointed(
        self,
        rs: RolloutState[S],
        operation: Callable[[RolloutState[S]], Awaitable[RolloutState[S]]],
    ) -> RolloutState[S]:
        if self.checkpointer is None:
            return await operation(rs)
        rs.check_continuation()
        if not rs.snapshot_available:
            # A context manager may return an analysis-only truncated endpoint.
            return await operation(rs)
        if rs.checkpoint_id is None:
            rs = self.save_checkpoint(rs)
        else:
            saved = self.checkpointer.info(rs.checkpoint_id)
            if (saved.episode_id, saved.branch_id, saved.invocation_id) != (
                rs.episode_id,
                rs.branch_id,
                rs.invocation_id,
            ):
                # A fork starts a new storage lineage from its source checkpoint.
                rs = self.save_checkpoint(rs)
        assert rs.checkpoint_id is not None
        for attempt in range(self.execution.retries + 1):
            try:
                # Protect the saved/input state from in-place workspace/tool updates.
                result = await operation(rs.snapshot())
            except Exception as exc:
                if attempt == self.execution.retries or not self._retryable(exc):
                    raise
                logger.warning(
                    "retrying from checkpoint=%s attempt=%d error=%r",
                    rs.checkpoint_id,
                    attempt + 1,
                    exc,
                )
                rs = self.restore(rs.checkpoint_id)
            else:
                # Storage failures propagate: never rerun an operation just because
                # its commit failed. The previous committed snapshot stays available.
                return self.save_checkpoint(result)
        raise AssertionError("unreachable")

    def _retryable(self, exc: Exception) -> bool:
        if isinstance(exc, ExceptionGroup):
            return all(self._retryable(child) for child in exc.exceptions)
        return isinstance(exc, self.execution.retry_on)

    @property
    def env(self) -> Env[S, A]:
        return self._env

    @property
    def policy(self) -> Policy[A]:
        """The model this agent drives, whole.

        Never copied: a fold derived from this agent drives the same object, so its
        requests share this policy's identity, routing, and capacity accounting.
        """
        return self._policy

    @property
    def parser(self) -> ActionParser[A]:
        """How this agent turns a reply into its environment's action."""
        return self._parser

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        """The tool schemas rendered into this agent's prompts.

        A data collector needs the same schemas to render an exact critic context.
        Return a copy so an auditor cannot mutate the agent's generation contract.
        """
        return deepcopy(tuple(self._tools or ()))

    # `T` is scoped to this method: a derived agent's task state is a fold's own
    # world, unrelated to the `S` this agent drives.
    def derive[T](
        self,
        *,
        env: Env[T, A],
        system_prompt: str,
        max_steps: int,
        finish_prompt: str | None,
        sampling_params: SamplingParams | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Agent[T, A]:
        """The same agent over a different world: same policy, new env and prompt.

        A compaction sub-rollout shares this agent's policy and execution services.
        It uses the new environment's action parser and tools, with its own prompt
        and step limit. Model clients stay with the caller.

        The derived agent uses AppendOnly: the fold is the thing
        being run, and a fold that folds is a loop with no floor. ``tools``
        defaults to the new env's own schemas, and ``sampling_params`` to this agent's
        -- pass either to override, e.g. a colder temperature for a summarizer.
        """
        return Agent(
            policy=self._policy,
            env=env,
            system_prompt=system_prompt,
            context=AppendOnly(),
            sampling_params=(
                self._sampling_params if sampling_params is None else sampling_params
            ),
            max_steps=max_steps,
            finish_prompt=finish_prompt,
            tools=tools,
            # a nested rollout talks to the same backend, so it inherits the same
            # services and patience: a fold that hung would stall the turn that folded
            execution=Execution(
                services=self.services,
                generate_timeout=self.execution.generate_timeout,
            ),
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
        self._env.validate_workspace(ws)
        messages = tuple(seed)
        state = await self._env.reset()
        rs = RolloutState(
            messages=messages,
            seed=messages,
            state=state,
            workspace=ws,
            budget=self.execution.resource_budget,
            history=deepcopy(messages),
            context_refs=tuple(range(len(messages))),
            steps_remaining=None if self._max_steps < 0 else self._max_steps,
        )
        return self.save_checkpoint(await self.context.initialize(rs, self))

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
        """Generate one task turn or compact the context.

        The fold check runs first, on the current context. If it fires, this turn is the
        fold alone (no task generation): see :meth:`compact`. Otherwise: render ->
        generate -> parse -> ``env.step`` -> append the world's reply turns, and record
        the whole thing as one :class:`~focus.harness.turn.Turn` carrying the
        exact prefix that was sent.

        Returns a new checkpoint. The workspace is threaded to ``env.step`` (and thus
        its tools) and is mutated **in place** -- ``fork`` the checkpoint before
        branching.

        ``sampling_params`` overlays the agent's defaults for this turn only, so a
        search phase can vary temperature per expansion without a second ``Agent`` --
        the agent is shared across concurrent branches and must stay stateless for
        that to be safe.
        """
        return await self._checkpointed(
            rs, lambda current: self._advance(current, sampling_params=sampling_params)
        )

    async def _advance(
        self, rs: RolloutState[S], *, sampling_params: SamplingParams | None = None
    ) -> RolloutState[S]:
        rs.check_continuation()
        if rs.done or rs.truncated:
            return rs
        if self._env.should_finish(rs):
            return await self._finish(rs, sampling_params=sampling_params)
        cond = self.context.prepare(rs, self)
        if self.context.should_compact(rs, cond.tokens):
            # This decision is the compaction alone; generation resumes next time.
            return self._mark(await self._fold(rs))
        if self._near_context_limit(cond, sampling_params):
            return await self._finish(rs, sampling_params=sampling_params)
        return await self._generate_turn(
            rs, list(rs.messages), cond, sampling_params=sampling_params, ending=False
        )

    def _near_context_limit(
        self, cond: PreparedPrompt, sampling_params: SamplingParams | None
    ) -> bool:
        """Whether a full reply would not leave room for the final-answer turn.

        Only for a context manager that never rewrites context: one that compacts is
        responsible for keeping its prompts small.
        """
        if self.context_limit is None or self.context.manages_context:
            return False
        reply = self._reply_tokens(sampling_params)
        size = len(cond.tokens)
        return size < self.context_limit <= size + reply + _FINISH_MARGIN

    def _reply_tokens(self, sampling_params: SamplingParams | None) -> int:
        params = merge_sampling_params(
            self.policy.default_params, self._params(sampling_params)
        )
        return int(params.max_tokens or 0)

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
        params = self._params(sampling_params)
        if self.context_limit is not None:
            available = self.context_limit - len(cond.tokens)
            if available <= 0:
                raise ValueError(
                    f"prompt of {len(cond.tokens)} tokens exceeds "
                    + f"context_limit={self.context_limit}; refusing to truncate it"
                )
            if self._reply_tokens(sampling_params) > available:
                params = merge_sampling_params(
                    params or SamplingParams(), SamplingParams(max_tokens=available)
                )
        self.services.attach_budget(rs.owner, rs.budget)
        result = await self._generate(cond, params, owner=rs.owner)
        self.context.record_reply(result, messages, self)
        action = self._parser.parse(result)
        origin = len(rs.turns)
        draft = self._turn(
            rs,
            cond.tokens,
            result,
            deepcopy(merge_sampling_params(self.policy.default_params, params)),
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
        applied = await self.context.apply(current, action, self, ending=ending)
        current = applied.rollout
        turns = list(current.turns)
        turns[origin] = replace(turns[origin], transition=applied.transition)
        return self._mark(
            _spend_step(
                current,
                budget=self.services.remaining(rs.owner),
                turns=tuple(turns),
                truncated=current.truncated or ending,
            ),
        )

    def _mark(self, current: RolloutState[S]) -> RolloutState[S]:
        """Evaluate eligibility after the operation has committed its state."""
        selected = self._checkpoints.eligible(current)
        return replace(current, checkpoint_at=len(current.turns) if selected else -1)

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
        if self.generate_timeout is None:
            return await self.services.generate(self._policy, cond, params, owner=owner)
        try:
            return await asyncio.wait_for(
                self.services.generate(self._policy, cond, params, owner=owner),
                self.generate_timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"generation exceeded generate_timeout={self.generate_timeout}s "
                + f"(prompt_tokens={len(cond.tokens)}); the turn is abandoned and the "
                + "expansion "
                + "counts as one dead rollout"
            ) from exc

    def _turn(
        self,
        rs: RolloutState[S],
        cond: tuple[int, ...],
        result: GenerateResult,
        sampling_params: SamplingParams | None,
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
        returned a count that does not match the completion, stores ``None`` rather
        than a misaligned column. A completion that came back *empty* is the one case
        where an empty column is the right answer rather than an absence -- it is
        scored, and its contribution is zero.
        """
        tokens = tuple(result.tokens)
        lps = tuple(result.logprobs or ())
        return Turn(
            prefix=_chained_prefix(
                rs.turns[-1] if rs.turns else None,
                tuple(result.prefix_tokens) or cond,
            ),
            tokens=tokens,
            logprobs=lps if len(lps) == len(tokens) else None,
            exact=result.exact_generation and self._parser.trainable,
            sampling_params=sampling_params,
            tag="task",
            invocation_id=rs.invocation_id,
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

        ``sampling_params`` overlays the agent's defaults exactly as in
        :meth:`advance`: the forced turn is part of the continuation it ends, so a
        search phase's sampling override applies to it too.

        An ended episode is unchanged. With ``finish_prompt=None``, mark bare
        truncation without generating.
        """
        if rs.done:
            return rs
        rs.check_continuation()
        if rs.done or rs.truncated:
            return rs
        return await self._checkpointed(
            rs, lambda current: self._finish(current, sampling_params=sampling_params)
        )

    async def _finish(
        self, rs: RolloutState[S], *, sampling_params: SamplingParams | None = None
    ) -> RolloutState[S]:
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
        if self.context.manages_context and self.context.should_compact(
            rs, self.context.prepare(rs, self).tokens
        ):
            rs = await self._fold(rs)
        if rs.done:
            return rs
        logger.debug(
            "forcing an ending truncated=True turns_taken=%d folds=%d",
            rs.turns_taken,
            rs.folds,
        )
        prompt = self._finish_prompt
        messages: list[Message] = [
            *rs.messages,
            dict(prompt)
            if not isinstance(prompt, str)
            else {"role": "tool", "content": prompt},
        ]
        tools = tuple(self._env.finish_tools(self._tools))
        return await self._generate_turn(
            rs,
            messages,
            self.context.prepare(
                replace(rs, messages=tuple(messages)), self, tools=tools
            ),
            sampling_params=sampling_params,
            ending=True,
        )

    async def compact(self, rs: RolloutState[S]) -> RolloutState[S]:
        """Force one configured fold, even if its trigger has not fired.

        Uses the same compactor and bookkeeping as :meth:`advance`. No task step is
        generated. Like ``advance``, this may mutate the checkpoint's workspace;
        pass ``rs.snapshot()`` when retaining the original for comparison.
        """
        return await self._checkpointed(rs, self._compact)

    async def _compact(self, rs: RolloutState[S]) -> RolloutState[S]:
        rs.check_continuation()
        if rs.done or rs.truncated:
            raise ValueError("Cannot compact a finished rollout for continuation")
        return self._mark(await self._fold(rs))

    async def _fold(self, rs: RolloutState[S]) -> RolloutState[S]:
        """Compact the context and record it: one decision, a new context version.

        The context manager returns the context and task state to continue from, plus
        any turns it generated; they are recorded as ``"fold"`` turns without task
        transitions. A fold that generated nothing leaves one empty marker turn. A
        ``"compaction"`` effect marks the boundary for branch-point policies.
        """
        self.services.attach_budget(rs.owner, rs.budget)
        try:
            folded = await self.context.compact(
                Compaction(
                    messages=rs.messages,
                    seed=rs.seed,
                    state=rs.state,
                    workspace=rs.workspace,
                    agent=self,
                    checkpoint=rs,
                )
            )
        except IncompleteCompactionError as exc:
            if self.on_compaction_failure != "end" or not exc.turns:
                raise
            # The rejected fold keeps its exact generations, but never becomes context:
            # a token-free task transition ends the episode with zero outcome reward.
            state = self._env.on_compaction_failure(rs.state, exc.reasons)
            terminal: Turn[S] = Turn(
                prefix=(),
                logprobs=(),
                transition=StepResult(next_state=state, done=True, reward_outcome=0.0),
            )
            rejected = tuple(_as_fold(t) for t in exc.turns)
            return replace(rs, state=state, turns=(*rs.turns, *rejected, terminal))
        marker: Turn[Any] = Turn(
            prefix=(),
            tokens=(),
            logprobs=(),
            tag="fold",
            actor=self.actor,
            invocation_id=rs.invocation_id,
            context_version=rs.context_version,
        )
        added = tuple(_as_fold(t) for t in folded.turns) or (marker,)
        return _spend_step(
            rs,
            budget=self.services.remaining(rs.owner),
            messages=tuple(folded.messages),
            context_refs=(None,) * len(folded.messages),
            state=folded.state,
            turns=(*rs.turns, *added),
            context_version=rs.context_version + 1,
            effects=(
                *rs.effects,
                Effect(
                    "compaction",
                    len(rs.turns),
                    rs.context_version,
                    rs.context_version + 1,
                    invocation_id=rs.invocation_id,
                    turn_end=len(rs.turns) + len(added),
                ),
            ),
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
        """Run to completion, exposing selected continuations and the endpoint.

        A new local decision consumes one step, including a fold-only advance.
        Finalization is outside that budget and happens once. Every yielded
        checkpoint has an independent snapshot.
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
        # A crash must not replenish a shorter per-run horizon. Persist the effective
        # limit even when it was supplied on a continuation rather than a fresh root.
        remaining = None if budget < 0 else budget
        if max_steps is not None or remaining != rs.steps_remaining:
            rs = self.save_checkpoint(replace(rs, steps_remaining=remaining))
        step = 0
        while budget < 0 or step < budget:
            before = len(rs.turns)
            rs = await self.advance(rs, sampling_params=sampling_params)
            step += 1
            if rs.done or rs.truncated:
                break
            if len(rs.turns) == before:
                break
            if rs.forkable:
                # Bare truncation at this boundary is one endpoint, not two nodes
                # with identical logs and an empty edge between them.
                if step == budget and self._finish_prompt is None:
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


__all__ = ["DEFAULT_FINISH_PROMPT", "Agent", "Execution"]
