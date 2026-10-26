"""Serializable inference requests and shared local dispatch."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from focus.codec import Message
from focus.execution.budget import ResourceBudget
from focus.execution.pool import DEFAULT_OWNER, AdmissionPool, Owner
from focus.generation import GenerateResult, Policy, PreparedPrompt, SamplingParams
from focus.generation.interfaces import (
    _continuation_params,
    _validate_continuation_result,
    aclose_client,
)
from focus.reward import RewardModel, RewardResult


@dataclass(frozen=True)
class InferenceRequest:
    backend: str
    format_id: str
    tokens: tuple[int, ...]
    messages: tuple[Message, ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    params: SamplingParams | None = None
    owner: Owner = DEFAULT_OWNER
    queue_timeout: float | None = None
    execution_timeout: float | None = None


@runtime_checkable
class InferenceExecutor(Protocol):
    def bind(self, policy: Policy[Any]) -> str: ...

    async def submit(self, request: InferenceRequest) -> GenerateResult: ...

    async def aclose(self) -> None: ...


def policy_key(policy: Policy[Any]) -> str:
    """The name a remote coordinator routes a policy's requests by."""
    return f"{policy.model}:{policy.version}:{policy.format.name}"


class LocalInferenceExecutor:
    def __init__(self, pool: AdmissionPool | None = None, *, own_clients: bool = False):
        self.pool = pool or AdmissionPool()
        self.policies: dict[str, Policy[Any]] = {}
        self.own_clients = own_clients
        self._closed = False

    def bind(self, policy: Policy[Any]) -> str:
        """Register ``policy`` and return the key its requests carry.

        Policies are told apart by object identity: two snapshots of one model with the
        same label (one collecting while the other's successor is being trained) get
        distinct keys rather than colliding.
        """
        name = policy_key(policy)
        key, suffix = name, 1
        while (existing := self.policies.get(key)) is not None:
            if existing is policy:
                return key
            suffix += 1
            key = f"{name}#{suffix}"
        self.policies[key] = policy
        return key

    async def submit(self, request: InferenceRequest) -> GenerateResult:
        policy = self.policies[request.backend]
        if policy.format.name != request.format_id:
            raise ValueError("request format differs from registered policy")

        async def generate() -> GenerateResult:
            prompt = PreparedPrompt(
                request.tokens, policy.format, request.messages, request.tools
            )
            return await policy.agenerate(prompt, request.params)

        return await self.pool.run(
            generate,
            owner=request.owner,
            resource=request.backend,
            queue_timeout=request.queue_timeout,
            execution_timeout=request.execution_timeout,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.pool.aclose()
        errors: list[Exception] = []
        if self.own_clients:
            for policy in {id(p): p for p in self.policies.values()}.values():
                try:
                    close = getattr(policy, "aclose", None)
                    if close is not None:
                        await close()
                    else:
                        await aclose_client(policy)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("backend cleanup failed", errors)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    tool_calls: int = 0
    value_batches: int = 0


@dataclass
class ExecutionServices:
    """Explicitly shared services; derived runners inherit this same object."""

    inference: InferenceExecutor = field(default_factory=LocalInferenceExecutor)
    auxiliary: AdmissionPool | None = None
    usage: dict[str, Usage] = field(default_factory=dict)
    budgets: dict[tuple[str, str], ResourceBudget] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.auxiliary is None:
            self.auxiliary = (
                self.inference.pool
                if isinstance(self.inference, LocalInferenceExecutor)
                else AdmissionPool()
            )

    def attach_budget(self, owner: Owner, budget: ResourceBudget) -> None:
        self.budgets.setdefault((owner.episode, owner.branch), budget)

    def remaining(self, owner: Owner) -> ResourceBudget:
        return self.budgets.get((owner.episode, owner.branch), ResourceBudget())

    def charge(self, owner: Owner, **costs: int) -> None:
        key = (owner.episode, owner.branch)
        self.budgets[key] = self.remaining(owner).charge(**costs)

    async def astart(self) -> None:
        """Local resources start lazily; this hook gives factories one lifecycle API."""
        return None

    async def generate(
        self,
        policy: Policy[Any],
        prompt: PreparedPrompt,
        params: SamplingParams | None,
        *,
        owner: Owner = DEFAULT_OWNER,
        timeout: float | None = None,
        queue_timeout: float | None = None,
    ) -> GenerateResult:
        if prompt.exact_continuation:
            if not policy.provides_exact_tokens or policy.provides_native_tool_calls:
                raise ValueError(
                    "Token continuation requires an exact token-input policy"
                )
            params = _continuation_params(policy.default_params, params)
        key = self.inference.bind(policy)
        usage = self.usage.setdefault(owner.episode, Usage())
        self.charge(owner, model_calls=1, prompt_tokens=len(prompt.tokens))
        usage.calls += 1
        usage.prompt_tokens += len(prompt.tokens)
        request = InferenceRequest(
            key,
            prompt.format.name,
            prompt.tokens,
            tuple(prompt.messages),
            tuple(prompt.tools),
            deepcopy(params),
            owner,
            execution_timeout=timeout,
            queue_timeout=queue_timeout,
        )
        try:
            result = await self.inference.submit(request)
        except BaseException:
            usage.failures += 1
            raise
        usage.output_tokens += len(result.tokens)
        self.charge(owner, output_tokens=len(result.tokens))
        if prompt.exact_continuation:
            assert params is not None
            try:
                _validate_continuation_result(result, params)
            except ValueError:
                usage.failures += 1
                raise
        return result

    async def score(
        self, model: RewardModel, contexts: list[str], *, owner: Owner = DEFAULT_OWNER
    ) -> list[RewardResult]:
        usage = self.usage.setdefault(owner.episode, Usage())
        usage.calls += 1
        usage.value_batches += 1
        assert self.auxiliary is not None

        async def invoke() -> list[RewardResult]:
            owned = getattr(model, "ascore_owned", None)
            if owned is not None:
                result: list[RewardResult] = await owned(contexts, owner)
                return result
            return await model.ascore_batch(contexts)

        try:
            return await self.auxiliary.run(invoke, owner=owner, resource="value")
        except BaseException:
            usage.failures += 1
            raise

    async def __aenter__(self) -> ExecutionServices:
        await self.astart()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def release_episode(self, episode: str) -> Usage | None:
        """After all work settles, retire accounting; checkpoints retain budgets."""
        for key in tuple(self.budgets):
            if key[0] == episode:
                del self.budgets[key]
        return self.usage.pop(episode, None)

    async def aclose(self) -> None:
        assert self.auxiliary is not None
        results = await asyncio.gather(
            self.inference.aclose(),
            self.auxiliary.aclose(),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise ExceptionGroup("execution service cleanup failed", errors)
