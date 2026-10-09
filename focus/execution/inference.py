"""Serializable inference requests and shared local dispatch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from focus.codec import Message
from focus.execution.pool import DEFAULT_OWNER, AdmissionPool, Owner
from focus.execution.process import ProcessExecutor
from focus.generation import GenerateResult, Policy, PreparedPrompt, SamplingParams
from focus.generation.interfaces import aclose_client


@dataclass(frozen=True)
class InferenceRequest:
    backend: str
    format_id: str
    tokens: tuple[int, ...]
    messages: tuple[Message, ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    params: SamplingParams | None = None
    owner: Owner = DEFAULT_OWNER
    token_input: bool = False
    queue_timeout: float | None = None
    execution_timeout: float | None = None


class InferenceExecutor(Protocol):
    def bind(self, policy: Policy[Any]) -> str: ...

    async def submit(self, request: InferenceRequest) -> GenerateResult: ...

    async def aclose(self) -> None: ...


def policy_key(policy: Policy[Any]) -> str:
    return f"{policy.model}:{policy.version}:{policy.format.name}"


class LocalInferenceExecutor:
    def __init__(self, pool: AdmissionPool | None = None, *, own_clients: bool = False):
        self.pool = pool or AdmissionPool()
        self.policies: dict[str, Policy[Any]] = {}
        self.own_clients = own_clients

    def bind(self, policy: Policy[Any]) -> str:
        key = policy_key(policy)
        existing = self.policies.get(key)
        if existing is not None and existing is not policy:
            raise ValueError(
                f"different policies registered under the same identity: {key}"
            )
        self.policies[key] = policy
        return key

    async def submit(self, request: InferenceRequest) -> GenerateResult:
        policy = self.policies[request.backend]
        if policy.format.name != request.format_id:
            raise ValueError("request format differs from registered policy")

        async def generate() -> GenerateResult:
            if request.token_input:
                return await policy.agenerate_tokens(request.tokens, request.params)
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
        await self.pool.aclose()
        if self.own_clients:
            for policy in self.policies.values():
                close = getattr(policy, "aclose", None)
                if close is not None:
                    await close()
                else:
                    await aclose_client(policy)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0


@dataclass
class ExecutionServices:
    """Explicitly shared services; derived runners inherit this same object."""

    inference: InferenceExecutor = field(default_factory=LocalInferenceExecutor)
    auxiliary: AdmissionPool | None = None
    processes: ProcessExecutor = field(default_factory=ProcessExecutor)
    usage: dict[str, Usage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.auxiliary is None:
            self.auxiliary = (
                self.inference.pool
                if isinstance(self.inference, LocalInferenceExecutor)
                else AdmissionPool()
            )

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
        token_input: bool = False,
        queue_timeout: float | None = None,
    ) -> GenerateResult:
        key = self.inference.bind(policy)
        usage = self.usage.setdefault(owner.episode, Usage())
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
            token_input=token_input,
            execution_timeout=timeout,
            queue_timeout=queue_timeout,
        )
        try:
            result = await self.inference.submit(request)
        except BaseException:
            usage.failures += 1
            raise
        usage.output_tokens += len(result.tokens)
        return result

    async def __aenter__(self) -> ExecutionServices:
        await self.astart()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.processes.aclose()
        await self.inference.aclose()
        assert self.auxiliary is not None
        await self.auxiliary.aclose()
