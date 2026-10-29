"""Local request admission, continuation validation, and usage accounting."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from focus.execution.budget import ResourceBudget
from focus.execution.pool import DEFAULT_OWNER, AdmissionPool, Owner
from focus.generation import GenerateResult, Policy, PreparedPrompt, SamplingParams
from focus.generation._common import (
    _continuation_params,
    _validate_continuation_result,
)

if TYPE_CHECKING:
    from focus.critic import Critic


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
    """Explicitly shared services; derived agents inherit this same object."""

    pool: AdmissionPool = field(default_factory=AdmissionPool)
    usage: dict[str, Usage] = field(default_factory=dict)
    budgets: dict[tuple[str, str], ResourceBudget] = field(default_factory=dict)

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
        usage = self.usage.setdefault(owner.episode, Usage())
        self.charge(owner, model_calls=1, prompt_tokens=len(prompt.tokens))
        usage.calls += 1
        usage.prompt_tokens += len(prompt.tokens)
        params = deepcopy(params)

        async def generate() -> GenerateResult:
            return await policy.agenerate(prompt, params)

        try:
            result = await self.pool.run(
                generate,
                owner=owner,
                resource="generation",
                queue_timeout=queue_timeout,
                execution_timeout=timeout,
            )
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

    async def predict(
        self, critic: Critic, contexts: list[str], *, owner: Owner = DEFAULT_OWNER
    ) -> Sequence[float]:
        """Run one critic batch under the shared admission pool."""
        usage = self.usage.setdefault(owner.episode, Usage())
        usage.calls += 1
        usage.value_batches += 1

        async def invoke() -> Sequence[float]:
            return await critic.predict(contexts)

        try:
            return await self.pool.run(invoke, owner=owner, resource="value")
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
        """Cancel owned requests. Policy and critic clients belong to the caller."""
        await self.pool.aclose()


__all__ = ["ExecutionServices", "Usage"]
