"""Expand isolated rollouts through the agent's completion loop.

The agent exposes folds and the endpoint; the expander returns them with the
metadata each new node records, such as the search pass that produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from focus.generation import SamplingParams
from focus.harness import Agent, RolloutState, Workspace

#: Node-metadata key holding the index of the search pass that created the node.
PASS_KEY = "pass"


class Expander[P](ABC):
    """Produce a node's children from its payload -- the scheduler's execution seam."""

    @abstractmethod
    async def expand(self, payload: P) -> tuple[list[P], Mapping[str, object]]:
        """The children, plus metadata to record on each node they become.

        The metadata rides back with the payloads rather than being written by the
        expander itself, because the payloads exist before the nodes do. An expander
        with nothing to record answers ``{}``.
        """


class RolloutExpander[S](Expander[RolloutState[S]]):
    """Fork an isolated branch, run to completion, and keep its checkpoints."""

    def __init__(
        self,
        agent: Agent[S, Any],
        *,
        sampling_params: SamplingParams | None = None,
        annotation: Mapping[str, object] | None = None,
    ) -> None:
        self._agent = agent
        self._sampling_params = sampling_params
        self._annotation: Mapping[str, object] = MappingProxyType(
            dict(annotation or {})
        )

    async def expand(
        self, payload: RolloutState[S]
    ) -> tuple[list[RolloutState[S]], Mapping[str, object]]:
        # fork() copies the workspace so concurrent expansions of the same node stay
        # isolated; the region then advances in place on that private copy.
        chain = [
            checkpoint
            async for checkpoint in self._agent.iter_run(
                payload.fork(), sampling_params=self._sampling_params
            )
        ]
        if chain and len(chain[-1].turns) == len(payload.turns):
            chain = []  # let the scheduler account for a no-progress expansion
        return chain, self._annotation

    async def start(
        self, prompt: str, *, workspace: Workspace | None = None
    ) -> RolloutState[S]:
        """Convenience: build the search root from a prompt (keeps the Agent here)."""
        return await self._agent.start(prompt, workspace=workspace)


__all__ = ["PASS_KEY", "Expander", "RolloutExpander"]
