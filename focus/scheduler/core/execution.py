"""Expand isolated rollouts through the runner's completion loop.

Proposal policies select the sampling parameters for each expansion. The runner
exposes folds and the endpoint; the expander returns them with edge provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from focus.harness import RolloutState, Runner, Workspace
from focus.scheduler.core.proposal import ANCHOR_ID, ProposalPolicy
from focus.scheduler.core.tree import Node

# -- provenance: what produced an edge, recorded on the node it produced -------------

#: Node-metadata key holding an edge's :class:`EdgeProvenance`. Written and read only
#: through :func:`stamp_provenance` / :func:`provenance_of`, never by string literal, so
#: the tree's general-purpose metadata stays a typed seam rather than loose keys.
PROVENANCE_KEY = "provenance"


@dataclass(frozen=True)
class EdgeProvenance:
    """How the edge into a node was produced -- stamped when the node is created.

    ``pass_id`` is launch identity (``"pass-0"``, ``"pass-1"``, ...) for budgeting.
    ``proposal_id`` names the draw law so off-anchor edges can be importance-corrected.
    """

    proposal_id: str
    continuation_id: str
    behavior_version: str
    reward_config_id: str
    pass_id: str = ""


def stamp_provenance(node: Node[Any], provenance: EdgeProvenance) -> None:
    """Record how ``node``'s incoming edge was produced."""
    node.metadata[PROVENANCE_KEY] = provenance


def provenance_of(node: Node[Any]) -> EdgeProvenance | None:
    """The recorded provenance of ``node``'s incoming edge, if it has one."""
    found = node.metadata.get(PROVENANCE_KEY)
    return found if isinstance(found, EdgeProvenance) else None


def drawn_from_anchor(node: Node[Any], anchor_version: str = ANCHOR_ID) -> bool:
    """Whether this edge was drawn from the anchor itself -- so needs no correction.

    Read off the recorded proposal, not off version names or a temperature: whether an
    importance ratio is needed is a fact about *which law produced the edge*, and the
    v1 configuration draws everything from the anchor, which is exactly why the anchor
    logprob channel is deliberately never filled there. Deciding this by comparing
    channel names would demand that unfilled channel and fail.

    One comparison, against the name the caller passed, and no second spelling: also
    accepting the literal ``"anchor"`` would make *two* laws the anchor for a run that
    had renamed its channel, so an edge drawn from a proposal called ``anchor`` would
    skip the correction it is owed under an ``anchor_version`` of some other name. The
    proposals default their ``proposal_id`` to the same
    :data:`~focus.scheduler.core.proposal.ANCHOR_ID` this defaults to, so the
    out-of-box configuration needs neither name written down.
    """
    stamp = provenance_of(node)
    return stamp is not None and stamp.proposal_id == anchor_version


# -- expansion: the injected node-payload -> children operation ----------------------


class Expander[P](ABC):
    """Produce a node's children from its payload -- the scheduler's execution seam."""

    @abstractmethod
    async def expand(self, payload: P) -> tuple[list[P], Mapping[str, object]]:
        """The children, plus metadata to record on each node they become.

        The metadata rides back with the payloads rather than being written by the
        expander itself, because the payloads exist before the nodes do: an expander
        knows how it drew an edge but cannot reach the node that edge will produce, and
        returning the two together is what keeps that knowledge from having to be
        stashed on the expander between calls -- where concurrent expansions would race
        over it. An expander with nothing to record answers ``{}``.
        """


class RolloutExpander[S](Expander[RolloutState[S]]):
    """Fork an isolated branch, run to completion, and keep fold checkpoints.

    One class, configured per pass rather than subclassed: every search pass shares the
    same expander and proposal policy, and each stamps its own pass provenance.
    """

    def __init__(
        self,
        runner: Runner[S, Any],
        *,
        proposal: ProposalPolicy | None = None,
        provenance: EdgeProvenance | None = None,
    ) -> None:
        self._runner = runner
        self._proposal = proposal
        self._provenance = provenance

    async def expand(
        self, payload: RolloutState[S]
    ) -> tuple[list[RolloutState[S]], Mapping[str, object]]:
        proposal_id, sampling_params = (
            self._proposal.draw() if self._proposal is not None else (None, None)
        )
        # fork() copies the workspace so concurrent expansions of the same node stay
        # isolated; the region then advances in place on that private copy.
        chain = [
            checkpoint
            async for checkpoint in self._runner.iter_run(
                payload.fork(), sampling_params=sampling_params
            )
        ]
        if chain and len(chain[-1].turns) == len(payload.turns):
            chain = []  # let the scheduler account for a no-progress expansion
        return chain, self._annotation(proposal_id)

    def _annotation(self, proposal_id: str | None) -> Mapping[str, object]:
        """This expansion's provenance, with the drawn proposal folded in."""
        if self._provenance is None:
            return {}
        stamp = self._provenance
        if proposal_id is not None and proposal_id != stamp.proposal_id:
            stamp = replace(stamp, proposal_id=proposal_id)
        return {PROVENANCE_KEY: stamp}

    async def start(
        self, prompt: str, *, workspace: Workspace | None = None
    ) -> RolloutState[S]:
        """Convenience: build the search root from a prompt (keeps the Runner here)."""
        return await self._runner.start(prompt, workspace=workspace)


__all__ = [
    "PROVENANCE_KEY",
    "EdgeProvenance",
    "Expander",
    "RolloutExpander",
    "drawn_from_anchor",
    "provenance_of",
    "stamp_provenance",
]
