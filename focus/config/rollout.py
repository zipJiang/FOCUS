"""Serializable search configuration. Training recipes are supplied to preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from focus.allocation import Allocator
from focus.registry import registrable
from focus.reward.config import RewardConfig
from focus.scheduler.core.gating import GatingPolicy
from focus.scheduler.core.proposal import AnchorProposal, ProposalPolicy


@registrable(builds_self=True)
@dataclass(frozen=True)
class SearchPass:
    """One allocation algorithm with an explicit turn budget and branching gate."""

    allocator: Allocator
    budget: int
    gating: GatingPolicy

    def __post_init__(self) -> None:
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int)
            or self.budget < 0
        ):
            raise ValueError("SearchPass.budget must be a nonnegative integer")


@registrable(builds_self=True)
@dataclass(frozen=True, kw_only=True)
class RolloutConfig:
    """Every knob one rollout is run with, with the value it runs at if unwritten.

    Real defaults, not ``None`` markers: what a field says *is* what the run does, so a
    config read back after loading is the configuration, with nothing left to resolve
    somewhere downstream.

    A ``builds_self`` registry namespace, so a nested mapping becomes this record
    in one call -- ``build(RolloutConfig, mapping)``. Each field is annotated
    with the interface it holds, which is what lets the registry turn a nested mapping
    (a proposal, a list of search passes, a ``reward_config:``
    block) into the object the field is typed for, rather than leaving a ``dict`` that
    would look configured and behave defaulted.

    Keyword-only, like every self-building config record here: the fields are knobs in
    no meaningful order, and the authored form is a mapping anyway. So there is no
    positional call to keep working -- and none to silently re-bind when a knob is
    added or two are reordered.
    """

    passes: tuple[SearchPass, ...] = ()
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    #: The law an expansion draws under. A ``mixture`` here obliges whoever builds the
    #: :class:`~focus.loop.Runtime` to set its live ``scorer`` too: an
    #: off-anchor draw owes an importance ratio, and that needs the anchor's logprobs.
    #:
    #: Its ``proposal_id`` and :attr:`reward_config`'s ``anchor_version`` are one name
    #: written twice -- "was this edge drawn from the anchor?" is that one comparison
    #: (:func:`~focus.scheduler.core.execution.drawn_from_anchor`). Both
    #: default to :data:`~focus.scheduler.core.proposal.ANCHOR_ID`, so
    #: renaming *one* of them makes every anchor draw look off-anchor and the run stops
    #: in ``rescore_anchor`` asking for a scorer it should not need.
    proposal: ProposalPolicy = field(default_factory=AnchorProposal)
    #: The continuation identity stamped on every edge. The *behavior* channel is not
    #: authored beside it: it is whatever the actor's runner files its generations
    #: under, and :func:`~focus.loop.run_search` reads it off that runner
    #: rather than trusting a second copy here.
    continuation_id: str = "nu-0"
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.passes, tuple):
            object.__setattr__(self, "passes", tuple(self.passes))


@registrable(builds_self=True)
@dataclass(frozen=True)
class RolloutSpec:
    """The picklable whole of an experiment: what to run it on, and how to run it.

    Ships to a Ray actor unchanged, because nothing here is live: :attr:`env` is a
    *registry config*, and the runner built from it -- with the actor's own generator
    and tokenizer -- belongs to the factory inside the actor.

    Three fields, and the third is the configuration entire: a spec is a
    :class:`RolloutConfig` plus what the factory needs to build a runner, and nothing
    else.
    """

    #: Registry config for the env, and the system prompt it is prompted with: inputs
    #: to the ``runtime_factory``, which is what builds the runner around them. Nothing
    #: here reads them.
    env: Mapping[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    config: RolloutConfig = field(default_factory=RolloutConfig)


__all__ = [
    "RolloutConfig",
    "RolloutSpec",
    "SearchPass",
]
