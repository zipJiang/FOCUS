"""Serializable search configuration. Training recipes are supplied to preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from focus.allocation import Allocator
from focus.generation import SamplingParams
from focus.scheduler.core.gating import GatingPolicy
from focus.values import RewardConfig


@dataclass(frozen=True)
class SearchPass:
    """One allocation algorithm with an explicit turn budget and branching gate.

    ``sampling_params`` overrides the runner's sampling for this pass's continuations.
    """

    allocator: Allocator
    budget: int
    gating: GatingPolicy
    sampling_params: SamplingParams | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int)
            or self.budget < 0
        ):
            raise ValueError("SearchPass.budget must be a nonnegative integer")


@dataclass(frozen=True, kw_only=True)
class RolloutConfig:
    """Every knob one rollout is run with, with the value it runs at if unwritten.

    Real defaults, not ``None`` markers: what a field says *is* what the run does, so a
    config read back after loading is the configuration, with nothing left to resolve
    somewhere downstream.

    Keyword-only: the fields are knobs in
    no meaningful order. So there is no
    positional call to keep working -- and none to silently re-bind when a knob is
    added or two are reordered.
    """

    passes: tuple[SearchPass, ...] = ()
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.passes, tuple):
            object.__setattr__(self, "passes", tuple(self.passes))


@dataclass(frozen=True)
class RolloutSpec:
    """The picklable whole of an experiment: what to run it on, and how to run it.

    Ships to a Ray actor unchanged, because nothing here is live: :attr:`env` is a
    plain mapping, and the runner built from it -- with the actor's own generator
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
