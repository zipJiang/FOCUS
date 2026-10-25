"""Build typed rollout settings from nested plain mappings."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from focus import (
    ConcurrencyGating,
    FifoAllocator,
    ValueRefinement,
    WidthGating,
)
from focus.config import RolloutConfig, RolloutSpec
from focus.loop import Runtime
from focus.registry import build
from focus.reward.config import RewardConfig
from focus.scheduler import MixtureProposal


def _settings() -> dict[str, Any]:
    return {
        "system_prompt": "solve it",
        "env": {"name": "toy", "size": 3},
        "config": {
            "reward_config": {
                "gamma": 0.99,
                "beta_step": 0.5,
                "value_version": "v-1",
                "clip": 4.0,
            },
            "proposal": {
                "name": "mixture",
                "epsilon": 0.2,
                "explore": {"temperature": 1.3},
            },
            "passes": [
                {
                    "allocator": {"name": "fifo"},
                    "budget": 12,
                    "gating": {"name": "width", "width": 4},
                },
                {
                    "allocator": {"name": "fifo"},
                    "budget": 16,
                    "gating": {"name": "concurrency", "limit": 4},
                },
                {
                    "allocator": {"name": "value_refinement"},
                    "budget": 8,
                    "gating": {"name": "concurrency", "limit": 4},
                },
            ],
            "continuation_id": "nu-7",
            "max_concurrency": 6,
        },
    }


def test_a_mapping_builds_a_spec_with_nested_components() -> None:
    spec = build(RolloutSpec, _settings())

    assert spec.system_prompt == "solve it"
    assert spec.env == {"name": "toy", "size": 3}
    config = spec.config
    assert config.passes[0].budget == 12
    assert config.reward_config == RewardConfig(
        gamma=0.99, beta_step=0.5, value_version="v-1", clip=4.0
    )
    assert isinstance(config.proposal, MixtureProposal)
    assert isinstance(config.passes[0].gating, WidthGating)
    # a list of estimator configs is a tuple of estimators, not a tuple of dicts
    assert [type(w) for w in [p.allocator for p in config.passes[1:]]] == [
        FifoAllocator,
        ValueRefinement,
    ]
    assert config.continuation_id == "nu-7"
    assert config.max_concurrency == 6


def test_a_configuration_mapping_builds_the_record_itself() -> None:
    """``env`` and ``system_prompt`` are what an *actor's factory* needs; a run that
    builds its own runner has nothing to say about them and skips the spec entirely."""
    config = build(RolloutConfig, {"passes": []})

    assert config.passes == ()  # authored: no estimators at all


def test_the_built_config_is_what_the_actor_s_runtime_runs_with() -> None:
    """The end of the path: what the trainer's actor would hold.

    One assignment, not a field-by-field transfer -- the configuration is one record,
    and the factory's half of the runtime is the live objects it does not touch.
    """
    spec = build(RolloutSpec, _settings())

    runtime = replace(Runtime(runner=None), config=spec.config)  # type: ignore[arg-type]

    assert runtime.config is spec.config
    assert runtime.config.passes[0].budget == 12


def test_a_mapping_builds_named_components_and_keeps_defaults() -> None:
    config = build(
        RolloutConfig,
        {
            "passes": [
                {
                    "allocator": {"name": "fifo"},
                    "budget": 12,
                    "gating": {"name": "width", "width": 3},
                },
                {
                    "allocator": {"name": "fifo"},
                    "budget": 4,
                    "gating": {"name": "concurrency", "limit": 2},
                },
            ]
        },
    )

    assert config.passes[0].budget == 12
    assert [type(w).__name__ for w in [p.allocator for p in config.passes[1:]]] == [
        "FifoAllocator"
    ]
    assert isinstance(config.passes[-1].gating, ConcurrencyGating)
    # and what the file did not say is the documented default, with nothing left to
    # resolve later: a config read back is the configuration
    assert config.max_concurrency == 4
    assert config.reward_config == RewardConfig()
