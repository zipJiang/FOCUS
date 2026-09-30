"""A YAML experiment file, all the way to the objects a rollout runs with.

Two hops, and the second is the one that has actually broken: ``load_yaml`` is a thin
``OmegaConf.to_container``, but what comes out of it is nested *plain containers* --
lists where a field says tuple, dicts where it says a policy or a frozen record -- and a
config that accepted those without coercing them would look configured and behave
defaulted.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

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

YAML = """\
system_prompt: solve it
env:
  name: toy
  size: 3
config:
  reward_config:
    gamma: 0.99
    beta_step: 0.5
    value_version: v-1
    clip: 4.0
  proposal:
    name: mixture
    epsilon: 0.2
    explore:
      temperature: 1.3
  passes:
    - allocator: {name: fifo}
      budget: 12
      gating: {name: width, width: 4}
    - allocator: {name: fifo}
      budget: 16
      gating: {name: concurrency, limit: 4}
    - allocator: {name: value_refinement}
      budget: 8
      gating: {name: concurrency, limit: 4}
  continuation_id: nu-7
  max_concurrency: 6
"""


def _loaded(tmp_path: Path, text: str) -> dict[str, Any]:
    omegaconf = pytest.importorskip("omegaconf")
    del omegaconf  # only its presence matters; load_yaml imports it itself

    from focus.config.omegaconf import load_yaml

    path = tmp_path / "experiment.yaml"
    path.write_text(text)
    return load_yaml(path)


def test_a_yaml_file_loads_into_a_spec_with_everything_it_names_built(
    tmp_path: Path,
) -> None:
    spec = build(RolloutSpec, _loaded(tmp_path, YAML))

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


def test_a_file_that_is_only_configuration_loads_into_the_record_itself(
    tmp_path: Path,
) -> None:
    """``env`` and ``system_prompt`` are what an *actor's factory* needs; a run that
    builds its own runner has nothing to say about them and skips the spec entirely."""
    config = build(RolloutConfig, _loaded(tmp_path, "passes: []\n"))

    assert config.passes == ()  # authored: no estimators at all


def test_the_loaded_config_is_what_the_actor_s_runtime_runs_with(
    tmp_path: Path,
) -> None:
    """The end of the path: what the trainer's actor would hold.

    One assignment, not a field-by-field transfer -- the configuration is one record,
    and the factory's half of the runtime is the live objects it does not touch.
    """
    spec = build(RolloutSpec, _loaded(tmp_path, YAML))

    runtime = replace(Runtime(runner=None), config=spec.config)  # type: ignore[arg-type]

    assert runtime.config is spec.config
    assert runtime.config.passes[0].budget == 12


def test_load_yaml_says_which_extra_to_install_when_omegaconf_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The import is inside the function precisely so this is the failure users see."""
    import builtins

    from focus.config.omegaconf import load_yaml

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "omegaconf":
            raise ImportError("no omegaconf")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"focus\[config\]"):
        load_yaml(tmp_path / "missing.yaml")


def test_a_plain_mapping_takes_the_same_path_without_omegaconf() -> None:
    """``load_yaml`` only ever produces plain containers, so the coercion this file is
    about is testable -- and is tested -- with the dependency absent."""
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
