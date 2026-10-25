"""Every registered name, built from the plain mapping a config file can express.

A registry name exists so an experiment can be *written down*. A name that only
constructs when the caller imports the class and hands it live objects is therefore
half-registered -- and nothing else in the suite would notice, because every other test
constructs its policies directly. So this file is a table: one row per registered name,
with the minimal kwargs that name accepts, built through ``build`` and nothing else.

The table is checked against the registry itself (:func:`registered_names`), so a new
implementation cannot land without either appearing here or failing the sweep.

The generation and reward backends are the one place a live object is unavoidable: they
take an injected client (``post``, ``sampling_client``, ``codec``) or, for the lazily
imported ``openai`` SDK, a stubbed import. That stands in for a socket, not for
configuration; every row still goes through the registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from focus import (
    Allocator,
    BestFirstAllocator,
    FifoAllocator,
    ValueRefinement,
)
from focus.allocation import CalibrationModel
from focus.config import RolloutConfig

# Importing a package is what fires its registrations, so every slot's home package is
# imported below -- the interfaces this file names happen to cover all of them.
from focus.execution import InferenceExecutor
from focus.generation import Policy, SamplingParams
from focus.generation.parsing.base import ActionParser
from focus.harness.checkpoint import CheckpointPolicy
from focus.harness.compaction import (
    AgenticCompactor,
    AnyOf,
    Compactor,
    PromptTokens,
    Trigger,
)
from focus.harness.workspace import Workspace
from focus.preparation import AdvantageEstimator
from focus.protocols import ContextProtocol
from focus.registry import (
    _REGISTRY,
    build,
    registered_interfaces,
    registered_names,
)
from focus.reward import RewardModel
from focus.reward.config import RewardConfig
from focus.scheduler import (
    CriticSelection,
    GatingPolicy,
    MixtureProposal,
    ProposalPolicy,
    Termination,
)
from tests.helpers import FakeChatTokenizer, native_format, test_format


async def _post(url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The injected transport of the slime backend -- never called here."""
    del url, payload
    return {}


#: ``(interface, config)`` for every registered name, with the minimal kwargs that name
#: accepts -- so a row doubles as the documented shape of that slot's entry.
TABLE: tuple[tuple[type, Mapping[str, Any]], ...] = (
    (InferenceExecutor, {"name": "local"}),
    (ContextProtocol, {"name": "append_only"}),
    (ContextProtocol, {"name": "summary"}),
    (ContextProtocol, {"name": "clm"}),
    (ContextProtocol, {"name": "memex"}),
    (ContextProtocol, {"name": "rlm"}),
    (Policy, {"name": "token_continuation", "backend": Policy(format=test_format())}),
    # -- generation and reward backends ----------------------------------------------
    (CheckpointPolicy, {"name": "sparse"}),
    (CheckpointPolicy, {"name": "every_decision"}),
    (
        Policy,
        {
            "name": "vllm",
            "base_url": "http://x/v1",
            "api_key": "k",
            "model": "m",
            "format": test_format(),
        },
    ),
    (
        Policy,
        {
            "name": "openai_completion",
            "base_url": "http://x/v1",
            "api_key": "k",
            "model": "m",
            "format": test_format(),
        },
    ),
    (
        Policy,
        {
            "name": "openai_chat",
            "model": "m",
            "format": native_format(FakeChatTokenizer()),
        },
    ),
    (
        Policy,
        {
            "name": "openai",
            "model": "m",
            "format": native_format(FakeChatTokenizer()),
        },
    ),
    (
        Policy,
        {
            "name": "openai_responses",
            "model": "m",
            "format": native_format(FakeChatTokenizer()),
        },
    ),
    (
        Policy,
        {
            "name": "sglang",
            "base_url": "http://x",
            "model": "m",
            "format": test_format(),
        },
    ),
    (
        Policy,
        {
            "name": "slime",
            "model": "m",
            "format": test_format(),
            "post": _post,
            "url": "http://x/generate",
        },
    ),
    (
        Policy,
        {
            "name": "tinker",
            "sampling_client": MagicMock(),
            "model": "m",
            "profile": "text",
        },
    ),
    (RewardModel, {"name": "vllm", "base_url": "http://x", "model": "rm"}),
    # -- harness ---------------------------------------------------------------------
    (Workspace, {"name": "null"}),
    (Workspace, {"name": "memory"}),
    (ActionParser, {"name": "text"}),
    (ActionParser, {"name": "token_chunk", "chunk_size": 256}),
    (ActionParser, {"name": "hermes"}),
    (ActionParser, {"name": "native"}),
    (ActionParser, {"name": "qwen_xml"}),
    (Trigger, {"name": "prompt_tokens", "max_prompt_tokens": 128}),
    (Trigger, {"name": "state_counter", "remaining": "left", "maximum": "cap"}),
    (Trigger, {"name": "any", "triggers": [{"name": "prompt_tokens"}]}),
    (Compactor, {"name": "agentic", "max_interactions": 2}),
    (Compactor, {"name": "budget", "max_prompt_tokens": 128}),
    # -- scheduler -------------------------------------------------------------------
    (ProposalPolicy, {"name": "anchor"}),
    (ProposalPolicy, {"name": "mixture", "explore": {"temperature": 1.5}}),
    (GatingPolicy, {"name": "width", "width": 3}),
    (GatingPolicy, {"name": "concurrency", "limit": 3}),
    (Allocator, {"name": "fifo"}),
    (Allocator, {"name": "best_first", "key": "reward"}),
    (Allocator, {"name": "windowed_token_entropy", "window": 4}),
    (Termination, {"name": "budget", "max_turns": 8}),
    (Allocator, {"name": "value_refinement"}),
    (Allocator, {"name": "fifo"}),
    (Allocator, {"name": "fifo"}),
    (CriticSelection, {"name": "terminals"}),
    (CriticSelection, {"name": "all"}),
    # -- weighting -------------------------------------------------------------------
    (AdvantageEstimator, {"name": "refined_td"}),
    (AdvantageEstimator, {"name": "direct_branch_td"}),
    (AdvantageEstimator, {"name": "grpo"}),
    (AdvantageEstimator, {"name": "vine_ppo"}),
    (CalibrationModel, {"name": "beta_bernoulli"}),
    (CalibrationModel, {"name": "gaussian", "v_prior": 0.5}),
)


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """The vLLM backend builds its client in ``__init__``; the SDK is an extra."""
    monkeypatch.setattr("focus.generation.vllm._import_openai", lambda: MagicMock())
    monkeypatch.setattr("focus.generation.sglang._import_httpx", lambda: MagicMock())


@pytest.mark.parametrize(
    ("interface", "config"),
    TABLE,
    ids=[f"{interface.__name__}-{config['name']}" for interface, config in TABLE],
)
def test_every_registered_name_builds_from_a_mapping(
    interface: type,
    config: Mapping[str, Any],
    stub_openai: None,
) -> None:
    built: Any = build(interface, config)

    assert isinstance(built, interface)


def _shipped(interface: type) -> set[str]:
    """The names of ``interface`` whose implementation ships with the package.

    The registry is one process-wide set of tables, and other test modules register
    fakes into it (a ``toy`` env, a ``unit_custom`` generator). Those are not part of
    the config surface this file and the docs are the inventory of, so the entry's
    defining module is what decides -- read off the private table because that is the
    only place the *provenance* of a registration lives.
    """
    return {
        name
        for name, entry in _REGISTRY.get(interface, {}).items()
        if getattr(entry.target, "__module__", "").startswith("focus")
    }


def _shipped_interfaces() -> list[type]:
    return sorted(
        (i for i in registered_interfaces() if _shipped(i)),
        key=lambda cls: cls.__name__,
    )


def test_the_table_covers_every_registered_name() -> None:
    """Exhaustiveness, read off the registry rather than trusted to a reviewer."""
    covered: dict[type, set[str]] = {}
    for interface, config in TABLE:
        covered.setdefault(interface, set()).add(str(config["name"]))

    missing = {
        interface.__name__: sorted(_shipped(interface) - covered.get(interface, set()))
        for interface in _shipped_interfaces()
        if _shipped(interface) - covered.get(interface, set())
    }

    assert not missing
    assert set(_shipped_interfaces()) == set(covered)
    # and the public accessor agrees about what it can see
    assert set(registered_names(Trigger)) >= _shipped(Trigger)


# -- nested construction --------------------------------------------------------------
#
# The rows above are flat. What a real config file has is a policy holding another
# policy -- or a whole list of them -- and the registry only descends where something
# told it how: an annotation it can resolve, or explicit ``nested_builders`` wiring for
# the shapes that carry no annotation at all (``**kwargs``).


def test_a_dataclass_config_is_coerced_from_its_annotation() -> None:
    """A ``builds_self`` namespace (``SamplingParams``) annotated on a parameter
    turns a mapping into the record, not a dict the constructor then holds."""
    proposal = build(
        ProposalPolicy,
        {"name": "mixture", "explore": {"temperature": 1.4, "max_tokens": 64}},
    )
    allocator = build(Allocator, {"name": "value_refinement"})

    assert isinstance(proposal, MixtureProposal)
    assert isinstance(allocator, ValueRefinement)
    assert proposal._explore == SamplingParams(temperature=1.4, max_tokens=64)
    assert (
        build(RewardConfig, {"value_prior_strength": 2.0}).value_prior_strength == 2.0
    )


def test_a_compactor_takes_its_trigger_and_sampling_params_as_mappings() -> None:
    compactor = build(
        Compactor,
        {
            "name": "agentic",
            "trigger": {"name": "state_counter", "remaining": "left"},
            "sampling_params": {"temperature": 0.0},
            "max_interactions": 3,
        },
    )

    assert isinstance(compactor, AgenticCompactor)
    assert isinstance(compactor.trigger, Trigger)
    assert compactor._sampling_params == SamplingParams(temperature=0.0)


def test_the_budget_compactor_forwards_a_mapping_through_its_kwargs() -> None:
    """``**agentic_kwargs`` carries no annotation, so this needs explicit wiring."""
    compactor = build(
        Compactor, {"name": "budget", "sampling_params": {"temperature": 0.2}}
    )

    assert isinstance(compactor, AgenticCompactor)
    assert compactor._sampling_params == SamplingParams(temperature=0.2)


def test_a_composite_trigger_builds_its_members() -> None:
    trigger = build(
        Trigger,
        {
            "name": "any",
            "triggers": [
                {"name": "prompt_tokens", "max_prompt_tokens": 16},
                {"name": "state_counter", "remaining": "left", "maximum": "cap"},
            ],
        },
    )

    assert isinstance(trigger, AnyOf)
    assert [type(t).__name__ for t in trigger._triggers] == [
        "PromptTokens",
        "StateCounter",
    ]


def test_an_already_built_member_passes_through_the_composite() -> None:
    """Half a config and half an object is the shape a Python-side override takes."""
    inner = PromptTokens(4)
    trigger = build(
        Trigger,
        {"name": "any", "triggers": [inner, {"name": "prompt_tokens"}]},
    )

    assert isinstance(trigger, AnyOf)
    assert trigger._triggers[0] is inner


def test_a_sequence_field_builds_every_member_from_its_annotation() -> None:
    """``weighters: tuple[AdvantageEstimator, ...]`` is as buildable as a single field.

    A field holding *several* implementations is the shape a real config file has, and
    without the annotation walk descending into it a list of ``{"name": ...}`` mappings
    arrives as a list of dicts -- configured-looking, defaulted-behaving. The container
    comes back as the tuple the annotation asks for, so a built config compares equal to
    a hand-written one.
    """
    config = build(
        RolloutConfig,
        {
            "passes": [
                {
                    "allocator": {"name": "fifo"},
                    "budget": 4,
                    "gating": {"name": "concurrency", "limit": 4},
                },
                {
                    "allocator": {"name": "value_refinement"},
                    "budget": 4,
                    "gating": {"name": "concurrency", "limit": 4},
                },
            ]
        },
    )

    assert isinstance(config.passes, tuple)
    assert [type(w) for w in (p.allocator for p in config.passes)] == [
        FifoAllocator,
        ValueRefinement,
    ]


def test_an_already_built_member_passes_through_a_sequence_field() -> None:
    """Half a config and half an object, as a Python-side override writes it."""
    inner = FifoAllocator()

    config = build(
        RolloutConfig,
        {
            "passes": [
                {
                    "allocator": inner,
                    "budget": 3,
                    "gating": {"name": "concurrency", "limit": 4},
                },
                {
                    "allocator": {"name": "value_refinement"},
                    "budget": 4,
                    "gating": {"name": "concurrency", "limit": 4},
                },
            ]
        },
    )

    assert config.passes
    assert config.passes[0].allocator is inner


def test_an_optional_interface_field_still_builds_from_a_mapping() -> None:
    """``| None`` says what an *absent* value means, not that the present one is opaque:
    ``selection`` is unset by default because the scheduler owns that default."""
    selection = {"name": "best_first", "key": "reward"}

    config = build(
        RolloutConfig,
        {
            "passes": [
                {
                    "allocator": selection,
                    "budget": 32,
                    "gating": {"name": "width", "width": 3},
                }
            ]
        },
    )

    assert isinstance(config.passes[0].allocator, BestFirstAllocator)
    assert RolloutConfig().passes == ()


def test_a_sequence_of_plain_values_is_left_exactly_as_it_was_written() -> None:
    """The walk descends only where the members are themselves buildable. Every other
    list on a config is data, and retyping it would be a silent change of meaning."""
    params = build(SamplingParams, {"stop": ["</s>", "\n\n"]})

    assert params.stop == ["</s>", "\n\n"]


def test_an_unknown_name_says_what_was_registered() -> None:
    with pytest.raises(ValueError, match="unknown trigger 'nope'; registered: any"):
        build(Trigger, {"name": "nope"})
