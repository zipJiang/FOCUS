"""Running the step controller as a slime rollout function.

slime trains with many Ray actors, each calling a rollout function against a shared
inference server, and it hands each actor its endpoint **at runtime**. So a live
generator cannot be baked into an authored config; it has to be built inside the actor.
Hence two layers:

* :class:`~focus.config.RolloutSpec` -- picklable configuration only. Ships
  to every actor. Vendor-neutral, so it lives in ``focus.config`` and is
  re-exported here for the imports that name it through this module.
* :class:`~focus.loop.Runtime` -- the live bundle, built once per actor
  from the actor's own generator and tokenizer, carrying the spec's config.

Flattening a result into trainer samples is *not* one of those layers. slime consumes
the same span-per-sample shape any trainer would -- one sample per packed span, the head
it trains and the estimator that scored it named in its metadata -- so
:func:`~focus.export.to_samples` lives beside the records it flattens and is
re-exported here for the imports that already name it through this module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from focus.config import RolloutConfig, RolloutSpec
from focus.export import CriticSampleFactory, SampleFactory, to_samples
from focus.harness.workspace import Workspace
from focus.loop import Runtime, run_search
from focus.preparation import AdvantageEstimator, prepare_samples
from focus.preparation.records import ESTIMATOR_KEY, LANE_KEY

logger = logging.getLogger(__name__)


def make_generate(
    spec: RolloutSpec,
    runtime_factory: Callable[[Any], Runtime],
    *,
    estimator: AdvantageEstimator | None,
    to_prompt: Callable[[Any], Any] = lambda sample: sample,
    workspace_factory: Callable[[Any], Workspace] | None = None,
    sample_factory: SampleFactory[Any] | None = None,
    critic_sample_factory: CriticSampleFactory[Any] | None = None,
    min_abs_advantage: float | None = None,
) -> Callable[..., Any]:
    """Build the coroutine slime's ``--custom-generate-function-path`` expects.

    ``min_abs_advantage`` is forwarded to :func:`to_samples` after preparation;
    it filters actor edges without changing search or critic targets.

    **The precedence rule, whole:** the runtime a factory returns keeps its live objects
    and gets ``spec.config`` -- the authored configuration replaces whatever the factory
    left there. One assignment, because the configuration is one record: an actor-local
    override would make two actors run different knobs while both claimed the same spec.

    ``to_prompt`` says how to get the prompt out of the trainer's own sample type, and
    defaults to the identity because a caller whose ``sample`` *is* the prompt string
    should not have to say so. slime's ``Sample`` is an object, so a slime actor passes
    ``to_prompt=lambda sample: sample.prompt``; leaving it unset there hands the object
    itself to :func:`~focus.loop.run_search`, which refuses it by type.

    ``workspace_factory`` is the same question for the *workspace*, and it is a
    per-prompt one: a workspace is a rollout's own scratch -- notes a ``storage`` call
    writes, forked per branch -- so an env whose tools came from ``workspace.tools()``
    needs one built per sample and threaded into
    :func:`~focus.loop.run_search`. Left unset, a rollout gets a
    :class:`~focus.harness.workspace.NullWorkspace`, which is right for an
    env with no workspace tools and wrong for one that has them -- silently wrong, in a
    tool-result string the model reads, so
    :meth:`~focus.harness.runner.Runner.root` refuses that pairing outright.
    It takes the trainer's own sample, as ``to_prompt`` does, so a workspace can be
    seeded from the case it belongs to. It is not a spec
    field for the same reason a runner is not: the *factory* is authored where the env
    is built, and the workspace it returns holds live per-prompt state that no picklable
    record may carry. And it is not built once per actor: one workspace shared across
    prompts would carry one prompt's notes into the next and race between the concurrent
    rollouts of a single actor.

    ``runtime_factory`` owns everything that needs the actor's live endpoint, and the
    anchor scorer is one of those things. A ``mixture`` proposal on the config buys
    coverage and owes an importance ratio on its explore draws, which needs the anchor
    policy's logprobs over tokens it did not generate -- so a factory configuring one
    must also set :attr:`~focus.loop.Runtime.scorer`, e.g.
    ``TokenScorer(anchor_policy)``, or the run fails with a clear ``ValueError``
    before export. Not a config field, because a scorer holds a live policy, which is
    exactly what a picklable spec may not carry.

    :attr:`~focus.loop.Runtime.value_model` -- the value head's reward model
    -- belongs to the factory and supplies the network prior for refined TD.
    There is no version to
    keep in step with it: ``run_search`` files its estimates under the config's
    ``reward_config.value_version`` itself. Its bound belongs to the factory too --
    :attr:`~focus.loop.Runtime.score_timeout`, how long *that* server may take
    before a batch is abandoned as a scoring failure, set beside
    ``Runner(generate_timeout=...)`` for the same reason: a deadline is a fact about an
    endpoint, and a spec travels to actors that may not share one. A pass drains its
    in-flight expansions rather than cancelling them, so an unbounded backend of either
    kind holds the training step, not just its own rollout.
    """
    cached: tuple[Any, Runtime] | None = None

    async def generate(
        args: Any, sample: Any, sampling_params: Any = None
    ) -> list[Any]:
        nonlocal cached
        if cached is None or cached[0] is not args:
            cached = (args, replace(runtime_factory(args), config=spec.config))
        # Another call may replace the cache while this call awaits search.
        runtime = cached[1]
        index = getattr(sample, "index", 0)
        # Per sample, never per actor: the workspace is this prompt's scratch, and the
        # tree forks it per branch from whatever it starts as.
        workspace = None if workspace_factory is None else workspace_factory(sample)
        result = prepare_samples(
            await run_search(to_prompt(sample), runtime, workspace=workspace),
            estimator=estimator,
            reward_config=runtime.config.reward_config,
            behavior_version=runtime.runner.policy.version,
        )
        samples = to_samples(
            result,
            group_index=index,
            sample_factory=sample_factory,
            critic_sample_factory=critic_sample_factory,
            min_abs_advantage=min_abs_advantage,
        )
        # The one line a slime actor emits per prompt. The samples themselves are
        # unchanged by this -- an actor is a black box under Ray, and how many rollouts
        # a prompt cost (and how many died) is otherwise only visible in the tree it
        # threw away. WARNING when the group came back empty: the trainer takes an empty
        # list without complaint, so this line is the only thing that says a prompt
        # contributed nothing to the step, and the failure count beside it says why.
        logger.log(
            logging.WARNING if not samples else logging.INFO,
            (
                "rollout finished group=%s samples=%d actor=%d critic=%d stats=%s "
                + "failures=%d"
            ),
            index,
            len(samples),
            len(result.actor),
            len(result.critic),
            dict(result.stats),
            len(result.failures),
        )
        return samples

    return generate


__all__ = [
    "ESTIMATOR_KEY",
    "LANE_KEY",
    "RolloutConfig",
    "RolloutSpec",
    "make_generate",
    "to_samples",
]
