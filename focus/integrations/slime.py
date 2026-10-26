"""Plug FOCUS into Slime's rollout and training hooks.

Slime owns the loop; FOCUS is called through hooks named on Slime's command line. The
user supplies the same :data:`~focus.integrations.common.CollectorFor` as for Tinker:

.. code-block:: python

    generate = make_generate(collector_for, critic_sink=JsonlDirectory(path))
    evaluate = make_evaluate(collector_for)

and points Slime at them, plus this module's :func:`preserve_advantages`,
:func:`advantages`, and :func:`sum_policy_tokens`::

    --custom-generate-function-path my_task.generate
    --custom-reward-post-process-path focus.integrations.slime.preserve_advantages
    --custom-advantage-function-path focus.integrations.slime.advantages
    --custom-pg-loss-reducer-function-path focus.integrations.slime.sum_policy_tokens

The hooks follow Slime's customization interfaces at revision ``18868a79``. Each
generation builds a :class:`~focus.backends.SlimePolicy` for Slime's SGLang router,
wraps the sample as a :class:`~focus.integrations.common.Problem`, and collects it with
a fresh collector, so concurrent prompts keep their own reference answers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from copy import copy
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Protocol

from focus.generation.policy import PolicyFormat
from focus.generation.slime import SlimePolicy
from focus.harness.packing import regions
from focus.harness.workspace import Workspace
from focus.integrations.common import (
    CollectorFor,
    CriticSink,
    Problem,
    collect_records,
)
from focus.preparation.records import LANE_KEY

logger = logging.getLogger(__name__)


class GenerateHook(Protocol):
    async def __call__(
        self, args: Any, sample: Any, sampling_params: Any = None
    ) -> Any: ...


@lru_cache(maxsize=4)
def _tokenizer(checkpoint: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint)


def policy_for(args: Any, *, profile: str | None = None) -> SlimePolicy:
    """The policy Slime's router serves: ``args.hf_checkpoint`` behind its URL."""
    from slime.utils.http_utils import post

    checkpoint = args.hf_checkpoint
    format = PolicyFormat.resolve(
        checkpoint, tokenizer=_tokenizer(checkpoint), profile=profile
    )
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    return SlimePolicy(post, url, model=checkpoint, format=format)


def problem_of(sample: Any) -> Problem:
    """A Slime sample as a :class:`Problem`: its prompt, label, index, and metadata."""
    return Problem(
        prompt=sample.prompt,
        label=getattr(sample, "label", None),
        id=getattr(sample, "index", None),
        metadata=MappingProxyType(dict(getattr(sample, "metadata", None) or {})),
    )


def to_sample(
    record: Mapping[str, Any], source: Any, tokenizer: Any, sample_type: Any
) -> Any:
    """One actor record as a Slime ``Sample``, response-aligned.

    The response starts at the first generated token. Later conditioning tokens -- tool
    results, say -- stay in the response with loss mask zero. ``reward`` carries the
    FOCUS advantage; every segment of one tree shares the source prompt's
    ``rollout_id``.
    """
    if record["metadata"][LANE_KEY] != "actor":
        raise ValueError("Only actor records belong in the policy training batch")
    mask, tokens, logprobs = record["loss_mask"], record["tokens"], record["logprobs"]
    if not (len(mask) == len(tokens) == len(logprobs)):
        raise ValueError("Misaligned actor record")
    if not any(mask):
        raise ValueError("Expected a trainable segment")
    start = list(mask).index(True)
    if start == 0:
        raise ValueError("An actor segment must follow conditioning context")
    sample = copy(source)
    sample.metadata = dict(source.metadata or {})
    sample.metadata["focus"] = record["metadata"]
    sample.tokens = list(tokens)
    sample.response_length = len(mask) - start
    sample.response = tokenizer.decode(sample.tokens[start:], skip_special_tokens=True)
    sample.loss_mask = list(map(int, mask[start:]))
    sample.rollout_log_probs = list(logprobs[start:])
    sample.reward = float(record["reward"])
    rollout_id = source.rollout_id if source.rollout_id is not None else source.index
    if rollout_id is None:
        raise ValueError(
            "Slime must assign an index or rollout_id to the source prompt"
        )
    sample.rollout_id = rollout_id
    sample.status = sample_type.Status.COMPLETED
    return sample


def make_generate(
    collector_for: CollectorFor,
    *,
    profile: str | None = None,
    critic_sink: CriticSink | None = None,
    workspace_for: Callable[[Problem], Workspace] | None = None,
    min_abs_advantage: float | None = None,
    allow_partial: bool = False,
) -> GenerateHook:
    """Slime's custom generate function: collect one prompt's tree with FOCUS.

    Returns the tree's actor segments as Slime ``Sample`` objects and sends its critic
    records to ``critic_sink``. Sampling comes from the agent and the search phases;
    Slime's ``sampling_params`` argument is accepted but does not override them.
    ``profile`` resolves the served model's format. Collector lifecycle is the
    factory's: each collector is used once, inside ``async with``.
    """

    async def generate(args: Any, sample: Any, sampling_params: Any = None) -> Any:
        from slime.utils.types import Sample

        del sampling_params
        policy = policy_for(args, profile=profile)
        problem = problem_of(sample)
        actor, critic = await collect_records(
            collector_for(policy, problem),
            problem,
            group_index=int(sample.index or 0),
            workspace_for=workspace_for,
            min_abs_advantage=min_abs_advantage,
            allow_partial=allow_partial,
        )
        if critic_sink is not None:
            critic_sink(critic)
        if not actor:
            raise ValueError(f"FOCUS collected no actor segments for {problem.id!r}")
        tokenizer = policy.format.codec.tokenizer
        return [to_sample(record, sample, tokenizer, Sample) for record in actor]

    return generate


def make_evaluate(
    collector_for: CollectorFor,
    *,
    profile: str | None = None,
    workspace_for: Callable[[Problem], Workspace] | None = None,
) -> GenerateHook:
    """Slime's evaluation generate function: one unbranched rollout, scored.

    Runs the collector's agent once from the prompt and returns a ``Sample`` whose
    reward is the outcome -- for a correctness reward, 0 or 1 -- rather than a signed
    advantage.
    """

    async def evaluate(args: Any, sample: Any, sampling_params: Any = None) -> Any:
        from slime.utils.types import Sample

        del sampling_params
        policy = policy_for(args, profile=profile)
        problem = problem_of(sample)
        collector = collector_for(policy, problem)
        workspace = None if workspace_for is None else workspace_for(problem)
        async with collector:
            endpoint = await collector.agent.run_from(
                problem.prompt, workspace=workspace
            )
        packed = regions(endpoint.turns)
        if not packed:
            raise ValueError(f"evaluation generated nothing for {problem.id!r}")
        last = packed[-1]
        start = last.trainable.index(True) if any(last.trainable) else len(last.tokens)
        result = copy(sample)
        result.tokens = list(last.tokens)
        result.response_length = len(last.tokens) - start
        result.response = policy.decode(last.tokens[start:], skip_special_tokens=True)
        result.loss_mask = list(map(int, last.trainable[start:]))
        logprobs = last.logprobs or (0.0,) * len(last.tokens)
        result.rollout_log_probs = list(logprobs[start:])
        result.reward = float(endpoint.reward_outcome)
        result.status = (
            Sample.Status.TRUNCATED if endpoint.truncated else Sample.Status.COMPLETED
        )
        return result

    return evaluate


def preserve_advantages(args: Any, samples: list[Any]) -> tuple[list[float], ...]:
    """Reward post-process hook: keep FOCUS advantages unchanged by group centering."""
    del args
    values = [float(sample.reward) for sample in samples]
    return values, values


def advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    """Advantage hook: broadcast each segment's FOCUS advantage over its tokens.

    Overrides the estimator Slime's argument parser selected. Advantage normalization
    must stay off; returns are zero because critic targets are exported separately.
    """
    import torch

    if getattr(args, "normalize_advantages", False):
        raise ValueError("Keep FOCUS advantages unnormalized (normalize_advantages)")
    probabilities = (
        rollout_data["rollout_log_probs"]
        if args.use_rollout_logprobs
        else rollout_data["log_probs"]
    )
    rollout_data["advantages"] = [
        torch.full_like(lp, float(weight))
        for lp, weight in zip(probabilities, rollout_data["rewards"], strict=True)
    ]
    rollout_data["returns"] = [torch.zeros_like(lp) for lp in probabilities]


def sum_policy_tokens(
    total_lengths: Any,
    response_lengths: Any,
    loss_masks: list[Any],
    calculate_per_token_loss: bool = False,
) -> Callable[[Any], Any]:
    """Policy-loss reducer hook: sum masked token losses instead of averaging them.

    Each generated token then contributes its segment's advantage once, whatever the
    batch's total length.
    """
    import torch

    del total_lengths, response_lengths
    if calculate_per_token_loss:
        raise ValueError("Use the rollout-based batch reduction with FOCUS advantages")
    mask = torch.cat(loss_masks)

    def reduce(loss: Any) -> Any:
        if loss.numel() != mask.numel():
            raise ValueError("Expected unsplit response tokens; use context parallel 1")
        return (loss * mask).sum()

    return reduce


__all__ = [
    "GenerateHook",
    "advantages",
    "make_evaluate",
    "make_generate",
    "policy_for",
    "preserve_advantages",
    "problem_of",
    "sum_policy_tokens",
    "to_sample",
]
