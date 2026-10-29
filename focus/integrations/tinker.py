"""Train a Tinker actor on FOCUS collections, with one-step asynchronous overlap.

Your loop owns training; this module supplies its pieces. :func:`to_datum` turns an
actor record into Tinker's next-token training data, :func:`update_actor` takes one
PPO step, and :func:`train` runs the loop that overlaps the next collection with the
current update. Collection is the user's
:data:`~focus.integrations.common.CollectorFor`, called with a
:class:`~focus.backends.TinkerPolicy` bound to one frozen sampler.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from focus.generation.tinker import TinkerPolicy
from focus.harness.workspace import Workspace
from focus.integrations.common import (
    CollectorFor,
    CriticSink,
    Problem,
    collect_records,
)
from focus.preparation.records import LANE_KEY

logger = logging.getLogger(__name__)

#: PPO with the paper's ratio clip, :math:`1 \pm 0.2`.
PPO_CLIP: Mapping[str, float] = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}


def _next_token_inputs(
    record: Mapping[str, Any],
) -> tuple[list[int], dict[str, list[Any]]]:
    """Shift an actor record into Tinker's model input and per-token loss inputs.

    The input is ``tokens[:-1]``; targets, behavior log probabilities, and advantages
    are read one position later. Conditioning tokens get advantage zero.
    """
    if record["metadata"][LANE_KEY] != "actor":
        raise ValueError("Tinker actor data must come from the actor lane")
    tokens, mask, logprobs = (record[k] for k in ("tokens", "loss_mask", "logprobs"))
    if not (len(tokens) == len(mask) == len(logprobs)) or len(tokens) < 2:
        raise ValueError(
            "Expected aligned tokens, masks, and behavior log probabilities"
        )
    if mask[0]:
        raise ValueError("The first token must be conditioning context")
    return list(tokens[:-1]), {
        "target_tokens": list(tokens[1:]),
        "logprobs": list(logprobs[1:]),
        "advantages": [record["reward"] if keep else 0.0 for keep in mask[1:]],
    }


def to_datum(record: Mapping[str, Any]) -> Any:
    """One actor record as a ``tinker.types.Datum``."""
    from tinker import types

    inputs, loss_inputs = _next_token_inputs(record)
    # Tinker converts plain lists to its tensor type on validation.
    loss_fn_inputs: Any = loss_inputs
    return types.Datum(
        model_input=types.ModelInput.from_ints(inputs), loss_fn_inputs=loss_fn_inputs
    )


async def update_actor(
    training_client: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    learning_rate: float,
    loss_fn: str = "ppo",
    loss_fn_config: Mapping[str, float] = PPO_CLIP,
) -> Any:
    """Take one optimizer step on ``records``' actor lane; return the backward result.

    The optimizer step is submitted only after the gradients are ready, and awaited
    before returning, so a sampler published afterwards holds the updated weights.
    """
    import tinker

    data = [to_datum(r) for r in records if r["metadata"][LANE_KEY] == "actor"]
    if not data:
        raise ValueError("No actor segments to train on")
    backward = await training_client.forward_backward_async(
        data, loss_fn=loss_fn, loss_fn_config=dict(loss_fn_config)
    )
    result = await backward.result_async()
    step = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=learning_rate)
    )
    await step.result_async()
    return result


@dataclass(frozen=True)
class _CollectedBatch:
    """One problem's records, collected under the sampler of ``behavior_step``."""

    index: int
    behavior_step: int
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class StepReport:
    """What one actor update trained on, and how far behind the learner it was."""

    batch: int
    behavior_step: int
    learner_step: int
    policy_lag: int
    metrics: Mapping[str, Any] = field(default_factory=dict)


async def _collect_batch(
    sampler: Any,
    problem: Problem,
    index: int,
    behavior_step: int,
    *,
    model: str,
    collector_for: CollectorFor,
    profile: str | None,
    workspace_for: Callable[[Problem], Workspace] | None,
    min_abs_advantage: float | None,
    allow_partial: bool,
) -> _CollectedBatch:
    """Collect ``problem`` with one immutable sampler for the whole tree."""
    # The constructor reads the sampler's tokenizer, a blocking SDK call.
    policy = await asyncio.to_thread(
        TinkerPolicy, sampler, model=model, profile=profile
    )
    actor, critic = await collect_records(
        collector_for(policy, problem),
        problem,
        group_index=index,
        workspace_for=workspace_for,
        min_abs_advantage=min_abs_advantage,
        allow_partial=allow_partial,
    )
    records = [*actor, *critic]
    for record in records:
        record["metadata"]["training"] = {
            "batch_index": index,
            "problem_id": problem.id if problem.id is not None else index,
            "behavior_step": behavior_step,
        }
    return _CollectedBatch(index, behavior_step, records)


async def _publish_sampler(training_client: Any) -> Any:
    # The SDK's synchronous snapshot call waits remotely; keep the event loop free.
    return await asyncio.to_thread(training_client.save_weights_and_get_sampling_client)


async def train(
    training_client: Any,
    *,
    model: str,
    problems: Sequence[Problem],
    collector_for: CollectorFor,
    steps: int,
    learning_rate: float,
    critic_sink: CriticSink | None = None,
    async_depth: Literal[0, 1] = 1,
    profile: str | None = None,
    loss_fn: str = "ppo",
    loss_fn_config: Mapping[str, float] = PPO_CLIP,
    workspace_for: Callable[[Problem], Workspace] | None = None,
    min_abs_advantage: float | None = None,
    allow_partial: bool = False,
    on_step: Callable[[StepReport], None] | None = None,
) -> list[StepReport]:
    """Take ``steps`` actor updates, one per collected problem, cycling ``problems``.

    With ``async_depth=1`` the next problem is collected from the current sampler while
    the learner updates, so each batch is at most one update behind the learner and
    every batch trains exactly once; ``0`` publishes a fresh sampler before each
    collection. Critic records go to ``critic_sink`` with their learner step.
    ``profile`` resolves the sampler's format, as for
    :class:`~focus.backends.TinkerPolicy`. An outstanding collection is cancelled and
    joined when the loop stops early.
    """
    if steps < 1 or async_depth not in (0, 1) or not problems:
        raise ValueError(
            "Use positive steps, a nonempty problem list, and async depth 0 or 1"
        )

    def collect(sampler: Any, index: int, behavior_step: int) -> asyncio.Task[Any]:
        problem = problems[index % len(problems)]
        return asyncio.create_task(
            _collect_batch(
                sampler,
                problem,
                index,
                behavior_step,
                model=model,
                collector_for=collector_for,
                profile=profile,
                workspace_for=workspace_for,
                min_abs_advantage=min_abs_advantage,
                allow_partial=allow_partial,
            )
        )

    reports: list[StepReport] = []
    sampler = await _publish_sampler(training_client)
    pending: asyncio.Task[Any] | None = collect(sampler, 0, 0)
    try:
        for step in range(steps):
            assert pending is not None
            batch = await pending
            pending = None
            lag = step - batch.behavior_step
            if batch.index != step or not 0 <= lag <= async_depth:
                raise ValueError(
                    f"Unexpected batch or policy lag: batch={batch.index}, lag={lag}"
                )
            has_next = step + 1 < steps
            if async_depth == 1 and has_next:
                # Capture the current sampler before the learner takes its next step.
                pending = collect(sampler, step + 1, step)
            result = await update_actor(
                training_client,
                batch.records,
                learning_rate=learning_rate,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            critic = [r for r in batch.records if r["metadata"][LANE_KEY] == "critic"]
            for record in critic:
                record["metadata"]["training"]["learner_step"] = step
            if critic_sink is not None:
                critic_sink(critic)
            report = StepReport(
                batch=step,
                behavior_step=batch.behavior_step,
                learner_step=step,
                policy_lag=lag,
                metrics=dict(getattr(result, "metrics", {}) or {}),
            )
            reports.append(report)
            logger.info("tinker step %s", report)
            if on_step is not None:
                on_step(report)
            if has_next and (async_depth == 0 or step + 2 < steps):
                sampler = await _publish_sampler(training_client)
                if async_depth == 0:
                    pending = collect(sampler, step + 1, step + 1)
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    return reports


__all__ = ["PPO_CLIP", "StepReport", "to_datum", "train", "update_actor"]
