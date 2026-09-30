"""VinePPO collection: retained root paths and disposable Monte Carlo probes."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from focus.allocation import RolloutLedger
from focus.harness import AnyRollout
from focus.scheduler.core.execution import EdgeProvenance, stamp_provenance
from focus.scheduler.core.gating import ConcurrencyGating
from focus.scheduler.core.tree import Node, SchedulerState, SchedulerView

if TYPE_CHECKING:
    from focus.loop import Runtime

VINE_VALUE = "vine_outcome_value"
VINE_VALUE_SAMPLES = "vine_outcome_value_samples"
VINE_READY = "vine_complete"


async def _bounded_map[T, R](
    items: Sequence[T], operation: Callable[[T], Awaitable[R]], concurrency: int
) -> list[R]:
    # TaskGroup cancels siblings on failure; no detached sampling after an error.
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: T) -> R:
        async with semaphore:
            return await operation(item)

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(item)) for item in items]
    return [task.result() for task in tasks]


async def run_vine_search(
    root: AnyRollout, runtime: Runtime
) -> SchedulerState[AnyRollout]:
    """Collect actor paths and their independent probes as one pipeline.

    A probe starts as soon as its state exists: the shared root probe at once, and
    each intermediate probe when the actor yields that checkpoint (an independent
    snapshot), rather than after every actor path has finished. The probe set, each
    probe's horizon, and the estimate are unchanged; only waiting is removed.

    The root probe is shared across root trajectories, independent of every actor
    action. Intermediate estimates are shared by the incoming and outgoing edge.
    Only scalar outcomes and compute counters survive the probe phase.
    """
    config = runtime.config
    vine = config.vine
    assert vine is not None
    if not root.forkable:
        raise ValueError("VinePPO requires a live forkable root")
    state: SchedulerState[AnyRollout] = SchedulerState.root(
        root, gating=ConcurrencyGating(1)
    )
    RolloutLedger(config.reward_config).bind(SchedulerView(state))
    state.nodes[0].metadata[VINE_READY] = False
    for kind in ("actor", "value"):
        for metric in ("rollouts", "turns", "generated_tokens", "prompt_tokens"):
            state.stats[f"vine_{kind}_{metric}"] = 0.0

    def account(before: AnyRollout, after: AnyRollout, kind: str) -> None:
        turns = after.edge(before)
        cost = {
            "rollouts": 1,
            "turns": sum(t.transition is not None for t in turns),
            "generated_tokens": sum(len(t.tokens) for t in turns),
            "prompt_tokens": sum(len(t.prefix) for t in turns),
        }
        for metric, value in cost.items():
            state.stats[f"vine_{kind}_{metric}"] += value
        state.stats["rollouts"] += 1
        state.stats["turns"] += cost["turns"]

    def outcome(leaf: AnyRollout) -> float:
        # Do not silently treat censored continuations or backend failures as zero.
        if not leaf.done:
            raise ValueError(
                "VinePPO requires completed outcomes, not truncated rollouts"
            )
        value = leaf.reward_outcome
        if not math.isfinite(value):
            raise ValueError("VinePPO outcome must be finite")
        return value

    # Actors and probes are bounded separately, as the two former phases were, so
    # probes never delay the actor paths that create further probe states.
    actor_slots = asyncio.Semaphore(config.max_concurrency)
    probe_slots = asyncio.Semaphore(config.max_concurrency)

    async def probe(checkpoint: AnyRollout, max_steps: int) -> float:
        async with probe_slots:
            _, params = config.proposal.draw()
            leaf = await runtime.runner.run(
                checkpoint.fork(), sampling_params=params, max_steps=max_steps
            )
        value = outcome(leaf)
        account(checkpoint, leaf, "value")
        return value  # discard the entire auxiliary trajectory here

    Probes = list[asyncio.Task[float]]

    # TaskGroup cancels siblings on failure; no detached sampling after an error.
    async with asyncio.TaskGroup() as group:

        def probes(checkpoint: AnyRollout, max_steps: int) -> Probes:
            return [
                group.create_task(probe(checkpoint, max_steps))
                for _ in range(vine.value_rollouts_per_state)
            ]

        root_probes = probes(root, runtime.runner.max_steps)

        async def actor() -> list[tuple[AnyRollout, int, Probes]]:
            async with actor_slots:
                _, params = config.proposal.draw()
                chain: list[tuple[AnyRollout, int, Probes]] = []
                previous, remaining = root, runtime.runner.max_steps
                async for rs in runtime.runner.iter_run(
                    root.fork(), sampling_params=params
                ):
                    # iter_run yields each fold separately. Each task transition and
                    # each fold-only advance consumes one unit of its local horizon. A
                    # probe must not receive a fresh horizon merely because it starts
                    # mid-path.
                    consumed = sum(t.transition is not None for t in rs.edge(previous))
                    if not rs.done:
                        consumed += 1
                    remaining = max(0, remaining - consumed) if remaining >= 0 else -1
                    chain.append(
                        (rs, remaining, [] if rs.done else probes(rs, remaining))
                    )
                    previous = rs
            if not chain or len(chain[-1][0].turns) <= len(root.turns):
                raise ValueError("VinePPO actor rollout made no progress")
            outcome(chain[-1][0])
            account(root, chain[-1][0], "actor")
            return chain

        actors = [group.create_task(actor()) for _ in range(vine.group_size)]

    def estimate(node: Node[AnyRollout], tasks: Probes) -> None:
        samples = [task.result() for task in tasks]
        if len(samples) != vine.value_rollouts_per_state:
            raise RuntimeError("VinePPO value probe cohort is incomplete")
        node.metadata[VINE_VALUE] = sum(samples) / len(samples)
        node.metadata[VINE_VALUE_SAMPLES] = len(samples)

    stamp = EdgeProvenance(
        proposal_id=config.proposal.proposal_id,
        continuation_id=config.continuation_id,
        behavior_version=runtime.runner.policy.version,
        reward_config_id=config.reward_config.config_id,
        pass_id="vine-actor",
    )
    estimate(state.nodes[state.root_id], root_probes)
    for task in actors:
        chain = task.result()
        nodes = state.attach_chain(state.root_id, [rs for rs, _, _ in chain])
        for node, (_, _, tasks) in zip(nodes, chain, strict=True):
            stamp_provenance(node, stamp)
            if node.payload.done:
                node.metadata[VINE_VALUE] = outcome(node.payload)
                node.metadata[VINE_VALUE_SAMPLES] = 0
            else:
                estimate(node, tasks)
    state.stats["nodes"] = float(len(state.nodes) - 1)
    state.stats["vine_trainable_edges"] = float(len(state.nodes) - 1)
    state.nodes[0].metadata[VINE_READY] = True
    return state
