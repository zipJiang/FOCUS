"""Prepare one training recipe from a settled tree without a live runtime."""

from __future__ import annotations

import logging
from types import MappingProxyType

from focus.allocation.evidence import evidence_of
from focus.harness import AnyRollout
from focus.harness.turn import logprob
from focus.preparation.base import Advantage, Edge, Estimator
from focus.preparation.direct_branch import DirectBranchTd
from focus.preparation.records import (
    ActorSample,
    Batch,
    CriticSample,
    _refuse_non_finite,
    _trainable_edges,
    edge_spans,
)
from focus.scheduler.core.execution import PASS_KEY
from focus.scheduler.core.returns import (
    edge_duration,
    edge_return,
    finished_through,
)
from focus.scheduler.core.tree import SchedulerState, SchedulerView
from focus.values import RewardConfig, TreeValues, tree_values

logger = logging.getLogger(__name__)


def critic_targets(values: TreeValues) -> list[CriticSample]:
    """Branch-mean targets :math:`\\bar G` at every expanded checkpoint that was scored.

    The target is held fixed during critic training. Trees searched without a value
    model carry no critic inputs and yield no targets; checkpoints whose backup depends
    on a failed critic score are skipped, so a partial collection still trains the
    critic on the rest.
    """
    result = []
    for node in values.tree.nodes():
        value = values[node.id]
        if (
            value.backup is None
            or value.refined is None
            or value.prior is None
            or node.value_context is None
        ):
            continue
        result.append(
            CriticSample(
                node_id=node.id,
                context=node.value_context,
                target=value.backup,
                diagnostics=MappingProxyType(
                    {
                        "prior": value.prior,
                        "direct_branches": float(value.branches),
                        "mean_return": value.backup,
                        "refined_value": value.refined,
                    }
                ),
            )
        )
    return result


def _actor_samples(
    view: SchedulerView[AnyRollout], values: TreeValues, estimator: Estimator
) -> list[ActorSample]:
    """One sample per trainable searched edge, weighted by ``estimator``."""
    reward = values.reward
    edges: list[Edge] = []
    spans_by_edge = []
    for node, parent in _trainable_edges(view):
        if estimator.requires_completed_paths and not finished_through(view, node):
            continue
        spans = edge_spans(node, parent)
        if not spans:
            continue
        if any(not span.exact for span in spans):
            raise ValueError(
                f"edge into node {node.id} is not trainable: inexact generated tokens"
            )
        logprob(node.payload.edge(parent.payload))  # behavior logprobs are required
        edges.append(
            Edge(
                node,
                parent,
                edge_return(node, parent, reward),
                edge_duration(node, parent),
            )
        )
        spans_by_edge.append(spans)
    weights = estimator.advantages(tuple(edges), values)
    if len(weights) != len(edges):
        raise ValueError(
            f"estimator {estimator.name!r} returned {len(weights)} advantages "
            + f"for {len(edges)} edges"
        )
    samples = []
    for edge, spans, weight in zip(edges, spans_by_edge, weights, strict=True):
        advantage = weight if isinstance(weight, Advantage) else Advantage(weight)
        samples.append(
            ActorSample(
                node_id=edge.node.id,
                spans=spans,
                group_id=edge.parent.id,
                estimator=estimator.name,
                weight=float(advantage.value),
                diagnostics=MappingProxyType(dict(advantage.diagnostics)),
                search_pass=edge.node.metadata.get(PASS_KEY),
            )
        )
    return samples


def prepare_samples(
    state: SchedulerState[AnyRollout],
    *,
    reward_config: RewardConfig,
    estimator: Estimator | None = DirectBranchTd(),  # noqa: B008 (stateless)
) -> Batch:
    """Compute one actor recipe and the critic's branch-mean targets.

    The default recipe is the paper's direct-branch TD. Critic targets do not depend
    on the actor recipe: every expanded checkpoint the critic scored regresses on the
    mean of its direct-branch backups. ``estimator=None`` prepares only those targets.
    """
    view = SchedulerView(state)
    evidence_of(view, reward_config)  # refuses trees searched under other rewards
    values = tree_values(view, reward_config)
    actor = _actor_samples(view, values, estimator) if estimator is not None else []
    critic = critic_targets(values)
    for record in actor:
        _refuse_non_finite(record)
    for target in critic:
        _refuse_non_finite(target)
    result = Batch(
        actor=tuple(actor),
        critic=tuple(critic),
        reward_config=reward_config,
        stats=MappingProxyType(dict(view.stats)),
        failures=view.failures,
    )
    logger.log(
        logging.INFO if actor or critic else logging.WARNING,
        "samples prepared actor=%d critic=%d estimator=%s stats=%s failures=%d",
        len(actor),
        len(critic),
        estimator.name if estimator is not None else None,
        dict(result.stats),
        len(result.failures),
    )
    return result
