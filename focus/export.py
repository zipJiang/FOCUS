"""Flatten prepared records into trainer samples and format trees for inspection."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, overload

from focus.harness import AnyRollout
from focus.harness.packing import PackedSequence
from focus.preparation.records import (
    ESTIMATOR_KEY,
    LANE_KEY,
    ActorSample,
    CriticSample,
    Lane,
    PreparedBatch,
)
from focus.scheduler.core.tree import SchedulerView

# -- flattening: from two record tuples to the one list a trainer consumes -----------


class SampleFactory[T](Protocol):
    """How a framework builds its own sample type out of one span.

    The five keywords named here are the actor contract. ``**extra`` carries
    the actor's own keys
    (``estimator``, ``group_id``, ``diagnostics``) and the edge-shape
    keys (``node_id``, ``edge_tokens``, ``span_index``, ``span_count``,
    ``group_edge_count``), which arrive by
    name on every call and may be ignored. A factory *must* take ``**extra``, because
    this module adds a key when the export learns to say something new and a factory
    that enumerated them would break on the next one.

    Generic in what it returns, so ``to_samples(result, i, sample_factory=MySample)``
    is a ``list[MySample]`` rather than a ``list[Any]`` a trainer then has to cast.

    ``span`` is the packed region itself, and ``logprobs`` is the behavior column
    already selected out of it (:attr:`PreparedBatch.behavior_version`).
    """

    def __call__(
        self,
        *,
        span: PackedSequence,
        logprobs: tuple[float, ...],
        reward: float,
        lane: Lane,
        group_index: int,
        **extra: Any,
    ) -> T: ...


class CriticSampleFactory[T](Protocol):
    def __call__(
        self, *, context: str, target: float, group_index: int, **extra: Any
    ) -> T: ...


def _critic_dict(
    *, context: str, target: float, group_index: int, **extra: Any
) -> dict[str, Any]:
    return {
        "context": context,
        "target": target,
        "group_index": group_index,
        "metadata": {LANE_KEY: "critic", **extra},
    }


def _record_samples[T](
    make: SampleFactory[T],
    record: ActorSample,
    *,
    version: str,
    reward: float,
    lane: Lane,
    group_index: int,
    **fields: Any,
) -> Iterator[T]:
    """One sample per span of ``record``, each saying where in the edge it sits.

    Every sample of an edge repeats that edge's identity and size, because the trainer
    sees a flat list: ``node_id`` says which edge, ``edge_tokens`` how big it is, and
    ``span_index`` / ``span_count`` where this piece of it falls.
    """
    # Both are properties of the *edge*, so they are read once rather than per span:
    # `edge_tokens` counts the mask over every span, which an edge split four ways
    # would otherwise recount four times.
    edge_tokens = record.edge_tokens
    span_count = len(record.spans)
    for index, span in enumerate(record.spans):
        values = span.logprobs.get(version, ())
        if any(span.trainable) and (
            len(values) != len(span.tokens)
            or len(span.trainable) != len(span.tokens)
            or any(
                keep and not math.isfinite(value)
                for value, keep in zip(values, span.trainable, strict=True)
            )
        ):
            raise ValueError("actor export requires aligned finite behavior logprobs")
        yield make(
            span=span,
            # Selected here rather than at packing: a pack carries every version it was
            # scored in, and which one is *behavior* is the run's fact, not the span's.
            logprobs=span.logprobs.get(version, ()),
            reward=reward,
            lane=lane,
            group_index=group_index,
            node_id=record.node_id,
            edge_tokens=edge_tokens,
            span_index=index,
            span_count=span_count,
            **fields,
        )


@overload
def to_samples(
    result: PreparedBatch,
    group_index: int,
    *,
    sample_factory: None = None,
    critic_sample_factory: None = None,
    min_abs_advantage: float | None = None,
) -> list[dict[str, Any]]: ...


@overload
def to_samples[T](
    result: PreparedBatch,
    group_index: int,
    *,
    sample_factory: SampleFactory[T] | None = None,
    critic_sample_factory: CriticSampleFactory[T] | None = None,
    min_abs_advantage: float | None = None,
) -> list[T]: ...


def to_samples(
    result: PreparedBatch,
    group_index: int,
    *,
    sample_factory: SampleFactory[Any] | None = None,
    critic_sample_factory: CriticSampleFactory[Any] | None = None,
    min_abs_advantage: float | None = None,
) -> list[Any]:
    """Export actor spans and checkpoint critic contexts, one group per prompt.

    ``reward`` is the actor weight its estimator wrote; whatever the
    estimator said about how it reached the weight rides in
    ``metadata["diagnostics"]``, where no trainer can mistake it for a coefficient.

    ``sample_factory`` builds the trainer's own sample type instead of a dict -- see
    :class:`SampleFactory` for the call contract, and :func:`_dict_sample` for what the
    built-in one does with it. The overloads say the obvious thing in types: no factory
    means ``list[dict[str, Any]]``, a factory means a list of whatever it returns.

    ``min_abs_advantage`` excludes whole actor edges whose absolute estimator weight
    is at most the cutoff. ``None`` disables filtering;
    zero excludes exactly zero weights. Edges containing malformed compaction
    attempts are retained to preserve their small shaping signal.
    Critic records and the prepared batch remain
    intact, including when every actor edge is excluded. The caller must support that
    empty actor group; this function does not retain a below-threshold fallback edge.

    Every actor sample carries ``group_edge_count``, the original number of actor
    edges in this prompt. A trainer averaging tokens within edges and edges within
    prompts must use that count, rather than recounting the surviving edges, to keep
    retained coefficients unchanged. Excluding samples also excludes any auxiliary
    actor losses, such as reference KL, computed on those samples.
    """
    if min_abs_advantage is not None and (
        not math.isfinite(min_abs_advantage) or min_abs_advantage < 0
    ):
        raise ValueError("min_abs_advantage must be finite and nonnegative")
    if min_abs_advantage is not None and any(
        not math.isfinite(actor.weight) for actor in result.actor
    ):
        raise ValueError("cannot filter non-finite actor advantages")
    actors = tuple(
        actor
        for actor in result.actor
        if min_abs_advantage is None
        or abs(actor.weight) > min_abs_advantage
        or any(span.retain_for_training for span in actor.spans)
    )
    if result.critic and sample_factory is not None and critic_sample_factory is None:
        raise ValueError("mixed custom exports require critic_sample_factory")
    if actors and critic_sample_factory is not None and sample_factory is None:
        raise ValueError("mixed custom exports require sample_factory")
    make: SampleFactory[Any] = sample_factory or _dict_sample
    make_critic: CriticSampleFactory[Any] = critic_sample_factory or _critic_dict
    version = result.behavior_version
    samples: list[Any] = []

    for actor in actors:
        reward = actor.weight
        samples.extend(
            _record_samples(
                make,
                actor,
                version=version,
                reward=reward,
                lane="actor",
                group_index=group_index,
                diagnostics=actor.diagnostics,
                estimator=actor.estimator,
                group_id=actor.group_id,
                group_edge_count=len(result.actor),
            )
        )

    samples.extend(
        make_critic(
            context=critic.context,
            target=critic.target,
            group_index=group_index,
            node_id=critic.node_id,
            value_version=critic.value_version,
            reward_config_id=critic.reward_config_id,
            diagnostics=dict(critic.diagnostics),
        )
        for critic in result.critic
    )
    return samples


def _dict_sample(
    *,
    span: PackedSequence,
    logprobs: tuple[float, ...],
    reward: float,
    lane: Lane,
    group_index: int,
    estimator: str | None = None,
    diagnostics: Mapping[str, float] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The framework-free sample shape: exactly what a trainer needs, named plainly."""
    # `tag` rides in the metadata rather than as a column, and so do the edge-shape keys
    # (`edge_tokens`, `span_index`, `span_count`) that arrive in `extra`: a trainer that
    # does not distinguish task work from fold work, or that already normalises per
    # token, simply never reads them.
    metadata: dict[str, Any] = {
        LANE_KEY: lane,
        "tag": span.tag,
        "invocation_id": span.invocation_id,
        "policy_id": span.policy_id,
        "format_id": span.format_id,
        **extra,
    }
    if estimator is not None:
        metadata[ESTIMATOR_KEY] = estimator
    if diagnostics is not None:
        # Copied out of the record's read-only proxy into a plain dict: this is the
        # object a trainer serialises, and `json.dumps` has no idea what a
        # `mappingproxy` is. Nested under one key rather than spread across the
        # metadata, so a recipe naming a diagnostic `tag` or `lane` cannot overwrite
        # what the export said.
        metadata["diagnostics"] = dict(diagnostics)
    return {
        "tokens": list(span.tokens),
        # `map`, not a comprehension: the coercion is the same one, run in C rather
        # than as a bytecode loop over every token of every span -- and this is the one
        # place in the library that touches all exported tokens one by one.
        "loss_mask": list(map(bool, span.trainable)),
        "logprobs": list(logprobs),
        "reward": reward,
        "group_index": group_index,
        "metadata": metadata,
    }


# -- reading a run: the two human-facing renderings -----------------------------------


def format_tree(
    view: SchedulerView[AnyRollout], result: PreparedBatch | None = None
) -> str:
    """One line per node of the search tree, indented by depth.

    The other half of :func:`format_edges`, and the half that shows what the *search*
    did: which nodes it created, which of them are still forkable, which ended the
    episode, and what the value head thought of each. An edge report can only speak for
    the finished edges that became records, so a tree whose expansions all died prints
    nothing there and its whole shape here.

    Each line is ``node <id>  parent=<id>  <flags>  reward=<payload reward>`` followed
    by the node's critic estimates (``critic[<version>=<estimate>]``, one entry per
    reward-model channel that scored it) if it has any. ``flags`` is two characters,
    ``f`` for forkable and ``d`` for done, a dash where the node is neither.

    Pass ``result`` to join each estimator's actor weight onto the node it was written
    for. A node with no weights is one no estimator scored -- the root, an interior node
    no finished path runs through, an edge that packed to no tokens -- and it stays on
    the report, because *which* nodes went unscored is the question this answers.

    Children print under their parent in the order they were attached, so the reading
    order is the tree's own rather than the id order :func:`format_edges` uses.
    """
    weights: dict[int, list[str]] = {}
    if result is not None:
        for record in result.actor:
            weights.setdefault(record.node_id, []).append(
                f"{record.estimator}={record.weight:+.4f}"
            )
    lines = []
    stack = [view.root]
    while stack:
        node = stack.pop()
        # reversed, because the stack pops last-in first: children print in the order
        # they were attached, which is the order the passes created them in
        stack.extend(reversed(view.children(node)))
        flags = ("f" if node.payload.forkable else "-") + (
            "d" if node.payload.done else "-"
        )
        parent = "-" if node.parent_id is None else str(node.parent_id)
        critic = " ".join(
            f"{version}={value:+.3f}" for version, value in sorted(node.critic.items())
        )
        lines.append(
            (
                (
                    f"{'  ' * node.depth}node {node.id:>3}  parent={parent:>3}  "
                    + f"{flags}  "
                    + f"reward={node.payload.reward:>6.3f}"
                )
                + (f"  critic[{critic}]" if critic else "")
                + ("  " + "  ".join(weights.get(node.id, ())))
            ).rstrip()
        )
    return "\n".join(lines)


def _tag_of(spans: Sequence[PackedSequence]) -> str:
    """What to print in an edge line's ``tag=`` column, for any span tuple.

    A pack never mixes tags, but an *edge* is several packs and may: a fold and the task
    turns after it land in the same edge whenever the template's boundaries fall that
    way. Reading the first pack's tag and calling it the edge's would print ``task`` for
    an edge that is half fold, so a mixed edge names every tag it carries
    (``task+fold``), in the order the packs run.

    An edge with no spans at all prints ``-``. :func:`build_result` never makes one, but
    a hand-built record in a test or a notebook can, and a reporter that raises on the
    record someone is trying to look at is the reporter failing at its only job.
    """
    tags: list[str] = []
    for span in spans:
        if span.tag not in tags:
            tags.append(span.tag)
    return "+".join(tags) if tags else "-"


def format_edges(result: PreparedBatch) -> str:
    """Show actor edges with diagnostics, followed by checkpoint value targets."""
    scored: dict[int, list[str]] = {}
    for record in result.actor:
        shown = " ".join(
            f"{name}={value:+.3f}"
            for name, value in record.diagnostics.items()
            if name != "advantage" or value != record.weight
        )
        scored.setdefault(record.node_id, []).append(
            f"{record.estimator}={record.weight:+.4f}"
            + (f" [{shown}]" if shown else "")
        )
    lines = []
    for edge in sorted(result.actor, key=lambda r: r.node_id):
        stamp = "-" if edge.search_pass is None else f"pass-{edge.search_pass}"
        lines.append(
            f"edge {edge.node_id:>3}  {stamp:<9} tag={_tag_of(edge.spans):<5} "
            f"tokens={edge.edge_tokens:>4}  " + "  ".join(scored.get(edge.node_id, ()))
        )
    lines.extend(
        f"value {checkpoint.node_id:>3}  target={checkpoint.target:>6.3f}  "
        f"version={checkpoint.value_version}"
        for checkpoint in sorted(result.critic, key=lambda r: r.node_id)
    )
    return "\n".join(lines)


__all__ = [
    "ESTIMATOR_KEY",
    "LANE_KEY",
    "ActorSample",
    "CriticSample",
    "CriticSampleFactory",
    "Lane",
    "PreparedBatch",
    "SampleFactory",
    "format_edges",
    "format_tree",
    "to_samples",
]
