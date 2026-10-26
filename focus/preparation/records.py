"""Prepared edge records, token regions, and validation helpers."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from focus.harness import AnyRollout
from focus.harness.packing import PackedSequence, regions
from focus.harness.turn import Turn
from focus.scheduler.core.tree import Node, SchedulerView
from focus.values import RewardConfig

#: Which head a sample trains. Two, and only two: the actor's scalar is a weight to
#: push a generation up or down, the critic's is a return to regress on, and a third
#: name here would be a head nothing in this module knows how to fill.
type Lane = Literal["actor", "critic"]

#: Metadata key naming which head a sample trains: ``"actor"`` or ``"critic"``.
LANE_KEY = "lane"

#: Metadata key naming the estimator that computed an actor sample's scalar.
ESTIMATOR_KEY = "estimator"


#: What an estimator that explains nothing publishes. A proxy rather than a fresh
#: ``{}``: the default is shared by every record whose recipe wrote no diagnostics, and
#: a writable one would let a single record's edit rewrite all of them.
_NO_DIAGNOSTICS: Mapping[str, float] = MappingProxyType({})


def trainable_tokens(spans: Sequence[PackedSequence]) -> int:
    """Tokens the actor takes a gradient on across ``spans`` -- the size of the edge."""
    return sum(sum(span.trainable) for span in spans)


@dataclass
class ActorSample:
    """One edge the actor may train on, under the weight its estimator wrote.

    One scalar reaches the loss, and the estimator's
    :meth:`~focus.preparation.base.Estimator.advantages` supplies it. Export
    composes nothing out of parts -- a confidence times an advantage would make every
    recipe declare a factor it may have no opinion about and put an arithmetic step
    between what an estimator
    meant and what the actor trained on. A recipe that wants to *show* its parts puts
    them in :attr:`diagnostics`, which nothing reads back into the weight.
    """

    node_id: int
    spans: tuple[PackedSequence, ...]
    #: The node the edge leaves -- the fork this record belongs to for sibling grouping.
    group_id: int
    #: The name of the estimator that weighted this record.
    estimator: str
    #: The actor weight :math:`w` from :meth:`Estimator.advantages`, used as-is.
    weight: float = 0.0
    #: Whatever the estimator wants said about how it reached :attr:`weight` -- e.g.
    #: ``{"advantage": A, "baseline": V}``. Metadata, not arithmetic: it is carried into
    #: the trainer's ``metadata["diagnostics"]`` and never multiplied into anything.
    #: Frozen at export, so a record cannot be rewritten through the dict its estimator
    #: kept.
    diagnostics: Mapping[str, float] = _NO_DIAGNOSTICS
    #: Index of the search pass that produced this edge, when one did.
    search_pass: int | None = None

    @property
    def edge_tokens(self) -> int:
        """Trainable tokens over the whole edge -- what one unit of credit is worth.

        A trainer normalising loss per *sample* needs this to normalise per edge
        instead, because the span count is a property of the template, not of the edge.
        """
        return trainable_tokens(self.spans)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle with a plain dict where the live record holds a read-only proxy.

        The same trade :class:`Batch` makes for ``stats``, and for the same
        reason: a ``MappingProxyType`` has no pickle support at all, so freezing
        :attr:`diagnostics` at export would otherwise make every actor record -- and
        the result holding it -- unshippable.
        """
        return {**self.__dict__, "diagnostics": dict(self.diagnostics)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.diagnostics = MappingProxyType(state["diagnostics"])


@dataclass(frozen=True)
class CriticSample:
    """One checkpoint input and its recipe-defined remaining-return target."""

    node_id: int
    context: str
    target: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "diagnostics": dict(self.diagnostics)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        object.__setattr__(self, "diagnostics", MappingProxyType(state["diagnostics"]))


#: The stats of a result built without a search behind it (a hand-made result, a test).
#: A proxy rather than ``{}`` so the shared default cannot be written through.
_NO_STATS: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True)
class Batch:
    """Everything one prompt produced for training, and what producing it cost.

    The records are the deliverable; :attr:`stats` and :attr:`failures` are the search's
    own account of the run, copied off the scheduler as the result is built. They ride
    here because a trainer logging throughput or alerting on a dead endpoint needs
    exactly those numbers and nothing else from the search -- and the alternative,
    keeping the tree alive per prompt to read them off later, holds every rollout's
    tokens in memory for a metric.
    """

    actor: tuple[ActorSample, ...] = ()
    critic: tuple[CriticSample, ...] = ()
    reward_config: RewardConfig | None = None
    #: The scheduler's running counters at export -- ``rollouts``, ``nodes``, ``turns``,
    #: ``failures``, ``score_failures`` (see :mod:`...scheduler.core.tree`). A frozen
    #: copy: the tree may keep growing, this is what the export saw.
    stats: Mapping[str, float] = field(default_factory=lambda: _NO_STATS)
    #: Every expansion that died, as ``(node_id, repr(exc))``. The count is in
    #: ``stats["failures"]``; these are the reasons.
    failures: tuple[tuple[int, str], ...] = ()

    def __len__(self) -> int:
        return len(self.actor) + len(self.critic)

    def to_records(
        self,
        group_index: int,
        *,
        actor_factory: Any = None,
        critic_factory: Any = None,
        min_abs_advantage: float | None = None,
    ) -> list[Any]:
        """Flatten into trainer records, one per actor span and one per critic target.

        Without factories each record is a plain dict; see
        :func:`~focus.export.to_samples` for the record schema and the factory
        contract.
        """
        from focus.export import to_samples

        return to_samples(
            self,
            group_index,
            sample_factory=actor_factory,
            critic_sample_factory=critic_factory,
            min_abs_advantage=min_abs_advantage,
        )

    def describe(self) -> str:
        """Actor edges with their diagnostics, then critic targets, one per line."""
        from focus.export import format_edges

        return format_edges(self)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle with a plain dict where the live record holds a read-only proxy.

        ``MappingProxyType`` cannot be pickled. Store its contents as a dict so
        callers can save a batch or pass it to a trainer in another process.
        Restore the read-only proxy when loading the record.
        """
        return {**self.__dict__, "stats": dict(self.stats)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # frozen, so this is the only way to land the proxy back on the field
        object.__setattr__(self, "stats", MappingProxyType(state["stats"]))


def spans_of(turns: Sequence[Turn[object]], /) -> tuple[PackedSequence, ...]:
    """The regions of a run of turns that hold any tokens.

    A :class:`~focus.harness.packing.PackedSequence` already *is* the
    trainer's
    view of a region -- ids, loss mask, behavior logprobs -- so a record carries
    the packs themselves rather than a projection of them.

    Token-less packs are dropped: a fold that generated nothing leaves a marker turn
    that packs to no tokens, and it is a fact about the log, not a training sample.
    """
    return tuple(seq for seq in regions(turns) if seq.tokens and seq.actor)


def edge_spans(
    node: Node[AnyRollout], parent: Node[AnyRollout]
) -> tuple[PackedSequence, ...]:
    """Every token-bearing region on the edge into ``node``.

    The edge is a slice of the turn log, and its regions are derived from that slice --
    so a fork's shared history is excluded structurally rather than by comparing what
    the parent already had.
    """
    return spans_of(node.payload.edge(parent.payload))


def _trainable_edges(
    view: SchedulerView[AnyRollout],
) -> Iterator[tuple[Node[AnyRollout], Node[AnyRollout]]]:
    """Searched edges ending at live checkpoints or terminals."""
    for node in view.nodes():
        parent = None if node.parent_id is None else view.node(node.parent_id)
        if parent is None or not (node.payload.done or node.payload.forkable):
            continue
        yield node, parent


def _refuse_non_finite(record: ActorSample | CriticSample) -> None:
    """Raise unless every scalar on ``record`` is a real number.

    Refused here, not sanitised downstream. A ``nan`` advantage is not a coarse number
    to be rounded away -- it is an estimator's arithmetic having come apart, and a zero
    written in its place trains on the edge as though the recipe had scored it neutral.
    An infinity is worse: it survives :func:`json.dumps`, which emits a bare ``NaN`` /
    ``Infinity`` that no strict JSON reader accepts, and reaches the optimizer as a step
    of unbounded size. Either way the cheapest place to hear about it is beside the
    estimator that produced it, not in a loss that went ``nan`` three steps later.
    """
    if isinstance(record, CriticSample):
        scalars = [("target", record.target, "the direct-branch backup")]
        scalars.extend(
            (name, value, "critic diagnostic")
            for name, value in record.diagnostics.items()
        )
    else:
        estimator = f"estimator {record.estimator!r}"
        scalars = [
            ("weight", record.weight, estimator),
            # Diagnostics are checked too, though nothing trains on them: they are the
            # numbers a run is debugged by and they go out over the same JSON encoder,
            # which writes a bare `NaN` no strict reader accepts. A `nan` here also
            # says the estimator's arithmetic came apart, whatever the weight ended up
            # being.
            *(
                (f"diagnostic {name!r}", value, estimator)
                for name, value in record.diagnostics.items()
            ),
        ]
    for name, value, source in scalars:
        if not math.isfinite(value):
            raise ValueError(
                f"{source} produced a non-finite {name} ({value}) on the edge into "
                + f"node {record.node_id}. Export will not write a zero in its place: "
                + "check the env's rewards and the critic's predictions."
            )
