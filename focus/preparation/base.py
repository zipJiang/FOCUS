"""Actor recipes: training coefficients for searched edges, from tree values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from focus.harness import AnyRollout
from focus.scheduler.core.tree import Node
from focus.values import TreeValues


@dataclass(frozen=True)
class Edge:
    """One searched continuation: from ``parent`` to ``node``.

    ``reward`` is the edge return :math:`R` under the batch's reward configuration and
    ``duration`` the number of task transitions :math:`d` it took. Compaction turns
    shape the reward but take no discount step.
    """

    node: Node[AnyRollout]
    parent: Node[AnyRollout]
    reward: float
    duration: int


@dataclass(frozen=True)
class Advantage:
    """An actor coefficient with optional diagnostics that never enter the loss."""

    value: float
    diagnostics: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )


class Estimator(ABC):
    """Assign an actor coefficient to every edge of a settled tree.

    Implement :meth:`advantages` as a pure function of the edges and the tree's
    values. Return one float or :class:`Advantage` per edge, in order.
    """

    #: Recorded on every actor sample. Defaults to the class name.
    name: str = ""
    #: Outcome-based recipes may require a completed path through each actor edge.
    requires_completed_paths: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("name"):
            cls.name = cls.__name__

    @abstractmethod
    def advantages(
        self, edges: Sequence[Edge], values: TreeValues
    ) -> Sequence[float | Advantage]: ...


__all__ = ["Advantage", "Edge", "Estimator"]
