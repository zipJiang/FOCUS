"""Group-level training coefficients, independent of search allocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

from focus.harness import AnyRollout
from focus.registry import register, registrable
from focus.reward.config import RewardConfig
from focus.scheduler.core.tree import Node, SchedulerView

if TYPE_CHECKING:
    from focus.preparation.records import ActorSample


@registrable(slot="advantage_estimator")
class AdvantageEstimator(ABC):
    """Assign training coefficients to a finished group of edge records.

    Implementations hold configuration only and never mutate the tree. Write weight
    and optional diagnostics without adding, removing, or re-keying records.
    """

    name: str = ""
    #: Outcome-based recipes may require a completed path through each actor edge.
    requires_completed_paths: bool = False

    @abstractmethod
    def assign(
        self,
        records: Sequence[ActorSample],
        *,
        view: SchedulerView[AnyRollout],
        reward_config: RewardConfig,
    ) -> None: ...


def _resolved(
    records: Sequence[ActorSample], view: SchedulerView[AnyRollout]
) -> Iterator[tuple[ActorSample, Node[AnyRollout]]]:
    for record in records:
        if (node := view.node(record.node_id)) is not None:
            yield record, node


def register_estimator[E: AdvantageEstimator](
    name: str,
) -> Callable[[type[E]], type[E]]:
    def decorator(cls: type[E]) -> type[E]:
        cls.name = name
        return register(AdvantageEstimator, name, cls)

    return decorator
