"""What every trainer integration asks of the user, and the collection they share.

A trainer integration owns the policy -- it knows which weights are sampling -- and
asks the user one question per training item: given this policy and this
:class:`Problem`, which :class:`~focus.collector.Collector` should collect it? That
callable, :data:`CollectorFor`, is the whole task-specific supply; the Tinker and Slime
integrations take the same one. Critic records leave through a :data:`CriticSink`,
because neither trainer trains the critic.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from focus.collector import Collector
from focus.generation.policy import Policy
from focus.harness.workspace import Workspace
from focus.preparation.records import LANE_KEY


@dataclass(frozen=True)
class Problem:
    """One training item: the prompt the policy sees and what grading needs.

    ``label`` stays out of the prompt -- typically a reference answer the environment
    grades against. ``id`` names the item in exported metadata.
    """

    prompt: Any
    label: Any = None
    id: Any = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


#: Build the collector for one problem under the trainer's current policy.
type CollectorFor = Callable[[Policy[Any], Problem], Collector]
#: Receive one collection's critic records; the critic is trained elsewhere.
type CriticSink = Callable[[Sequence[dict[str, Any]]], None]


class JsonlDirectory:
    """A critic sink writing each collection to its own JSONL file in ``directory``.

    One file per call, so concurrent collectors and worker processes never interleave
    lines. A collection without critic records writes nothing.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def __call__(self, records: Sequence[dict[str, Any]]) -> None:
        if not records:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{uuid4().hex}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))


async def collect_records(
    collector: Collector,
    problem: Problem,
    *,
    group_index: int,
    workspace_for: Callable[[Problem], Workspace] | None = None,
    min_abs_advantage: float | None = None,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect ``problem`` and return its ``(actor, critic)`` records.

    The collector's own ``async with`` lifecycle applies. A collection with failed
    expansions or critic predictions raises unless ``allow_partial`` keeps its
    successful records.
    """
    workspace = None if workspace_for is None else workspace_for(problem)
    async with collector:
        batch = await collector.collect(problem.prompt, workspace=workspace)
    failed = batch.stats.get("failures", 0) or batch.stats.get("score_failures", 0)
    if failed and not allow_partial:
        raise RuntimeError(
            f"collection failed for problem {problem.id!r}: {dict(batch.stats)}"
        )
    records = batch.to_records(group_index, min_abs_advantage=min_abs_advantage)
    actor = [r for r in records if r["metadata"][LANE_KEY] == "actor"]
    critic = [r for r in records if r["metadata"][LANE_KEY] == "critic"]
    return actor, critic


__all__ = [
    "CollectorFor",
    "CriticSink",
    "JsonlDirectory",
    "Problem",
    "collect_records",
]
