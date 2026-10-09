"""Branch-local indexed experience archives backed by immutable history references."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from focus.harness.rollout import AnyRollout
from focus.protocols.base import ContextProtocol, ProtocolError, ToolProtocol, schema
from focus.registry import register

if TYPE_CHECKING:
    from focus.harness.runner import Runner


@dataclass(frozen=True)
class Archive:
    index: int
    references: tuple[int, ...]
    summary: str


@register(ContextProtocol, "memex")
class Memex(ToolProtocol):
    def __init__(self, operation_cost: float = 0, max_archives: int = 1024):
        if not math.isfinite(operation_cost) or operation_cost < 0 or max_archives < 1:
            raise ProtocolError(
                "nonnegative finite cost and positive archive limit required"
            )
        self.operation_cost = operation_cost
        self.max_archives = max_archives

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return (
            schema(
                "memory_status",
                "List live history indices and archive summaries.",
                {},
                [],
            ),
            schema(
                "memory_compress",
                "Archive original history records and replace "
                "their live occurrences with a summary. Use memory_status for indices.",
                {
                    "indices": {"type": "array", "items": {"type": "integer"}},
                    "summary": {"type": "string"},
                },
                ["indices", "summary"],
            ),
            schema(
                "memory_retrieve",
                "Retrieve exact original records by archive index.",
                {"index": {"type": "integer"}},
                ["index"],
            ),
        )

    async def operation(
        self,
        rs: AnyRollout,
        name: str,
        arguments: dict[str, Any],
        runner: Runner[Any, Any],
    ) -> tuple[AnyRollout, str, str]:
        archives: tuple[Archive, ...] = rs.execution.get("memex", ())
        if name == "memory_status":
            status = {
                "history_records": len(rs.history),
                "live_indices": rs.context_refs,
                "live_characters": sum(
                    len(str(m.get("content", ""))) for m in rs.messages
                ),
                "archives": [
                    {"index": a.index, "summary": a.summary, "references": a.references}
                    for a in archives
                ],
            }
            return rs, json.dumps(status), ""
        if name == "memory_retrieve":
            index = arguments["index"]
            if type(index) is not int or not 0 <= index < len(archives):
                raise ProtocolError("unknown archive index")
            original = [rs.history[i] for i in archives[index].references]
            return self._charge(rs), json.dumps(original, ensure_ascii=False), "memory"
        indices, summary = arguments["indices"], arguments["summary"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or not 0 <= i < len(rs.history) for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ProtocolError(
                "indices must be distinct existing history record numbers"
            )
        if not isinstance(summary, str) or not summary.strip():
            raise ProtocolError("summary must be nonempty text")
        if len(archives) >= self.max_archives:
            raise ProtocolError("archive capacity exhausted")
        refs = rs.context_refs or (None,) * len(rs.messages)
        if len(refs) != len(rs.messages):
            raise ProtocolError("context reference alignment is invalid")
        selected = set(indices)
        kept = [
            (m, ref)
            for m, ref in zip(rs.messages, refs, strict=True)
            if ref not in selected
        ]
        index = len(archives)
        summary_message = {"role": "user", "content": f"Memory [{index}]: {summary}"}
        # All validation and construction precede the single immutable state commit.
        updated = replace(
            rs,
            messages=(*[m for m, _ in kept], summary_message),
            context_refs=(*[ref for _, ref in kept], None),
            context_version=rs.context_version + 1,
            execution={
                **rs.execution,
                "memex": (*archives, Archive(index, tuple(indices), summary)),
            },
        )
        return self._charge(updated), f"Archived as memory [{index}].", "memory"

    def _charge(self, rs: AnyRollout) -> AnyRollout:
        if not rs.turns or not self.operation_cost:
            return rs
        turn = rs.turns[-1]
        components = dict(turn.reward_components)
        components["memory"] = components.get("memory", 0) - self.operation_cost
        return replace(
            rs, turns=(*rs.turns[:-1], replace(turn, reward_components=components))
        )
