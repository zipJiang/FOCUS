"""Existing compaction policy behind the context protocol contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from focus.harness.compaction.base import Compactor
from focus.harness.compaction.triggers import Trigger
from focus.harness.rollout import AnyRollout
from focus.protocols.base import ContextProtocol
from focus.registry import register

if TYPE_CHECKING:
    from focus.harness.runner import Runner


@register(ContextProtocol, "summary")
class Summary(ContextProtocol):
    manages_context = True

    def __init__(self, compactor: Compactor[Any] | None = None):
        if compactor is not None and not isinstance(
            getattr(compactor, "trigger", None), Trigger
        ):
            raise TypeError(f"{type(compactor).__name__} has no `trigger`")
        self.compactor = compactor
        self.manages_context = compactor is not None

    async def initialize(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        if self.compactor is not None:
            self.compactor.trigger.check(rs.state)
        return rs

    def needs_prepare(self, rs: AnyRollout, runner: Runner[Any, Any]) -> bool:
        return self.compactor is not None and self.compactor.trigger.fires(
            runner.policy.prepare(rs.messages, tools=runner.tools).tokens,
            rs.state,
        )

    async def before(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        return await runner.compact(rs) if self.needs_prepare(rs, runner) else rs

    async def compact(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        if self.compactor is None:
            return await super().compact(rs, runner)
        return await runner._fold(rs, self.compactor)
