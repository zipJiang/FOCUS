"""Existing compaction policy behind the context protocol contract."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from focus.harness.checkpoint import Effect
from focus.harness.compaction.base import Compaction, Compactor
from focus.harness.compaction.triggers import Trigger
from focus.harness.rollout import AnyRollout
from focus.harness.turn import Turn
from focus.protocols.base import ContextProtocol

if TYPE_CHECKING:
    from focus.harness.runner import Runner


class Summary(ContextProtocol):
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
            self.prepare(rs, runner).tokens,
            rs.state,
        )

    async def before(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        # The enclosing runner operation commits this fold with its complete state.
        # Calling runner.compact here would nest a persistence/retry transaction.
        return await runner._compact(rs) if self.needs_prepare(rs, runner) else rs

    async def compact(self, rs: AnyRollout, runner: Runner[Any, Any]) -> AnyRollout:
        if self.compactor is None:
            return await super().compact(rs, runner)
        runner.services.attach_budget(rs.owner, rs.budget)
        folded = await self.compactor.compact(
            Compaction(
                messages=rs.messages,
                seed=rs.seed,
                state=rs.state,
                workspace=rs.workspace,
                runner=runner,
                checkpoint=rs,
            )
        )
        # Marker records a silent rewrite, with no generated tokens or task time.
        marker: Turn[Any] = Turn(
            prefix=(),
            tokens=(),
            logprobs=(),
            tag="fold",
            actor=runner.actor,
            invocation_id=rs.invocation_id,
            context_version=rs.context_version,
        )
        turns = tuple(replace(t, tag="fold", transition=None) for t in folded.turns)
        return replace(
            rs,
            budget=runner.services.remaining(rs.owner),
            messages=tuple(folded.messages),
            context_refs=(None,) * len(folded.messages),
            state=folded.state,
            turns=(*rs.turns, *(turns or (marker,))),
            context_version=rs.context_version + 1,
            effects=(
                *rs.effects,
                Effect(
                    "compaction",
                    len(rs.turns),
                    rs.context_version,
                    rs.context_version + 1,
                    invocation_id=rs.invocation_id,
                    turn_end=len(rs.turns) + len(turns or (marker,)),
                ),
            ),
            advances=rs.advances + 1,
            steps_remaining=None
            if rs.steps_remaining is None
            else max(0, rs.steps_remaining - 1),
        )
