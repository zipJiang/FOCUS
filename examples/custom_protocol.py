"""Example extension: model-controlled retention without scheduler changes."""

from dataclasses import replace
from typing import Any

from focus.harness.rollout import AnyRollout
from focus.harness.runner import Runner
from focus.protocols import ContextProtocol, ToolProtocol
from focus.protocols.base import ProtocolError, schema
from focus.registry import register


@register(ContextProtocol, "example_keep_recent")
class KeepRecent(ToolProtocol):
    def schemas(self) -> tuple[dict[str, Any], ...]:
        return (
            schema(
                "keep_recent",
                "Keep the last n messages.",
                {"n": {"type": "integer", "minimum": 1}},
                ["n"],
            ),
        )

    async def operation(
        self,
        rs: AnyRollout,
        name: str,
        arguments: dict[str, Any],
        runner: Runner[Any, Any],
    ) -> tuple[AnyRollout, str, str]:
        n = arguments["n"]
        if not isinstance(n, int) or n < 1:
            raise ProtocolError("n must be a positive integer")
        return (
            replace(
                rs,
                messages=rs.messages[-n:],
                context_refs=rs.context_refs[-n:],
                context_version=rs.context_version + 1,
            ),
            "Retained.",
            "context",
        )
