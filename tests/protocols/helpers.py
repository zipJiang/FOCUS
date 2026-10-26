"""Test protocol for context edits, native tool history, and checkpoint recovery."""

from dataclasses import replace
from typing import Any

from focus.harness.agent import Agent
from focus.harness.rollout import AnyRollout
from focus.protocols import ToolProtocol
from focus.protocols.base import ProtocolError, schema


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
        runner: Agent[Any, Any],
    ) -> tuple[AnyRollout, str, str]:
        n = arguments["n"]
        if type(n) is not int or n < 1:
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
