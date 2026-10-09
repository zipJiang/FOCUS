"""Editable live context, committed transactionally from a managed Python cell."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from focus.codec import Message
from focus.execution import Owner
from focus.execution.process import ProcessSnapshot
from focus.harness.rollout import AnyRollout
from focus.protocols.base import ContextProtocol, ProtocolError, ToolProtocol, schema
from focus.registry import register

if TYPE_CHECKING:
    from focus.harness.runner import Runner


def validate_document(data: bytes, max_bytes: int) -> tuple[Message, ...]:
    if len(data) > max_bytes:
        raise ProtocolError("context.json exceeds document byte limit")
    try:
        document = json.loads(data)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    if not isinstance(document, list) or not document:
        raise ProtocolError("context.json must be a nonempty JSON array of messages")
    for i, message in enumerate(document):
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ProtocolError(f"message {i} must have a valid role")
        if not isinstance(message.get("content"), str):
            raise ProtocolError(f"message {i}.content must be a string")
        if any(not isinstance(key, str) for key in message):
            raise ProtocolError("message keys must be strings")
    return tuple(document)


@register(ContextProtocol, "clm")
class CLM(ToolProtocol):
    """Edit, delete, insert or reorder live messages without rewriting the trace."""

    def __init__(self, timeout: float = 30, max_document_bytes: int = 4 * 1024 * 1024):
        if timeout <= 0 or max_document_bytes < 1:
            raise ProtocolError("positive execution and document limits required")
        self.timeout = timeout
        self.max_document_bytes = max_document_bytes

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return (
            schema(
                "context_edit",
                "Run Python to edit context.json, a JSON array of messages. "
                "You may insert, delete, rewrite or reorder. Only valid edits commit.",
                {"code": {"type": "string"}},
                ["code"],
            ),
        )

    async def operation(
        self,
        rs: AnyRollout,
        name: str,
        arguments: dict[str, Any],
        runner: Runner[Any, Any],
    ) -> tuple[AnyRollout, str, str]:
        code = arguments["code"]
        if not isinstance(code, str):
            raise ProtocolError("code must be a string")
        previous: ProcessSnapshot = rs.execution.get("clm_session", ProcessSnapshot())
        files = dict(previous.files)
        files["context.json"] = json.dumps(rs.messages, ensure_ascii=False).encode()
        result = await runner.services.execute(
            code,
            ProcessSnapshot(previous.namespace, files),
            timeout=self.timeout,
            owner=Owner(rs.episode_id, rs.branch_id, rs.invocation_id),
        )
        if result.snapshot is None or result.error:
            raise ProtocolError(f"cell was not committed: {result.error}")
        document = validate_document(
            result.snapshot.files.get("context.json", b""), self.max_document_bytes
        )
        return (
            replace(
                rs,
                messages=document,
                context_refs=(None,) * len(document),
                context_version=rs.context_version + 1,
                execution={**rs.execution, "clm_session": result.snapshot},
            ),
            "Context committed. " + result.output,
            "context",
        )
