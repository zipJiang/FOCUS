"""Prompt preparation, context edits, and compaction strategies."""

from focus.context.base import (
    AppendOnly,
    Compaction,
    CompactionResult,
    ContextError,
    ContextManager,
    ContextTools,
    IncompleteCompactionError,
    Segments,
)
from focus.context.compaction import Summarize
from focus.context.triggers import AnyOf, PromptTokens, StateCounter, Trigger

__all__ = [
    "ContextManager",
    "AppendOnly",
    "Segments",
    "Summarize",
    "ContextTools",
    "Compaction",
    "CompactionResult",
    "ContextError",
    "IncompleteCompactionError",
    "Trigger",
    "AnyOf",
    "PromptTokens",
    "StateCounter",
]
