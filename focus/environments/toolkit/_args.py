"""Argument helpers shared by the toolkit's tools."""

from __future__ import annotations

from typing import Any


def _as_int(args: dict[str, Any], key: str, default: int, *, lo: int, hi: int) -> int:
    """An integer argument, clamped into range.

    The *type* is already settled: ``BaseTool.verify_args`` coerces against the schema
    the tool declares, so a model that writes ``"3"`` (Qwen3.5's XML dialect has
    nowhere to put a type) has been read as ``3`` before this sees it, and a model that
    writes nonsense raised there. All that is left is the range.
    """
    raw = args.get(key)
    return default if raw is None or raw == "" else max(lo, min(hi, int(raw)))
