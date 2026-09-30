"""Canonical identities for the fixed repository tool interface."""

from __future__ import annotations

import json

from .metrics import relative_path
from .repository import MAX_HITS, READ_LINES


def _integer(value: object) -> int:
    # Match the tool schema coercion: fractional numbers are invalid, and bools
    # remain bools so they cannot be mistaken for accepted integer arguments.
    return value if isinstance(value, bool) else int(str(value).strip())


def call_key(name: str, arguments: str) -> tuple[str, str]:
    """Canonicalize valid defaults so omitted page sizes don't hide exact repeats."""
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            return name, arguments
        if name in {"list", "grep", "read"} and args.get("path") is not None:
            args["path"] = relative_path(args["path"], allow_root=name != "read")
        if name == "read":
            args["start"] = _integer(args.get("start", 1))
            args["lines"] = _integer(args.get("lines", READ_LINES))
        elif name == "grep":
            args["max_hits"] = _integer(args.get("max_hits", MAX_HITS))
            args.setdefault("path", None)
        if name == "grep" and args.get("path") in {None, "", "."}:
            args["path"] = None
        if name == "list":
            # Explicit root lists to depth two; omitted/null path lists depth one.
            args.setdefault("path", None)
        return name, json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return name, arguments
