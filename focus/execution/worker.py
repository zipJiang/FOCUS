"""Private managed-cell subprocess. IPC uses stdin/stdout; user output is captured."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import types
from pathlib import Path
from typing import Any

import cloudpickle


class LimitedOutput(io.StringIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, text: str) -> int:
        super().write(text[: max(0, self.limit - self.tell())])
        return len(text)


def restore_functions(namespace: dict[str, Any]) -> None:
    """cloudpickle copies function globals; a REPL needs the live namespace instead."""
    memo: dict[int, Any] = {}

    def bind(value: Any) -> Any:
        if id(value) in memo:
            return memo[id(value)]
        memo[id(value)] = value
        if (
            isinstance(value, types.FunctionType)
            and value.__module__ == "__focus_cell__"
        ):
            restored = types.FunctionType(
                value.__code__,
                namespace,
                value.__name__,
                value.__defaults__,
                value.__closure__,
            )
            restored.__kwdefaults__ = value.__kwdefaults__
            restored.__annotations__ = value.__annotations__
            restored.__dict__.update(value.__dict__)
            restored.__module__ = value.__module__
            restored.__qualname__ = value.__qualname__
            memo[id(value)] = restored
            return restored
        if isinstance(value, dict):
            for key in tuple(value):
                value[key] = bind(value[key])
        elif isinstance(value, list):
            value[:] = [bind(item) for item in value]
        elif isinstance(value, tuple):
            memo[id(value)] = tuple(bind(item) for item in value)
        elif isinstance(value, set):
            items = [bind(item) for item in value]
            value.clear()
            value.update(items)
        return memo[id(value)]

    bind(namespace)


def main() -> None:
    import sys

    wire_in, wire_out = sys.stdin, sys.stdout

    def send(message: dict[str, Any]) -> None:
        wire_out.write(json.dumps(message) + "\n")
        wire_out.flush()

    data = json.loads(wire_in.readline())
    namespace: dict[str, Any] = cloudpickle.loads(base64.b64decode(data["state"]))
    restore_functions(namespace)
    output = LimitedOutput(data["max_output"])

    def query(kind: str, prompts: Any, model: str | None = None) -> Any:
        send({"kind": "call", "operation": kind, "prompts": prompts, "model": model})
        reply = json.loads(wire_in.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    callbacks = {
        "llm_query": lambda prompt, model=None: query("lm", [prompt], model)[0],
        "llm_query_batched": lambda prompts, model=None: query("lm", prompts, model),
        "rlm_query": lambda prompt, model=None: query("rlm", [prompt], model)[0],
        "rlm_query_batched": lambda prompts, model=None: query("rlm", prompts, model),
    }
    namespace.update(callbacks)
    namespace["__name__"] = "__focus_cell__"
    error = ""
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            exec(compile(data["code"], "<focus-cell>", "exec"), namespace)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    for key in (*callbacks, "__builtins__", "__name__"):
        namespace.pop(key, None)
    try:
        # Live handles cannot be restored faithfully, even if a serializer happens
        # to support reconstructing a similar object.
        def validate(value: Any, seen: set[int]) -> None:
            if id(value) in seen:
                return
            seen.add(id(value))
            if (
                isinstance(value, types.FunctionType)
                and value.__module__ == "__focus_cell__"
            ):
                if value.__closure__:
                    raise TypeError(
                        "closures are not checkpointable; use explicit globals"
                    )
                validate(value.__defaults__, seen)
                validate(value.__kwdefaults__, seen)
            if isinstance(value, type) and value.__module__ == "__focus_cell__":
                raise TypeError(
                    "cell-defined classes are not checkpointable; use data records"
                )
            if type(value).__module__ == "__focus_cell__":
                raise TypeError(
                    "cell-defined instances are not checkpointable; use data records"
                )
            if isinstance(value, io.IOBase):
                raise TypeError("live file handles are not checkpointable; close them")
            if isinstance(value, dict):
                for item in value.values():
                    validate(item, seen)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    validate(item, seen)

        validate(namespace, set())
        payload = cloudpickle.dumps(namespace)
        state = base64.b64encode(payload).decode()
        files = {}
        size = len(payload)
        for path in Path.cwd().rglob("*"):
            if path.is_symlink():
                raise ValueError("workspace symlinks are not checkpointable")
            if path.is_file():
                size += path.stat().st_size
                if size > data["max_bytes"]:
                    raise ValueError("workspace snapshot exceeds byte limit")
                files[str(path.relative_to(Path.cwd()))] = base64.b64encode(
                    path.read_bytes()
                ).decode()
        if len(state) > data["max_bytes"] * 4 // 3:
            raise ValueError("Python snapshot exceeds byte limit")
        send(
            {
                "kind": "done",
                "state": state,
                "files": files,
                "output": output.getvalue(),
                "error": error,
            }
        )
    except Exception as exc:
        send({"kind": "unsupported", "output": output.getvalue(), "error": str(exc)})
    # Descendants may inherit the IPC pipe. The host kills the process group after
    # receiving the committed result, so none can outlive the managed cell.
    wire_in.read()  # Keep the group leader alive until host termination.
    os._exit(0)


if __name__ == "__main__":
    main()
