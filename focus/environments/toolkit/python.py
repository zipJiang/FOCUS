"""``python``: run a calculation in a subprocess that cannot outlive its limits.

The point is arithmetic, not general execution. An LLM's silent slips live in bracket
tables, per-item fees and date differences, so an env that wants those right gives the
model somewhere to compute them; everything here exists to make that cheap to offer and
hard to abuse.

The screen is deliberately an allow/deny pass over the AST rather than an interpreter:
functions, loops, comprehensions and f-strings all stay available, because real
iteration is exactly what the arithmetic needs. What it blocks is anything that would
reach *outside* the process -- which, in an eval harness, is usually the metric rather
than the security posture: the answer keys tend to sit on the same filesystem, and a
model that reads one has not solved anything.

The runtime side is belt and braces: ``-I -S`` (no site, no user paths, no
``PYTHONPATH`` or ``$CWD`` on ``sys.path``), a scrubbed environment, CPU / address
space / file size / process rlimits the child sets on itself before running a byte of
agent code, and a wall-clock timeout that kills the process. It is not a container and
does not pretend to be one.
"""

from __future__ import annotations

import ast
import asyncio
import sys
import textwrap
from collections.abc import Iterable
from typing import Any

from focus.harness.tools import BaseTool, ToolError

#: Worth having for arithmetic and safe to expose: exact decimals, dates, and light data
#: wrangling. An env with other needs passes its own set -- but every name in it is
#: reachable by the agent, so add only what the task actually computes with.
DEFAULT_IMPORTS = frozenset(
    {
        "math",
        "decimal",
        "fractions",
        "itertools",
        "datetime",
        "statistics",
        "json",
        "re",
        "collections",
    }
)

#: Names that would reach outside the process (or rebuild a way to). Denied whatever the
#: allowed imports are, because they need no import to reach.
_DENIED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
        "memoryview",
        "exit",
        "quit",
    }
)

#: The child sets its own limits before running anything, which is both simpler and
#: safer than ``preexec_fn`` (deprecated, and unsound with threads).
#:
#: Each limit is set best-effort: macOS refuses ``RLIMIT_AS`` outright ("current limit
#: exceeds maximum limit"), and a strict loop there would make every single call fail
#: with a traceback from the bootstrap rather than run the agent's arithmetic. Whatever
#: the platform grants still applies, and the wall-clock timeout below is enforced by
#: the parent either way.
_BOOTSTRAP = textwrap.dedent(
    """
    import resource, sys
    for _name, _limit in (
        ("RLIMIT_CPU", (2, 2)),
        ("RLIMIT_AS", (512 * 1024 * 1024,) * 2),
        ("RLIMIT_FSIZE", (0, 0)),
        ("RLIMIT_NPROC", (64, 64)),
    ):
        try:
            resource.setrlimit(getattr(resource, _name), _limit)
        except (AttributeError, OSError, ValueError):
            pass
    exec(compile(sys.stdin.read(), "<agent>", "exec"), {"__name__": "__main__"})
    """
)


def _describe(allowed_imports: Iterable[str]) -> str:
    return (
        "Run Python to compute something. Print what you want back (a trailing bare "
        + "expression is echoed). Each call is a fresh process, so nothing carries "
        + "over between calls. Imports available: "
        + f"{', '.join(sorted(allowed_imports))}."
    )


#: The temporary the echo binds the trailing value to. Leading underscores and a name
#: no agent writes, because it lands in the code's own namespace -- and it is the last
#: statement, so nothing the agent wrote can read it afterwards.
_ECHOED = "__echoed"


def _echo(value: ast.expr) -> ast.stmt:
    """``print(repr(value))`` -- unless the value is ``None``, which printed nothing.

    The test is the *value*, not the call. ``print(x)`` is a bare trailing expression
    and the overwhelmingly common last line of agent code, and echoing it appended
    ``None`` -- the value ``print`` returns -- to every such result, which reads as the
    tool malfunctioning and is exactly the lesson the echo exists to prevent. But
    ``print`` is only the commonest way there: a helper that prints and returns nothing,
    ``xs.sort()``, ``d.update(...)`` and a ``pprint`` from a custom import list all
    reach the same ``None``, and a guard spelled as "is this the name ``print``" catches
    none of them.

    The value is bound once, so the expression is evaluated exactly once however the
    test goes -- a trailing call with a side effect must not happen twice.
    """
    bound = ast.NamedExpr(target=ast.Name(id=_ECHOED, ctx=ast.Store()), value=value)
    shown = ast.Call(
        func=ast.Name(id="repr", ctx=ast.Load()),
        args=[ast.Name(id=_ECHOED, ctx=ast.Load())],
        keywords=[],
    )
    return ast.If(
        test=ast.Compare(
            left=bound, ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]
        ),
        body=[
            ast.Expr(
                ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[shown],
                    keywords=[],
                )
            )
        ],
        orelse=[],
    )


def screen_code(
    code: str, *, allowed_imports: Iterable[str] = DEFAULT_IMPORTS
) -> ast.Module:
    """Parse ``code`` and reject what would leave the sandbox.

    A trailing bare expression is rewritten to ``print(repr(...))`` -- models forget
    ``print`` constantly, and a tool that silently returns nothing teaches them it is
    broken. A value of ``None`` is left unsaid: a line that returned nothing (a
    ``print``, a helper that printed, an in-place ``sort``) already had its say, and
    appending ``None`` to it reads as the tool malfunctioning. See :func:`_echo`.
    """
    allowed = frozenset(allowed_imports)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolError(f"syntax error: {exc}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            bad = [n for n in names if n.split(".")[0] not in allowed]
            if bad:
                available = ", ".join(sorted(allowed))
                raise ToolError(
                    f"cannot import {', '.join(bad)}. Available: {available}"
                )
        if isinstance(node, ast.Name) and node.id in _DENIED_NAMES:
            raise ToolError(f"{node.id} is not available in this sandbox")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ToolError("attributes starting with __ are not available")
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr):
        echo = ast.copy_location(_echo(last.value), last)
        tree.body[-1] = ast.fix_missing_locations(echo)
    return tree


class PythonTool(BaseTool):
    """``python``: run a calculation in a subprocess that cannot outlive its limits."""

    name = "python"
    description = _describe(DEFAULT_IMPORTS)
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "the code to run"}},
        "required": ["code"],
    }

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_chars: int = 3000,
        allowed_imports: Iterable[str] = DEFAULT_IMPORTS,
        description: str | None = None,
    ):
        self._allowed = frozenset(allowed_imports)
        self.description = description or _describe(self._allowed)
        self._timeout = timeout
        self._max_chars = max_chars

    async def call(self, params: str | dict[str, Any], **kwargs: object) -> str:
        code = str(self.verify_args(params)["code"])
        source = ast.unparse(screen_code(code, allowed_imports=self._allowed))
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-c",
            _BOOTSTRAP,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={"PATH": "/usr/bin", "PYTHONHASHSEED": "0"},
        )
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(source.encode()), timeout=self._timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return f"[timed out after {self._timeout:g}s -- avoid unbounded loops]"
        text = out.decode(errors="replace").strip()
        if len(text) > self._max_chars:
            text = text[: self._max_chars] + "\n[output truncated]"
        if proc.returncode and proc.returncode < 0:
            # killed by a signal: the CPU or memory rlimit fired, which reads as a
            # mysterious crash unless we say so
            return (
                f"{text}\n[killed: the calculation exceeded its 2s CPU or 512MB memory "
                + "limit -- compute it directly rather than by search]"
            ).strip()
        if proc.returncode:
            return f"{text}\n[exited with code {proc.returncode}]".strip()
        return text or "(no output -- the tool returns only what you print)"


__all__ = ["DEFAULT_IMPORTS", "PythonTool", "screen_code"]
