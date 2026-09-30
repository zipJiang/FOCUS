"""The front door: `import focus` must work, from any starting point.

`focus/__init__.py` re-exports across every layer -- harness, loop, export,
scheduler, allocation -- which is exactly the shape that grows an import cycle without
anyone noticing: the package body runs before any submodule of it, so a submodule that
reaches back up (directly, or through a sibling that does) deadlocks on a
half-initialized module. The failure is *order-dependent* -- `import focus`
may be fine while `import focus.harness.runner` as the very first import of
the process is not -- so a suite that has already imported half the library cannot see
it.

Hence a subprocess per module, each a fresh interpreter whose first import is that
module and nothing else. It is the only way to make the claim the front door needs:
whichever module a user's own first import happens to be, it works.
"""

from __future__ import annotations

import ast
import os
import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import focus


def _module_names() -> list[str]:
    """Every importable module under the package, the package itself included.

    ``walk_packages`` imports each *package* to find its children, which is fine here:
    this runs in the collecting interpreter only to build the name list, and every
    import that matters happens in a subprocess below.
    """
    names = [focus.__name__]
    names += [
        info.name
        for info in pkgutil.walk_packages(focus.__path__, prefix=f"{focus.__name__}.")
    ]
    return names


def _import_failure(module: str) -> str | None:
    """``None`` if a fresh interpreter can import ``module`` first, else why not."""
    done = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode == 0:
        return None
    return f"`import {module}` failed:\n{done.stderr.strip()}"


def test_every_module_imports_first_in_a_fresh_interpreter() -> None:
    """No module may need another one to have been imported before it.

    One test rather than a parametrized case per module, because the interpreters are
    spawned concurrently: seventy-odd serial process starts cost seconds, and the pool
    turns that into a fraction of one. The guarantee is untouched -- a thread here only
    waits on a subprocess, and each subprocess is still a fresh interpreter whose first
    import is its own module -- and collecting every failure before asserting says
    *which* modules broke rather than only the first pytest happened to reach.
    """
    modules = _module_names()
    workers = min(len(modules), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        failures = [f for f in pool.map(_import_failure, modules) if f is not None]

    assert not failures, "\n\n".join(failures)


def test_the_front_door_exports_everything_it_claims_to() -> None:
    """``__all__`` is grouped by concern rather than sorted, so nothing checks it.

    RUF022 is off for `__init__` files here (the grouping is documentation), which
    means a name that no longer resolves -- a typo, or one that moved out of its
    subpackage -- would sit in the list unnoticed until a reader's ``from
    focus import`` failed.
    """
    assert len(set(focus.__all__)) == len(focus.__all__)
    for name in focus.__all__:
        assert hasattr(focus, name), f"{name} is exported but unresolved"


def test_a_seam_whose_implementations_are_exported_exports_its_interface() -> None:
    """A user who can name a recipe should be able to name what to subclass.

    `register_estimator` without `AdvantageEstimator` was the shape of the gap: the
    front door taught how to register an estimator and hid the class to register.
    """
    pairs = {
        "AdvantageEstimator": ("ChildRelative", "register_estimator"),
        "Trigger": ("PromptTokens", "StateCounter"),
        "Compactor": ("AgenticCompactor", "build_budget"),
        "ProposalPolicy": ("AnchorProposal", "MixtureProposal"),
        "GatingPolicy": ("WidthGating", "ConcurrencyGating"),
        "Allocator": ("FifoAllocator", "BestFirstAllocator", "TokenEntropyAllocator"),
    }
    missing = [
        interface
        for interface, implementations in pairs.items()
        if any(name in focus.__all__ for name in implementations)
        and interface not in focus.__all__
    ]
    assert not missing, f"exported implementations without their interface: {missing}"


def test_generation_has_no_reverse_dependency_on_the_harness() -> None:
    """Policies and parsers must be usable without runtime contracts."""
    package = Path(focus.__file__).parent
    violations = []
    for path in (package / "generation").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(
                module == "focus" or module.startswith("focus.harness")
                for module in modules
            ):
                violations.append(f"{path.relative_to(package)}:{node.lineno}")
    assert not violations, f"generation imports runtime contracts: {violations}"
