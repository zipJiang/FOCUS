"""Every supported module declares its public entry points deliberately."""

import ast
from pathlib import Path

import focus


def test_public_modules_declare_exports() -> None:
    root = Path(focus.__file__).parent
    for path in root.rglob("*.py"):
        if any(
            part.startswith("_") and part != "__init__.py"
            for part in path.relative_to(root).parts
        ):
            continue
        tree = ast.parse(path.read_text())
        exports = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ]
        assert len(exports) == 1, f"Declare the supported exports in {path}"
        names = ast.literal_eval(exports[0].value)
        assert len(names) == len(set(names)), f"Duplicate exports in {path}"
        assert all(not name.startswith("_") for name in names), (
            f"Private export in {path}"
        )
