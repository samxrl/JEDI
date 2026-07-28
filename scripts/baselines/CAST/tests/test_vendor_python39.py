from __future__ import annotations

import ast
from pathlib import Path

from src.paths import VENDOR_ROOT


PYTHON39_COMPAT_FILES = (
    "utils.py",
    "malleable_model.py",
    "steering_vector.py",
)


def test_vendor_union_annotations_are_deferred_for_python39() -> None:
    package = VENDOR_ROOT / "activation_steering"
    for filename in PYTHON39_COMPAT_FILES:
        path = package / filename
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path), feature_version=(3, 9))
        future_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ]
        assert any(
            any(alias.name == "annotations" for alias in node.names)
            for node in future_imports
        ), f"{filename} must defer annotations under Python 3.9"
