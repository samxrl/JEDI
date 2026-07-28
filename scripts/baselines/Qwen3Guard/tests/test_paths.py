from __future__ import annotations

from pathlib import Path

import pytest

from src.paths import QWEN3GUARD_ROOT, REPO_ROOT, require_local_write


def test_write_guard_accepts_only_baseline_tree() -> None:
    accepted = require_local_write(QWEN3GUARD_ROOT / "runs" / "smoke")
    assert QWEN3GUARD_ROOT.resolve() in accepted.parents

    with pytest.raises(ValueError):
        require_local_write(Path(REPO_ROOT) / "data" / "evaluations")
