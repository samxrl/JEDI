from __future__ import annotations

from io import StringIO

import pytest

from src.progress import StageProgress, progress_iter


def test_stage_progress_reaches_completion() -> None:
    stream = StringIO()
    progress = StageProgress("prepare", 2, file=stream)
    progress.begin("Read data")
    progress.advance()
    progress.begin("Write artifacts")
    progress.advance()
    progress.finish()

    output = stream.getvalue()
    assert "CAST prepare" in output
    assert "100%" in output
    assert "Complete" in output


def test_stage_progress_rejects_empty_total() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        StageProgress("invalid", 0)


def test_progress_iter_preserves_items() -> None:
    assert list(
        progress_iter(
            [1, 2, 3],
            desc="test",
            total=3,
            leave=False,
        )
    ) == [1, 2, 3]
