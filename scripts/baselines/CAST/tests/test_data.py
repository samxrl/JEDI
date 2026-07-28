from __future__ import annotations

from src.data import (
    assign_splits,
    build_overlap_report,
    normalize_prompt,
    prompt_hash,
)


def _rows(prefix: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"{prefix}:{index}",
            "prompt": f"{prefix} prompt {index}",
            "normalized_prompt_hash": prompt_hash(f"{prefix} prompt {index}"),
            "label": prefix,
        }
        for index in range(count)
    ]


def test_prompt_normalization_is_unicode_and_whitespace_stable() -> None:
    assert normalize_prompt("  Ａ\t B\n") == "A B"
    assert prompt_hash("Ａ  B") == prompt_hash("A B")


def test_split_assignment_is_deterministic_and_disjoint() -> None:
    sizes = {"vector_train": 6, "condition_calibration": 2, "steering_validation": 2}
    first = assign_splits(_rows("harmful", 15), sizes, seed=42)
    second = assign_splits(_rows("harmful", 15), sizes, seed=42)
    assert first == second
    hashes_by_split = {
        split: {
            row["normalized_prompt_hash"]
            for row in first
            if row["split"] == split
        }
        for split in sizes
    }
    assert not (
        hashes_by_split["vector_train"]
        & hashes_by_split["condition_calibration"]
    )
    assert not (
        hashes_by_split["vector_train"]
        & hashes_by_split["steering_validation"]
    )


def test_overlap_report_detects_evaluation_leakage() -> None:
    rows = assign_splits(
        _rows("harmful", 3),
        {"vector_train": 1, "condition_calibration": 1, "steering_validation": 1},
        seed=1,
    )
    leaked_hash = rows[0]["normalized_prompt_hash"]
    report = build_overlap_report(rows, {"evaluation": {leaked_hash}})
    assert report["all_internal_disjoint"]
    assert not report["all_evaluation_disjoint"]
    assert sum(report["evaluation_overlap_counts"][rows[0]["split"]].values()) == 1

