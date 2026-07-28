from __future__ import annotations

import numpy as np
import pytest
import torch

from src.calibration import (
    condition_matches,
    condition_similarity,
    effective_operator,
    search_condition_point,
)


def test_condition_similarity_matches_official_projection_formula() -> None:
    hidden = torch.tensor([2.0, 1.0, -0.5])
    direction = torch.tensor([1.0, 0.5, 0.25])
    projector = torch.outer(direction, direction) / torch.dot(direction, direction)
    projected = torch.tanh(projector @ hidden)
    expected = torch.dot(hidden, projected) / (
        torch.linalg.vector_norm(hidden) * torch.linalg.vector_norm(projected)
    )
    assert condition_similarity(hidden, direction).item() == pytest.approx(
        expected.item(),
        abs=1e-7,
    )


def test_official_comparator_names_are_preserved() -> None:
    assert effective_operator("smaller") == ">"
    assert effective_operator("larger") == "<"
    assert condition_matches(0.8, 0.5, "smaller")
    assert condition_matches(0.2, 0.5, "larger")


def test_condition_point_search_selects_perfect_harmful_separator() -> None:
    scores = {
        1: [0.9, 0.8, 0.2, 0.1],
        2: [0.5, 0.5, 0.5, 0.5],
    }
    labels = [1, 1, 0, 0]
    best, report = search_condition_point(
        scores,
        labels,
        np.asarray([0.5]),
    )
    assert best.layer == 1
    assert best.official_direction == "smaller"
    assert best.effective_operator == ">"
    assert best.f1 == pytest.approx(1.0)
    assert report["1"]["auroc_harmful_positive"] == pytest.approx(1.0)

