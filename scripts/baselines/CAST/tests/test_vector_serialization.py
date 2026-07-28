from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from activation_steering import SteeringVector
from src.paths import CAST_ROOT
from src.vectors import save_vector


def test_numpy_explained_variance_is_json_serializable() -> None:
    vector = SteeringVector(
        model_type="toy",
        directions={0: np.asarray([1.0, 0.0], dtype=np.float32)},
        explained_variances={0: np.float32(0.75)},
    )
    target = CAST_ROOT / "cache" / "tests" / "numpy-variance.svec"

    saved = save_vector(vector, target)
    with saved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["explained_variances"]["0"] == pytest.approx(0.75)
    assert isinstance(payload["explained_variances"]["0"], float)
    assert SteeringVector.load(str(saved)).explained_variances[0] == pytest.approx(
        0.75
    )


def test_failed_vector_save_keeps_existing_target() -> None:
    target = CAST_ROOT / "cache" / "tests" / "existing-vector.svec"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing", encoding="utf-8")

    class FailingVector:
        def save(self, file_path: str) -> None:
            Path(file_path).write_text("partial", encoding="utf-8")
            raise RuntimeError("simulated save failure")

    with pytest.raises(RuntimeError, match="simulated"):
        save_vector(FailingVector(), target)  # type: ignore[arg-type]

    assert target.read_text(encoding="utf-8") == "existing"
    assert not (target.parent / ".existing-vector.tmp.svec").exists()
