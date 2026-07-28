from __future__ import annotations

import pytest

from src.config import replace_model_path_name, select_model


def _single_model_config(path: str) -> dict[str, object]:
    return {
        "model": {
            "name": "vicuna_7b_v1_5",
            "path": path,
            "harmful_calibration_file": (
                "data/raw/vicuna_7b_v1_5_sampled_jailbreaks.json"
            ),
        }
    }


def test_llm_name_overrides_name_and_final_forward_path_component() -> None:
    config = _single_model_config("../../../models/vicuna_7b_v1_5")
    selected = select_model(config, "Qwen2.5-7B-Instruct")

    assert selected["name"] == "Qwen2.5-7B-Instruct"
    assert selected["path"] == "../../../models/Qwen2.5-7B-Instruct"
    assert config["model"]["name"] == "vicuna_7b_v1_5"
    assert config["model"]["path"] == "../../../models/vicuna_7b_v1_5"


def test_path_override_also_supports_windows_separator_and_bare_name() -> None:
    assert (
        replace_model_path_name(
            r"R:\models\vicuna_7b_v1_5",
            "qwen2_5_7b",
        )
        == r"R:\models\qwen2_5_7b"
    )
    assert replace_model_path_name("vicuna_7b_v1_5", "qwen3_8b") == "qwen3_8b"
    with pytest.raises(ValueError, match="model.path"):
        replace_model_path_name("../../../models/", "qwen3_8b")


@pytest.mark.parametrize("invalid_name", ["", "org/model", r"org\model", ".."])
def test_llm_name_rejects_paths(invalid_name: str) -> None:
    with pytest.raises(ValueError, match="--llm-name"):
        select_model(
            _single_model_config("../../../models/vicuna_7b_v1_5"),
            invalid_name,
        )
