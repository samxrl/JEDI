from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    guard_backend_kind,
    guard_reference,
    input_dir,
    load_config,
    override_guard_model,
    override_target_llm,
    run_dir,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "periodic_qwen3guard.yaml"
)


def test_default_config_and_target_override() -> None:
    config = load_config(CONFIG)
    resolved = override_target_llm(config, "Qwen3-8B")

    assert resolved["models"][0]["name"] == "Qwen3-8B"
    assert resolved["models"][0]["tokenizer_path"].endswith(
        "/models/Qwen3-8B"
    )
    assert config["models"][0]["name"] == "vicuna_7b_v1_5"
    assert config["baseline"]["condition_prefix"] == "qwen3guard_gen"
    assert config["guard"]["model_path"].endswith("/Qwen3Guard-Gen-8B")
    assert config["guard"]["controversial_policy"] == "block"
    assert config["guard"]["context_window_tokens"] == 32768
    assert config["guard"]["max_input_tokens"] == 32576
    assert config["execution"]["show_progress"] is True
    assert "GCG" in config["source"]["attack_methods"]
    assert all(
        not method.casefold().startswith("adaptive_")
        for method in config["source"]["attack_methods"]
    )


def test_guard_model_override_switches_backend_path_and_limits() -> None:
    config = load_config(CONFIG)
    resolved = override_guard_model(config, "Qwen3Guard-Stream-8B")

    assert resolved["guard"]["model_name"] == "Qwen3Guard-Stream-8B"
    assert resolved["guard"]["model_id"] == "Qwen/Qwen3Guard-Stream-8B"
    assert guard_backend_kind(resolved) == "stream"
    assert resolved["guard"]["model_path"].endswith(
        "/Qwen3Guard-Stream-8B"
    )
    assert resolved["guard"]["context_window_tokens"] == 8192
    assert resolved["guard"]["max_input_tokens"] == 8192
    assert resolved["execution"]["batch_size"] == 1
    assert resolved["baseline"]["condition_prefix"] == (
        "qwen3guard_stream"
    )
    assert Path(guard_reference(resolved)).resolve() == Path(
        r"R:\models\Qwen3Guard-Stream-8B"
    ).resolve()
    assert config["guard"]["model_name"] == "Qwen3Guard-Gen-8B"


def test_guard_model_override_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported guard model"):
        override_guard_model(load_config(CONFIG), "OtherGuard-8B")


def test_target_override_supports_windows_separator() -> None:
    config = load_config(CONFIG)
    config["models"][0]["tokenizer_path"] = r"R:\models\old"
    resolved = override_target_llm(config, "mistral_7b_v2")
    assert resolved["models"][0]["tokenizer_path"] == (
        r"R:\models\mistral_7b_v2"
    )


def test_artifact_paths_are_stable_per_target_and_guard() -> None:
    base = override_target_llm(load_config(CONFIG), "Qwen3-8B")
    stream = override_guard_model(base, "Qwen3Guard-Stream-8B")
    other_target = override_target_llm(base, "mistral_7b_v2")

    assert input_dir(base).parts[-2:] == (
        "Qwen3-8B",
        "Qwen3Guard-Gen-8B",
    )
    assert run_dir(stream).parts[-2:] == (
        "Qwen3-8B",
        "Qwen3Guard-Stream-8B",
    )
    assert run_dir(base) == run_dir(dict(base))
    assert run_dir(base) != run_dir(stream)
    assert run_dir(base) != run_dir(other_target)


@pytest.mark.parametrize("name", ["", "..", "org/model", r"org\model"])
def test_target_override_rejects_non_directory_name(name: str) -> None:
    with pytest.raises(ValueError):
        override_target_llm(load_config(CONFIG), name)
