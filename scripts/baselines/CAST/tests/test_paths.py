from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from src.config import (
    load_config,
    model_root,
    output_root,
    select_model,
    validate_batch_invariants,
)
from src.paths import (
    CAST_ROOT,
    atomic_write_text,
    configure_local_environment,
    require_cast_path,
)


def test_write_guard_accepts_only_cast_tree() -> None:
    inside = require_cast_path(CAST_ROOT / "runs" / "unit-test")
    assert CAST_ROOT.resolve() in inside.parents
    with pytest.raises(ValueError, match="refuses to write"):
        require_cast_path(CAST_ROOT.parent / "outside-cast")


def test_atomic_text_write_does_not_require_path_write_text_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = CAST_ROOT / "cache" / "tests" / "atomic-write.txt"

    def unsupported_write_text(*args: object, **kwargs: object) -> None:
        raise TypeError("write_text() got an unexpected keyword argument 'newline'")

    monkeypatch.setattr(Path, "write_text", unsupported_write_text)
    atomic_write_text(target, "line-1\nline-2\n")
    assert target.read_bytes() == b"line-1\nline-2\n"


def test_all_cache_paths_are_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRANSFORMERS_CACHE", "outside-cast")
    configured = configure_local_environment()
    root = CAST_ROOT.resolve()
    for value in configured.values():
        path = Path(value).resolve()
        assert path == root or root in path.parents
    assert Path(sys.pycache_prefix).resolve() == Path(
        configured["PYTHONPYCACHEPREFIX"]
    ).resolve()
    assert Path(tempfile.gettempdir()).resolve() == Path(
        configured["TMPDIR"]
    ).resolve()
    assert "TRANSFORMERS_CACHE" not in os.environ


def test_checked_in_config_preserves_batch_invariants() -> None:
    config = load_config(CAST_ROOT / "configs" / "cast_config.yaml")
    validate_batch_invariants(config)
    assert config["extraction"]["batch_size"] == 1
    assert config["evaluation"]["batch_size"] == 1
    assert all(
        dataset["fpr_strategy"] == "refusal_keywords"
        for dataset in config["evaluation"]["utility_dataset_config"]["datasets"]
    )


def test_model_outputs_are_partitioned_directly_by_model_name() -> None:
    config = load_config(CAST_ROOT / "configs" / "cast_config.yaml")
    model = select_model(config, "qwen2_5_7b")
    root = model_root(config, model)

    assert "run" not in config
    assert root == output_root(config) / "qwen2_5_7b"
    assert root == CAST_ROOT / "runs" / "qwen2_5_7b"
