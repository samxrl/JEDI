from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from transformers import AutoTokenizer

from src.input_adapter import _iter_samples
from src.paths import QWEN3GUARD_ROOT


class FakeTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "".join(chr(value) for value in token_ids)


def _config(root: Path, tokenizer_path: Path) -> dict[str, Any]:
    return {
        "paths": {"source_evaluation_dir": str(root)},
        "source": {
            "condition": "baseline",
            "include_safety": True,
            "include_utility": False,
            "require_baseline_rows": True,
            "max_samples_per_file": 0,
            "attack_methods": ["GCG"],
        },
        "models": [
            {
                "name": "model",
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_revision": None,
                "trust_remote_code": True,
                "local_files_only": True,
            }
        ],
        "streaming": {"chunk_sizes": [16, 32]},
    }


def _write_attack(
    model_dir: Path,
    name: str,
    condition: str,
) -> None:
    pd.DataFrame(
        [
            {
                "condition": condition,
                "prompt": f"{name} prompt",
                "assistant_output": f"{name} response",
                "attack_method": name,
            }
        ]
    ).to_csv(
        model_dir / f"model_evaluation_detailed_attack_{name}.csv",
        index=False,
        encoding="utf-8",
    )


def test_prepare_excludes_adaptive_attack_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = QWEN3GUARD_ROOT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        root = Path(temporary)
        model_dir = root / "evaluations" / "model"
        model_dir.mkdir(parents=True)
        tokenizer_path = root / "tokenizer"
        tokenizer_path.mkdir()
        _write_attack(model_dir, "adaptive_gcg", "guarded")
        _write_attack(model_dir, "GCG", "baseline")
        manifest: dict[str, Any] = {"source_files": [], "models": {}}

        rows = list(
            _iter_samples(
                _config(root / "evaluations", tokenizer_path),
                manifest,
            )
        )

    assert len(rows) == 1
    assert manifest["source_file_count"] == 1
    assert len(manifest["source_files"]) == 1
    selected = manifest["source_files"][0]
    assert selected["path"].endswith(
        "model_evaluation_detailed_attack_GCG.csv"
    )
    assert selected["available_conditions"] == ["baseline"]
    assert selected["matched_rows"] == 1


def test_prepare_rejects_allowlisted_main_file_without_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = QWEN3GUARD_ROOT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        root = Path(temporary)
        model_dir = root / "evaluations" / "model"
        model_dir.mkdir(parents=True)
        tokenizer_path = root / "tokenizer"
        tokenizer_path.mkdir()
        _write_attack(model_dir, "GCG", "guarded")
        manifest: dict[str, Any] = {"source_files": [], "models": {}}

        with pytest.raises(ValueError, match="has no rows with condition=baseline"):
            list(
                _iter_samples(
                    _config(root / "evaluations", tokenizer_path),
                    manifest,
                )
            )
