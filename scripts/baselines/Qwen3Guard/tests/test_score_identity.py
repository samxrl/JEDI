from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from src.commands import (
    _legacy_score_identity,
    _load_scores_for_materialization,
    _score_identity,
    _score_identity_matches,
)
from src.config import load_config, override_target_llm
from src.paths import (
    QWEN3GUARD_ROOT,
    sha256_file,
    write_json,
    write_jsonl_gz,
    write_yaml,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "periodic_qwen3guard.yaml"
)


def _configs() -> tuple[dict, dict]:
    original = override_target_llm(
        load_config(CONFIG),
        "vicuna_7b_v1_5",
    )
    changed_batch = copy.deepcopy(original)
    changed_batch["execution"]["batch_size"] = 8
    return original, changed_batch


def test_batch_size_is_not_part_of_semantic_score_identity() -> None:
    original, changed_batch = _configs()
    manifest = {"samples_sha256": "samples"}

    assert _score_identity(original, manifest) == _score_identity(
        changed_batch,
        manifest,
    )
    assert _legacy_score_identity(
        original,
        manifest,
    ) != _legacy_score_identity(changed_batch, manifest)


def test_legacy_identity_uses_resolved_score_config_for_migration() -> None:
    current, scored = _configs()
    manifest = {"samples_sha256": "samples"}
    stored = _legacy_score_identity(scored, manifest)
    cache = QWEN3GUARD_ROOT / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        root = Path(temporary)
        write_yaml(root / "config.resolved.yaml", scored)

        assert _score_identity_matches(
            stored,
            current,
            manifest,
            root=root,
        )

        incompatible = copy.deepcopy(current)
        incompatible["guard"]["controversial_policy"] = "allow"
        assert not _score_identity_matches(
            stored,
            incompatible,
            manifest,
            root=root,
        )


def test_score_loader_keeps_zero_token_sample_with_empty_scores() -> None:
    config, _ = _configs()
    input_manifest = {"samples_sha256": "samples"}
    samples = [
        {
            "sample_id": "empty",
            "checkpoint_ends": [],
        }
    ]
    cache = QWEN3GUARD_ROOT / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        root = Path(temporary)
        scores_path = root / "raw" / "prefix_scores.jsonl.gz"
        write_jsonl_gz(scores_path, [])
        write_json(
            root / "score_manifest.json",
            {
                "score_identity": _score_identity(
                    config,
                    input_manifest,
                ),
                "prefix_scores_sha256": sha256_file(scores_path),
            },
        )

        grouped = _load_scores_for_materialization(
            config,
            root,
            input_manifest,
            samples,
        )

    assert grouped == {"empty": {}}
