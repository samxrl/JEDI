# -*- coding: utf-8 -*-
"""Extract both CAST vectors with the pinned official activation-steering source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.paths import configure_local_environment, ensure_vendor_on_path  # noqa: E402

configure_local_environment()
ensure_vendor_on_path()

from activation_steering import SteeringVector  # noqa: E402

from src.config import (  # noqa: E402
    artifact_dir,
    data_dir,
    load_config,
    select_model,
    validate_batch_invariants,
)
from src.data import (  # noqa: E402
    CAST_UPSTREAM_COMMIT,
    load_prepared_splits,
    prompts_for,
)
from src.modeling import load_model_and_tokenizer, model_metadata  # noqa: E402
from src.paths import sha256_file, write_json  # noqa: E402
from src.progress import StageProgress  # noqa: E402
from src.vectors import (  # noqa: E402
    build_behavior_dataset,
    build_condition_dataset,
    load_prefix_pairs,
    save_vector,
    vector_layer_norms,
)


def _validate_extraction_config(config: dict[str, Any]) -> None:
    extraction = config["extraction"]
    if str(extraction.get("method", "pca_pairwise")) != "pca_pairwise":
        raise ValueError("The formal CAST reproduction requires extraction.method=pca_pairwise.")
    if str(extraction.get("behavior_aggregation", "suffix-only")) != "suffix-only":
        raise ValueError("The behavior vector must use suffix-only aggregation.")
    if str(extraction.get("condition_aggregation", "all")) != "all":
        raise ValueError("The condition vector must use all-token mean aggregation.")


def _load_inputs(config: dict[str, Any], model: dict[str, Any]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    prepared = data_dir(config, model)
    splits_path = prepared / "calibration_splits.jsonl"
    prefixes_path = prepared / "prefix_pairs.json"
    if not splits_path.exists() or not prefixes_path.exists():
        raise FileNotFoundError(
            "Prepared CAST data does not exist; run prepare_data.py first."
        )
    return load_prepared_splits(splits_path), load_prefix_pairs(prefixes_path)


def run(
    config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    validate_batch_invariants(config)
    _validate_extraction_config(config)
    if dry_run:
        progress = StageProgress("extract", 1)
        progress.begin("Validate vector-extraction scale")
        data_cfg = config["data"]
        split_sizes = data_cfg.get("split_sizes", {})
        train_size = int(split_sizes.get("vector_train", 60))
        prefix_count = int(data_cfg.get("prefix_limit", 100))
        summary = {
            "model": model_config["name"],
            "behavior_contrastive_pairs": train_size * prefix_count,
            "behavior_forward_sequences": train_size * prefix_count * 2,
            "condition_contrastive_pairs": train_size,
            "condition_forward_sequences": train_size * 2,
            "batch_size": 1,
            "would_load_model": model_config["path"],
        }
        progress.advance()
        progress.finish()
        return summary

    progress = StageProgress("extract", 6)
    progress.begin("Load artifacts from the prepare stage")
    rows, prefix_pairs = _load_inputs(config, model_config)
    harmful = prompts_for(rows, split="vector_train", label="harmful")
    benign = prompts_for(rows, split="vector_train", label="benign")
    if len(harmful) != len(benign) or not harmful:
        raise ValueError("vector_train must contain equal, nonempty harmful/benign prompt sets.")
    progress.advance()

    progress.begin("Load target model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(model_config)
    num_layers = int(model.config.num_hidden_layers)
    raw_layers = config["extraction"].get("hidden_layer_ids")
    layer_ids = (
        list(range(num_layers))
        if raw_layers is None
        else [int(value) for value in raw_layers]
    )
    if not layer_ids or min(layer_ids) < 0 or max(layer_ids) >= num_layers:
        raise ValueError(f"Invalid hidden_layer_ids: {layer_ids}")
    progress.advance()

    progress.begin("Build behavior/condition contrast data")
    behavior_dataset = build_behavior_dataset(tokenizer, benign, prefix_pairs)
    condition_dataset = build_condition_dataset(tokenizer, harmful, benign)
    progress.advance()
    extraction = config["extraction"]
    common = {
        "model": model,
        "tokenizer": tokenizer,
        "hidden_layer_ids": layer_ids,
        "batch_size": 1,
        "method": "pca_pairwise",
        "save_analysis": False,
    }
    progress.begin("Extract behavior vector")
    behavior_vector = SteeringVector.train(
        steering_dataset=behavior_dataset,
        accumulate_last_x_tokens="suffix-only",
        **common,
    )
    progress.advance()

    progress.begin("Extract condition vector")
    condition_vector = SteeringVector.train(
        steering_dataset=condition_dataset,
        accumulate_last_x_tokens="all",
        **common,
    )
    progress.advance()

    progress.begin("Atomically save vectors and metadata")
    destination = artifact_dir(config, model_config)
    destination.mkdir(parents=True, exist_ok=True)
    behavior_path = save_vector(
        behavior_vector,
        destination / "behavior_vector.svec",
    )
    condition_path = save_vector(
        condition_vector,
        destination / "condition_vector.svec",
    )
    prepared = data_dir(config, model_config)
    metadata = {
        "format_version": 1,
        "cast_source_commit": CAST_UPSTREAM_COMMIT,
        "model": model_metadata(model, tokenizer, str(model_config["name"])),
        "method": "pca_pairwise",
        "behavior": {
            "positive": "refusal",
            "negative": "compliance",
            "aggregation": "suffix-only",
            "contrastive_pairs": len(behavior_dataset.formatted_dataset),
            "layer_norms": vector_layer_norms(behavior_vector),
        },
        "condition": {
            "positive": "harmful",
            "negative": "benign",
            "aggregation": "all",
            "contrastive_pairs": len(condition_dataset.formatted_dataset),
            "layer_norms": vector_layer_norms(condition_vector),
        },
        "layer_index_semantics": {
            "vector_extraction": (
                "official batched_get_hiddens reads hidden_states[layer_id + 1], "
                "the output of decoder block layer_id"
            ),
            "official_runtime": (
                "LeashLayer evaluates/adds vectors at the input of decoder block layer_id"
            ),
            "compatibility_choice": "preserve_official_52be602_behavior",
        },
        "training_data_sha256": {
            "calibration_splits.jsonl": sha256_file(
                prepared / "calibration_splits.jsonl"
            ),
            "prefix_pairs.json": sha256_file(prepared / "prefix_pairs.json"),
        },
        "vector_files_sha256": {
            behavior_path.name: sha256_file(behavior_path),
            condition_path.name: sha256_file(condition_path),
        },
        "extraction_config": dict(extraction),
    }
    write_json(destination / "vector_metadata.json", metadata)
    summary = {
        "model": model_config["name"],
        "layers": layer_ids,
        "behavior_vector": str(behavior_path),
        "condition_vector": str(condition_path),
    }
    progress.advance()
    progress.finish()
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract CAST behavior/condition vectors.")
    parser.add_argument(
        "--config",
        default="scripts/baselines/CAST/configs/cast_config.yaml",
    )
    parser.add_argument(
        "--llm-name",
        help="Override model.name and replace the final path component of model.path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    model = select_model(config, args.llm_name)
    print(json.dumps(run(config, model, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
