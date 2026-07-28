# -*- coding: utf-8 -*-
"""Calibrate the CAST condition layer/threshold/comparator on an independent prompt split."""

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

from src.calibration import (  # noqa: E402
    collect_condition_scores,
    resolve_behavior_layers,
    resolve_candidate_layers,
    search_condition_point,
    threshold_values,
)
from src.config import (  # noqa: E402
    artifact_dir,
    data_dir,
    load_config,
    select_model,
    validate_batch_invariants,
)
from src.data import CAST_UPSTREAM_COMMIT, load_prepared_splits, prompts_for  # noqa: E402
from src.modeling import load_model_and_tokenizer, model_metadata  # noqa: E402
from src.paths import sha256_file, write_json, write_yaml  # noqa: E402
from src.progress import StageProgress  # noqa: E402


def run(
    config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    validate_batch_invariants(config)
    destination = artifact_dir(config, model_config)
    condition_path = destination / "condition_vector.svec"
    behavior_path = destination / "behavior_vector.svec"
    metadata_path = destination / "vector_metadata.json"
    missing = [
        str(path)
        for path in (condition_path, behavior_path, metadata_path)
        if not path.exists()
    ]
    if missing and not dry_run:
        raise FileNotFoundError(f"Run extract_vectors.py first; missing {missing}")
    calibration_cfg = config["calibration"]
    if str(calibration_cfg.get("comparison_mode", "mean")) != "mean":
        raise ValueError("The main CAST configuration requires condition comparison_mode=mean.")
    if dry_run:
        progress = StageProgress("calibrate", 1)
        progress.begin("Validate calibration search space and vector artifacts")
        summary = {
            "model": model_config["name"],
            "condition_vector_exists": condition_path.exists(),
            "threshold_range": calibration_cfg.get("threshold_range", [0.0, 0.1]),
            "threshold_step": calibration_cfg.get("threshold_step", 0.0001),
            "would_load_model": model_config["path"],
        }
        progress.advance()
        progress.finish()
        return summary

    progress = StageProgress("calibrate", 5)
    progress.begin("Load calibration split and condition vector")
    prepared_path = data_dir(config, model_config) / "calibration_splits.jsonl"
    if not prepared_path.exists():
        raise FileNotFoundError("Run prepare_data.py first.")
    rows = load_prepared_splits(prepared_path)
    harmful = prompts_for(
        rows,
        split="condition_calibration",
        label="harmful",
    )
    benign = prompts_for(
        rows,
        split="condition_calibration",
        label="benign",
    )
    if len(harmful) != len(benign) or not harmful:
        raise ValueError("condition_calibration must contain two equal, nonempty prompt classes.")

    condition_vector = SteeringVector.load(str(condition_path))
    progress.advance()

    progress.begin("Load target model and validate candidate layers")
    model, tokenizer = load_model_and_tokenizer(model_config)
    num_layers = int(model.config.num_hidden_layers)
    candidate_layers = resolve_candidate_layers(num_layers, calibration_cfg)
    missing_layers = [
        layer
        for layer in candidate_layers
        if layer not in condition_vector.directions
    ]
    if missing_layers:
        raise KeyError(f"Condition vector is missing candidate layers: {missing_layers}")
    progress.advance()

    progress.begin("Calculate per-prompt condition scores")
    prompts = harmful + benign
    labels = [1] * len(harmful) + [0] * len(benign)
    raw_max_length = calibration_cfg.get("max_prompt_length")
    max_prompt_length = (
        None if raw_max_length is None else int(raw_max_length)
    )
    scores = collect_condition_scores(
        model,
        tokenizer,
        prompts,
        condition_vector.directions,
        candidate_layers,
        max_length=max_prompt_length,
    )
    progress.advance()

    progress.begin("Search layer, threshold, and comparator")
    thresholds = threshold_values(
        calibration_cfg.get("threshold_range", [0.0, 0.1]),
        float(calibration_cfg.get("threshold_step", 0.0001)),
    )
    best, layer_reports = search_condition_point(scores, labels, thresholds)
    behavior_layers = resolve_behavior_layers(num_layers, config["runtime"])
    if best.layer >= min(behavior_layers):
        raise ValueError(
            "The condition layer must precede the first behavior layer so the prefill decision completes first."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        vector_metadata = json.load(handle)
    current_model_metadata = model_metadata(
        model,
        tokenizer,
        str(model_config["name"]),
    )
    if vector_metadata.get("model") != current_model_metadata:
        raise ValueError("Calibration model/tokenizer binding differs from the vector-extraction stage.")
    progress.advance()

    progress.begin("Save condition point and CAST parameters")
    rounded_threshold = round(best.threshold, 3)
    params = {
        "format_version": 1,
        "method": "CAST",
        "cast_source_commit": CAST_UPSTREAM_COMMIT,
        "model_name": str(model_config["name"]),
        "condition_layers": [best.layer],
        # Official find_best_condition_point rounds the threshold to three decimal places before returning.
        "condition_threshold": rounded_threshold,
        "condition_threshold_search_exact": best.threshold,
        "condition_official_direction": best.official_direction,
        "condition_effective_operator": best.effective_operator,
        "condition_comparison_mode": "mean",
        "condition_validation_f1": best.f1,
        "condition_validation_auroc_harmful_positive": best.auroc,
        "behavior_layers": behavior_layers,
        "behavior_strength": float(
            config["runtime"].get("behavior_strength", 1.5)
        ),
        "apply_behavior_on_first_call": bool(
            config["runtime"].get("apply_behavior_on_first_call", True)
        ),
        "use_ooi_preventive_normalization": bool(
            config["runtime"].get("use_ooi_preventive_normalization", False)
        ),
        "evaluation_batch_size": 1,
        "uses_whitening": False,
        "uses_cusum": False,
        "uses_dynamic_strength": False,
    }
    write_yaml(destination / "cast_params.yaml", params)
    report = {
        "best": params,
        "candidate_layers": candidate_layers,
        "threshold_search": {
            "start": float(thresholds[0]),
            "stop_exclusive": float(
                calibration_cfg.get("threshold_range", [0.0, 0.1])[1]
            ),
            "step": float(calibration_cfg.get("threshold_step", 0.0001)),
            "count": int(len(thresholds)),
        },
        "labels": {"harmful": len(harmful), "benign": len(benign)},
        "layers": layer_reports,
        "condition_vector_sha256": sha256_file(condition_path),
        "behavior_vector_sha256": sha256_file(behavior_path),
    }
    write_json(destination / "condition_point.json", report)
    summary = {
        "model": model_config["name"],
        "condition_layer": best.layer,
        "threshold": rounded_threshold,
        "threshold_search_exact": best.threshold,
        "official_direction": best.official_direction,
        "effective_operator": best.effective_operator,
        "f1": best.f1,
        "behavior_layers": behavior_layers,
    }
    progress.advance()
    progress.finish()
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate the CAST condition point.")
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
