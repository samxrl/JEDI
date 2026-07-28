from __future__ import annotations

import json
import random
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .config import data_dir, model_root
from .paths import (
    CAST_ROOT,
    REPO_ROOT,
    resolve_repo_path,
    sha256_file,
    sha256_text,
    stable_json_hash,
    write_json,
    write_jsonl,
    write_yaml,
)
from .progress import StageProgress


CAST_UPSTREAM_COMMIT = "52be60235ee309b46c49d6d5877f36e20c52e6ab"


def cast_source_hashes() -> dict[str, str]:
    """Record reproducibility fingerprints for the adapter, pinned third-party source, and configuration."""
    paths: list[Path] = []
    paths.extend(sorted(CAST_ROOT.glob("*.py")))
    paths.extend(sorted((CAST_ROOT / "src").glob("*.py")))
    paths.extend(sorted((CAST_ROOT / "configs").glob("*.yaml")))
    paths.extend(
        sorted(
            (
                CAST_ROOT
                / "third_party"
                / "activation-steering"
                / "activation_steering"
            ).glob("*.py")
        )
    )
    return {
        str(path.relative_to(CAST_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in paths
    }


def normalize_prompt(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.strip().split())


def prompt_hash(value: Any) -> str:
    return sha256_text(normalize_prompt(value))


def _nonempty_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = normalize_prompt(value)
    return text if text else None


def _deduplicate_records(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        digest = str(row["normalized_prompt_hash"])
        if digest in seen:
            continue
        seen.add(digest)
        result.append(dict(row))
    return result


def load_harmful_records(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"The harmful calibration file must be a list or contain records: {resolved}")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        text = _nonempty_text(
            row.get("prompt")
            or row.get("instruction")
            or row.get("query")
            or row.get("behavior")
        )
        if text is None:
            continue
        source = row.get("source", {})
        records.append(
            {
                "sample_id": f"harmful:{index}",
                "source_row_id": index,
                "source": source,
                "prompt": text,
                "normalized_prompt_hash": prompt_hash(text),
                "label": "harmful",
            }
        )
    return _deduplicate_records(records)


def load_benign_records(
    path: str | Path,
    *,
    prompt_field: str = "instruction",
    allowed_category: str | None = "regular",
    excluded_source_substrings: Sequence[str] = ("alpaca_eval",),
) -> list[dict[str, Any]]:
    resolved = resolve_repo_path(path)
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            if allowed_category is not None and str(row.get("category", "")) != allowed_category:
                continue
            source_id = str(row.get("source_id", ""))
            if any(token.lower() in source_id.lower() for token in excluded_source_substrings):
                continue
            text = _nonempty_text(row.get(prompt_field))
            if text is None:
                continue
            records.append(
                {
                    "sample_id": f"benign:{row.get('id', line_number - 1)}",
                    "source_row_id": row.get("id", line_number - 1),
                    "source": {
                        "source_id": source_id,
                        "dataset": row.get("dataset"),
                        "category": row.get("category"),
                    },
                    "prompt": text,
                    "normalized_prompt_hash": prompt_hash(text),
                    "label": "benign",
                }
            )
    return _deduplicate_records(records)


def load_prefix_pairs(
    path: str | Path,
    *,
    refusal_key: str,
    compliance_key: str,
    limit: int,
) -> list[dict[str, str]]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    refusal = payload.get(refusal_key)
    compliance = payload.get(compliance_key)
    if not isinstance(refusal, list) or not isinstance(compliance, list):
        raise KeyError(
            f"{resolved} must contain the lists {refusal_key!r} and {compliance_key!r}."
        )
    count = min(len(refusal), len(compliance), limit)
    if count < limit:
        raise ValueError(f"Insufficient prefix pairs: {limit} required, {count} available.")
    return [
        {
            "pair_id": f"prefix:{index}",
            "refusal": str(refusal[index]),
            "compliance": str(compliance[index]),
        }
        for index in range(count)
    ]


def assign_splits(
    rows: Sequence[Mapping[str, Any]],
    split_sizes: Mapping[str, Any],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    required = sum(int(value) for value in split_sizes.values())
    if len(rows) < required:
        raise ValueError(f"Insufficient available examples: {required} required, {len(rows)} available.")
    shuffled = [dict(row) for row in rows]
    random.Random(seed).shuffle(shuffled)
    result: list[dict[str, Any]] = []
    cursor = 0
    for split, raw_size in split_sizes.items():
        size = int(raw_size)
        for row in shuffled[cursor : cursor + size]:
            row["split"] = str(split)
            result.append(row)
        cursor += size
    return result


def _load_json_rows(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("records", "data", "train"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
    raise TypeError(f"Could not locate a record list in JSON: {path}")


def load_evaluation_hashes(
    evaluation_sets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, set[str]], list[dict[str, Any]], dict[str, str]]:
    hashes_by_dataset: dict[str, set[str]] = {}
    manifest: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for spec in evaluation_sets:
        name = str(spec["name"])
        path = resolve_repo_path(spec["path"])
        fields = [str(field) for field in spec["text_fields"]]
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows: Iterable[Mapping[str, Any]] = pd.read_csv(path).to_dict("records")
        elif suffix == ".json":
            rows = _load_json_rows(path)
        elif suffix == ".jsonl":
            parsed: list[Mapping[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        value = json.loads(line)
                        if isinstance(value, Mapping):
                            parsed.append(value)
            rows = parsed
        else:
            raise ValueError(f"Unsupported evaluation-data format: {path}")

        dataset_hashes: set[str] = set()
        for row_index, row in enumerate(rows):
            for field in fields:
                text = _nonempty_text(row.get(field))
                if text is None:
                    continue
                digest = prompt_hash(text)
                dataset_hashes.add(digest)
                manifest.append(
                    {
                        "dataset": name,
                        "source_row_id": row_index,
                        "text_field": field,
                        "normalized_prompt_hash": digest,
                    }
                )
        hashes_by_dataset[name] = dataset_hashes
        source_hashes[str(path)] = sha256_file(path)
    return hashes_by_dataset, manifest, source_hashes


def build_overlap_report(
    calibration_rows: Sequence[Mapping[str, Any]],
    evaluation_hashes: Mapping[str, set[str]],
) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in calibration_rows:
        by_split[str(row["split"])].add(str(row["normalized_prompt_hash"]))

    split_names = sorted(by_split)
    internal: dict[str, int] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            internal[f"{left}__{right}"] = len(by_split[left] & by_split[right])

    eval_overlap: dict[str, dict[str, int]] = {}
    for split, hashes in sorted(by_split.items()):
        eval_overlap[split] = {
            name: len(hashes & eval_hashes)
            for name, eval_hashes in sorted(evaluation_hashes.items())
        }
    return {
        "internal_split_overlap_counts": internal,
        "evaluation_overlap_counts": eval_overlap,
        "all_internal_disjoint": all(value == 0 for value in internal.values()),
        "all_evaluation_disjoint": all(
            value == 0
            for per_split in eval_overlap.values()
            for value in per_split.values()
        ),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_model_data(
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    data_cfg = config["data"]
    split_sizes = data_cfg.get(
        "split_sizes",
        {
            "vector_train": 60,
            "condition_calibration": 20,
            "steering_validation": 20,
        },
    )
    seed = int(config.get("seed", 42))
    harmful_path = resolve_repo_path(model["harmful_calibration_file"])
    benign_path = resolve_repo_path(data_cfg["benign_file"])
    prefix_path = resolve_repo_path(data_cfg["prefix_file"])

    progress = StageProgress("prepare", 5)
    progress.begin("Load harmful/benign calibration candidates")
    harmful_candidates = load_harmful_records(harmful_path)
    benign_candidates = load_benign_records(
        benign_path,
        prompt_field=str(data_cfg.get("benign_prompt_field", "instruction")),
        allowed_category=data_cfg.get("benign_allowed_category", "regular"),
        excluded_source_substrings=tuple(
            data_cfg.get("benign_excluded_source_substrings", ["alpaca_eval"])
        ),
    )
    progress.advance()

    progress.begin("Scan formal evaluation sets and calculate prompt hashes")
    evaluation_hashes, evaluation_manifest, evaluation_source_hashes = (
        load_evaluation_hashes(data_cfg.get("evaluation_sets", []))
    )
    progress.advance()

    progress.begin("Remove evaluation overlaps and perform prompt-level splitting")
    dropped_overlaps: list[dict[str, Any]] = []
    if bool(data_cfg.get("drop_evaluation_overlaps", True)):
        filtered: dict[str, list[dict[str, Any]]] = {}
        for label, candidates in (
            ("harmful", harmful_candidates),
            ("benign", benign_candidates),
        ):
            kept: list[dict[str, Any]] = []
            for row in candidates:
                digest = str(row["normalized_prompt_hash"])
                matching = sorted(
                    name
                    for name, hashes in evaluation_hashes.items()
                    if digest in hashes
                )
                if matching:
                    dropped_overlaps.append(
                        {
                            "sample_id": row["sample_id"],
                            "label": label,
                            "normalized_prompt_hash": digest,
                            "overlap_datasets": matching,
                        }
                    )
                else:
                    kept.append(row)
            filtered[label] = kept
        harmful_candidates = filtered["harmful"]
        benign_candidates = filtered["benign"]

    harmful = assign_splits(harmful_candidates, split_sizes, seed=seed)
    benign = assign_splits(benign_candidates, split_sizes, seed=seed + 1)
    calibration_rows = harmful + benign
    harmful_hashes = {row["normalized_prompt_hash"] for row in harmful}
    benign_hashes = {row["normalized_prompt_hash"] for row in benign}
    if harmful_hashes & benign_hashes:
        raise ValueError("Normalized prompts overlap between harmful and benign calibration examples.")
    progress.advance()

    progress.begin("Load prefix pairs and complete overlap audit")
    prefixes = load_prefix_pairs(
        prefix_path,
        refusal_key=str(data_cfg.get("refusal_key", "non_compliant_responses")),
        compliance_key=str(data_cfg.get("compliance_key", "compliant_responses")),
        limit=int(data_cfg.get("prefix_limit", 100)),
    )
    overlap = build_overlap_report(calibration_rows, evaluation_hashes)
    if not overlap["all_internal_disjoint"]:
        raise ValueError("Prompts overlap across CAST calibration splits.")
    if bool(data_cfg.get("fail_on_evaluation_overlap", True)) and not overlap[
        "all_evaluation_disjoint"
    ]:
        raise ValueError(
            "CAST calibration data overlaps with formal evaluation data; see the overlap report."
        )
    progress.advance()

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in calibration_rows:
        counts[str(row["split"])][str(row["label"])] += 1
    summary = {
        "model": str(model["name"]),
        "seed": seed,
        "counts": {split: dict(value) for split, value in counts.items()},
        "prefix_pairs": len(prefixes),
        "candidate_counts_after_decontamination": {
            "harmful": len(harmful_candidates),
            "benign": len(benign_candidates),
        },
        "dropped_evaluation_overlap_counts": {
            "harmful": sum(row["label"] == "harmful" for row in dropped_overlaps),
            "benign": sum(row["label"] == "benign" for row in dropped_overlaps),
        },
        "evaluation_unique_prompts": {
            name: len(values) for name, values in evaluation_hashes.items()
        },
        "overlap": overlap,
    }
    if dry_run:
        progress.begin("Complete dry-run summary")
        progress.advance()
        progress.finish()
        return summary

    progress.begin("Write model-specific data and manifest")
    destination = data_dir(config, model)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "calibration_splits.jsonl", calibration_rows)
    write_json(
        destination / "prefix_pairs.json",
        {
            "source": str(prefix_path),
            "source_sha256": sha256_file(prefix_path),
            "pairs": prefixes,
        },
    )
    write_jsonl(destination / "evaluation_manifest.jsonl", evaluation_manifest)
    write_jsonl(
        destination / "dropped_evaluation_overlaps.jsonl",
        dropped_overlaps,
    )
    write_json(destination / "overlap_report.json", overlap)
    source_hashes = {
        str(harmful_path): sha256_file(harmful_path),
        str(benign_path): sha256_file(benign_path),
        str(prefix_path): sha256_file(prefix_path),
        **evaluation_source_hashes,
    }
    write_json(
        destination / "data_manifest.json",
        {
            **summary,
            "jedi_source_commit": _git_commit(),
            "cast_source_commit": CAST_UPSTREAM_COMMIT,
            "source_hashes": source_hashes,
        },
    )
    root = model_root(config, model)
    root.mkdir(parents=True, exist_ok=True)
    resolved_config = dict(config)
    resolved_config["model"] = dict(model)
    resolved_config.pop("models", None)
    write_yaml(root / "resolved_config.yaml", resolved_config)
    write_json(
        root / "run_manifest.json",
        {
            "model": str(model["name"]),
            "jedi_source_commit": _git_commit(),
            "cast_source_commit": CAST_UPSTREAM_COMMIT,
            "resolved_config_sha256": stable_json_hash(resolved_config),
            "cast_adapter_source_hashes": cast_source_hashes(),
            "source_hashes": source_hashes,
        },
    )
    progress.advance()
    progress.finish()
    return summary


def load_prepared_splits(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Invalid calibration row: {value!r}")
                rows.append(value)
    return rows


def prompts_for(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    label: str,
) -> list[str]:
    return [
        str(row["prompt"])
        for row in rows
        if row.get("split") == split and row.get("label") == label
    ]
