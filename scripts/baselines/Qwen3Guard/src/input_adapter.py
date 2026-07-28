from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

from .config import (
    configured_guard_model,
    configured_target_llm,
    input_dir,
    selected_model,
    source_evaluation_dir,
    target_tokenizer_reference,
)
from .paths import (
    REPO_ROOT,
    sha256_file,
    sha256_text,
    stable_json_hash,
    write_json,
    write_jsonl_gz,
    write_yaml,
)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _relative_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def discover_source_files(
    evaluation_root: Path,
    model_name: str,
    *,
    include_safety: bool,
    include_utility: bool,
    attack_methods: Iterable[str] | None = None,
) -> list[Path]:
    model_dir = evaluation_root / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"JEDI evaluation directory not found: {model_dir}")
    files: list[Path] = []
    if include_safety:
        prefix = f"{model_name}_evaluation_detailed_attack_"
        candidates = model_dir.glob(f"{prefix}*.csv")
        allowed = (
            {str(method).casefold() for method in attack_methods}
            if attack_methods is not None
            else None
        )
        files.extend(
            path
            for path in candidates
            if allowed is None
            or path.stem[len(prefix) :].casefold() in allowed
        )
    if include_utility:
        files.extend(
            model_dir.glob(f"{model_name}_evaluation_detailed_utility_*.csv")
        )
    return sorted({path.resolve() for path in files})


def infer_eval_split(path: Path) -> str:
    lowered = path.name.lower()
    if "_detailed_attack_" in lowered:
        return "safety"
    if "_detailed_utility_" in lowered:
        return "utility"
    raise ValueError(f"Unable to infer eval_split from filename: {path}")


def candidate_checkpoint_ends(total_tokens: int, chunk_size: int) -> list[int]:
    """Return pre-release checkpoints for a chunk size, always including the final remainder."""
    total = int(total_tokens)
    size = int(chunk_size)
    if total < 0:
        raise ValueError("total_tokens cannot be negative.")
    if size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if total == 0:
        return []
    ends = list(range(size, total + 1, size))
    if not ends or ends[-1] != total:
        ends.append(total)
    return ends


def build_checkpoint_plan(
    total_tokens: int,
    chunk_sizes: Iterable[int],
) -> tuple[list[int], dict[str, list[int]]]:
    """Build a shared check grid and the strict subsequence for each chunk size."""
    by_chunk = {
        str(int(size)): candidate_checkpoint_ends(total_tokens, int(size))
        for size in chunk_sizes
    }
    union = sorted({end for ends in by_chunk.values() for end in ends})
    return union, by_chunk


def _iter_samples(
    config: Mapping[str, Any],
    manifest: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    from transformers import AutoTokenizer

    model = selected_model(config)
    model_name = str(model["name"])
    tokenizer_reference = target_tokenizer_reference(model)
    revision = model.get("tokenizer_revision") or None
    local_only = bool(model.get("local_files_only", True))
    if local_only and not Path(tokenizer_reference).exists():
        raise FileNotFoundError(f"Missing target tokenizer: {tokenizer_reference}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_reference,
        revision=revision,
        trust_remote_code=bool(model.get("trust_remote_code", True)),
        local_files_only=local_only,
    )
    evaluation_root = source_evaluation_dir(config)
    source_cfg = config["source"]
    files = discover_source_files(
        evaluation_root,
        model_name,
        include_safety=bool(source_cfg.get("include_safety", True)),
        include_utility=bool(source_cfg.get("include_utility", True)),
        attack_methods=source_cfg.get("attack_methods"),
    )
    if not files:
        raise FileNotFoundError(f"Model {model_name} has no detailed evaluation CSVs to replay.")
    if bool(source_cfg.get("include_safety", True)) and not any(
        infer_eval_split(path) == "safety" for path in files
    ):
        configured = source_cfg.get("attack_methods")
        raise FileNotFoundError(
            f"Model {model_name} has no detailed evaluation CSV matching "
            f"the main-experiment attack allowlist {configured}."
        )

    condition = str(source_cfg.get("condition", "baseline"))
    chunk_sizes = [int(value) for value in config["streaming"]["chunk_sizes"]]
    model_count = 0
    total_tokens = 0
    unique_calls = 0
    calls_by_chunk = {str(size): 0 for size in chunk_sizes}
    for source_file in files:
        source_hash = sha256_file(source_file)
        split = infer_eval_split(source_file)
        frame = pd.read_csv(source_file, encoding="utf-8-sig", low_memory=False)
        if "condition" not in frame.columns or "assistant_output" not in frame.columns:
            raise KeyError(
                f"{source_file} is missing the condition or assistant_output column."
            )
        available_conditions = sorted(
            {
                str(value).strip()
                for value in frame["condition"].dropna().unique().tolist()
                if str(value).strip()
            }
        )
        selected = frame.loc[
            frame["condition"].astype(str).str.strip() == condition
        ]
        matched_rows = int(len(selected))
        if selected.empty and bool(
            source_cfg.get("require_baseline_rows", True)
        ):
            raise ValueError(
                f"{source_file} has no rows with condition={condition}."
            )
        limit = int(source_cfg.get("max_samples_per_file", 0))
        if limit > 0:
            selected = selected.head(limit)
        manifest["source_files"].append(
            {
                "path": _relative_source(source_file),
                "sha256_before": source_hash,
                "eval_split": split,
                "available_conditions": available_conditions,
                "matched_rows": matched_rows,
                "selected_rows": int(len(selected)),
            }
        )

        for source_row_index, row in selected.iterrows():
            source_record = {
                str(key): _json_value(value) for key, value in row.to_dict().items()
            }
            source_record["eval_split"] = source_record.get("eval_split") or split
            response = str(source_record.get("assistant_output") or "")
            prompt = str(source_record.get("prompt") or "")
            token_ids = [
                int(value)
                for value in tokenizer.encode(
                    response,
                    add_special_tokens=False,
                )
            ]
            roundtrip = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            checkpoints, by_chunk = build_checkpoint_plan(
                len(token_ids),
                chunk_sizes,
            )
            source_rel = _relative_source(source_file)
            sample_id = stable_json_hash(
                {
                    "target_model": model_name,
                    "source_file": source_rel,
                    "source_row_index": int(source_row_index),
                    "prompt_sha256": sha256_text(prompt),
                    "response_sha256": sha256_text(response),
                }
            )[:24]
            yield {
                "sample_id": sample_id,
                "target_model": model_name,
                "source_file": source_rel,
                "source_file_sha256": source_hash,
                "source_row_index": int(source_row_index),
                "source_record": source_record,
                "source_response": response,
                "source_response_sha256": sha256_text(response),
                "prompt_sha256": sha256_text(prompt),
                "target_token_ids": token_ids,
                "total_target_tokens": len(token_ids),
                "checkpoint_ends": checkpoints,
                "checkpoint_ends_by_chunk": by_chunk,
                "target_tokenizer_path": tokenizer_reference,
                "target_tokenizer_revision": revision,
                "roundtrip_decoded_sha256": sha256_text(roundtrip),
                "roundtrip_exact_match": roundtrip == response,
            }
            model_count += 1
            total_tokens += len(token_ids)
            unique_calls += len(checkpoints)
            for size, ends in by_chunk.items():
                calls_by_chunk[size] += len(ends)

    manifest["source_file_count"] = len(files)
    manifest["models"][model_name] = {
        "sample_count": model_count,
        "total_target_tokens": total_tokens,
        "unique_guard_calls": unique_calls,
        "guard_calls_if_run_separately": calls_by_chunk,
        "target_tokenizer_path": tokenizer_reference,
        "target_tokenizer_revision": revision,
    }
    manifest["sample_count"] = model_count
    manifest["total_target_tokens"] = total_tokens
    manifest["unique_guard_calls"] = unique_calls
    manifest["guard_calls_if_run_separately"] = calls_by_chunk


def prepare_input_snapshot(
    config: Mapping[str, Any],
) -> Path:
    """Read JEDI evaluation CSVs and generate token snapshots inside the Qwen3Guard baseline."""
    destination = input_dir(config)
    destination.mkdir(parents=True, exist_ok=True)
    chunk_sizes = [int(value) for value in config["streaming"]["chunk_sizes"]]
    manifest: dict[str, Any] = {
        "format_version": 2,
        "artifact_layout": "target_model/guard_model",
        "target_model": configured_target_llm(config),
        "guard_model": configured_guard_model(config),
        "prepare_identity": prepare_identity(config),
        "source_condition": config["source"].get("condition", "baseline"),
        "max_samples_per_file": int(
            config["source"].get("max_samples_per_file", 0)
        ),
        "base_checkpoint_stride": int(
            config["streaming"]["base_checkpoint_stride"]
        ),
        "chunk_sizes": chunk_sizes,
        "sample_count": 0,
        "total_target_tokens": 0,
        "unique_guard_calls": 0,
        "guard_calls_if_run_separately": {
            str(size): 0 for size in chunk_sizes
        },
        "models": {},
        "source_files": [],
    }
    samples_path = destination / "samples.jsonl.gz"
    write_jsonl_gz(samples_path, _iter_samples(config, manifest))
    manifest["samples_sha256"] = sha256_file(samples_path)
    write_json(destination / "source_manifest.json", manifest)
    write_yaml(destination / "config.resolved.yaml", dict(config))
    return destination


def call_budget_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, summary in manifest["models"].items():
        sample_count = int(summary["sample_count"])
        unique_calls = int(summary["unique_guard_calls"])
        separate = {
            str(key): int(value)
            for key, value in summary["guard_calls_if_run_separately"].items()
        }
        rows.append(
            {
                "target_model": model_name,
                "sample_count": sample_count,
                "total_target_tokens": int(summary["total_target_tokens"]),
                "unique_shared_guard_calls": unique_calls,
                "separate_guard_calls_total": sum(separate.values()),
                "guard_calls_saved": sum(separate.values()) - unique_calls,
                "mean_shared_guard_calls": (
                    unique_calls / sample_count if sample_count else math.nan
                ),
                **{
                    f"guard_calls_c{size}": count
                    for size, count in separate.items()
                },
            }
        )
    return rows


def prepare_identity(config: Mapping[str, Any]) -> str:
    """Identify configuration that changes input-snapshot content or the checkpoint plan."""
    model = selected_model(config)
    return stable_json_hash(
        {
            "format_version": 2,
            "target_model": configured_target_llm(config),
            "guard_model": configured_guard_model(config),
            "source_evaluation_dir": str(source_evaluation_dir(config)),
            "source": dict(config["source"]),
            "target_tokenizer": {
                "name": model["name"],
                "path": target_tokenizer_reference(model),
                "revision": model.get("tokenizer_revision") or None,
                "trust_remote_code": bool(
                    model.get("trust_remote_code", True)
                ),
                "local_files_only": bool(
                    model.get("local_files_only", True)
                ),
            },
            "streaming": {
                "base_checkpoint_stride": int(
                    config["streaming"]["base_checkpoint_stride"]
                ),
                "chunk_sizes": [
                    int(value)
                    for value in config["streaming"]["chunk_sizes"]
                ],
            },
        }
    )
