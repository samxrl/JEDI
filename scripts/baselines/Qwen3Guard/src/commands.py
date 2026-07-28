from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm

from .backend import build_guard_backend
from .config import (
    configured_target_llm,
    configured_guard_model,
    guard_slug,
    input_dir,
    run_dir,
    selected_model,
    target_tokenizer_reference,
)
from .evaluation import (
    atomic_to_csv,
    export_run,
    judge_run,
    split_detail_outputs,
    summarize_run,
)
from .input_adapter import (
    call_budget_rows,
    prepare_identity,
    prepare_input_snapshot,
)
from .materializer import materialize_sample
from .paths import (
    REPO_ROOT,
    append_jsonl,
    load_yaml,
    read_jsonl,
    read_jsonl_gz,
    require_local_write,
    sha256_file,
    stable_json_hash,
    write_json,
    write_jsonl_gz,
    write_yaml,
)
from .scorer import expected_score_keys, score_key, score_prepared_samples


def _source_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def verify_source_hashes(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for source in manifest["source_files"]:
        path = _source_path(str(source["path"]))
        actual = sha256_file(path)
        expected = str(source["sha256_before"])
        checks.append(
            {
                "path": str(source["path"]),
                "sha256_before": expected,
                "sha256_after": actual,
                "unchanged": actual == expected,
            }
        )
    return checks


def prepare_command(
    config: Mapping[str, Any],
) -> Path:
    destination = input_dir(config)
    manifest_path = destination / "source_manifest.json"
    samples_path = destination / "samples.jsonl.gz"
    if manifest_path.exists() or samples_path.exists():
        if not manifest_path.is_file() or not samples_path.is_file():
            raise RuntimeError(
                "The current stable input directory contains incomplete prepare artifacts; "
                "inspect this target-LLM/guard-model directory first."
            )
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest.get("prepare_identity") != prepare_identity(
            config
        ):
            raise RuntimeError(
                "The current target-LLM/guard-model input directory is bound to different prepare settings; "
                "restore the original settings or remove old inputs and run artifacts for this combination before retrying."
            )
        if (
            sha256_file(samples_path)
            != existing_manifest.get("samples_sha256")
        ):
            raise RuntimeError("Existing samples.jsonl.gz does not match the manifest hash.")
        if not all(
            item["unchanged"]
            for item in verify_source_hashes(existing_manifest)
        ):
            raise RuntimeError(
                "JEDI source evaluation files have changed relative to the existing stable snapshot; "
                "handle old inputs and run artifacts for this combination first."
            )

    destination = prepare_input_snapshot(config)
    manifest_path = destination / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    after_checks = verify_source_hashes(manifest)
    by_path = {item["path"]: item for item in after_checks}
    for source in manifest["source_files"]:
        check = by_path[source["path"]]
        source["sha256_after"] = check["sha256_after"]
        source["unchanged"] = check["unchanged"]
    if not all(item["unchanged"] for item in after_checks):
        raise RuntimeError("A JEDI source evaluation file changed during prepare.")
    write_json(manifest_path, manifest)
    atomic_to_csv(
        pd.DataFrame(call_budget_rows(manifest)),
        destination / "call_budget.csv",
    )
    return destination


def _load_prepared(
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    prepared = input_dir(config)
    manifest_path = prepared / "source_manifest.json"
    samples_path = prepared / "samples.jsonl.gz"
    if not manifest_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError(
            "Prepare artifacts are missing for the current target-LLM/guard-model combination; "
            "run prepare first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_model = configured_target_llm(config)
    expected_guard = configured_guard_model(config)
    prepared_models = set(manifest.get("models", {}))
    if prepared_models != {expected_model}:
        raise RuntimeError(
            f"The prepare snapshot targets {sorted(prepared_models)}, "
            f"but this run uses --target-llm {expected_model}."
        )
    if manifest.get("target_model") != expected_model:
        raise RuntimeError("target_model in the prepare manifest does not match the current path.")
    if manifest.get("guard_model") != expected_guard:
        raise RuntimeError("guard_model in the prepare manifest does not match the current path.")
    if manifest.get("prepare_identity") != prepare_identity(config):
        raise RuntimeError("The prepare manifest does not match the current input-adapter settings.")
    if sha256_file(samples_path) != manifest["samples_sha256"]:
        raise RuntimeError("samples.jsonl.gz does not match the source_manifest.json hash.")
    source_checks = verify_source_hashes(manifest)
    if not all(item["unchanged"] for item in source_checks):
        raise RuntimeError("JEDI source evaluation files have changed since prepare.")
    samples = read_jsonl_gz(samples_path)
    if len(samples) != int(manifest["sample_count"]):
        raise RuntimeError("The sample count in samples.jsonl.gz does not match the manifest.")
    actual_models = {str(sample["target_model"]) for sample in samples}
    if actual_models and actual_models != {expected_model}:
        raise RuntimeError(
            f"The prepare snapshot contains incorrect target models: {sorted(actual_models)}."
        )
    return prepared, manifest, samples


def _load_target_tokenizer(config: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model = selected_model(config)
    reference = target_tokenizer_reference(model)
    if bool(model.get("local_files_only", True)) and not Path(reference).exists():
        raise FileNotFoundError(f"Missing target tokenizer: {reference}")
    return AutoTokenizer.from_pretrained(
        reference,
        revision=model.get("tokenizer_revision") or None,
        trust_remote_code=bool(model.get("trust_remote_code", True)),
        local_files_only=bool(model.get("local_files_only", True)),
    )


def _score_identity(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    """Identify inputs and guard configuration that change prefix-classification semantics."""
    return stable_json_hash(
        {
            "samples_sha256": manifest["samples_sha256"],
            "target_model": configured_target_llm(config),
            "guard": dict(config["guard"]),
            "streaming": {
                "prefix_mode": config["streaming"]["prefix_mode"],
                "include_user_prompt": config["streaming"][
                    "include_user_prompt"
                ],
            },
        }
    )


def _legacy_score_identity(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    """Reproduce the legacy hash that included performance settings in score identity for artifact migration."""
    return stable_json_hash(
        {
            "samples_sha256": manifest["samples_sha256"],
            "target_model": configured_target_llm(config),
            "guard": dict(config["guard"]),
            "streaming": {
                "prefix_mode": config["streaming"]["prefix_mode"],
                "include_user_prompt": config["streaming"][
                    "include_user_prompt"
                ],
            },
            "execution": {
                "batch_size": int(
                    config["execution"].get("batch_size", 1)
                ),
                "store_raw_guard_outputs": bool(
                    config["execution"].get(
                        "store_raw_guard_outputs",
                        True,
                    )
                ),
            },
        }
    )


def _score_identity_matches(
    stored_identity: Any,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> bool:
    """Support legacy identity while strictly comparing configuration that truly affects labels."""
    expected = _score_identity(config, manifest)
    if stored_identity == expected:
        return True
    resolved_path = root / "config.resolved.yaml"
    if not resolved_path.is_file():
        return False
    previous = load_yaml(resolved_path)
    return (
        stored_identity == _legacy_score_identity(previous, manifest)
        and _score_identity(previous, manifest) == expected
    )


def _read_existing_scores(
    final_path: Path,
    partial_path: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if final_path.is_file():
        for row in read_jsonl_gz(final_path):
            rows[score_key(row)] = row
    if partial_path.is_file():
        for row in read_jsonl(partial_path):
            rows[score_key(row)] = row
    return rows


def score_prefixes_command(
    config: Mapping[str, Any],
) -> Path:
    """Score the shared cumulative-prefix grid; partial JSONL supports batch-level resumption."""
    _, input_manifest, samples = _load_prepared(config)
    root = run_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    raw_dir = require_local_write(root / "raw")
    final_path = raw_dir / "prefix_scores.jsonl.gz"
    partial_path = raw_dir / "prefix_scores.partial.jsonl"
    score_manifest_path = root / "score_manifest.json"
    identity = _score_identity(config, input_manifest)
    if score_manifest_path.is_file():
        old_manifest = json.loads(
            score_manifest_path.read_text(encoding="utf-8")
        )
        if not _score_identity_matches(
            old_manifest.get("score_identity"),
            config,
            input_manifest,
            root=root,
        ):
            raise RuntimeError(
                "The current target-LLM/guard-model directory contains an incompatible scoring configuration; "
                "restore the original settings or remove old run artifacts for this combination before retrying."
            )
    else:
        if final_path.exists() or partial_path.exists():
            raise RuntimeError(
                "Found orphaned score files without score_manifest.json; "
                "handle unknown artifacts in the current target-LLM/guard-model directory first."
            )
    write_yaml(root / "config.resolved.yaml", dict(config))
    if not score_manifest_path.is_file():
        write_json(
            score_manifest_path,
            {
                "format_version": 3,
                "artifact_layout": "target_model/guard_model",
                "target_model": configured_target_llm(config),
                "guard_model": configured_guard_model(config),
                "score_identity_version": 2,
                "score_identity": identity,
                "samples_sha256": input_manifest["samples_sha256"],
                "status": "in_progress",
                "score_execution": {
                    "configured_batch_size": int(
                        config["execution"].get("batch_size", 1)
                    ),
                    "store_raw_guard_outputs": bool(
                        config["execution"].get(
                            "store_raw_guard_outputs",
                            True,
                        )
                    ),
                },
            },
        )

    expected = expected_score_keys(samples)
    existing = _read_existing_scores(final_path, partial_path)
    unexpected = set(existing) - expected
    if unexpected:
        raise RuntimeError(
            f"Existing scores contain {len(unexpected)} checkpoints outside the current snapshot."
        )
    completed = set(existing)
    if completed != expected:
        tokenizer = _load_target_tokenizer(config)
        backend = build_guard_backend(config)
        progress = tqdm(
            total=len(expected),
            initial=len(completed),
            desc=(
                f"{configured_guard_model(config)} "
                f"{configured_target_llm(config)} prefixes"
            ),
            unit="checkpoint",
            dynamic_ncols=True,
            disable=(
                None
                if bool(
                    config["execution"].get("show_progress", True)
                )
                else True
            ),
        )

        def persist_batch(rows: list[dict[str, Any]]) -> None:
            append_jsonl(partial_path, rows)
            progress.update(len(rows))

        try:
            new_rows = score_prepared_samples(
                samples,
                tokenizer=tokenizer,
                backend=backend,
                batch_size=int(config["execution"].get("batch_size", 1)),
                completed=completed,
                error_policy=str(
                    config["guard"].get(
                        "backend_error_policy",
                        "fail_closed",
                    )
                ),
                store_raw_outputs=bool(
                    config["execution"].get(
                        "store_raw_guard_outputs",
                        True,
                    )
                ),
                guard_model=configured_guard_model(config),
                on_batch=persist_batch,
            )
        finally:
            progress.close()
            backend.close()
        for row in new_rows:
            existing[score_key(row)] = row

    if set(existing) != expected:
        missing = expected - set(existing)
        raise RuntimeError(f"Prefix scores are still missing {len(missing)} checkpoints.")
    sample_order = {
        str(sample["sample_id"]): index
        for index, sample in enumerate(samples)
    }
    ordered = sorted(
        existing.values(),
        key=lambda row: (
            sample_order[str(row["sample_id"])],
            int(row["checkpoint_end"]),
        ),
    )
    write_jsonl_gz(final_path, ordered)
    source_checks = verify_source_hashes(input_manifest)
    if not all(item["unchanged"] for item in source_checks):
        raise RuntimeError("A JEDI source evaluation file changed during scoring.")
    write_json(
        score_manifest_path,
        {
            "format_version": 3,
            "artifact_layout": "target_model/guard_model",
            "target_model": configured_target_llm(config),
            "guard_model": configured_guard_model(config),
            "score_identity_version": 2,
            "score_identity": identity,
            "samples_sha256": input_manifest["samples_sha256"],
            "status": "completed",
            "score_count": len(ordered),
            "expected_score_count": len(expected),
            "guard_error_count": sum(
                1 for row in ordered if row.get("guard_error")
            ),
            "score_execution": {
                "configured_batch_size": int(
                    config["execution"].get("batch_size", 1)
                ),
                "observed_batch_sizes": sorted(
                    {
                        int(row.get("batch_size") or 1)
                        for row in ordered
                    }
                ),
                "store_raw_guard_outputs": bool(
                    config["execution"].get(
                        "store_raw_guard_outputs",
                        True,
                    )
                ),
            },
            "prefix_scores_sha256": sha256_file(final_path),
            "source_hash_checks": source_checks,
        },
    )
    return final_path


def _load_scores_for_materialization(
    config: Mapping[str, Any],
    root: Path,
    input_manifest: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    final_path = root / "raw" / "prefix_scores.jsonl.gz"
    score_manifest_path = root / "score_manifest.json"
    if not final_path.is_file() or not score_manifest_path.is_file():
        raise FileNotFoundError(
            "score-prefixes artifacts are missing; complete prefix scoring first."
        )
    score_manifest = json.loads(
        score_manifest_path.read_text(encoding="utf-8")
    )
    if not _score_identity_matches(
        score_manifest.get("score_identity"),
        config,
        input_manifest,
        root=root,
    ):
        raise RuntimeError("Prefix scores do not match the current inputs or configuration.")
    if sha256_file(final_path) != score_manifest["prefix_scores_sha256"]:
        raise RuntimeError("The prefix_scores.jsonl.gz hash does not match the manifest.")
    rows = read_jsonl_gz(final_path)
    expected = expected_score_keys(samples)
    actual = {score_key(row) for row in rows}
    if actual != expected:
        raise RuntimeError(
            f"Prefix-score keys are incomplete: {len(expected - actual)} missing, "
            f"{len(actual - expected)} extra."
        )
    # Zero-token responses have no checkpoints and produce no score rows. Preserve
    # an empty mapping so materialization can handle them as zero checks and release unchanged.
    grouped: dict[str, dict[int, dict[str, Any]]] = {
        str(sample["sample_id"]): {} for sample in samples
    }
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), {})[
            int(row["checkpoint_end"])
        ] = row
    return grouped


def materialize_command(
    config: Mapping[str, Any],
    *,
    chunk_sizes: Sequence[int] | None = None,
) -> Path:
    _, input_manifest, samples = _load_prepared(config)
    root = run_dir(config)
    scores = _load_scores_for_materialization(
        config,
        root,
        input_manifest,
        samples,
    )
    tokenizer = _load_target_tokenizer(config)
    sizes = [
        int(value)
        for value in (chunk_sizes or config["streaming"]["chunk_sizes"])
    ]
    configured_sizes = {
        int(value) for value in config["streaming"]["chunk_sizes"]
    }
    unknown = sorted(set(sizes) - configured_sizes)
    if unknown:
        raise ValueError(f"The following chunk sizes were not generated during prepare: {unknown}")

    defended_dir = require_local_write(root / "defended")
    raw_dir = require_local_write(root / "raw")
    for chunk_size in sizes:
        results = [
            materialize_sample(
                sample,
                scores[str(sample["sample_id"])],
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                refusal_text=str(config["streaming"]["refusal_text"]),
                condition_prefix=str(
                    guard_slug(config)
                ),
                guard_model=configured_guard_model(config),
            )
            for sample in samples
        ]
        frame = pd.DataFrame(results)
        atomic_to_csv(
            frame,
            defended_dir / f"detailed_c{chunk_size}.csv",
        )
        split_detail_outputs(
            frame,
            defended_dir,
            chunk_size=chunk_size,
            stage="evaluation",
            guard_name=guard_slug(config),
        )
        write_jsonl_gz(
            raw_dir / f"sample_results_c{chunk_size}.jsonl.gz",
            results,
        )

    source_checks = verify_source_hashes(input_manifest)
    if not all(item["unchanged"] for item in source_checks):
        raise RuntimeError("A JEDI source evaluation file changed during materialization.")
    write_json(
        root / "manifest.json",
        {
            "format_version": 2,
            "artifact_layout": "target_model/guard_model",
            "baseline": dict(config["baseline"]),
            "target_model": configured_target_llm(config),
            "guard_model": configured_guard_model(config),
            "samples_sha256": input_manifest["samples_sha256"],
            "prefix_scores_sha256": sha256_file(
                root / "raw" / "prefix_scores.jsonl.gz"
            ),
            "sample_count": len(samples),
            "chunk_sizes": sizes,
            "source_hash_checks": source_checks,
        },
    )
    return root


def validate_command(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = input_dir(config)
    root = run_dir(config)
    report: dict[str, Any] = {
        "config_valid": True,
        "target_model": configured_target_llm(config),
        "guard_model": configured_guard_model(config),
        "input_dir": str(prepared),
        "run_dir": str(root),
        "errors": [],
    }
    if not prepared.exists():
        report["status"] = "config_only"
        return report
    try:
        _, manifest, samples = _load_prepared(config)
    except Exception as exc:
        report["errors"].append(f"Prepare validation failed: {type(exc).__name__}: {exc}")
        report["status"] = "failed"
        return report

    samples_by_id = {
        str(sample["sample_id"]): sample for sample in samples
    }
    report["source_hash_checks"] = verify_source_hashes(manifest)
    score_path = root / "raw" / "prefix_scores.jsonl.gz"
    if score_path.is_file():
        score_rows = read_jsonl_gz(score_path)
        expected = expected_score_keys(samples)
        actual = {score_key(row) for row in score_rows}
        if actual != expected:
            report["errors"].append(
                f"Prefix-score keys are incomplete: {len(expected - actual)} missing, "
                f"{len(actual - expected)} extra."
            )
        if any(row.get("guard_error") for row in score_rows):
            report["errors"].append("Prefix scores contain guard_error.")
    else:
        report["errors"].append("prefix_scores.jsonl.gz is missing.")

    run_manifest_path = root / "manifest.json"
    expected_sizes = [
        int(value) for value in config["streaming"]["chunk_sizes"]
    ]
    if run_manifest_path.is_file():
        run_manifest = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        expected_sizes = [
            int(value) for value in run_manifest.get("chunk_sizes", [])
        ]
    for chunk_size in expected_sizes:
        detail_path = root / "defended" / f"detailed_c{chunk_size}.csv"
        if not detail_path.is_file():
            report["errors"].append(
                f"Detailed defended results for c{chunk_size} are missing."
            )
            continue
        frame = pd.read_csv(
            detail_path,
            encoding="utf-8-sig",
            low_memory=False,
        )
        if len(frame) != len(samples):
            report["errors"].append(
                f"c{chunk_size} sample count {len(frame)} != {len(samples)}."
            )
        if frame["sample_id"].duplicated().any():
            report["errors"].append(f"c{chunk_size} contains duplicate sample_id values.")
        actual_guards = {
            str(value)
            for value in frame.get(
                "guard_model",
                pd.Series(dtype=str),
            ).dropna().unique().tolist()
        }
        expected_guard = configured_guard_model(config)
        if actual_guards != {expected_guard}:
            report["errors"].append(
                f"c{chunk_size} guard models {sorted(actual_guards)} "
                f"!= {expected_guard}."
            )
        for row in frame.to_dict(orient="records"):
            sample = samples_by_id.get(str(row.get("sample_id")))
            if sample is None:
                report["errors"].append(
                    f"c{chunk_size} contains unknown sample_id={row.get('sample_id')}."
                )
                continue
            triggered = str(row.get("triggered")).strip().lower() == "true"
            if not triggered:
                if (
                    str(row.get("protected_output_sha256"))
                    != sample["source_response_sha256"]
                ):
                    report["errors"].append(
                        f"{row['sample_id']} did not trigger, but its output changed."
                    )
            else:
                candidates = [
                    int(value)
                    for value in sample["checkpoint_ends_by_chunk"][
                        str(chunk_size)
                    ]
                ]
                trigger_end = int(row["trigger_token_end"])
                if trigger_end not in candidates:
                    report["errors"].append(
                        f"{row['sample_id']} has a trigger point outside the c{chunk_size} grid."
                    )
                else:
                    offset = candidates.index(trigger_end)
                    expected_start = candidates[offset - 1] if offset else 0
                    if int(row["released_token_count"]) != expected_start:
                        report["errors"].append(
                            f"{row['sample_id']} has an invalid released_token_count."
                        )
                    if int(row["trigger_token_start"]) != expected_start:
                        report["errors"].append(
                            f"{row['sample_id']} has an invalid trigger_token_start."
                        )
            reused = (
                str(row.get("source_label_reused")).strip().lower() == "true"
            )
            if reused and (
                str(row.get("protected_output_sha256"))
                != str(row.get("source_response_sha256"))
            ):
                report["errors"].append(
                    f"{row['sample_id']} incorrectly reused a label after its output changed."
                )
    report["status"] = "passed" if not report["errors"] else "failed"
    if root.exists():
        write_json(root / "validation_report.json", report)
    return report


__all__ = [
    "export_run",
    "judge_run",
    "materialize_command",
    "prepare_command",
    "score_prefixes_command",
    "summarize_run",
    "validate_command",
]
