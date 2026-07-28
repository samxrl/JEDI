# -*- coding: utf-8 -*-
"""Evaluate CAST with JEDI's data and metrics while writing only inside CAST."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
JEDI_SRC_DIR = REPO_ROOT / "src"
JEDI_SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.paths import configure_local_environment  # noqa: E402

configure_local_environment()

for path in (JEDI_SRC_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(JEDI_SCRIPTS_DIR) in sys.path:
    sys.path.remove(str(JEDI_SCRIPTS_DIR))
sys.path.insert(0, str(JEDI_SCRIPTS_DIR))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import GenerationConfig  # noqa: E402
from transformers.generation.streamers import BaseStreamer  # noqa: E402

main_eval_path = JEDI_SCRIPTS_DIR / "run_evaluation.py"
main_eval_spec = importlib.util.spec_from_file_location(
    "jedi_main_evaluation_for_cast",
    main_eval_path,
)
if main_eval_spec is None or main_eval_spec.loader is None:
    raise ImportError(f"Unable to load JEDI's main evaluation script: {main_eval_path}")
main_eval = importlib.util.module_from_spec(main_eval_spec)
main_eval_spec.loader.exec_module(main_eval)

from src.config import (  # noqa: E402
    artifact_dir,
    load_config,
    result_dir,
    select_model,
    validate_batch_invariants,
)
from src.data import prompt_hash  # noqa: E402
from src.modeling import model_input_device, resolve_model_reference  # noqa: E402
from src.paths import (  # noqa: E402
    require_cast_path,
    resolve_repo_path,
    stable_json_hash,
    write_json,
)
from src.progress import StageProgress  # noqa: E402
from src.runtime import CASTGuard  # noqa: E402


load_model_and_tokenizer = main_eval.load_model_and_tokenizer
load_utility_dataset = main_eval.load_utility_dataset
load_safety_dataset = main_eval.load_safety_dataset
run_classification = main_eval.run_classification
calculate_metrics = main_eval.calculate_metrics
should_compute_keyword_fpr = main_eval.should_compute_keyword_fpr
annotate_keyword_false_positives = main_eval.annotate_keyword_false_positives
calculate_keyword_fpr_metrics = main_eval.calculate_keyword_fpr_metrics
build_generation_config = main_eval.build_generation_config
get_override_max_new_tokens = main_eval.get_override_max_new_tokens
apply_chat_template_compat = main_eval.apply_chat_template_compat


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
EVALUATION_RUNTIME_VERSION = 1


def clear_runtime_memory(context: str) -> None:
    logger.info("Releasing runtime memory: %s", context)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FirstTokenTimer(BaseStreamer):
    """Ignore the prompt submitted by generate and record TTFT when the first new token arrives."""

    def __init__(self, started_at: float) -> None:
        self.started_at = started_at
        self.put_calls = 0
        self.ttft_ms: float | None = None

    def put(self, value: Any) -> None:
        self.put_calls += 1
        if self.put_calls == 2 and self.ttft_ms is None:
            cuda_synchronize()
            self.ttft_ms = (time.perf_counter() - self.started_at) * 1000.0

    def end(self) -> None:
        return None


def build_base_generation_config(
    evaluation_config: Mapping[str, Any],
    tokenizer: Any,
) -> GenerationConfig:
    kwargs = dict(evaluation_config.get("generation_kwargs", {}))
    if bool(kwargs.pop("do_sample", False)):
        raise ValueError("Formal CAST evaluation requires do_sample=false.")
    if int(kwargs.pop("num_beams", 1)) != 1:
        raise ValueError("The current CAST runtime requires num_beams=1.")
    config = GenerationConfig(
        max_new_tokens=int(kwargs.pop("max_new_tokens", 256)),
        do_sample=False,
        num_beams=1,
        **kwargs,
    )
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    return config


def atomic_to_csv(df: pd.DataFrame, path: str | Path) -> Path:
    target = require_cast_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(target)
    return target


def prepare_detailed_output(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "trigger_step" in output.columns:
        output["trigger_step"] = output["trigger_step"].apply(
            lambda value: pd.NA
            if pd.isna(value) or int(float(value)) < 0
            else int(float(value))
        ).astype("Int64")
    return output


def utility_detail_path(output_dir: Path, model_name: str, dataset: str) -> Path:
    return output_dir / f"{model_name}_cast_evaluation_detailed_utility_{dataset}.csv"


def safety_detail_path(output_dir: Path, model_name: str, attack: str) -> Path:
    return output_dir / f"{model_name}_cast_evaluation_detailed_attack_{attack}.csv"


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def split_resume_rows(
    source_df: pd.DataFrame,
    detail_path: Path,
    *,
    artifact_fingerprint: str,
    runtime_config_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not detail_path.exists():
        return _empty_like(source_df), source_df.reset_index(drop=True)
    try:
        existing = pd.read_csv(detail_path)
    except Exception as exc:
        logger.warning("Ignoring unreadable checkpoint %s: %s", detail_path, exc)
        return _empty_like(source_df), source_df.reset_index(drop=True)
    required = {
        "prompt",
        "assistant_output",
        "cast_artifact_fingerprint",
        "cast_runtime_config_sha256",
    }
    if not required.issubset(existing.columns):
        logger.warning("Ignoring legacy or incomplete checkpoint: %s", detail_path)
        return _empty_like(source_df), source_df.reset_index(drop=True)
    existing = existing[
        (existing["cast_artifact_fingerprint"].astype(str) == artifact_fingerprint)
        & (
            existing["cast_runtime_config_sha256"].astype(str)
            == runtime_config_sha256
        )
        & existing["assistant_output"].notna()
    ].copy()
    rows_by_hash: dict[str, list[int]] = {}
    for index, row in existing.iterrows():
        digest = str(row.get("prompt_hash") or prompt_hash(row["prompt"]))
        rows_by_hash.setdefault(digest, []).append(index)
    completed_indices: list[int] = []
    pending_indices: list[int] = []
    for index, row in source_df.iterrows():
        digest = prompt_hash(row["prompt"])
        candidates = rows_by_hash.get(digest)
        if candidates:
            completed_indices.append(candidates.pop(0))
        else:
            pending_indices.append(index)
    completed = (
        existing.loc[completed_indices].reset_index(drop=True)
        if completed_indices
        else _empty_like(source_df)
    )
    pending = source_df.loc[pending_indices].reset_index(drop=True)
    logger.info(
        "Checkpoint %s: %d complete, %d remaining.",
        detail_path,
        len(completed),
        len(pending),
    )
    return completed, pending


def save_alpaca_format(df: pd.DataFrame, path: Path, generator: str) -> None:
    records = [
        {
            "dataset": row.get("dataset", "alpaca_eval"),
            "instruction": row.get("instruction", row.get("prompt", "")),
            "output": row.get("assistant_output", ""),
            "generator": generator,
        }
        for _, row in df.iterrows()
    ]
    write_json(path, records)


def save_xstest_format(df: pd.DataFrame, path: Path) -> None:
    output = pd.DataFrame()
    output["id"] = df["id"] if "id" in df.columns else range(1, len(df) + 1)
    output["type"] = df["type"] if "type" in df.columns else "N/A"
    output["prompt"] = df["prompt"]
    output["completion"] = df["assistant_output"]
    for column in ("annotation_1", "annotation_2", "agreement", "final_label"):
        output[column] = None
    atomic_to_csv(
        output[
            [
                "id",
                "type",
                "prompt",
                "completion",
                "annotation_1",
                "annotation_2",
                "agreement",
                "final_label",
            ]
        ],
        path,
    )


def save_utility_files(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    dataset_name: str,
) -> Path:
    detail = prepare_detailed_output(df)
    path = atomic_to_csv(
        detail,
        utility_detail_path(output_dir, model_name, dataset_name),
    )
    if dataset_name == "alpaca_eval":
        save_alpaca_format(
            detail,
            output_dir / f"{model_name}-alpaca_eval-CAST.json",
            f"{model_name}-CAST",
        )
    if "xstest" in dataset_name.lower():
        save_xstest_format(
            detail,
            output_dir / f"{model_name}_xstest_CAST.csv",
        )
    return path


def save_safety_file(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    attack: str,
) -> Path:
    return atomic_to_csv(
        prepare_detailed_output(df),
        safety_detail_path(output_dir, model_name, attack),
    )


def _parse_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return (
        df[column]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def condition_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    trigger = _parse_bool_series(df, "cast_condition_met")
    scores = pd.to_numeric(
        df.get("cast_condition_score", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    ).dropna()
    result: dict[str, Any] = {
        "total": int(total),
        "condition_trigger_count": int(trigger.sum()),
        "condition_trigger_rate": float(trigger.mean()) if total else 0.0,
        "condition_score_mean": float(scores.mean()) if not scores.empty else None,
        "condition_score_median": float(scores.median()) if not scores.empty else None,
    }
    ttft = pd.to_numeric(
        df.get("ttft_ms", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    ).dropna()
    result["ttft_ms_median"] = float(ttft.median()) if not ttft.empty else None
    result["ttft_ms_p95"] = float(ttft.quantile(0.95)) if not ttft.empty else None
    if "label" in df.columns:
        harmful = df["label"].astype(str).str.lower().eq("yes")
        for name, mask in (
            ("triggered", trigger),
            ("not_triggered", ~trigger),
        ):
            subset_size = int(mask.sum())
            result[f"dsr_among_{name}"] = (
                float(1.0 - harmful[mask].mean()) if subset_size else None
            )
    return result


def aggregate_evaluation_metrics(
    summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Aggregate benign FPR and safety DSR by sample count so macro averages do not hide size differences."""
    utility_metrics = list(summary.get("utility", {}).values())
    benign_total = sum(int(metric.get("total", 0)) for metric in utility_metrics)
    false_positive_count = sum(
        int(
            metric.get(
                "false_positive_count",
                metric.get("harmful_count", 0),
            )
        )
        for metric in utility_metrics
    )
    benign_trigger_count = sum(
        int(metric.get("trigger_count", 0)) for metric in utility_metrics
    )
    benign_fpr = (
        false_positive_count / benign_total if benign_total else 0.0
    )

    safety_metrics = list(summary.get("safety", {}).values())
    safety_total = sum(int(metric.get("total", 0)) for metric in safety_metrics)
    harmful_count = sum(
        int(metric.get("harmful_count", 0)) for metric in safety_metrics
    )
    safety_trigger_count = sum(
        int(metric.get("trigger_count", 0)) for metric in safety_metrics
    )
    safety_asr = harmful_count / safety_total if safety_total else 0.0
    return {
        "benign": {
            "metric": "FPR",
            "fpr": benign_fpr,
            "rate": benign_fpr,
            "false_positive_count": false_positive_count,
            "total": benign_total,
            "trigger_count": benign_trigger_count,
            "trigger_rate": (
                benign_trigger_count / benign_total if benign_total else 0.0
            ),
        },
        "safety": {
            "metric": "DSR",
            "dsr": 1.0 - safety_asr,
            "asr": safety_asr,
            "harmful_count": harmful_count,
            "total": safety_total,
            "trigger_count": safety_trigger_count,
            "trigger_rate": (
                safety_trigger_count / safety_total if safety_total else 0.0
            ),
        },
    }


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def terminal_metric_lines(summary: Mapping[str, Any]) -> list[str]:
    """Build stable, testable terminal text printed after evaluation."""
    model_name = str(summary.get("model", "unknown"))
    lines = [f"=== CAST Final Evaluation Metrics | {model_name} ==="]
    lines.append("[Benign Datasets: FPR]")
    for dataset_name, metrics in summary.get("utility", {}).items():
        false_positives = int(
            metrics.get(
                "false_positive_count",
                metrics.get("harmful_count", 0),
            )
        )
        total = int(metrics.get("total", 0))
        lines.append(
            f"{dataset_name}: FPR={_format_percent(metrics.get('fpr', metrics.get('rate')))} "
            f"({false_positives}/{total}), "
            f"Trigger={_format_percent(metrics.get('trigger_rate'))}"
        )
    aggregate = summary.get("aggregate") or aggregate_evaluation_metrics(summary)
    benign = aggregate["benign"]
    lines.append(
        "Benign Overall: "
        f"FPR={_format_percent(benign.get('fpr'))} "
        f"({int(benign.get('false_positive_count', 0))}/"
        f"{int(benign.get('total', 0))}), "
        f"Trigger={_format_percent(benign.get('trigger_rate'))}"
    )

    lines.append("[Safety Datasets: DSR]")
    for attack_name, metrics in summary.get("safety", {}).items():
        harmful = int(metrics.get("harmful_count", 0))
        total = int(metrics.get("total", 0))
        lines.append(
            f"{attack_name}: DSR={_format_percent(metrics.get('dsr'))}, "
            f"ASR={_format_percent(metrics.get('asr', metrics.get('rate')))} "
            f"({harmful}/{total}), "
            f"Trigger={_format_percent(metrics.get('trigger_rate'))}"
        )
    safety = aggregate["safety"]
    lines.append(
        "Safety Overall: "
        f"DSR={_format_percent(safety.get('dsr'))}, "
        f"ASR={_format_percent(safety.get('asr'))} "
        f"({int(safety.get('harmful_count', 0))}/"
        f"{int(safety.get('total', 0))}), "
        f"Trigger={_format_percent(safety.get('trigger_rate'))}"
    )
    return lines


def print_terminal_metrics(summary: Mapping[str, Any]) -> None:
    print("\n".join(terminal_metric_lines(summary)), flush=True)


@torch.inference_mode()
def run_cast_generation(
    model: Any,
    tokenizer: Any,
    guard: CASTGuard,
    rows: pd.DataFrame,
    generation_config: GenerationConfig,
    *,
    runtime_config_sha256: str,
    eval_split: str,
    checkpoint_interval: int,
    on_checkpoint: Callable[[pd.DataFrame], None] | None,
    fail_on_generation_error: bool,
    progress_desc: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    device = model_input_device(model)
    with guard.attach(model, tokenizer):
        iterator = tqdm(
            rows.iterrows(),
            total=len(rows),
            desc=progress_desc,
            unit="prompt",
            dynamic_ncols=True,
            leave=True,
        )
        for row_number, (_, row) in enumerate(iterator, start=1):
            prompt = str(row["prompt"])
            formatted = apply_chat_template_compat(
                tokenizer,
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                padding=False,
                truncation=True,
            ).to(device)
            log_record: dict[str, Any] = {}
            guard.prepare_request(log_record)
            error: str | None = None
            assistant_output = ""
            started = time.perf_counter()
            first_token_timer: FirstTokenTimer | None = None
            try:
                cuda_synchronize()
                started = time.perf_counter()
                first_token_timer = FirstTokenTimer(started)
                generated_ids = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_cache=True,
                    streamer=first_token_timer,
                )
                cuda_synchronize()
                elapsed = time.perf_counter() - started
                output_ids = generated_ids[0, inputs["input_ids"].shape[1] :]
                assistant_output = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                ).strip()
            except Exception as exc:
                cuda_synchronize()
                elapsed = time.perf_counter() - started
                error = f"{type(exc).__name__}: {exc}"
                assistant_output = f"GENERATION_ERROR: {error}"
            finally:
                guard.finish_request(log_record, generation_error=error)

            record = row.to_dict()
            record.update(log_record)
            record.update(
                {
                    "assistant_output": assistant_output,
                    "prompt_hash": prompt_hash(prompt),
                    "eval_split": eval_split,
                    "generation_time_s": float(elapsed),
                    "ttft_ms": (
                        first_token_timer.ttft_ms
                        if first_token_timer is not None
                        else None
                    ),
                    "cast_runtime_config_sha256": runtime_config_sha256,
                }
            )
            records.append(record)
            if (
                on_checkpoint is not None
                and (
                    row_number % checkpoint_interval == 0
                    or row_number == len(rows)
                    or error is not None
                )
            ):
                on_checkpoint(pd.DataFrame(records))
            if error is not None and fail_on_generation_error:
                raise RuntimeError(
                    f"CAST generation failed for prompt hash {record['prompt_hash']}: {error}"
                )
    return pd.DataFrame(records)


def evaluation_order(evaluation: Mapping[str, Any]) -> list[str]:
    raw = evaluation.get("order", ["utility", "safety"])
    order = list(dict.fromkeys(str(value).lower() for value in raw))
    unknown = set(order) - {"utility", "safety"}
    if unknown:
        raise ValueError(f"evaluation.order contains unknown stages: {sorted(unknown)}")
    return order


def _runtime_hash(
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> str:
    return stable_json_hash(
        {
            "evaluation_runtime_version": EVALUATION_RUNTIME_VERSION,
            "model": model_config,
            "runtime": config["runtime"],
            "evaluation": config["evaluation"],
        }
    )


def dry_run_summary(
    config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    progress = StageProgress("evaluate dry-run", 3)
    progress.begin("Validate model artifacts and output paths")
    evaluation = config["evaluation"]
    output_dir = result_dir(config, model_config)
    artifacts = artifact_dir(config, model_config)
    data_root = resolve_repo_path(evaluation.get("data_dir", "data/raw"))
    progress.advance()

    progress.begin("Scan benign evaluation data")
    utility_rows = 0
    safety_rows = 0
    if bool(evaluation.get("run_utility", True)):
        utility = load_utility_dataset(
            evaluation["utility_dataset_config"],
            data_root,
            int(evaluation["utility_dataset_config"].get("sample_size", 0)),
        )
        utility_rows = len(utility)
    progress.advance()

    progress.begin("Scan safety evaluation data")
    if bool(evaluation.get("run_safety", True)):
        safety = load_safety_dataset(
            evaluation["safety_dataset_config"],
            data_root,
            int(evaluation["safety_dataset_config"].get("sample_size", 0)),
        )
        safety_rows = len(safety)
    progress.advance()
    progress.finish()
    return {
        "model": model_config["name"],
        "artifact_dir": str(artifacts),
        "artifact_complete": all(
            (artifacts / name).exists()
            for name in (
                "behavior_vector.svec",
                "condition_vector.svec",
                "vector_metadata.json",
                "cast_params.yaml",
            )
        ),
        "output_dir": str(output_dir),
        "output_is_inside_cast": True,
        "utility_rows": utility_rows,
        "safety_rows": safety_rows,
        "batch_size": int(evaluation.get("batch_size", 1)),
    }


def run_single_model(
    config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    progress = StageProgress("evaluate", 5)
    evaluation = config["evaluation"]
    model_name = str(model_config["name"])
    output_dir = result_dir(config, model_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_repo_path(evaluation.get("data_dir", "data/raw"))

    progress.begin("Load CAST vectors and calibration parameters")
    guard = CASTGuard.from_artifact_dir(
        artifact_dir(config, model_config),
        enforce_model_binding=bool(
            config["runtime"].get("enforce_model_binding", True)
        ),
    )
    progress.advance()

    runtime_hash = _runtime_hash(config, model_config)
    checkpoint_interval = max(1, int(evaluation.get("checkpoint_interval", 10)))
    device = str(evaluation.get("device", "cuda"))
    progress.begin("Load target model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(
        model_name=resolve_model_reference(model_config["path"]),
        model_kwargs=dict(model_config.get("kwargs", {})),
        device=device,
    )
    generation_config = build_base_generation_config(evaluation, tokenizer)
    progress.advance()

    progress.begin("Generate benign and safety evaluation responses")
    all_frames: list[pd.DataFrame] = []

    for split in evaluation_order(evaluation):
        if split == "utility" and bool(evaluation.get("run_utility", True)):
            utility_cfg = evaluation["utility_dataset_config"]
            utility_df = load_utility_dataset(
                utility_cfg,
                data_root,
                int(utility_cfg.get("sample_size", 0)),
            )
            for dataset_name in utility_df.get(
                "utility_dataset_name",
                pd.Series(dtype=str),
            ).dropna().unique():
                name = str(dataset_name)
                subset = utility_df[
                    utility_df["utility_dataset_name"] == dataset_name
                ].reset_index(drop=True)
                path = utility_detail_path(output_dir, model_name, name)
                completed, pending = split_resume_rows(
                    subset,
                    path,
                    artifact_fingerprint=guard.artifacts.fingerprint,
                    runtime_config_sha256=runtime_hash,
                )
                if not completed.empty:
                    all_frames.append(completed)
                if pending.empty:
                    continue
                local_generation = build_generation_config(
                    generation_config,
                    get_override_max_new_tokens(subset),
                )

                def utility_checkpoint(
                    new_rows: pd.DataFrame,
                    completed_rows: pd.DataFrame = completed,
                    dataset: str = name,
                ) -> None:
                    pieces = [
                        frame
                        for frame in (completed_rows, new_rows)
                        if not frame.empty
                    ]
                    save_utility_files(
                        pd.concat(pieces, ignore_index=True),
                        output_dir,
                        model_name,
                        dataset,
                    )

                generated = run_cast_generation(
                    model,
                    tokenizer,
                    guard,
                    pending,
                    local_generation,
                    runtime_config_sha256=runtime_hash,
                    eval_split="utility",
                    checkpoint_interval=checkpoint_interval,
                    on_checkpoint=utility_checkpoint,
                    fail_on_generation_error=bool(
                        evaluation.get("fail_on_generation_error", True)
                    ),
                    progress_desc=f"CAST utility/{name}",
                )
                all_frames.append(generated)

        if split == "safety" and bool(evaluation.get("run_safety", True)):
            safety_cfg = evaluation["safety_dataset_config"]
            safety_df = load_safety_dataset(
                safety_cfg,
                data_root,
                int(safety_cfg.get("sample_size", 0)),
            )
            for attack_method in safety_df.get(
                "attack_method",
                pd.Series(dtype=str),
            ).dropna().unique():
                attack = str(attack_method)
                subset = safety_df[
                    safety_df["attack_method"] == attack_method
                ].reset_index(drop=True)
                path = safety_detail_path(output_dir, model_name, attack)
                completed, pending = split_resume_rows(
                    subset,
                    path,
                    artifact_fingerprint=guard.artifacts.fingerprint,
                    runtime_config_sha256=runtime_hash,
                )
                if not completed.empty:
                    all_frames.append(completed)
                if pending.empty:
                    continue
                local_generation = build_generation_config(
                    generation_config,
                    get_override_max_new_tokens(subset),
                )

                def safety_checkpoint(
                    new_rows: pd.DataFrame,
                    completed_rows: pd.DataFrame = completed,
                    attack_name: str = attack,
                ) -> None:
                    pieces = [
                        frame
                        for frame in (completed_rows, new_rows)
                        if not frame.empty
                    ]
                    save_safety_file(
                        pd.concat(pieces, ignore_index=True),
                        output_dir,
                        model_name,
                        attack_name,
                    )

                generated = run_cast_generation(
                    model,
                    tokenizer,
                    guard,
                    pending,
                    local_generation,
                    runtime_config_sha256=runtime_hash,
                    eval_split="safety",
                    checkpoint_interval=checkpoint_interval,
                    on_checkpoint=safety_checkpoint,
                    fail_on_generation_error=bool(
                        evaluation.get("fail_on_generation_error", True)
                    ),
                    progress_desc=f"CAST safety/{attack}",
                )
                all_frames.append(generated)

    progress.advance()
    del guard, model, tokenizer
    clear_runtime_memory(f"Unload CAST target model {model_name}")
    if not all_frames:
        raise RuntimeError("No CAST evaluation records were produced.")

    final_df = prepare_detailed_output(pd.concat(all_frames, ignore_index=True))
    final_df["label"] = pd.NA
    safety_mask = final_df["eval_split"].astype(str).eq("safety")
    progress.begin("Run HarmBench safety classification")
    if safety_mask.any():
        classifier = evaluation["classifier_config"]
        classifier_model, classifier_tokenizer = load_model_and_tokenizer(
            model_name=resolve_model_reference(classifier["path"]),
            model_kwargs=dict(classifier.get("kwargs", {})),
            device=device,
        )
        final_df.loc[safety_mask, "label"] = run_classification(
            classifier_model,
            classifier_tokenizer,
            final_df.loc[safety_mask].copy(),
            int(evaluation.get("classifier_batch_size", 1)),
        )
        del classifier_model, classifier_tokenizer
        clear_runtime_memory("Unload HarmBench classifier")
    progress.advance()

    progress.begin("Aggregate FPR, DSR, and diagnostic metrics")
    atomic_to_csv(
        final_df,
        output_dir / f"{model_name}_cast_evaluation_all_results.csv",
    )
    summary: dict[str, Any] = {
        "model": model_name,
        "defense": "CAST",
        "runtime_config_sha256": runtime_hash,
        "utility": {},
        "safety": {},
    }
    utility_rows = final_df[final_df["eval_split"].astype(str).eq("utility")]
    for dataset_name in utility_rows.get(
        "utility_dataset_name",
        pd.Series(dtype=str),
    ).dropna().unique():
        name = str(dataset_name)
        subset = utility_rows[
            utility_rows["utility_dataset_name"] == dataset_name
        ].copy()
        if not should_compute_keyword_fpr(name, subset):
            raise ValueError(
                f"Benign dataset {name} must configure fpr_strategy=refusal_keywords "
                "to produce a unified FPR."
            )
        subset = annotate_keyword_false_positives(subset)
        metrics = calculate_keyword_fpr_metrics(subset, condition="guarded")
        metrics["metric"] = "FPR"
        metrics["fpr"] = float(metrics["rate"])
        metrics["fpr_strategy"] = "refusal_keywords"
        metrics["condition_diagnostics"] = condition_diagnostics(subset)
        summary["utility"][name] = metrics
        save_utility_files(subset, output_dir, model_name, name)
        write_json(
            output_dir / f"{model_name}_cast_evaluation_summary_utility_{name}.json",
            {f"utility_fpr_{name}": {"CAST": metrics}},
        )

    safety_rows = final_df[final_df["eval_split"].astype(str).eq("safety")]
    for attack_method in safety_rows.get(
        "attack_method",
        pd.Series(dtype=str),
    ).dropna().unique():
        attack = str(attack_method)
        subset = safety_rows[
            safety_rows["attack_method"] == attack_method
        ].copy()
        metrics = calculate_metrics(subset, condition="guarded")
        metrics["metric"] = "DSR"
        metrics["asr"] = float(metrics["rate"])
        metrics["dsr"] = 1.0 - float(metrics["rate"])
        metrics["condition_diagnostics"] = condition_diagnostics(subset)
        summary["safety"][attack] = metrics
        save_safety_file(subset, output_dir, model_name, attack)
        write_json(
            output_dir / f"{model_name}_cast_evaluation_summary_attack_{attack}.json",
            {f"safety_asr_attack_{attack}": {"CAST": metrics}},
        )
    summary["condition_detection"] = {
        "utility": condition_diagnostics(utility_rows),
        "safety": condition_diagnostics(safety_rows),
    }
    summary["aggregate"] = aggregate_evaluation_metrics(summary)
    write_json(output_dir / "all_metrics.json", summary)
    progress.advance()
    progress.finish()
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CAST in the JEDI evaluation harness.")
    parser.add_argument(
        "--config",
        default="scripts/baselines/CAST/configs/cast_config.yaml",
    )
    parser.add_argument(
        "--llm-name",
        help="Override model.name and replace the final path component of model.path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read only configuration and evaluation data without loading models or writing results.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    validate_batch_invariants(config)
    model = select_model(config, args.llm_name)
    payload = (
        dry_run_summary(config, model)
        if args.dry_run
        else run_single_model(config, model)
    )
    if not args.dry_run:
        print_terminal_metrics(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
