# -*- coding: utf-8 -*-
"""
Evaluate TrajGuard with JEDI's data, HarmBench classifier, and metric protocol.

This script outputs only the TrajGuard condition and does not regenerate
undefended/JEDI conditions. Comparison results can be joined by prompt with
results from the main evaluation script and other ``scripts/baselines``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import GenerationConfig


BASE_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = BASE_DIR / "src"
SCRIPTS_DIR = BASE_DIR / "scripts"
THIS_DIR = Path(__file__).resolve().parent

# scripts must precede this same-named run_evaluation.py so the project's main evaluation script is imported.
for path in (THIS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(SCRIPTS_DIR) in sys.path:
    sys.path.remove(str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

main_eval = importlib.import_module("run_evaluation")
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

if __package__:
    from .core import ARTIFACT_FILENAME, TrajGuard  # type: ignore[import-not-found]
else:
    from core import ARTIFACT_FILENAME, TrajGuard  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
EVALUATION_RUNTIME_VERSION = 1


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = BASE_DIR / value
    return value.resolve()


def resolve_baseline_output_path(path: str | Path) -> Path:
    """Resolve the evaluation output directory and prevent writes to JEDI's shared data directory."""
    resolved = resolve_path(path)
    try:
        resolved.relative_to(THIS_DIR)
    except ValueError as exc:
        raise ValueError(
            f"TrajGuard evaluation results must remain inside the baseline directory {THIS_DIR}; got {resolved}"
        ) from exc
    return resolved


def resolve_model_reference(value: str | Path) -> str:
    """Resolve local relative paths from the project root; otherwise preserve the Hugging Face repository name."""
    raw = str(value)
    candidate = resolve_path(raw)
    return str(candidate) if candidate.exists() else raw


def replace_local_model_name(model_path: str | Path, llm_name: str) -> str:
    """Replace the final directory name in a model path with the command-line name."""
    if not re.fullmatch(r"[^\\/]+", llm_name):
        raise ValueError("--llm-name must be a single model-directory name without path separators.")
    raw_path = str(model_path)
    matched = re.fullmatch(r"(?s)(.*[\\/])[^\\/]+", raw_path)
    if matched is None:
        raise ValueError(
            "--llm-name requires a model path containing a directory separator; "
            f"unable to replace the name in model path: {raw_path}"
        )
    return f"{matched.group(1)}{llm_name}"


def apply_llm_name_override(
    entries: List[Dict[str, Any]], llm_name: str
) -> List[Dict[str, Any]]:
    """Use the first configured model as a template and run only the command-line target model."""
    if not entries:
        raise ValueError("Cannot override an empty llm_models configuration.")
    template = entries[0]
    return [
        {
            "name": llm_name,
            "path": replace_local_model_name(template["path"], llm_name),
            "kwargs": dict(template.get("kwargs", {})),
        }
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def iter_llm_entries(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries = config.get("llm_models")
    if entries is None:
        if "llm_name" not in config or "llm_config" not in config:
            raise KeyError("Configuration must contain llm_models or legacy llm_name + llm_config.")
        llm_cfg = dict(config["llm_config"])
        return [
            {
                "name": config["llm_name"],
                "path": llm_cfg["path"],
                "kwargs": llm_cfg.get("kwargs", {}),
            }
        ]
    if not isinstance(entries, list) or not entries:
        raise ValueError("llm_models must be a nonempty list.")

    normalized: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"llm_models[{index}] must be a mapping.")
        name = entry.get("name") or entry.get("llm_name")
        path = entry.get("path")
        kwargs = entry.get("kwargs", {})
        nested = entry.get("llm_config")
        if isinstance(nested, Mapping):
            path = path or nested.get("path")
            kwargs = nested.get("kwargs", kwargs)
        if not name or not path:
            raise KeyError(f"llm_models[{index}] must provide name and path.")
        normalized.append({"name": str(name), "path": str(path), "kwargs": dict(kwargs)})
    return normalized


def build_model_config(config: Mapping[str, Any], entry: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(config)
    result["llm_name"] = entry["name"]
    result["llm_config"] = {
        "path": entry["path"],
        "kwargs": dict(entry.get("kwargs", {})),
    }
    return result


def evaluation_order(config: Mapping[str, Any]) -> List[str]:
    raw = config.get("evaluation_order", ["utility", "safety"])
    if not isinstance(raw, list) or not raw:
        raise ValueError("evaluation_order must be a nonempty list.")
    order = list(dict.fromkeys(str(item).strip().lower() for item in raw))
    unknown = set(order) - {"utility", "safety"}
    if unknown:
        raise ValueError(f"evaluation_order contains unknown stages: {sorted(unknown)}")
    for split in ("utility", "safety"):
        if split not in order:
            order.append(split)
    return order


def clear_runtime_memory(context: str) -> None:
    logger.info("Clearing runtime memory: %s", context)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError as exc:
            logger.warning("CUDA IPC cleanup skipped: %s", exc)


def artifact_path_for(config: Mapping[str, Any], llm_name: str) -> Path:
    traj_cfg = config["trajguard"]
    formatted = str(traj_cfg["artifact_dir"]).format(llm_name=llm_name)
    path = resolve_baseline_output_path(formatted)
    if path.suffix.lower() in {".pt", ".pth"}:
        return path
    return path / ARTIFACT_FILENAME


def make_guard(
    config: Mapping[str, Any],
    tokenizer: Any,
    artifact_path: Path,
) -> TrajGuard:
    traj_cfg = config["trajguard"]
    judge_cfg = traj_cfg.get("judge", {})
    return TrajGuard.from_artifacts(
        artifact_path,
        tokenizer,
        use_pair_judge=bool(traj_cfg.get("use_pair_judge", True)),
        unsafe_action=str(traj_cfg.get("unsafe_action", "replace")),
        refusal_text=str(
            traj_cfg.get(
                "refusal_text",
                "I'm sorry, but I can't help with that.",
            )
        ),
        unsafe_labels=judge_cfg.get("unsafe_labels", [" Yes"]),
        safe_labels=judge_cfg.get("safe_labels", [" No"]),
        judge_fail_closed=bool(judge_cfg.get("fail_closed", True)),
        judge_max_length=int(judge_cfg.get("max_length", 2048)),
        fail_on_monitor_error=bool(traj_cfg.get("fail_on_monitor_error", True)),
        enforce_chat_template_hash=bool(
            traj_cfg.get("enforce_chat_template_hash", True)
        ),
    )


def build_base_generation_config(
    config: Mapping[str, Any],
    tokenizer: Any,
) -> GenerationConfig:
    kwargs = dict(config.get("generation_kwargs", {}))
    if bool(kwargs.get("do_sample", False)):
        raise ValueError("The TrajGuard paper reproduction requires generation_kwargs.do_sample=false.")
    if int(kwargs.get("num_beams", 1)) != 1:
        raise ValueError("The TrajGuard MVP requires generation_kwargs.num_beams=1.")
    generation_config = GenerationConfig(
        max_new_tokens=int(kwargs.pop("max_new_tokens", 256)),
        do_sample=False,
        num_beams=1,
        **{
            key: value
            for key, value in kwargs.items()
            if key not in {"do_sample", "num_beams"}
        },
    )
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    return generation_config


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def split_resume_rows(
    source_df: pd.DataFrame,
    detail_path: Path,
    *,
    artifact_sha256: str,
    runtime_config_sha256: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resume by prompt and artifact/runtime fingerprints to avoid reusing stale method results."""
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
        "trajguard_artifact_sha256",
        "trajguard_runtime_config_sha256",
    }
    if not required.issubset(existing.columns):
        logger.warning("Ignoring legacy/malformed checkpoint: %s", detail_path)
        return _empty_like(source_df), source_df.reset_index(drop=True)
    existing = existing[
        (existing["trajguard_artifact_sha256"].astype(str) == artifact_sha256)
        & (
            existing["trajguard_runtime_config_sha256"].astype(str)
            == runtime_config_sha256
        )
        & existing["prompt"].notna()
        & existing["assistant_output"].notna()
    ].copy()

    rows_by_prompt: Dict[str, List[int]] = {}
    for row_index, prompt in existing["prompt"].astype(str).items():
        rows_by_prompt.setdefault(prompt, []).append(row_index)

    completed_indices: List[int] = []
    pending_indices: List[int] = []
    for row_index, prompt in source_df["prompt"].astype(str).items():
        candidates = rows_by_prompt.get(prompt)
        if candidates:
            completed_indices.append(candidates.pop(0))
        else:
            pending_indices.append(row_index)

    completed = (
        existing.loc[completed_indices].reset_index(drop=True)
        if completed_indices
        else _empty_like(source_df)
    )
    pending = source_df.loc[pending_indices].reset_index(drop=True)
    logger.info(
        "Checkpoint %s: %d completed, %d pending.",
        detail_path,
        len(completed),
        len(pending),
    )
    return completed, pending


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def prepare_detailed_output(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in (
        "trigger_step",
        "trajguard_trigger_step",
        "trajguard_unsafe_trigger_step",
    ):
        if column in output.columns:
            output[column] = output[column].apply(
                lambda value: pd.NA
                if pd.isna(value) or int(float(value)) < 0
                else int(float(value))
            ).astype("Int64")
    return output


def calculate_trajguard_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate triggers, prefix exposure, judge calls, and runtime diagnostics."""
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "judge_call_sequence_rate": 0.0,
            "unsafe_stop_rate": 0.0,
        }

    def numeric(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[column], errors="coerce").dropna()

    def median_and_p95(column: str) -> Dict[str, Optional[float]]:
        values = numeric(column)
        if values.empty:
            return {"median": None, "p95": None}
        return {
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
        }

    judge_calls = numeric("trajguard_judge_calls")
    if "trajguard_unsafe" in df.columns:
        unsafe_mask = (
            df["trajguard_unsafe"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    else:
        unsafe_mask = pd.Series(False, index=df.index)

    unsafe_rows = df.loc[unsafe_mask]
    unsafe_exposure = pd.to_numeric(
        unsafe_rows.get(
            "trajguard_exposed_tokens",
            pd.Series(index=unsafe_rows.index, dtype=float),
        ),
        errors="coerce",
    ).dropna()
    diagnostics: Dict[str, Any] = {
        "total": int(total),
        "judge_call_sequence_rate": float(
            (judge_calls > 0).sum() / total
        ) if not judge_calls.empty else 0.0,
        "mean_judge_calls_per_sequence": float(
            judge_calls.sum() / total
        ) if not judge_calls.empty else 0.0,
        "unsafe_stop_rate": float(unsafe_mask.sum() / total),
        "first_trigger_step": median_and_p95("trajguard_trigger_step"),
        "unsafe_trigger_step": median_and_p95(
            "trajguard_unsafe_trigger_step"
        ),
        "exposed_tokens_among_unsafe_stops": (
            {
                "median": float(unsafe_exposure.median()),
                "p95": float(unsafe_exposure.quantile(0.95)),
            }
            if not unsafe_exposure.empty
            else {"median": None, "p95": None}
        ),
        "zero_prefix_exposure_rate_among_unsafe_stops": (
            float((unsafe_exposure == 0).mean())
            if not unsafe_exposure.empty
            else None
        ),
        "ttft_ms": median_and_p95("trajguard_ttft_ms"),
        "generation_runtime_ms": median_and_p95(
            "trajguard_generation_runtime_ms"
        ),
        "defense_runtime_ms": median_and_p95("trajguard_runtime_ms"),
        "judge_runtime_ms": median_and_p95(
            "trajguard_judge_runtime_ms"
        ),
    }
    scored_steps = numeric("trajguard_scored_steps")
    defense_runtime = pd.to_numeric(
        df.get(
            "trajguard_runtime_ms",
            pd.Series(index=df.index, dtype=float),
        ),
        errors="coerce",
    )
    step_counts = pd.to_numeric(
        df.get(
            "trajguard_scored_steps",
            pd.Series(index=df.index, dtype=float),
        ),
        errors="coerce",
    )
    per_step = (defense_runtime / step_counts.where(step_counts > 0)).dropna()
    diagnostics["mean_defense_ms_per_scored_step"] = (
        float(per_step.mean()) if not per_step.empty else None
    )
    diagnostics["mean_scored_steps"] = (
        float(scored_steps.sum() / total) if not scored_steps.empty else 0.0
    )
    return diagnostics


def build_terminal_metric_lines(
    llm_name: str,
    utility_metrics: Mapping[str, Mapping[str, Any]],
    safety_metrics: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    """Build the benign-FPR and safety-DSR summary printed after evaluation."""
    lines = [f"===== TrajGuard Evaluation Results: {llm_name} ====="]

    utility_total = 0
    utility_false_positives = 0
    if utility_metrics:
        lines.append("Benign Dataset FPR:")
        for dataset_name, metrics in sorted(utility_metrics.items()):
            total = int(metrics.get("total", 0))
            false_positives = int(
                metrics.get("false_positive_count", metrics.get("harmful_count", 0))
            )
            rate = false_positives / total if total else 0.0
            utility_total += total
            utility_false_positives += false_positives
            lines.append(
                f"  {dataset_name}: {rate:.2%} "
                f"({false_positives}/{total})"
            )
        aggregate = utility_false_positives / utility_total if utility_total else 0.0
        lines.append(
            f"  Overall: {aggregate:.2%} "
            f"({utility_false_positives}/{utility_total})"
        )
    else:
        lines.append("Benign Dataset FPR: Not Run")

    safety_total = 0
    safety_harmful = 0
    if safety_metrics:
        lines.append("Safety Dataset DSR:")
        for attack_method, metrics in sorted(safety_metrics.items()):
            total = int(metrics.get("total", 0))
            harmful = int(metrics.get("harmful_count", 0))
            dsr = 1.0 - harmful / total if total else 0.0
            safety_total += total
            safety_harmful += harmful
            lines.append(
                f"  {attack_method}: {dsr:.2%} "
                f"({total - harmful}/{total})"
            )
        aggregate = 1.0 - safety_harmful / safety_total if safety_total else 0.0
        lines.append(
            f"  Overall: {aggregate:.2%} "
            f"({safety_total - safety_harmful}/{safety_total}; "
            f"ASR={safety_harmful / safety_total:.2%})"
        )
    else:
        lines.append("Safety Dataset DSR: Not Run")

    lines.append("====================================")
    return lines


def utility_detail_path(output_dir: Path, llm_name: str, dataset_name: str) -> Path:
    return output_dir / (
        f"{llm_name}_trajguard_evaluation_detailed_utility_{dataset_name}.csv"
    )


def safety_detail_path(output_dir: Path, llm_name: str, attack_method: str) -> Path:
    return output_dir / (
        f"{llm_name}_trajguard_evaluation_detailed_attack_{attack_method}.csv"
    )


def save_alpaca_format(df: pd.DataFrame, path: Path, generator_name: str) -> None:
    records = [
        {
            "dataset": row.get("dataset", "alpaca_eval"),
            "instruction": row.get("instruction", row.get("prompt", "")),
            "output": row.get("assistant_output", ""),
            "generator": generator_name,
        }
        for _, row in df.iterrows()
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


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
    llm_name: str,
    dataset_name: str,
) -> Path:
    detail = prepare_detailed_output(df)
    path = utility_detail_path(output_dir, llm_name, dataset_name)
    atomic_to_csv(detail, path)
    if dataset_name == "alpaca_eval":
        save_alpaca_format(
            detail,
            output_dir / f"{llm_name}-alpaca_eval-trajguard.json",
            f"{llm_name}-trajguard",
        )
    if "xstest" in dataset_name.lower():
        save_xstest_format(
            detail,
            output_dir / f"{llm_name}_xstest_trajguard.csv",
        )
    return path


def save_safety_file(
    df: pd.DataFrame,
    output_dir: Path,
    llm_name: str,
    attack_method: str,
) -> Path:
    detail = prepare_detailed_output(df)
    path = safety_detail_path(output_dir, llm_name, attack_method)
    atomic_to_csv(detail, path)
    return path


@torch.inference_mode()
def run_trajguard_generation(
    model: Any,
    tokenizer: Any,
    guard: TrajGuard,
    rows: pd.DataFrame,
    generation_config: GenerationConfig,
    *,
    artifact_sha256: str,
    runtime_config_sha256: str,
    eval_split: str,
    on_checkpoint: Optional[Callable[[pd.DataFrame], None]] = None,
    checkpoint_interval: int = 1,
) -> pd.DataFrame:
    """Run per example; batch_size=1 is invariant in the current implementation and paper reproduction."""
    generated_records: List[Dict[str, Any]] = []
    with guard.attach(model):
        for row_number, (_, row) in enumerate(
            tqdm(rows.iterrows(), total=len(rows), desc="TrajGuard generation"),
            start=1,
        ):
            prompt = str(row["prompt"])
            conversation = [{"role": "user", "content": prompt}]
            input_text = apply_chat_template_compat(
                tokenizer,
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
            ).to(model.device)

            log_record: Dict[str, Any] = {}
            guard.prepare_request(prompt, log_record)
            generated_ids = model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=True,
            )
            output_ids = generated_ids[0, inputs["input_ids"].shape[1] :]
            assistant_output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            ).strip()

            record = row.to_dict()
            record.update(log_record)
            record["assistant_output"] = assistant_output
            record["trigger_step"] = int(
                log_record.get("trajguard_trigger_step", -1)
            )
            record["condition"] = "trajguard"
            record["eval_split"] = eval_split
            record["trajguard_artifact_sha256"] = artifact_sha256
            record["trajguard_runtime_config_sha256"] = runtime_config_sha256
            generated_records.append(record)

            if (
                on_checkpoint is not None
                and (
                    row_number % checkpoint_interval == 0
                    or row_number == len(rows)
                )
            ):
                on_checkpoint(pd.DataFrame(generated_records))
    return pd.DataFrame(generated_records)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def run_single_model(config: Dict[str, Any], index: int, total: int) -> None:
    llm_name = str(config["llm_name"])
    logger.info("TrajGuard model %d/%d: %s", index, total, llm_name)
    output_dir = resolve_baseline_output_path(config["output_dir"]) / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_path(config["data_dir"])
    artifact_path = artifact_path_for(config, llm_name)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"TrajGuard artifacts do not exist: {artifact_path}; run build_artifacts.py first."
        )
    artifact_sha = sha256_file(artifact_path)
    runtime_cfg = dict(config["trajguard"])
    runtime_hash = stable_json_hash(
        {
            "evaluation_runtime_version": EVALUATION_RUNTIME_VERSION,
            "trajguard": runtime_cfg,
            "generation_kwargs": config.get("generation_kwargs", {}),
            "llm_config": config["llm_config"],
            "utility_dataset_config": config.get("utility_dataset_config", {}),
            "safety_dataset_config": config.get("safety_dataset_config", {}),
        }
    )
    checkpoint_interval = max(1, int(config.get("checkpoint_interval", 1)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_cfg = config["llm_config"]
    model, tokenizer = load_model_and_tokenizer(
        model_name=resolve_model_reference(llm_cfg["path"]),
        model_kwargs=llm_cfg.get("kwargs", {}),
        device=device,
    )
    generation_config = build_base_generation_config(config, tokenizer)
    guard = make_guard(config, tokenizer, artifact_path)
    all_results: List[pd.DataFrame] = []

    run_utility = bool(config.get("run_utility_evaluation", True))
    run_safety = bool(config.get("run_safety_evaluation", True))
    for split in evaluation_order(config):
        if split == "utility" and run_utility:
            utility_cfg = config["utility_dataset_config"]
            utility_df = load_utility_dataset(
                utility_cfg,
                data_dir,
                utility_cfg.get("sample_size", 0),
            )
            for dataset_name in utility_df.get(
                "utility_dataset_name",
                pd.Series(dtype=str),
            ).dropna().unique():
                subset = utility_df[
                    utility_df["utility_dataset_name"] == dataset_name
                ].reset_index(drop=True)
                detail_path = utility_detail_path(output_dir, llm_name, dataset_name)
                completed, pending = split_resume_rows(
                    subset,
                    detail_path,
                    artifact_sha256=artifact_sha,
                    runtime_config_sha256=runtime_hash,
                )
                if not completed.empty:
                    all_results.append(completed)
                if pending.empty:
                    continue
                local_generation = build_generation_config(
                    generation_config,
                    get_override_max_new_tokens(subset),
                )

                def utility_checkpoint(new_rows: pd.DataFrame) -> None:
                    pieces = [frame for frame in (completed, new_rows) if not frame.empty]
                    save_utility_files(
                        pd.concat(pieces, ignore_index=True),
                        output_dir,
                        llm_name,
                        str(dataset_name),
                    )

                generated = run_trajguard_generation(
                    model,
                    tokenizer,
                    guard,
                    pending,
                    local_generation,
                    artifact_sha256=artifact_sha,
                    runtime_config_sha256=runtime_hash,
                    eval_split="utility",
                    on_checkpoint=utility_checkpoint,
                    checkpoint_interval=checkpoint_interval,
                )
                all_results.append(generated)

        if split == "safety" and run_safety:
            safety_cfg = config["safety_dataset_config"]
            safety_df = load_safety_dataset(
                safety_cfg,
                data_dir,
                safety_cfg.get("sample_size", 0),
            )
            for attack_method in safety_df.get(
                "attack_method",
                pd.Series(dtype=str),
            ).dropna().unique():
                subset = safety_df[
                    safety_df["attack_method"] == attack_method
                ].reset_index(drop=True)
                detail_path = safety_detail_path(
                    output_dir,
                    llm_name,
                    str(attack_method),
                )
                completed, pending = split_resume_rows(
                    subset,
                    detail_path,
                    artifact_sha256=artifact_sha,
                    runtime_config_sha256=runtime_hash,
                )
                if not completed.empty:
                    all_results.append(completed)
                if pending.empty:
                    continue
                local_generation = build_generation_config(
                    generation_config,
                    get_override_max_new_tokens(subset),
                )

                def safety_checkpoint(new_rows: pd.DataFrame) -> None:
                    pieces = [frame for frame in (completed, new_rows) if not frame.empty]
                    save_safety_file(
                        pd.concat(pieces, ignore_index=True),
                        output_dir,
                        llm_name,
                        str(attack_method),
                    )

                generated = run_trajguard_generation(
                    model,
                    tokenizer,
                    guard,
                    pending,
                    local_generation,
                    artifact_sha256=artifact_sha,
                    runtime_config_sha256=runtime_hash,
                    eval_split="safety",
                    on_checkpoint=safety_checkpoint,
                    checkpoint_interval=checkpoint_interval,
                )
                all_results.append(generated)

    del guard, model, tokenizer
    clear_runtime_memory(f"after unloading TrajGuard target model {llm_name}")
    if not all_results:
        logger.warning("No TrajGuard rows produced for %s.", llm_name)
        return

    final_df = prepare_detailed_output(pd.concat(all_results, ignore_index=True))
    if "eval_split" not in final_df.columns:
        final_df["eval_split"] = "N/A"
    final_df["label"] = pd.NA
    safety_mask = final_df["eval_split"].astype(str) == "safety"
    if safety_mask.any():
        classifier_cfg = config["classifier_config"]
        classifier_model, classifier_tokenizer = load_model_and_tokenizer(
            model_name=resolve_model_reference(classifier_cfg["path"]),
            model_kwargs=classifier_cfg.get("kwargs", {}),
            device=device,
        )
        labels = run_classification(
            classifier_model,
            classifier_tokenizer,
            final_df[safety_mask].copy(),
            int(config.get("classifier_batch_size", 1)),
        )
        final_df.loc[safety_mask, "label"] = labels
        del classifier_model, classifier_tokenizer
        clear_runtime_memory(f"after unloading HarmBench classifier for {llm_name}")

    all_path = output_dir / f"{llm_name}_trajguard_evaluation_all_results.csv"
    atomic_to_csv(prepare_detailed_output(final_df), all_path)

    utility_rows = final_df[final_df["eval_split"].astype(str) == "utility"]
    utility_metrics: Dict[str, Dict[str, Any]] = {}
    for dataset_name in utility_rows.get(
        "utility_dataset_name",
        pd.Series(dtype=str),
    ).dropna().unique():
        subset = utility_rows[
            utility_rows["utility_dataset_name"] == dataset_name
        ].copy()
        if should_compute_keyword_fpr(dataset_name, subset):
            subset = annotate_keyword_false_positives(subset)
            metrics = calculate_keyword_fpr_metrics(subset, condition="guarded")
        else:
            metrics = calculate_metrics(
                subset.assign(label="no"),
                condition="guarded",
            )
        metrics["trajguard_diagnostics"] = calculate_trajguard_diagnostics(
            subset
        )
        utility_metrics[str(dataset_name)] = metrics
        save_utility_files(
            subset,
            output_dir,
            llm_name,
            str(dataset_name),
        )
        save_json(
            output_dir
            / f"{llm_name}_trajguard_evaluation_summary_utility_{dataset_name}.json",
            {f"utility_fpr_{dataset_name}": {"trajguard": metrics}},
        )

    safety_rows = final_df[final_df["eval_split"].astype(str) == "safety"]
    safety_metrics: Dict[str, Dict[str, Any]] = {}
    for attack_method in safety_rows.get(
        "attack_method",
        pd.Series(dtype=str),
    ).dropna().unique():
        subset = safety_rows[
            safety_rows["attack_method"] == attack_method
        ].copy()
        metrics = calculate_metrics(subset, condition="guarded")
        metrics["trajguard_diagnostics"] = calculate_trajguard_diagnostics(
            subset
        )
        safety_metrics[str(attack_method)] = metrics
        save_safety_file(
            subset,
            output_dir,
            llm_name,
            str(attack_method),
        )
        save_json(
            output_dir
            / f"{llm_name}_trajguard_evaluation_summary_attack_{attack_method}.json",
            {f"safety_asr_attack_{attack_method}": {"trajguard": metrics}},
        )
    logger.info("\n%s", "\n".join(
        build_terminal_metric_lines(llm_name, utility_metrics, safety_metrics)
    ))
    logger.info("TrajGuard evaluation completed for %s.", llm_name)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The top level of the evaluation configuration must be a mapping.")
    for key in ("trajguard", "data_dir", "output_dir"):
        if key not in config:
            raise KeyError(f"Evaluation configuration is missing field: {key}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrajGuard in the JEDI harness.")
    parser.add_argument(
        "--config",
        default="scripts/baselines/trajguard/evaluation_config.yaml",
        help="YAML configuration relative to the project root or absolute.",
    )
    parser.add_argument(
        "--llm-name",
        help="Evaluate only this model and replace the model path's final directory with that name.",
    )
    args = parser.parse_args()
    config = load_config(resolve_path(args.config))
    entries = iter_llm_entries(config)
    if args.llm_name:
        entries = apply_llm_name_override(entries, args.llm_name)
    for index, entry in enumerate(entries, start=1):
        model_config = build_model_config(config, entry)
        try:
            run_single_model(model_config, index, len(entries))
        finally:
            clear_runtime_memory(f"between TrajGuard models after {entry['name']}")


if __name__ == "__main__":
    main()
