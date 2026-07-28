# -*- coding: utf-8 -*-
"""
Build paper-consistent TrajGuard offline artifacts.

Procedure:
1. Load data and enforce prompt-level hash isolation.
2. Fit per-layer Incremental PCA to hidden states from the final k tokens of benign/malicious reference prompts.
3. Fit benign/malicious Gaussian regions in PCA space with Ledoit-Wolf.
4. Estimate MVD on an independent jailbreak set and select the Top-K layers.
5. Perform real streaming generation on an independent benign validation set and calibrate the threshold at the 99.5th percentile of streaming scores.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import roc_auc_score
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


BASE_DIR = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

if __package__:
    from .core import (  # type: ignore[import-not-found]
        ARTIFACT_FILENAME,
        SCHEMA_VERSION,
        TrajGuard,
        TrajGuardArtifacts,
        find_decoder_layers,
        tokenizer_chat_template_hash,
    )
else:
    from core import (  # noqa: E402
        ARTIFACT_FILENAME,
        SCHEMA_VERSION,
        TrajGuard,
        TrajGuardArtifacts,
        find_decoder_layers,
        tokenizer_chat_template_hash,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = BASE_DIR / value
    return value.resolve()


def resolve_baseline_output_path(path: str | Path) -> Path:
    """Resolve the artifact directory and prevent TrajGuard from writing to shared project directories."""
    resolved = resolve_path(path)
    try:
        resolved.relative_to(THIS_DIR)
    except ValueError as exc:
        raise ValueError(
            f"TrajGuard artifacts must remain inside the baseline directory {THIS_DIR}; got {resolved}"
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


def apply_llm_name_override(config: Dict[str, Any], llm_name: str) -> None:
    """Override the target-model name and local model directory in the build configuration in place."""
    model_cfg = config["model"]
    model_cfg["name"] = llm_name
    model_cfg["path"] = replace_local_model_name(model_cfg["path"], llm_name)


def normalize_text(text: str) -> str:
    """Normalize for data isolation with NFKC, whitespace collapse, trimming, and casefold."""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _nested_get(record: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _json_records(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("data", "records", "items", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise TypeError(f"The top level of JSON must be a list or mapping: {path}")


def _read_records(path: Path) -> List[Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _json_records(path)
    if suffix in {".jsonl", ".ndjson"}:
        records: List[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSONL.") from exc
        return records
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=separator).to_dict(orient="records")
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    raise ValueError(f"Unsupported data-file format: {path}")


def _expand_paths(patterns: str | Sequence[str]) -> List[Path]:
    if isinstance(patterns, str):
        patterns = [patterns]
    resolved: List[Path] = []
    for pattern in patterns:
        candidate = Path(pattern).expanduser()
        expanded_pattern = str(candidate if candidate.is_absolute() else BASE_DIR / candidate)
        matches = [
            Path(x).resolve()
            for x in sorted(glob.glob(expanded_pattern, recursive=True))
        ]
        if not matches and Path(expanded_pattern).is_file():
            matches = [Path(expanded_pattern).resolve()]
        resolved.extend(path for path in matches if path.is_file())
    unique = list(dict.fromkeys(resolved))
    if not unique:
        raise FileNotFoundError(f"Data paths matched no files: {list(patterns)}")
    return unique


def load_text_split(
    split_name: str,
    spec: Mapping[str, Any],
    *,
    default_seed: int,
    collect_all_fields: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load a split and return deduplicated original texts plus the source-file list."""
    if "paths" not in spec:
        raise KeyError(f"datasets.{split_name}.paths is missing.")
    paths = _expand_paths(spec["paths"])
    fields = spec.get(
        "text_fields",
        ["query", "prompt", "instruction", "Goal", "Behavior"],
    )
    if isinstance(fields, str):
        fields = [fields]

    texts: List[str] = []
    seen: set[str] = set()
    for path in paths:
        for record in _read_records(path):
            candidates: List[Any] = []
            if isinstance(record, str):
                candidates = [record]
            elif isinstance(record, Mapping):
                for field in fields:
                    value = _nested_get(record, str(field))
                    if value is not None:
                        candidates.append(value)
                        if not collect_all_fields:
                            break
            for value in candidates:
                if isinstance(value, list):
                    values = value
                else:
                    values = [value]
                for item in values:
                    if not isinstance(item, str) or not item.strip():
                        continue
                    digest = text_hash(item)
                    if digest not in seen:
                        seen.add(digest)
                        texts.append(item.strip())

    if not texts:
        raise ValueError(f"Split {split_name} yielded no text.")

    seed = int(spec.get("seed", default_seed))
    if bool(spec.get("shuffle", True)):
        rng = random.Random(seed)
        rng.shuffle(texts)
    max_samples = int(spec.get("max_samples", 0) or 0)
    if max_samples > 0:
        texts = texts[:max_samples]

    logger.info(
        "Loaded split %s: %d unique prompts from %d files.",
        split_name,
        len(texts),
        len(paths),
    )
    return texts, [str(path) for path in paths]


def validate_data_isolation(
    split_texts: Mapping[str, Sequence[str]],
    *,
    strict_internal_disjoint: bool,
) -> Dict[str, List[str]]:
    hashes = {
        name: sorted({text_hash(text) for text in texts})
        for name, texts in split_texts.items()
    }
    hash_sets = {name: set(values) for name, values in hashes.items()}
    names = list(hash_sets)
    violations: List[str] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if not strict_internal_disjoint and "evaluation" not in {left, right}:
                continue
            overlap = hash_sets[left] & hash_sets[right]
            if overlap:
                violations.append(f"{left} vs {right}: {len(overlap)}")
    if violations:
        raise ValueError(
            "TrajGuard data isolation failed; normalized prompt overlaps found: " + "; ".join(violations)
        )
    return hashes


def drop_evaluation_overlaps(
    split_texts: Dict[str, List[str]],
    *,
    max_fraction: float,
) -> Dict[str, List[str]]:
    """Explicitly remove construction-split texts matching evaluation prompts and return audit hashes."""
    evaluation = split_texts.get("evaluation")
    if not evaluation:
        return {}
    if not 0.0 <= max_fraction <= 1.0:
        raise ValueError("max_evaluation_overlap_fraction must be in [0, 1].")

    evaluation_hashes = {text_hash(text) for text in evaluation}
    excluded: Dict[str, List[str]] = {}
    for name in (
        "benign_reference",
        "malicious_reference",
        "layer_selection",
        "benign_validation",
    ):
        original = split_texts[name]
        overlap_hashes = sorted(
            {text_hash(text) for text in original} & evaluation_hashes
        )
        if not overlap_hashes:
            continue
        fraction = len(overlap_hashes) / max(1, len(original))
        if fraction > max_fraction:
            raise ValueError(
                f"{name} overlaps evaluation by {len(overlap_hashes)} examples "
                f"({fraction:.2%}), exceeding the automatic-removal limit of {max_fraction:.2%}."
            )
        split_texts[name] = [
            text for text in original if text_hash(text) not in evaluation_hashes
        ]
        if not split_texts[name]:
            raise ValueError(f"{name} became empty after removing evaluation overlaps.")
        excluded[name] = overlap_hashes
        logger.warning(
            "Excluded %d exact evaluation overlaps from %s (%.2f%%).",
            len(overlap_hashes),
            name,
            fraction * 100.0,
        )
    return excluded


def apply_chat_template(tokenizer: Any, prompts: Sequence[str]) -> List[str]:
    outputs: List[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "qwen3" in str(getattr(tokenizer, "name_or_path", "")).lower():
            kwargs["enable_thinking"] = False
        try:
            outputs.append(tokenizer.apply_chat_template(messages, **kwargs))
        except Exception:
            outputs.append(prompt)
    return outputs


def load_model_and_tokenizer(model_cfg: Mapping[str, Any]) -> Tuple[Any, Any]:
    model_path = resolve_model_reference(model_cfg["path"])
    kwargs = dict(model_cfg.get("kwargs", {}))
    dtype = kwargs.get("torch_dtype")
    if isinstance(dtype, str):
        if dtype == "auto":
            kwargs["torch_dtype"] = "auto"
        elif hasattr(torch, dtype):
            kwargs["torch_dtype"] = getattr(torch, dtype)
        else:
            raise ValueError(f"Invalid torch_dtype: {dtype}")
    if torch.cuda.is_available() and "device_map" not in kwargs:
        kwargs["device_map"] = "auto"

    logger.info("Loading target model for TrajGuard artifacts: %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(kwargs.get("trust_remote_code", True)),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if not torch.cuda.is_available() and "device_map" not in kwargs:
        model.to("cpu")
    return model, tokenizer


@torch.inference_mode()
def extract_hidden_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    context_tokens: int,
    max_length: int,
) -> List[np.ndarray]:
    """Use block hooks to return per-layer [batch, hidden] means over the final k tokens.

    This does not read ``outputs.hidden_states`` directly. Some architectures
    apply an extra final norm to the last item, placing it in a different
    coordinate system from the online decoder-block hook output.
    """
    input_texts = apply_chat_template(tokenizer, prompts)
    encoded = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    device = getattr(model, "device", next(model.parameters()).device)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    attention_mask = encoded["attention_mask"]
    decoder_layers = find_decoder_layers(model)
    per_layer: List[Optional[np.ndarray]] = [None] * len(decoder_layers)
    handles: List[Any] = []

    def make_hook(layer_id: int):
        def hook(_module: Any, _args: Tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, Tensor) or hidden.ndim != 3:
                shape = getattr(hidden, "shape", None)
                raise ValueError(
                    f"Offline hook output for layer {layer_id} must be [batch,tokens,hidden], "
                    f"got {shape}."
                )
            vectors: List[Tensor] = []
            for row in range(hidden.shape[0]):
                valid_positions = torch.nonzero(
                    attention_mask[row],
                    as_tuple=False,
                ).flatten().to(hidden.device)
                if valid_positions.numel() == 0:
                    raise ValueError("Encountered an example with an all-zero attention_mask.")
                selected = valid_positions[-context_tokens:]
                vectors.append(hidden[row, selected, :].mean(dim=0))
            per_layer[layer_id] = (
                torch.stack(vectors, dim=0).float().cpu().numpy()
            )

        return hook

    try:
        for layer_id, layer in enumerate(decoder_layers):
            handles.append(layer.register_forward_hook(make_hook(layer_id)))
        forward_model = getattr(model, "base_model", model)
        forward_model(
            **encoded,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer_id for layer_id, values in enumerate(per_layer) if values is None]
    if missing:
        raise RuntimeError(f"Offline hooks did not capture decoder layers: {missing}")
    return [values for values in per_layer if values is not None]


def _iter_batches(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def fit_incremental_pcas(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    pca_dim: int,
    batch_size: int,
    partial_fit_rows: int,
    context_tokens: int,
    max_length: int,
) -> List[IncrementalPCA]:
    if len(texts) < pca_dim:
        raise ValueError(f"PCA requires at least {pca_dim} examples; got {len(texts)}.")
    if partial_fit_rows < pca_dim:
        raise ValueError("partial_fit_rows cannot be smaller than pca_dim.")

    first = extract_hidden_batch(
        model,
        tokenizer,
        texts[: min(batch_size, len(texts))],
        context_tokens=context_tokens,
        max_length=max_length,
    )
    num_layers = len(first)
    pcas = [IncrementalPCA(n_components=pca_dim) for _ in range(num_layers)]
    buffers: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    buffered_rows = [0 for _ in range(num_layers)]
    fitted_once = [False for _ in range(num_layers)]

    processed_rows = 0
    for prompts in tqdm(
        _iter_batches(texts, batch_size),
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="Fitting per-layer PCA",
    ):
        features = extract_hidden_batch(
            model,
            tokenizer,
            prompts,
            context_tokens=context_tokens,
            max_length=max_length,
        )
        if len(features) != num_layers:
            raise RuntimeError("Different batches returned different hidden-state layer counts.")
        processed_rows += len(prompts)
        remaining_rows = len(texts) - processed_rows
        for layer_id, values in enumerate(features):
            buffers[layer_id].append(values)
            buffered_rows[layer_id] += len(values)
            # If the remaining examples are fewer than the PCA dimension, retain the
            # buffer so tail examples merge with the current chunk and IncrementalPCA does not discard a small final batch.
            can_flush = remaining_rows == 0 or remaining_rows >= pca_dim
            if buffered_rows[layer_id] >= partial_fit_rows and can_flush:
                chunk = np.concatenate(buffers[layer_id], axis=0).astype(np.float32)
                pcas[layer_id].partial_fit(chunk)
                fitted_once[layer_id] = True
                buffers[layer_id].clear()
                buffered_rows[layer_id] = 0

    for layer_id in range(num_layers):
        if buffered_rows[layer_id] >= pca_dim:
            chunk = np.concatenate(buffers[layer_id], axis=0).astype(np.float32)
            pcas[layer_id].partial_fit(chunk)
            fitted_once[layer_id] = True
        elif buffered_rows[layer_id] > 0:
            raise RuntimeError(
                f"The final PCA batch for layer {layer_id} has only {buffered_rows[layer_id]} rows; "
                "the batching logic failed to retain enough examples."
            )
        if not fitted_once[layer_id]:
            raise RuntimeError(f"PCA for layer {layer_id} never completed partial_fit.")
    return pcas


def project_prompts(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    pcas: Sequence[IncrementalPCA],
    *,
    batch_size: int,
    context_tokens: int,
    max_length: int,
    description: str,
) -> Dict[int, np.ndarray]:
    accumulators: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(pcas))}
    for batch in tqdm(
        _iter_batches(prompts, batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc=description,
    ):
        features = extract_hidden_batch(
            model,
            tokenizer,
            batch,
            context_tokens=context_tokens,
            max_length=max_length,
        )
        for layer_id, values in enumerate(features):
            projected = pcas[layer_id].transform(values.astype(np.float32))
            accumulators[layer_id].append(projected.astype(np.float32))
    return {
        layer_id: np.concatenate(chunks, axis=0)
        for layer_id, chunks in accumulators.items()
    }


def mahalanobis(values: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = values.astype(np.float32) - mean.astype(np.float32)
    squared = np.sum((delta @ precision.astype(np.float32)) * delta, axis=1)
    return np.sqrt(np.maximum(squared, 0.0))


def estimate_mvd(
    values: np.ndarray,
    malicious_mean: np.ndarray,
    malicious_precision: np.ndarray,
    malicious_radius: float,
    *,
    rng: np.random.Generator,
    trials: int,
    search_steps: int,
    r_max: float,
) -> float:
    """Reuse the official low-cost MVD estimate with random directions and linear radius search."""
    if len(values) == 0:
        raise ValueError("The MVD selection set cannot be empty.")
    if trials <= 0 or search_steps <= 0 or r_max <= 0.0:
        raise ValueError("MVD trials, search_steps, and r_max must be positive.")
    radii: List[float] = []
    for _ in range(trials):
        sample = values[int(rng.integers(0, len(values)))].astype(np.float32)
        direction = rng.normal(size=sample.shape[0]).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8

        current = mahalanobis(
            sample[None, :],
            malicious_mean,
            malicious_precision,
        )[0]
        if current > malicious_radius:
            radii.append(0.0)
            continue

        found = r_max
        for step in range(1, search_steps + 1):
            radius = r_max * step / search_steps
            candidate = sample + radius * direction
            distance = mahalanobis(
                candidate[None, :],
                malicious_mean,
                malicious_precision,
            )[0]
            if distance > malicious_radius:
                found = float(radius)
                break
        radii.append(found)
    return float(np.median(radii))


def fit_regions_and_select_layers(
    benign: Mapping[int, np.ndarray],
    malicious: Mapping[int, np.ndarray],
    selection: Mapping[int, np.ndarray],
    pcas: Sequence[IncrementalPCA],
    config: Mapping[str, Any],
) -> Tuple[List[int], Dict[int, Dict[str, Any]], Dict[int, Dict[str, float]]]:
    layer_cfg = config["layer_selection"]
    top_k = int(layer_cfg.get("top_k", 8))
    radius_quantile = float(layer_cfg.get("malicious_radius_quantile", 0.9))
    if top_k <= 0 or top_k > len(benign):
        raise ValueError(
            f"layer_selection.top_k must be in [1, {len(benign)}], got {top_k}."
        )
    if not 0.0 < radius_quantile < 1.0:
        raise ValueError("malicious_radius_quantile must be in (0, 1).")
    seed = int(layer_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    stats: Dict[int, Dict[str, Any]] = {}
    diagnostics: Dict[int, Dict[str, float]] = {}
    for layer_id in sorted(benign):
        benign_values = np.asarray(benign[layer_id], dtype=np.float32)
        malicious_values = np.asarray(malicious[layer_id], dtype=np.float32)
        lw_benign = LedoitWolf().fit(benign_values)
        lw_malicious = LedoitWolf().fit(malicious_values)
        benign_mean = lw_benign.location_.astype(np.float32)
        malicious_mean = lw_malicious.location_.astype(np.float32)
        benign_precision = lw_benign.precision_.astype(np.float32)
        malicious_precision = lw_malicious.precision_.astype(np.float32)
        for name, values in (
            ("benign_mean", benign_mean),
            ("malicious_mean", malicious_mean),
            ("benign_precision", benign_precision),
            ("malicious_precision", malicious_precision),
        ):
            if not np.isfinite(values).all():
                raise RuntimeError(f"Layer {layer_id} {name} contains NaN/Inf.")

        d_b_on_b = mahalanobis(benign_values, benign_mean, benign_precision)
        d_m_on_b = mahalanobis(benign_values, malicious_mean, malicious_precision)
        d_b_on_m = mahalanobis(malicious_values, benign_mean, benign_precision)
        d_m_on_m = mahalanobis(malicious_values, malicious_mean, malicious_precision)
        risk_b = d_b_on_b - d_m_on_b
        risk_m = d_b_on_m - d_m_on_m
        radius = float(np.quantile(d_m_on_m, radius_quantile))
        mvd = estimate_mvd(
            np.asarray(selection[layer_id], dtype=np.float32),
            malicious_mean,
            malicious_precision,
            radius,
            rng=rng,
            trials=int(layer_cfg.get("mvd_trials", 20)),
            search_steps=int(layer_cfg.get("mvd_search_steps", 20)),
            r_max=float(layer_cfg.get("mvd_r_max", 3.0)),
        )
        if not np.isfinite(mvd):
            raise RuntimeError(f"MVD for layer {layer_id} is not finite.")

        labels = np.concatenate(
            [np.zeros(len(risk_b), dtype=np.int64), np.ones(len(risk_m), dtype=np.int64)]
        )
        scores = np.concatenate([risk_b, risk_m])
        auroc = float(roc_auc_score(labels, scores))
        fisher = float(
            (risk_m.mean() - risk_b.mean()) ** 2
            / (risk_m.var() + risk_b.var() + 1e-12)
        )
        stats[layer_id] = {
            "pca_mean": np.asarray(pcas[layer_id].mean_, dtype=np.float32),
            "pca_components": np.asarray(pcas[layer_id].components_, dtype=np.float32),
            "benign_mean": benign_mean,
            "benign_precision": benign_precision,
            "malicious_mean": malicious_mean,
            "malicious_precision": malicious_precision,
            "malicious_radius": radius,
        }
        diagnostics[layer_id] = {
            "mvd": mvd,
            "auroc": auroc,
            "fisher_ratio": fisher,
            "benign_risk_mean": float(risk_b.mean()),
            "malicious_risk_mean": float(risk_m.mean()),
            "malicious_radius": radius,
        }
        logger.info(
            "Layer %d: MVD=%.4f AUROC=%.4f Fisher=%.4f risk(B/M)=%.3f/%.3f",
            layer_id,
            mvd,
            auroc,
            fisher,
            risk_b.mean(),
            risk_m.mean(),
        )

    selected = sorted(diagnostics, key=lambda layer: diagnostics[layer]["mvd"])[:top_k]
    quality_cfg = config.get("quality_gate", {})
    selected_mean_auc = float(np.mean([diagnostics[layer]["auroc"] for layer in selected]))
    minimum_auc = float(quality_cfg.get("min_selected_mean_auroc", 0.55))
    if selected_mean_auc < minimum_auc:
        message = (
            f"Mean AUROC of selected layers is {selected_mean_auc:.4f}, below the quality threshold {minimum_auc:.4f}."
        )
        if bool(quality_cfg.get("fail_on_low_separation", True)):
            raise RuntimeError(message)
        logger.warning(message)

    logger.info("Selected TrajGuard decoder layers: %s", selected)
    return selected, stats, diagnostics


def _generation_config(tokenizer: Any, calibration_cfg: Mapping[str, Any]) -> GenerationConfig:
    config = GenerationConfig(
        max_new_tokens=int(calibration_cfg.get("max_new_tokens", 64)),
        do_sample=False,
        num_beams=1,
    )
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    return config


def _stream_trigger_index(scores: Sequence[float], threshold: float, sustain_steps: int) -> int:
    count = 0
    for index, score in enumerate(scores):
        count = count + 1 if score >= threshold else 0
        if count >= sustain_steps:
            return index
    return -1


@torch.inference_mode()
def calibrate_streaming_threshold(
    model: Any,
    tokenizer: Any,
    payload: Dict[str, Any],
    prompts: Sequence[str],
    calibration_cfg: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    provisional = dict(payload)
    provisional["parameters"] = dict(payload["parameters"])
    provisional["parameters"]["threshold"] = 1.0e30
    artifacts = TrajGuardArtifacts(provisional)
    guard = TrajGuard(
        artifacts,
        tokenizer,
        use_pair_judge=False,
        monitor_only=True,
        enforce_chat_template_hash=True,
    )
    generation_config = _generation_config(tokenizer, calibration_cfg)
    sequences: List[List[float]] = []

    with guard.attach(model):
        for prompt in tqdm(prompts, desc="Calibrating benign streaming threshold"):
            text = apply_chat_template(tokenizer, [prompt])[0]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(calibration_cfg.get("max_prompt_length", 2048)),
            )
            device = getattr(model, "device", next(model.parameters()).device)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            guard.prepare_request(prompt)
            model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=True,
            )
            if guard.last_log is None:
                raise RuntimeError("No TrajGuard request log was produced after calibration generation.")
            sequences.append(list(guard.last_log.streaming_scores))

    flattened = [score for sequence in sequences for score in sequence]
    if not flattened:
        raise RuntimeError("Benign calibration produced no streaming scores.")
    quantile = float(calibration_cfg.get("threshold_quantile", 0.995))
    if not 0.0 < quantile < 1.0:
        raise ValueError("calibration.threshold_quantile must be in (0, 1).")
    threshold = float(np.quantile(np.asarray(flattened, dtype=np.float64), quantile))
    sustain_steps = int(payload["parameters"]["sustain_steps"])
    trigger_indices = [
        _stream_trigger_index(sequence, threshold, sustain_steps)
        for sequence in sequences
    ]
    triggered = [index for index in trigger_indices if index >= 0]
    report = {
        "threshold": threshold,
        "threshold_quantile": quantile,
        "num_validation_prompts": len(sequences),
        "num_streaming_scores": len(flattened),
        "token_level_exceedance_rate": float(
            np.mean(np.asarray(flattened) >= threshold)
        ),
        "sequence_level_trigger_rate": float(len(triggered) / max(1, len(sequences))),
        "triggered_sequences": len(triggered),
        "median_score": float(np.median(flattened)),
        "max_score": float(np.max(flattened)),
        "median_trigger_step": (
            float(np.median(triggered)) if triggered else None
        ),
        # Every sequence with a sustained trigger calls the judge at least once.
        # Retriggering after SAFE requires an actual judge result and cannot be inferred during SGS-only calibration.
        "minimum_judge_call_rate": float(
            len(triggered) / max(1, len(sequences))
        ),
    }
    return threshold, report


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    return value


def build_payload(
    config: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    selected_layers: Sequence[int],
    stats: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    extraction_cfg = config["extraction"]
    streaming_cfg = config["streaming"]
    pca_dim = int(extraction_cfg.get("pca_dim", 64))
    model_cfg = config["model"]
    config_hidden_size = getattr(model.config, "hidden_size", None)
    hidden_size = (
        int(config_hidden_size)
        if config_hidden_size is not None
        else int(np.asarray(stats[int(selected_layers[0])]["pca_mean"]).shape[0])
    )
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "trajguard_paper_consistent",
        "model": {
            "name": str(model_cfg["name"]),
            "path": str(model_cfg["path"]),
            "config_name_or_path": str(getattr(model.config, "name_or_path", "")),
            "num_layers": len(find_decoder_layers(model)),
            "hidden_size": hidden_size,
            "chat_template_sha256": tokenizer_chat_template_hash(tokenizer),
            "layer_index_semantics": (
                "decoder_block_0_based; offline_hidden_states_index=layer_id+1"
            ),
        },
        "parameters": {
            "context_tokens": int(extraction_cfg.get("context_tokens", 3)),
            "pca_dim": pca_dim,
            "window_size": int(streaming_cfg.get("window_size", 8)),
            "trim_count": int(streaming_cfg.get("trim_count", 1)),
            "ewma_lambda": float(streaming_cfg.get("ewma_lambda", 0.8)),
            "sustain_steps": int(streaming_cfg.get("sustain_steps", 3)),
            "threshold_quantile": float(
                config["calibration"].get("threshold_quantile", 0.995)
            ),
            "threshold": 0.0,
        },
        "selected_layers": [int(layer) for layer in selected_layers],
        "layers": {},
    }
    for layer_id in selected_layers:
        layer_stats = stats[int(layer_id)]
        payload["layers"][str(layer_id)] = {
            key: torch.as_tensor(layer_stats[key], dtype=torch.float32).cpu()
            for key in (
                "pca_mean",
                "pca_components",
                "benign_mean",
                "benign_precision",
                "malicious_mean",
                "malicious_precision",
            )
        }
    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            _json_ready(payload),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The top level of the TrajGuard artifact configuration must be a mapping.")
    for key in (
        "model",
        "datasets",
        "extraction",
        "layer_selection",
        "streaming",
        "calibration",
        "output",
    ):
        if key not in config:
            raise KeyError(f"Configuration is missing top-level field: {key}")
    return config


def load_all_splits(config: Mapping[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    seed = int(config.get("seed", 42))
    dataset_cfg = config["datasets"]
    required = (
        "benign_reference",
        "malicious_reference",
        "layer_selection",
        "benign_validation",
    )
    split_texts: Dict[str, List[str]] = {}
    sources: Dict[str, List[str]] = {}
    for name in required:
        texts, files = load_text_split(name, dataset_cfg[name], default_seed=seed)
        split_texts[name] = texts
        sources[name] = files

    evaluation_specs = dataset_cfg.get("evaluation_sets", [])
    evaluation_texts: List[str] = []
    evaluation_files: List[str] = []
    for index, spec in enumerate(evaluation_specs):
        texts, files = load_text_split(
            f"evaluation_{index}",
            spec,
            default_seed=seed,
            collect_all_fields=True,
        )
        evaluation_texts.extend(texts)
        evaluation_files.extend(files)
    if evaluation_texts:
        split_texts["evaluation"] = list(
            {text_hash(text): text for text in evaluation_texts}.values()
        )
        sources["evaluation"] = evaluation_files

    excluded_evaluation_overlaps: Dict[str, List[str]] = {}
    if bool(dataset_cfg.get("drop_evaluation_overlaps", False)):
        excluded_evaluation_overlaps = drop_evaluation_overlaps(
            split_texts,
            max_fraction=float(
                dataset_cfg.get("max_evaluation_overlap_fraction", 0.01)
            ),
        )

    hashes = validate_data_isolation(
        split_texts,
        strict_internal_disjoint=bool(
            dataset_cfg.get("strict_internal_disjoint", True)
        ),
    )
    manifest = {
        "sources": sources,
        "counts": {name: len(values) for name, values in split_texts.items()},
        "hashes": hashes,
        "excluded_evaluation_overlaps": excluded_evaluation_overlaps,
    }
    return split_texts, manifest


def run_build(config: Dict[str, Any], *, dry_run: bool = False) -> Path:
    split_texts, data_manifest = load_all_splits(config)
    output_cfg = config["output"]
    output_dir = resolve_baseline_output_path(
        str(output_cfg["artifact_dir"]).format(llm_name=config["model"]["name"])
    )

    if dry_run:
        logger.info(
            "Dry-run passed without writing files. Split counts: %s",
            data_manifest["counts"],
        )
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "data_manifest.json", data_manifest)
    hashes = data_manifest["hashes"]
    save_json(
        output_dir / "reference_hashes.json",
        {
            "benign_reference": hashes["benign_reference"],
            "malicious_reference": hashes["malicious_reference"],
        },
    )
    save_json(
        output_dir / "layer_selection_hashes.json",
        hashes["layer_selection"],
    )
    save_json(
        output_dir / "threshold_validation_hashes.json",
        hashes["benign_validation"],
    )
    save_json(
        output_dir / "evaluation_hashes.json",
        hashes.get("evaluation", []),
    )
    save_yaml(output_dir / "artifact_config_resolved.yaml", config)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_model_and_tokenizer(config["model"])
    extraction_cfg = config["extraction"]
    pca_dim = int(extraction_cfg.get("pca_dim", 64))
    batch_size = int(extraction_cfg.get("batch_size", 4))
    context_tokens = int(extraction_cfg.get("context_tokens", 3))
    max_length = int(extraction_cfg.get("max_length", 1024))
    reference_texts = (
        split_texts["benign_reference"] + split_texts["malicious_reference"]
    )

    pcas = fit_incremental_pcas(
        model,
        tokenizer,
        reference_texts,
        pca_dim=pca_dim,
        batch_size=batch_size,
        partial_fit_rows=int(extraction_cfg.get("partial_fit_rows", 256)),
        context_tokens=context_tokens,
        max_length=max_length,
    )
    benign_projected = project_prompts(
        model,
        tokenizer,
        split_texts["benign_reference"],
        pcas,
        batch_size=batch_size,
        context_tokens=context_tokens,
        max_length=max_length,
        description="Projecting benign reference",
    )
    malicious_projected = project_prompts(
        model,
        tokenizer,
        split_texts["malicious_reference"],
        pcas,
        batch_size=batch_size,
        context_tokens=context_tokens,
        max_length=max_length,
        description="Projecting malicious reference",
    )
    selection_projected = project_prompts(
        model,
        tokenizer,
        split_texts["layer_selection"],
        pcas,
        batch_size=batch_size,
        context_tokens=context_tokens,
        max_length=max_length,
        description="Projecting layer-selection jailbreaks",
    )

    if bool(output_cfg.get("cache_projected_activations", False)):
        cache_dir = output_dir / "projected_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, values in (
            ("benign", benign_projected),
            ("malicious", malicious_projected),
            ("selection", selection_projected),
        ):
            np.savez_compressed(
                cache_dir / f"{name}.npz",
                **{f"layer_{layer}": array for layer, array in values.items()},
            )

    selected, stats, diagnostics = fit_regions_and_select_layers(
        benign_projected,
        malicious_projected,
        selection_projected,
        pcas,
        config,
    )
    payload = build_payload(config, model, tokenizer, selected, stats)

    calibration_cfg = config["calibration"]
    if bool(calibration_cfg.get("enabled", True)):
        threshold, calibration_report = calibrate_streaming_threshold(
            model,
            tokenizer,
            payload,
            split_texts["benign_validation"],
            calibration_cfg,
        )
    else:
        if "manual_threshold" not in calibration_cfg:
            raise ValueError("manual_threshold is required when calibration.enabled=false.")
        threshold = float(calibration_cfg["manual_threshold"])
        calibration_report = {
            "threshold": threshold,
            "mode": "manual",
            "warning": "Not a main paper reproduction configuration: benign streaming-threshold calibration was not run.",
        }
    payload["parameters"]["threshold"] = threshold

    # Run the full runtime validation once more before saving.
    TrajGuardArtifacts(payload)
    artifact_path = output_dir / ARTIFACT_FILENAME
    temporary_artifact_path = artifact_path.with_suffix(
        artifact_path.suffix + ".tmp"
    )
    torch.save(payload, temporary_artifact_path)
    temporary_artifact_path.replace(artifact_path)
    save_json(
        output_dir / "selected_layers.json",
        {
            "selected_layers": selected,
            "diagnostics": diagnostics,
        },
    )
    save_json(output_dir / "calibration_report.json", calibration_report)
    save_json(
        output_dir / "artifact_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "method": payload["method"],
            "model": payload["model"],
            "parameters": payload["parameters"],
            "selected_layers": selected,
            "artifact_path": str(artifact_path),
        },
    )
    save_yaml(
        output_dir / "trajguard_params.yaml",
        {
            "method": payload["method"],
            "model": payload["model"],
            "parameters": payload["parameters"],
            "selected_layers": selected,
        },
    )
    logger.info("TrajGuard artifacts saved to %s", output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-consistent TrajGuard artifacts.")
    parser.add_argument(
        "--config",
        default="scripts/baselines/trajguard/artifact_config.yaml",
        help="YAML configuration relative to the project root or absolute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data paths, fields, sample counts, and hash isolation without loading a model.",
    )
    parser.add_argument(
        "--llm-name",
        help="Override the configured model name and replace the model path's final directory with that name.",
    )
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if args.llm_name:
        apply_llm_name_override(config, args.llm_name)
    run_build(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
