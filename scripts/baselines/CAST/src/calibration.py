from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .modeling import format_user_prompt, model_input_device
from .progress import progress_iter


OFFICIAL_DIRECTIONS = ("larger", "smaller")


def effective_operator(official_direction: str) -> str:
    """Return the actual operator explicitly because official names invert the comparison."""
    if official_direction == "smaller":
        return ">"
    if official_direction == "larger":
        return "<"
    raise ValueError(f"Unknown official comparator: {official_direction}")


def condition_matches(
    score: float,
    threshold: float,
    official_direction: str,
) -> bool:
    operator = effective_operator(official_direction)
    return score > threshold if operator == ">" else score < threshold


def condition_similarity(
    hidden_state: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Reproduce LeashLayer: cos(h, tanh(vv^T h / ||v||^2))."""
    hidden = hidden_state.flatten()
    vector = direction.to(device=hidden.device, dtype=hidden.dtype).flatten()
    denominator = torch.dot(vector, vector)
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("The condition vector must have a finite, nonzero norm.")
    projected = vector * (torch.dot(vector, hidden) / denominator)
    projected = torch.tanh(projected)
    norm_product = torch.linalg.vector_norm(hidden) * torch.linalg.vector_norm(projected)
    if not torch.isfinite(norm_product) or norm_product <= 0:
        return torch.tensor(float("nan"), device=hidden.device)
    return torch.dot(hidden, projected) / norm_product


def resolve_candidate_layers(
    num_layers: int,
    calibration_config: Mapping[str, Any],
) -> list[int]:
    start = int(calibration_config.get("layer_start", 1))
    explicit_end = calibration_config.get("layer_end_exclusive")
    if explicit_end is None:
        fraction = float(calibration_config.get("layer_end_fraction", 0.4))
        end = int(math.ceil(fraction * num_layers))
    else:
        end = int(explicit_end)
    end = min(num_layers, end)
    if not 0 <= start < end:
        raise ValueError(
            f"Invalid condition-layer range [{start}, {end}); the model has {num_layers} layers."
        )
    return list(range(start, end))


def resolve_behavior_layers(
    num_layers: int,
    runtime_config: Mapping[str, Any],
) -> list[int]:
    explicit = runtime_config.get("behavior_layers")
    if explicit is not None:
        layers = [int(value) for value in explicit]
    else:
        start = math.floor(
            float(runtime_config.get("behavior_start_fraction", 0.47)) * num_layers
        )
        end = math.ceil(
            float(runtime_config.get("behavior_end_fraction", 0.72)) * num_layers
        )
        layers = list(range(start, min(end, num_layers)))
    if not layers or min(layers) < 0 or max(layers) >= num_layers:
        raise ValueError(f"Invalid behavior layers: {layers}")
    return layers


@torch.inference_mode()
def collect_condition_scores(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    directions: Mapping[int, np.ndarray],
    layer_ids: Sequence[int],
    *,
    max_length: int | None,
) -> dict[int, list[float]]:
    scores = {int(layer): [] for layer in layer_ids}
    device = model_input_device(model)
    for prompt in progress_iter(
        prompts,
        desc="CAST condition scoring",
        total=len(prompts),
        unit="prompt",
        leave=False,
    ):
        formatted = format_user_prompt(
            tokenizer,
            prompt,
            add_generation_prompt=True,
        )
        tokenize_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
        }
        if max_length is not None:
            tokenize_kwargs["max_length"] = max_length
        inputs = tokenizer(formatted, **tokenize_kwargs).to(device)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        attention_mask = inputs.get("attention_mask")
        for layer_id in layer_ids:
            # hidden_states[layer_id] is the input to decoder block layer_id,
            # matching the tensor seen by the official LeashLayer before that block.
            hidden = outputs.hidden_states[layer_id][0]
            if attention_mask is None:
                pooled = hidden.mean(dim=0)
            else:
                mask = attention_mask[0].to(hidden.device, hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=0) / mask.sum().clamp_min(1)
            direction = torch.as_tensor(directions[layer_id], device=hidden.device)
            value = condition_similarity(pooled, direction).item()
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite condition score: layer={layer_id}, prompt={prompt[:80]!r}"
                )
            scores[int(layer_id)].append(float(value))
        del outputs
    return scores


@dataclass(frozen=True)
class ConditionPoint:
    layer: int
    threshold: float
    official_direction: str
    effective_operator: str
    f1: float
    auroc: float


def threshold_values(
    threshold_range: Sequence[float],
    threshold_step: float,
) -> np.ndarray:
    if len(threshold_range) != 2:
        raise ValueError("threshold_range must be [start, stop].")
    start, stop = map(float, threshold_range)
    if threshold_step <= 0 or stop <= start:
        raise ValueError("Invalid threshold_range/threshold_step.")
    return np.arange(start, stop, float(threshold_step))


def search_condition_point(
    scores_by_layer: Mapping[int, Sequence[float]],
    labels: Sequence[int],
    thresholds: Sequence[float],
) -> tuple[ConditionPoint, dict[str, Any]]:
    y_true = np.asarray(labels, dtype=np.int64)
    if len(set(y_true.tolist())) != 2:
        raise ValueError("Condition calibration must contain both positive and negative examples.")
    best: ConditionPoint | None = None
    layer_reports: dict[str, Any] = {}
    for layer_id in sorted(scores_by_layer):
        scores = np.asarray(scores_by_layer[layer_id], dtype=np.float64)
        if scores.shape[0] != y_true.shape[0]:
            raise ValueError(f"Score count and label count differ for layer {layer_id}.")
        auroc = float(roc_auc_score(y_true, scores))
        layer_best_f1 = -1.0
        layer_best: dict[str, Any] | None = None
        for threshold in thresholds:
            for direction in OFFICIAL_DIRECTIONS:
                predictions = np.fromiter(
                    (
                        int(condition_matches(score, float(threshold), direction))
                        for score in scores
                    ),
                    dtype=np.int64,
                    count=len(scores),
                )
                score_f1 = float(f1_score(y_true, predictions, zero_division=0))
                candidate = ConditionPoint(
                    layer=int(layer_id),
                    threshold=float(threshold),
                    official_direction=direction,
                    effective_operator=effective_operator(direction),
                    f1=score_f1,
                    auroc=auroc,
                )
                if best is None or candidate.f1 > best.f1:
                    best = candidate
                if score_f1 > layer_best_f1:
                    layer_best_f1 = score_f1
                    layer_best = {
                        "threshold": float(threshold),
                        "official_direction": direction,
                        "effective_operator": effective_operator(direction),
                        "f1": score_f1,
                    }
        layer_reports[str(layer_id)] = {
            "auroc_harmful_positive": auroc,
            "harmful_scores": scores[y_true == 1].tolist(),
            "benign_scores": scores[y_true == 0].tolist(),
            "best": layer_best,
        }
    if best is None:
        raise RuntimeError("No condition point was found.")
    return best, layer_reports
