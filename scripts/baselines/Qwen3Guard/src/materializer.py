from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .paths import sha256_text


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _categories(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
    return []


def _decode_prefix(
    tokenizer: Any,
    token_ids: Sequence[int],
    end: int,
) -> str:
    if end <= 0:
        return ""
    return tokenizer.decode(
        [int(value) for value in token_ids[:end]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def materialize_sample(
    sample: Mapping[str, Any],
    score_rows: Mapping[int, Mapping[str, Any]],
    *,
    tokenizer: Any,
    chunk_size: int,
    refusal_text: str,
    condition_prefix: str = "qwen3guard_gen",
    guard_model: str | None = None,
) -> dict[str, Any]:
    """Derive a user-visible response with real chunk-buffer semantics from shared checkpoint labels."""
    size_key = str(int(chunk_size))
    by_chunk = sample["checkpoint_ends_by_chunk"]
    if size_key not in by_chunk:
        raise KeyError(
            f"Sample {sample['sample_id']} was not prepared for chunk_size={chunk_size}."
        )
    candidates = [int(value) for value in by_chunk[size_key]]
    checked: list[Mapping[str, Any]] = []
    trigger: Mapping[str, Any] | None = None
    released_end = 0
    for checkpoint_end in candidates:
        if checkpoint_end not in score_rows:
            raise KeyError(
                f"Sample {sample['sample_id']} is missing a Qwen3Guard label "
                f"for checkpoint={checkpoint_end}."
            )
        score = score_rows[checkpoint_end]
        label = str(score.get("label", "")).strip().lower()
        if label not in {"safe", "unsafe", "controversial"}:
            raise ValueError(
                f"Sample {sample['sample_id']} at checkpoint={checkpoint_end} "
                f"has an unresolved label: {label!r}"
            )
        is_safe = score.get("is_safe")
        if not isinstance(is_safe, bool):
            raise TypeError(
                f"Sample {sample['sample_id']} at checkpoint={checkpoint_end} "
                "is missing a Boolean is_safe release decision."
            )
        checked.append(score)
        if not is_safe:
            trigger = score
            break
        released_end = checkpoint_end

    source = dict(sample["source_record"])
    source_response = str(sample["source_response"])
    if trigger is None:
        # Reuse the source response verbatim when not triggered to avoid tokenizer round trips changing spaces.
        protected = source_response
        trigger_end = None
        trigger_start = None
    else:
        trigger_end = int(trigger["checkpoint_end"])
        trigger_start = released_end
        released_prefix = _decode_prefix(
            tokenizer,
            sample["target_token_ids"],
            released_end,
        )
        protected = (
            str(refusal_text)
            if not released_prefix
            else released_prefix.rstrip() + "\n\n" + str(refusal_text)
        )

    unchanged = sha256_text(protected) == sample["source_response_sha256"]
    source_label = source.get("label")
    normalized_source_label = str(source_label or "").strip().lower()
    source_label_reused = (
        unchanged and normalized_source_label in {"yes", "no"}
    )
    amortized_generate = [
        float(row.get("amortized_guard_generate_ms") or 0.0)
        for row in checked
    ]
    amortized_e2e = [
        float(row.get("amortized_guard_e2e_ms") or 0.0)
        for row in checked
    ]
    blocking_generate = [
        float(row.get("guard_generate_ms") or 0.0) for row in checked
    ]
    blocking_e2e = [
        float(row.get("guard_e2e_ms") or 0.0) for row in checked
    ]
    errors = [
        str(row["guard_error"])
        for row in checked
        if row.get("guard_error")
    ]
    first = checked[0] if checked else None
    total_tokens = int(sample["total_target_tokens"])

    output = source
    output.update(
        {
            "sample_id": sample["sample_id"],
            "target_model": sample["target_model"],
            "guard_model": guard_model,
            "source_file": sample["source_file"],
            "source_file_sha256": sample["source_file_sha256"],
            "source_row_index": sample["source_row_index"],
            "source_response_sha256": sample["source_response_sha256"],
            "prompt_sha256": sample["prompt_sha256"],
            "source_condition": source.get("condition"),
            "source_label": source_label,
            "condition": f"{condition_prefix}_c{int(chunk_size)}",
            "assistant_output": protected,
            "protected_output_sha256": sha256_text(protected),
            "chunk_size": int(chunk_size),
            "prefix_mode": "cumulative",
            "total_target_tokens": total_tokens,
            "num_guard_checks": len(checked),
            "triggered": trigger is not None,
            "trigger_checkpoint": trigger_end,
            "trigger_token_start": trigger_start,
            "trigger_token_end": trigger_end,
            "released_token_count": released_end,
            "blocked_tokens_at_trigger": (
                trigger_end - released_end if trigger_end is not None else 0
            ),
            "trigger_fraction": (
                trigger_end / total_tokens
                if trigger_end is not None and total_tokens > 0
                else None
            ),
            "zero_original_token_exposure": (
                trigger is not None and released_end == 0
            ),
            "first_guard_label": (
                str(first["label"]) if first is not None else None
            ),
            "first_guard_is_safe": (
                bool(first["is_safe"]) if first is not None else None
            ),
            "first_guard_categories": (
                _categories(first.get("categories")) if first is not None else []
            ),
            "first_guard_refusal": (
                first.get("refusal") if first is not None else None
            ),
            "first_guard_risk_probability": (
                first.get("risk_probability")
                if first is not None
                else None
            ),
            "first_guard_category_probability": (
                first.get("category_probability")
                if first is not None
                else None
            ),
            "trigger_guard_label": (
                str(trigger["label"]) if trigger is not None else None
            ),
            "trigger_guard_is_safe": (
                bool(trigger["is_safe"]) if trigger is not None else None
            ),
            "trigger_guard_categories": (
                _categories(trigger.get("categories"))
                if trigger is not None
                else []
            ),
            "trigger_guard_refusal": (
                trigger.get("refusal") if trigger is not None else None
            ),
            "trigger_guard_risk_probability": (
                trigger.get("risk_probability")
                if trigger is not None
                else None
            ),
            "trigger_guard_category_probability": (
                trigger.get("category_probability")
                if trigger is not None
                else None
            ),
            "guard_generate_ms_total": sum(amortized_generate),
            "guard_e2e_ms_total": sum(amortized_e2e),
            "blocking_guard_generate_ms_total": sum(blocking_generate),
            "blocking_guard_e2e_ms_total": sum(blocking_e2e),
            "guard_generate_ms_mean": (
                statistics.fmean(amortized_generate)
                if amortized_generate
                else 0.0
            ),
            "guard_generate_ms_p95": _percentile(
                amortized_generate,
                0.95,
            ),
            "guard_e2e_ms_mean": (
                statistics.fmean(amortized_e2e) if amortized_e2e else 0.0
            ),
            "guard_e2e_ms_p95": _percentile(amortized_e2e, 0.95),
            "guard_ms_until_trigger": (
                sum(amortized_e2e) if trigger is not None else None
            ),
            "blocking_guard_ms_until_trigger": (
                sum(blocking_e2e) if trigger is not None else None
            ),
            "guard_error": " | ".join(errors) if errors else None,
            "source_label_reused": source_label_reused,
            "label": (
                normalized_source_label if source_label_reused else None
            ),
            "target_tokenizer_path": sample["target_tokenizer_path"],
            "target_tokenizer_revision": sample[
                "target_tokenizer_revision"
            ],
            "roundtrip_decoded_sha256": sample[
                "roundtrip_decoded_sha256"
            ],
            "roundtrip_exact_match": sample["roundtrip_exact_match"],
        }
    )
    # Compatible with JEDI detailed results; here this is the target-token boundary checked so far.
    output["trigger_step"] = trigger_end
    return output
