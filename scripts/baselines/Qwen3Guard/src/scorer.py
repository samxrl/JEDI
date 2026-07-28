from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .backend import GuardBackend
from .paths import sha256_text
from .schema import GuardBatchResult, GuardDecision


ScoreKey = tuple[str, int]


def score_key(row: Mapping[str, Any]) -> ScoreKey:
    return str(row["sample_id"]), int(row["checkpoint_end"])


def expected_score_keys(
    samples: Sequence[Mapping[str, Any]],
) -> set[ScoreKey]:
    return {
        (str(sample["sample_id"]), int(end))
        for sample in samples
        for end in sample["checkpoint_ends"]
    }


def _decode_prefix(
    tokenizer: Any,
    sample: Mapping[str, Any],
    checkpoint_end: int,
) -> str:
    return tokenizer.decode(
        [int(value) for value in sample["target_token_ids"][:checkpoint_end]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _error_batch(
    size: int,
    exc: Exception,
    *,
    error_policy: str,
) -> GuardBatchResult:
    if error_policy == "raise":
        raise exc
    is_safe = error_policy == "fail_open"
    status = (
        "backend_error_fail_open"
        if is_safe
        else "backend_error_fail_closed"
    )
    error = f"{type(exc).__name__}: {exc}"
    return GuardBatchResult(
        decisions=[
            GuardDecision(
                raw_output="",
                is_safe=is_safe,
                verdict="safe" if is_safe else "unsafe",
                categories=[],
                parse_status=status,
                error=error,
            )
            for _ in range(size)
        ],
        generate_ms=0.0,
        e2e_ms=0.0,
        batch_size=size,
    )


def score_prepared_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    backend: GuardBackend,
    batch_size: int,
    completed: set[ScoreKey] | None = None,
    error_policy: str = "fail_closed",
    store_raw_outputs: bool = True,
    guard_model: str | None = None,
    on_batch: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Score batches by checkpoint round, skipping existing keys and persisting each batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if error_policy not in {"fail_closed", "fail_open", "raise"}:
        raise ValueError("error_policy must be fail_closed/fail_open/raise.")
    done = set(completed or set())
    rows: list[dict[str, Any]] = []
    max_checkpoints = max(
        (len(sample["checkpoint_ends"]) for sample in samples),
        default=0,
    )
    for checkpoint_offset in range(max_checkpoints):
        pending: list[tuple[Mapping[str, Any], int]] = []
        for sample in samples:
            ends = sample["checkpoint_ends"]
            if checkpoint_offset >= len(ends):
                continue
            checkpoint_end = int(ends[checkpoint_offset])
            key = (str(sample["sample_id"]), checkpoint_end)
            if key not in done:
                pending.append((sample, checkpoint_end))

        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            prefixes = [
                _decode_prefix(tokenizer, sample, checkpoint_end)
                for sample, checkpoint_end in batch
            ]
            conversations = [
                (
                    str(sample["source_record"].get("prompt") or ""),
                    prefix,
                )
                for (sample, _), prefix in zip(batch, prefixes)
            ]
            try:
                result = backend.classify_batch(conversations)
            except Exception as exc:
                result = _error_batch(
                    len(batch),
                    exc,
                    error_policy=error_policy,
                )
            if len(result.decisions) != len(batch):
                raise RuntimeError(
                    "The Qwen3Guard backend returned a verdict count that differs from the input batch size."
                )
            actual_batch = max(int(result.batch_size), 1)
            amortized_generate = result.generate_ms / actual_batch
            amortized_e2e = result.e2e_ms / actual_batch
            batch_rows: list[dict[str, Any]] = []
            for (sample, checkpoint_end), prefix, decision in zip(
                batch,
                prefixes,
                result.decisions,
            ):
                verdict = str(decision.verdict).strip().lower()
                if verdict not in {"safe", "unsafe", "controversial"}:
                    raise RuntimeError(
                        "The Qwen3Guard backend still returned an unresolved label after policy handling: "
                        f"{decision.verdict!r}"
                    )
                row = {
                    "sample_id": sample["sample_id"],
                    "target_model": sample["target_model"],
                    "guard_model": guard_model,
                    "checkpoint_index": checkpoint_offset + 1,
                    "checkpoint_end": checkpoint_end,
                    "total_target_tokens": int(
                        sample["total_target_tokens"]
                    ),
                    "prompt_sha256": sample["prompt_sha256"],
                    "prefix_sha256": sha256_text(prefix),
                    "guard_input_tokens": decision.input_tokens,
                    "guard_output_tokens": decision.output_tokens,
                    "label": verdict,
                    "is_safe": bool(decision.is_safe),
                    "categories": list(decision.categories),
                    "refusal": decision.refusal,
                    "risk_probability": decision.risk_probability,
                    "category_probability": (
                        decision.category_probability
                    ),
                    "parse_status": decision.parse_status,
                    "raw_guard_output": (
                        decision.raw_output if store_raw_outputs else None
                    ),
                    "guard_generate_ms": float(result.generate_ms),
                    "guard_e2e_ms": float(result.e2e_ms),
                    "amortized_guard_generate_ms": amortized_generate,
                    "amortized_guard_e2e_ms": amortized_e2e,
                    "batch_size": int(result.batch_size),
                    "batch_retry_count": int(result.retry_count),
                    "guard_error": decision.error,
                }
                batch_rows.append(row)
                done.add(score_key(row))
            if on_batch is not None:
                on_batch(batch_rows)
            rows.extend(batch_rows)
    return rows
