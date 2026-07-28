from __future__ import annotations

import hashlib
from typing import Any

from src.input_adapter import (
    build_checkpoint_plan,
    candidate_checkpoint_ends,
)
from src.materializer import materialize_sample


class FakeTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(str(value) for value in token_ids)


def _sample(total: int = 70) -> dict[str, Any]:
    token_ids = list(range(1, total + 1))
    union, by_chunk = build_checkpoint_plan(total, [16, 32])
    raw = " ".join(str(value) for value in token_ids)
    return {
        "sample_id": "sample",
        "target_model": "model",
        "source_file": "source.csv",
        "source_file_sha256": "source-hash",
        "source_row_index": 0,
        "source_record": {
            "condition": "baseline",
            "assistant_output": raw,
            "prompt": "prompt",
            "label": "yes",
            "eval_split": "safety",
            "attack_method": "GCG",
        },
        "source_response": raw,
        "source_response_sha256": hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest(),
        "prompt_sha256": "prompt-hash",
        "target_token_ids": token_ids,
        "total_target_tokens": total,
        "checkpoint_ends": union,
        "checkpoint_ends_by_chunk": by_chunk,
        "target_tokenizer_path": "tokenizer",
        "target_tokenizer_revision": None,
        "roundtrip_decoded_sha256": "roundtrip",
        "roundtrip_exact_match": True,
    }


def _score(end: int, label: str) -> dict[str, Any]:
    return {
        "checkpoint_end": end,
        "label": label,
        "is_safe": label == "safe",
        "categories": ["Violent"] if label == "unsafe" else [],
        "refusal": "no",
        "amortized_guard_generate_ms": 1.0,
        "amortized_guard_e2e_ms": 2.0,
        "guard_generate_ms": 4.0,
        "guard_e2e_ms": 5.0,
        "guard_error": None,
    }


def test_checkpoint_plan_shares_grid_and_checks_final_partial() -> None:
    assert candidate_checkpoint_ends(70, 16) == [16, 32, 48, 64, 70]
    assert candidate_checkpoint_ends(70, 32) == [32, 64, 70]
    union, by_chunk = build_checkpoint_plan(70, [16, 32])
    assert union == [16, 32, 48, 64, 70]
    assert by_chunk["32"] == [32, 64, 70]
    assert candidate_checkpoint_ends(7, 16) == [7]


def test_zero_token_response_needs_no_guard_check() -> None:
    sample = _sample(total=0)
    result = materialize_sample(
        sample,
        {},
        tokenizer=FakeTokenizer(),
        chunk_size=16,
        refusal_text="REFUSE",
    )

    assert result["assistant_output"] == ""
    assert result["num_guard_checks"] == 0
    assert result["triggered"] is False
    assert result["released_token_count"] == 0


def test_materialize_16_and_32_use_their_own_checkpoints() -> None:
    sample = _sample()
    scores = {
        16: _score(16, "safe"),
        32: _score(32, "safe"),
        48: _score(48, "unsafe"),
        64: _score(64, "unsafe"),
        70: _score(70, "safe"),
    }
    result_16 = materialize_sample(
        sample,
        scores,
        tokenizer=FakeTokenizer(),
        chunk_size=16,
        refusal_text="REFUSE",
    )
    result_32 = materialize_sample(
        sample,
        scores,
        tokenizer=FakeTokenizer(),
        chunk_size=32,
        refusal_text="REFUSE",
    )

    assert result_16["trigger_token_end"] == 48
    assert result_16["released_token_count"] == 32
    assert result_16["num_guard_checks"] == 3
    assert result_32["trigger_token_end"] == 64
    assert result_32["released_token_count"] == 32
    assert result_32["num_guard_checks"] == 2
    assert result_32["assistant_output"].endswith("\n\nREFUSE")


def test_controversial_label_triggers_when_policy_decision_blocks() -> None:
    sample = _sample(total=16)
    scores = {
        16: {
            **_score(16, "controversial"),
            "is_safe": False,
            "categories": ["Political"],
        }
    }
    result = materialize_sample(
        sample,
        scores,
        tokenizer=FakeTokenizer(),
        chunk_size=16,
        refusal_text="REFUSE",
    )

    assert result["triggered"] is True
    assert result["trigger_guard_label"] == "controversial"
    assert result["released_token_count"] == 0


def test_controversial_label_releases_when_policy_decision_allows() -> None:
    sample = _sample(total=16)
    scores = {
        16: {
            **_score(16, "controversial"),
            "is_safe": True,
            "categories": ["Political"],
        }
    }
    result = materialize_sample(
        sample,
        scores,
        tokenizer=FakeTokenizer(),
        chunk_size=16,
        refusal_text="REFUSE",
    )

    assert result["triggered"] is False
    assert result["assistant_output"] == sample["source_response"]
    assert result["first_guard_label"] == "controversial"
    assert result["first_guard_is_safe"] is True


def test_final_partial_can_trigger_and_untriggered_output_is_exact() -> None:
    sample = _sample()
    final_trigger_scores = {
        end: _score(end, "unsafe" if end == 70 else "safe")
        for end in sample["checkpoint_ends"]
    }
    triggered = materialize_sample(
        sample,
        final_trigger_scores,
        tokenizer=FakeTokenizer(),
        chunk_size=32,
        refusal_text="REFUSE",
    )
    assert triggered["trigger_token_end"] == 70
    assert triggered["released_token_count"] == 64

    all_safe = {
        end: _score(end, "safe") for end in sample["checkpoint_ends"]
    }
    sample["source_response"] += "  "
    sample["source_record"]["assistant_output"] = sample["source_response"]
    sample["source_response_sha256"] = hashlib.sha256(
        sample["source_response"].encode("utf-8")
    ).hexdigest()
    unchanged = materialize_sample(
        sample,
        all_safe,
        tokenizer=FakeTokenizer(),
        chunk_size=16,
        refusal_text="REFUSE",
    )
    assert unchanged["assistant_output"].endswith("  ")
    assert unchanged["protected_output_sha256"] == sample[
        "source_response_sha256"
    ]
