from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.backend import (
    Qwen3GuardStreamBackend,
    ScriptedBackend,
    _model_dtype_kwargs,
    find_user_turn_end,
    parse_qwen3guard_output,
    parse_qwen3guard_stream_result,
)


def test_dtype_keyword_tracks_transformers_version() -> None:
    torch = SimpleNamespace(bfloat16=object())

    legacy = _model_dtype_kwargs(
        SimpleNamespace(__version__="4.55.4"),
        torch,
        "bfloat16",
    )
    current = _model_dtype_kwargs(
        SimpleNamespace(__version__="4.56.2"),
        torch,
        "bfloat16",
    )

    assert set(legacy) == {"torch_dtype"}
    assert set(current) == {"dtype"}


def test_parse_qwen3guard_three_line_output_strictly() -> None:
    safe = parse_qwen3guard_output(
        "\nSafety: Safe\nCategories: None\nRefusal: No\n"
    )
    unsafe = parse_qwen3guard_output(
        "Safety: Unsafe\nCategories: Violent, Illegal Acts\nRefusal: Yes"
    )

    assert safe.is_safe is True
    assert safe.verdict == "safe"
    assert safe.refusal == "no"
    assert unsafe.is_safe is False
    assert unsafe.verdict == "unsafe"
    assert list(unsafe.categories) == ["Violent", "Illegal Acts"]
    assert unsafe.refusal == "yes"


def test_parser_rejects_noncanonical_first_line() -> None:
    assert parse_qwen3guard_output("Safe").verdict == "unknown"
    assert parse_qwen3guard_output("Safety: Unsafe content").verdict == "unknown"
    assert parse_qwen3guard_output("probably safe").verdict == "unknown"
    assert parse_qwen3guard_output("").parse_status == "empty"


def test_parse_stream_result_preserves_last_token_risk_and_probabilities() -> None:
    decision = parse_qwen3guard_stream_result(
        {
            "risk_level": ["Safe", "Controversial"],
            "risk_prob": [0.91, 0.82],
            "category": ["Violent", "Political"],
            "category_prob": [0.55, 0.76],
        }
    )

    assert decision.verdict == "controversial"
    assert decision.is_safe is False
    assert list(decision.categories) == ["Political"]
    assert decision.risk_probability == 0.82
    assert decision.category_probability == 0.76
    assert decision.output_tokens == 0


def test_find_user_turn_end_matches_official_template_search() -> None:
    token_ids = [10, 20, 30, 31, 99, 40, 10, 50, 99, 60]
    assert find_user_turn_end(
        token_ids,
        im_start_id=10,
        user_id=50,
        im_end_id=99,
    ) == 8


def test_stream_backend_applies_policy_and_rejects_fake_batching() -> None:
    backend = object.__new__(Qwen3GuardStreamBackend)
    backend.unknown_policy = "retry_then_fail_closed"
    backend.controversial_policy = "block"
    decision = parse_qwen3guard_stream_result(
        {
            "risk_level": ["Controversial"],
            "risk_prob": [0.8],
            "category": ["Political"],
            "category_prob": [0.7],
        }
    )
    backend._moderate_once = lambda _prompt, _prefix: (decision, 3.0)

    result = backend.classify_batch([("prompt", "prefix")])

    assert result.decisions[0].verdict == "controversial"
    assert result.decisions[0].is_safe is False
    assert result.generate_ms == 3.0
    with pytest.raises(ValueError, match="single-stream"):
        backend.classify_batch([("a", "b"), ("c", "d")])


def test_controversial_severity_is_preserved_and_policy_controls_release() -> None:
    raw = (
        "Safety: Controversial\n"
        "Categories: Political\n"
        "Refusal: No"
    )
    blocked = ScriptedBackend(lambda _prompt, _prefix: raw).classify_batch(
        [("prompt", "prefix")]
    ).decisions[0]
    allowed = ScriptedBackend(
        lambda _prompt, _prefix: raw,
        controversial_policy="allow",
    ).classify_batch([("prompt", "prefix")]).decisions[0]

    assert blocked.verdict == "controversial"
    assert blocked.is_safe is False
    assert allowed.verdict == "controversial"
    assert allowed.is_safe is True
