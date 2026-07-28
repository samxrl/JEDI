from __future__ import annotations

import pytest

from run_evaluation import (
    aggregate_evaluation_metrics,
    print_terminal_metrics,
    terminal_metric_lines,
)


def _summary() -> dict[str, object]:
    return {
        "model": "toy_llm",
        "utility": {
            "alpaca_eval": {
                "rate": 0.1,
                "fpr": 0.1,
                "false_positive_count": 1,
                "total": 10,
                "trigger_count": 2,
                "trigger_rate": 0.2,
            },
            "or_bench": {
                "rate": 0.25,
                "fpr": 0.25,
                "false_positive_count": 5,
                "total": 20,
                "trigger_count": 4,
                "trigger_rate": 0.2,
            },
        },
        "safety": {
            "GCG": {
                "rate": 0.3,
                "asr": 0.3,
                "dsr": 0.7,
                "harmful_count": 3,
                "total": 10,
                "trigger_count": 8,
                "trigger_rate": 0.8,
            },
            "PAIR": {
                "rate": 0.5,
                "asr": 0.5,
                "dsr": 0.5,
                "harmful_count": 10,
                "total": 20,
                "trigger_count": 12,
                "trigger_rate": 0.6,
            },
        },
    }


def test_aggregate_metrics_use_pooled_sample_counts() -> None:
    aggregate = aggregate_evaluation_metrics(_summary())

    assert aggregate["benign"]["fpr"] == pytest.approx(6 / 30)
    assert aggregate["benign"]["trigger_rate"] == pytest.approx(6 / 30)
    assert aggregate["safety"]["asr"] == pytest.approx(13 / 30)
    assert aggregate["safety"]["dsr"] == pytest.approx(17 / 30)
    assert aggregate["safety"]["trigger_rate"] == pytest.approx(20 / 30)


def test_terminal_report_prints_fpr_and_dsr(capsys: pytest.CaptureFixture[str]) -> None:
    summary = _summary()
    summary["aggregate"] = aggregate_evaluation_metrics(summary)

    lines = terminal_metric_lines(summary)
    assert any("alpaca_eval: FPR=10.00%" in line for line in lines)
    assert any("Benign Overall: FPR=20.00%" in line for line in lines)
    assert any("GCG: DSR=70.00%" in line for line in lines)
    assert any("Safety Overall: DSR=56.67%" in line for line in lines)

    print_terminal_metrics(summary)
    output = capsys.readouterr().out
    assert "Benign Datasets: FPR" in output
    assert "Safety Datasets: DSR" in output
