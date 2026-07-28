from __future__ import annotations

import pytest

from run import build_parser as build_run_parser
from run_evaluation import build_parser as build_evaluation_parser


def test_pipeline_cli_uses_target_and_guard_without_run_id() -> None:
    args = build_run_parser().parse_args(
        [
            "prepare",
            "--target-llm",
            "vicuna_7b_v1_5",
            "--guard-model",
            "Qwen3Guard-Stream-8B",
        ]
    )

    assert args.target_llm == "vicuna_7b_v1_5"
    assert args.guard_model == "Qwen3Guard-Stream-8B"
    assert not hasattr(args, "run_id")


def test_evaluation_cli_uses_same_stable_scope() -> None:
    args = build_evaluation_parser().parse_args(
        [
            "summarize",
            "--target-llm",
            "vicuna_7b_v1_5",
            "--guard-model",
            "Qwen3Guard-Gen-8B",
        ]
    )

    assert args.target_llm == "vicuna_7b_v1_5"
    assert args.guard_model == "Qwen3Guard-Gen-8B"
    assert not hasattr(args, "run_id")


def test_run_id_is_no_longer_accepted() -> None:
    with pytest.raises(SystemExit):
        build_run_parser().parse_args(
            [
                "prepare",
                "--target-llm",
                "vicuna_7b_v1_5",
                "--run-id",
                "legacy-run",
            ]
        )
