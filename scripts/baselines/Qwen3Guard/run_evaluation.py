# -*- coding: utf-8 -*-
"""Run only the classify, export, and summarize stages for Periodic Qwen3Guard Gen/Stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.paths import configure_local_environment  # noqa: E402

configure_local_environment()

from src.config import (  # noqa: E402
    load_config,
    override_guard_model,
    override_target_llm,
    supported_guard_models,
)
from src.evaluation import export_run, judge_run, summarize_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate materialized Periodic Qwen3Guard Gen/Stream results"
    )
    parser.add_argument(
        "stage",
        choices=["judge", "export", "summarize", "all"],
        help="all runs judge -> export -> summarize.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=THIS_DIR / "configs" / "periodic_qwen3guard.yaml",
    )
    parser.add_argument(
        "--target-llm",
        required=True,
        help=(
            "Target LLM name; overrides the configured model name and the directory "
            "after the final path separator in tokenizer_path, and must match earlier stages."
        ),
    )
    parser.add_argument(
        "--guard-model",
        choices=supported_guard_models(),
        help=(
            "Override guard.model_name; it must match the prefix-scoring and materialization stages."
        ),
    )
    parser.add_argument("--chunk-sizes", nargs="+", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.guard_model is not None:
        config = override_guard_model(config, args.guard_model)
    config = override_target_llm(config, args.target_llm)
    stages = (
        ["judge", "export", "summarize"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        if stage == "judge":
            outputs = judge_run(
                config,
                chunk_sizes=args.chunk_sizes,
            )
        elif stage == "export":
            outputs = export_run(
                config,
                chunk_sizes=args.chunk_sizes,
            )
        else:
            outputs = summarize_run(
                config,
                chunk_sizes=args.chunk_sizes,
            )
        for output in outputs:
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
