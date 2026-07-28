# -*- coding: utf-8 -*-
"""Main entry point for Periodic Qwen3Guard Gen/Stream offline checkpoint replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.paths import configure_local_environment  # noqa: E402

configure_local_environment()

from src.commands import (  # noqa: E402
    export_run,
    judge_run,
    materialize_command,
    prepare_command,
    score_prefixes_command,
    summarize_run,
    validate_command,
)
from src.config import (  # noqa: E402
    guard_backend_kind,
    load_config,
    override_guard_model,
    override_target_llm,
    supported_guard_models,
)


DEFAULT_CONFIG = THIS_DIR / "configs" / "periodic_qwen3guard.yaml"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Periodic Qwen3Guard YAML configuration file.",
    )
    parser.add_argument(
        "--target-llm",
        required=True,
        help=(
            "Unique target LLM name; overrides the configured model name and "
            "replaces the name after the final path separator in tokenizer_path."
        ),
    )
    parser.add_argument(
        "--guard-model",
        choices=supported_guard_models(),
        help=(
            "Override guard.model_name and automatically switch the Gen/Stream "
            "backend, local model path, and model-specific limits."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JEDI Periodic Qwen3Guard Gen/Stream offline replay baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Read JEDI baseline details and generate target-token snapshots.",
    )
    _add_common(prepare)
    prepare.add_argument(
        "--max-samples-per-file",
        type=int,
        help="Maximum baseline rows to prepare per source CSV; 0 means all rows.",
    )

    score = subparsers.add_parser(
        "score-prefixes",
        help="Score the shared cumulative-prefix grid with the selected Qwen3Guard model.",
    )
    _add_common(score)
    score.add_argument(
        "--batch-size",
        type=int,
        help="Override execution.batch_size.",
    )

    materialize = subparsers.add_parser(
        "materialize",
        help="Derive 16/32-token user-visible responses from shared labels.",
    )
    _add_common(materialize)
    materialize.add_argument("--chunk-sizes", nargs="+", type=int)

    judge = subparsers.add_parser(
        "judge",
        help="Classify changed safety responses using JEDI's HarmBench protocol.",
    )
    _add_common(judge)
    judge.add_argument("--chunk-sizes", nargs="+", type=int)

    export = subparsers.add_parser(
        "export",
        help="Export AlpacaEval-, XSTest-, and OR-Bench-compatible files.",
    )
    _add_common(export)
    export.add_argument("--chunk-sizes", nargs="+", type=int)

    summarize = subparsers.add_parser(
        "summarize",
        help="Aggregate DSR, utility proxies, trigger positions, and classifier latency.",
    )
    _add_common(summarize)
    summarize.add_argument("--chunk-sizes", nargs="+", type=int)

    validate = subparsers.add_parser(
        "validate",
        help="Validate configuration, write isolation, source hashes, and replay semantics.",
    )
    _add_common(validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.guard_model is not None:
        config = override_guard_model(config, args.guard_model)
    config = override_target_llm(config, args.target_llm)
    if args.command == "prepare":
        if args.max_samples_per_file is not None:
            if args.max_samples_per_file < 0:
                raise ValueError("--max-samples-per-file cannot be negative.")
            config["source"]["max_samples_per_file"] = (
                args.max_samples_per_file
            )
        print(prepare_command(config))
    elif args.command == "score-prefixes":
        if args.batch_size is not None:
            if args.batch_size <= 0:
                raise ValueError("--batch-size must be a positive integer.")
            if (
                guard_backend_kind(config) == "stream"
                and args.batch_size != 1
            ):
                raise ValueError(
                    "Qwen3Guard-Stream supports only --batch-size 1."
                )
            config["execution"]["batch_size"] = args.batch_size
        print(score_prefixes_command(config))
    elif args.command == "materialize":
        print(
            materialize_command(
                config,
                chunk_sizes=args.chunk_sizes,
            )
        )
    elif args.command == "judge":
        for path in judge_run(
            config,
            chunk_sizes=args.chunk_sizes,
        ):
            print(path)
    elif args.command == "export":
        for path in export_run(
            config,
            chunk_sizes=args.chunk_sizes,
        ):
            print(path)
    elif args.command == "summarize":
        for path in summarize_run(
            config,
            chunk_sizes=args.chunk_sizes,
        ):
            print(path)
    elif args.command == "validate":
        report = validate_command(config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] != "failed" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
