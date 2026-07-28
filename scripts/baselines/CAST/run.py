# -*- coding: utf-8 -*-
"""Unified command entry point for all CAST stages."""

from __future__ import annotations

import argparse
import importlib
from typing import Sequence


COMMAND_MODULES = {
    "prepare": "prepare_data",
    "extract": "extract_vectors",
    "calibrate": "calibrate_condition",
    "evaluate": "run_evaluation",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one CAST reproduction stage.")
    parser.add_argument("command", choices=sorted(COMMAND_MODULES))
    parser.add_argument(
        "--config",
        default="scripts/baselines/CAST/configs/cast_config.yaml",
    )
    parser.add_argument(
        "--llm-name",
        help="Override model.name and replace the final path component of model.path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    module = importlib.import_module(COMMAND_MODULES[args.command])
    forwarded = ["--config", args.config]
    if args.llm_name:
        forwarded.extend(["--llm-name", args.llm_name])
    if args.dry_run:
        forwarded.append("--dry-run")
    module.main(forwarded)


if __name__ == "__main__":
    main()
