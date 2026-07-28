# -*- coding: utf-8 -*-
"""Build independent calibration splits, prefix pairs, and overlap audits for CAST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.paths import configure_local_environment  # noqa: E402

configure_local_environment()

from src.config import load_config, select_model, validate_batch_invariants  # noqa: E402
from src.data import prepare_model_data  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare independent CAST calibration data.")
    parser.add_argument(
        "--config",
        default="scripts/baselines/CAST/configs/cast_config.yaml",
        help="CAST YAML configuration, relative to the repository root or absolute.",
    )
    parser.add_argument(
        "--llm-name",
        help="Override model.name and replace the final path component of model.path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, splits, and overlaps without writing files.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    validate_batch_invariants(config)
    model = select_model(config, args.llm_name)
    summary = prepare_model_data(config, model, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
