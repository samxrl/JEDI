from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .modeling import format_user_prompt
from .paths import ensure_vendor_on_path, require_cast_path

ensure_vendor_on_path()

from activation_steering import SteeringVector  # noqa: E402
from activation_steering.utils import ContrastivePair  # noqa: E402


@dataclass
class PreparedSteeringDataset:
    """Minimal data interface required by the official SteeringVector.train."""

    formatted_dataset: list[ContrastivePair]
    suffixes: list[tuple[str, str]] | None


def build_behavior_dataset(
    tokenizer: Any,
    benign_prompts: Sequence[str],
    prefix_pairs: Sequence[Mapping[str, str]],
) -> PreparedSteeringDataset:
    """Build refusal (+)/compliance (-) pairs in the official suffix-major order."""
    formatted_prompts = [
        format_user_prompt(
            tokenizer,
            prompt,
            add_generation_prompt=False,
        )
        for prompt in benign_prompts
    ]
    suffixes = [
        (str(pair["refusal"]), str(pair["compliance"]))
        for pair in prefix_pairs
    ]
    pairs = [
        ContrastivePair(
            positive=formatted + refusal,
            negative=formatted + compliance,
        )
        for refusal, compliance in suffixes
        for formatted in formatted_prompts
    ]
    return PreparedSteeringDataset(
        formatted_dataset=pairs,
        suffixes=suffixes,
    )


def build_condition_dataset(
    tokenizer: Any,
    harmful_prompts: Sequence[str],
    benign_prompts: Sequence[str],
) -> PreparedSteeringDataset:
    if len(harmful_prompts) != len(benign_prompts):
        raise ValueError("The condition vector requires equal harmful/benign example counts.")
    pairs = [
        ContrastivePair(
            positive=format_user_prompt(
                tokenizer,
                harmful,
                add_generation_prompt=False,
            ),
            negative=format_user_prompt(
                tokenizer,
                benign,
                add_generation_prompt=False,
            ),
        )
        for harmful, benign in zip(harmful_prompts, benign_prompts)
    ]
    return PreparedSteeringDataset(formatted_dataset=pairs, suffixes=None)


def load_prefix_pairs(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("pairs")
    if not isinstance(rows, list):
        raise TypeError(f"prefix_pairs.json is missing the pairs list: {path}")
    return [dict(row) for row in rows]


def save_vector(vector: SteeringVector, path: str | Path) -> Path:
    target = require_cast_path(path)
    if target.suffix != ".svec":
        target = require_cast_path(Path(str(target) + ".svec"))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = require_cast_path(
        target.with_name(f".{target.stem}.tmp{target.suffix}")
    )
    try:
        vector.save(str(temporary))
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def vector_layer_norms(vector: SteeringVector) -> dict[str, float]:
    return {
        str(layer): float(np.linalg.norm(direction))
        for layer, direction in vector.directions.items()
    }
