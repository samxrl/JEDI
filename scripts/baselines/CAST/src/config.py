from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from .paths import CAST_ROOT, load_yaml, require_cast_path


def validate_llm_name(value: str) -> str:
    """Validate a command-line model name and prevent interpreting it as an extra path."""
    name = str(value).strip()
    if not name:
        raise ValueError("--llm-name cannot be empty.")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("--llm-name accepts only a model name and cannot contain path separators.")
    return name


def replace_model_path_name(path: str | Path, llm_name: str) -> str:
    """Preserve the parent portion of a model path and replace only its final component."""
    name = validate_llm_name(llm_name)
    raw_path = str(path)
    if not raw_path or raw_path.endswith(("/", "\\")):
        raise ValueError("model.path must end with the existing model name.")
    separator_index = max(raw_path.rfind("/"), raw_path.rfind("\\"))
    return raw_path[: separator_index + 1] + name


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    for key in (
        "output_root",
        "model",
        "data",
        "extraction",
        "calibration",
        "runtime",
    ):
        if key not in config:
            raise KeyError(f"CAST configuration is missing top-level field: {key}")
    return config


def select_model(
    config: Mapping[str, Any],
    llm_name: str | None = None,
) -> dict[str, Any]:
    """Select a model template, overriding its name and final path component when llm_name is provided."""
    target_name = validate_llm_name(llm_name) if llm_name is not None else None
    if "models" in config:
        candidates = config["models"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("models must be a nonempty list.")
        if target_name is None:
            if len(candidates) != 1:
                raise ValueError("--llm-name is required when the configuration contains multiple models.")
            model = candidates[0]
        else:
            matches = [
                item for item in candidates if item.get("name") == target_name
            ]
            if len(matches) == 1:
                model = matches[0]
            elif len(candidates) == 1:
                model = candidates[0]
            else:
                raise KeyError(
                    "No template in the multi-model configuration matches --llm-name; "
                    "the model settings to inherit cannot be determined."
                )
    else:
        model = config["model"]
    if not isinstance(model, Mapping):
        raise TypeError("The model configuration must be a mapping.")
    for key in ("name", "path", "harmful_calibration_file"):
        if not model.get(key):
            raise KeyError(f"Model configuration is missing field: {key}")
    selected = copy.deepcopy(dict(model))
    if target_name is not None:
        selected["name"] = target_name
        selected["path"] = replace_model_path_name(
            selected["path"],
            target_name,
        )
    return selected


def output_root(config: Mapping[str, Any]) -> Path:
    """Return the common root directory for all model directories."""
    return require_cast_path(
        config.get("output_root", CAST_ROOT / "runs")
    )


def model_root(config: Mapping[str, Any], model: Mapping[str, Any]) -> Path:
    name = str(model["name"])
    if (
        not name
        or name in {".", ".."}
        or not re.fullmatch(r"[^\\/]+", name)
    ):
        raise ValueError("model.name must be a single valid directory name.")
    return require_cast_path(output_root(config) / name)


def data_dir(config: Mapping[str, Any], model: Mapping[str, Any]) -> Path:
    return require_cast_path(model_root(config, model) / "data")


def artifact_dir(config: Mapping[str, Any], model: Mapping[str, Any]) -> Path:
    return require_cast_path(model_root(config, model) / "artifacts")


def result_dir(config: Mapping[str, Any], model: Mapping[str, Any]) -> Path:
    return require_cast_path(model_root(config, model) / "results")


def validate_batch_invariants(config: Mapping[str, Any]) -> None:
    extraction_batch = int(config["extraction"].get("batch_size", 1))
    if extraction_batch != 1:
        raise ValueError(
            "CAST suffix spans, padding exclusion, and official runtime state require extraction.batch_size=1."
        )
    evaluation = config.get("evaluation", {})
    if int(evaluation.get("batch_size", 1)) != 1:
        raise ValueError("The official CAST runtime supports only evaluation.batch_size=1.")
    generation = evaluation.get("generation_kwargs", {})
    if bool(generation.get("do_sample", False)):
        raise ValueError("The CAST comparison experiment requires generation_kwargs.do_sample=false.")
    if int(generation.get("num_beams", 1)) != 1:
        raise ValueError("The current CAST runtime requires generation_kwargs.num_beams=1.")
