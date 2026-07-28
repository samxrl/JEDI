from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from .paths import (
    load_yaml,
    require_local_write,
    resolve_local_path,
    resolve_repo_path,
)


REQUIRED_TOP_LEVEL = {
    "baseline",
    "paths",
    "source",
    "models",
    "streaming",
    "guard",
    "execution",
    "judgment",
    "evaluation",
}

SUPPORTED_GUARD_MODELS: dict[str, dict[str, Any]] = {
    "Qwen3Guard-Gen-8B": {
        "backend": "gen",
        "model_id": "Qwen/Qwen3Guard-Gen-8B",
        "slug": "qwen3guard_gen",
        "context_window_tokens": 32768,
    },
    "Qwen3Guard-Stream-8B": {
        "backend": "stream",
        "model_id": "Qwen/Qwen3Guard-Stream-8B",
        "slug": "qwen3guard_stream",
        "context_window_tokens": 8192,
    },
}


def validate_name(value: str, *, field: str = "name") -> str:
    name = value.strip()
    if not name or name in {".", ".."} or not re.fullmatch(r"[^\\/]+", name):
        raise ValueError(f"{field} must be a single valid directory name.")
    return name


def _replace_path_name(value: str, target_llm: str) -> str:
    """Replace the name after the final path separator, supporting both / and backslash."""
    path = value.rstrip("/\\")
    separator_index = max(path.rfind("/"), path.rfind("\\"))
    if separator_index < 0:
        return target_llm
    return path[: separator_index + 1] + target_llm


def supported_guard_models() -> tuple[str, ...]:
    return tuple(SUPPORTED_GUARD_MODELS)


def _guard_profile(model_name: str) -> dict[str, Any]:
    name = validate_name(model_name, field="guard.model_name")
    try:
        return SUPPORTED_GUARD_MODELS[name]
    except KeyError as exc:
        supported = ", ".join(supported_guard_models())
        raise ValueError(
            f"Unsupported guard model {name!r}; allowed values: {supported}."
        ) from exc


def _apply_guard_model(
    config: Mapping[str, Any],
    model_name: str,
    *,
    reset_model_limit: bool,
) -> dict[str, Any]:
    """Resolve the guard model, backend, model path, and model-specific context limits together."""
    resolved = copy.deepcopy(dict(config))
    profile = _guard_profile(model_name)
    guard = copy.deepcopy(dict(resolved.get("guard") or {}))
    if not guard.get("model_path"):
        raise KeyError("guard.model_path cannot be empty.")
    guard["model_name"] = model_name
    guard["model_id"] = profile["model_id"]
    guard["backend"] = profile["backend"]
    guard["model_path"] = _replace_path_name(
        str(guard["model_path"]),
        model_name,
    )
    guard["context_window_tokens"] = int(
        profile["context_window_tokens"]
    )
    configured_limit = guard.get("max_input_tokens", "auto")
    if reset_model_limit or str(configured_limit).lower() == "auto":
        if profile["backend"] == "gen":
            generated = max(
                int(guard.get("max_new_tokens", 128)),
                int(guard.get("retry_max_new_tokens", 192)),
            )
            guard["max_input_tokens"] = (
                int(profile["context_window_tokens"]) - generated
            )
        else:
            guard["max_input_tokens"] = int(
                profile["context_window_tokens"]
            )
    resolved["guard"] = guard

    baseline = copy.deepcopy(dict(resolved.get("baseline") or {}))
    baseline["name"] = f"periodic_{profile['slug']}"
    baseline["condition_prefix"] = profile["slug"]
    resolved["baseline"] = baseline

    if profile["backend"] == "stream":
        execution = copy.deepcopy(dict(resolved.get("execution") or {}))
        # Official stream_moderate_from_ids is a single-stream state interface; do not emulate batching.
        execution["batch_size"] = 1
        resolved["execution"] = execution
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(Path(path).expanduser().resolve())
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise KeyError(f"Qwen3Guard configuration is missing top-level fields: {missing}")
    guard = config.get("guard")
    if not isinstance(guard, Mapping):
        raise TypeError("guard must be a mapping.")
    config = _apply_guard_model(
        config,
        str(guard.get("model_name", "")),
        reset_model_limit=False,
    )
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    models = config.get("models")
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError(
            "models must contain exactly one target-model template; "
            "specify the actual model with --target-llm."
        )
    model = models[0]
    if not isinstance(model, Mapping):
        raise TypeError("The sole item in models must be a mapping.")
    validate_name(str(model.get("name", "")), field="models[].name")
    if not model.get("tokenizer_path"):
        raise KeyError("models[].tokenizer_path cannot be empty.")

    source = config["source"]
    if bool(source.get("include_safety", True)):
        attack_methods = source.get("attack_methods")
        if not isinstance(attack_methods, list) or not attack_methods:
            raise ValueError(
                "source.attack_methods must provide a nonempty allowlist of main-experiment attacks."
            )
        normalized_methods = [
            validate_name(str(method), field="source.attack_methods[]")
            for method in attack_methods
        ]
        if len({method.casefold() for method in normalized_methods}) != len(
            normalized_methods
        ):
            raise ValueError("source.attack_methods cannot contain duplicates.")

    streaming = config["streaming"]
    sizes = streaming.get("chunk_sizes")
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("streaming.chunk_sizes must be a nonempty list.")
    sizes = [int(value) for value in sizes]
    if any(value <= 0 for value in sizes):
        raise ValueError("Every streaming.chunk_sizes value must be a positive integer.")
    if len(sizes) != len(set(sizes)):
        raise ValueError("streaming.chunk_sizes cannot contain duplicates.")
    stride = int(streaming.get("base_checkpoint_stride", min(sizes)))
    if stride <= 0 or any(size % stride != 0 for size in sizes):
        raise ValueError(
            "base_checkpoint_stride must be positive and divide every chunk size."
        )
    if stride not in sizes:
        raise ValueError(
            "streaming.chunk_sizes must contain base_checkpoint_stride "
            "to generate a complete shared check grid."
        )
    if streaming.get("prefix_mode") != "cumulative":
        raise ValueError("The main Qwen3Guard baseline supports only prefix_mode=cumulative.")
    if not bool(streaming.get("include_user_prompt", True)):
        raise ValueError("Response classification must include the original user prompt.")
    if not bool(streaming.get("check_final_partial", True)):
        raise ValueError("Formal replay must check the final remainder shorter than one chunk.")
    if streaming.get("buffer_policy") != "hold_current_chunk":
        raise ValueError(
            "The current implementation supports only buffer_policy=hold_current_chunk."
        )
    if streaming.get("trigger_policy") != "block_current_and_remaining":
        raise ValueError(
            "The current implementation supports only trigger_policy=block_current_and_remaining."
        )

    guard = config["guard"]
    model_name = str(guard.get("model_name", ""))
    profile = _guard_profile(model_name)
    if guard.get("backend") != profile["backend"]:
        raise ValueError("guard.backend does not match guard.model_name.")
    if guard.get("model_id") != profile["model_id"]:
        raise ValueError("guard.model_id does not match guard.model_name.")
    if not guard.get("model_path"):
        raise KeyError("guard.model_path cannot be empty.")
    if int(guard.get("context_window_tokens", 0)) != int(
        profile["context_window_tokens"]
    ):
        raise ValueError(
            "guard.context_window_tokens does not match the guard model's official configuration."
        )
    max_new_tokens = int(guard.get("max_new_tokens", 128))
    retry_max_new_tokens = int(
        guard.get("retry_max_new_tokens", 192)
    )
    max_input_tokens = int(guard.get("max_input_tokens", 32576))
    context_window_tokens = int(
        guard.get("context_window_tokens", 32768)
    )
    if max_input_tokens <= 0:
        raise ValueError("guard.max_input_tokens must be a positive integer.")
    if context_window_tokens <= 0:
        raise ValueError("guard.context_window_tokens must be a positive integer.")
    if profile["backend"] == "gen":
        if bool(guard.get("do_sample", False)):
            raise ValueError(
                "Formal Qwen3Guard-Gen replay requires guard.do_sample=false."
            )
        if max_new_tokens <= 0:
            raise ValueError("guard.max_new_tokens must be a positive integer.")
        if retry_max_new_tokens <= 0:
            raise ValueError(
                "guard.retry_max_new_tokens must be a positive integer."
            )
        if (
            max_input_tokens + max(
                max_new_tokens,
                retry_max_new_tokens,
            )
            > context_window_tokens
        ):
            raise ValueError(
                "Gen guard.max_input_tokens must reserve context for generated output."
            )
    elif max_input_tokens > context_window_tokens:
        raise ValueError(
            "Stream guard.max_input_tokens cannot exceed the model context length."
        )
    if guard.get("controversial_policy", "block") not in {"block", "allow"}:
        raise ValueError(
            "guard.controversial_policy must be block or allow."
        )
    if guard.get("unknown_policy", "retry_then_fail_closed") not in {
        "retry_then_fail_closed",
        "fail_closed",
        "fail_open",
        "raise",
    }:
        raise ValueError("Invalid guard.unknown_policy value.")
    if guard.get("backend_error_policy", "fail_closed") not in {
        "fail_closed",
        "fail_open",
        "raise",
    }:
        raise ValueError("Invalid guard.backend_error_policy value.")

    if int(config["execution"].get("batch_size", 1)) <= 0:
        raise ValueError("execution.batch_size must be a positive integer.")
    if (
        profile["backend"] == "stream"
        and int(config["execution"].get("batch_size", 1)) != 1
    ):
        raise ValueError(
            "The official Qwen3Guard-Stream state interface requires execution.batch_size=1."
        )
    if int(config["judgment"].get("batch_size", 1)) <= 0:
        raise ValueError("judgment.batch_size must be a positive integer.")

    # Validate early that every runtime write directory is covered by isolation constraints.
    for key in ("input_root", "run_root"):
        resolve_local_path(config["paths"][key])


def override_target_llm(
    config: Mapping[str, Any],
    target_llm: str,
) -> dict[str, Any]:
    """Override the sole template's name and final tokenizer-path directory with a command-line model name."""
    name = validate_name(target_llm, field="--target-llm")
    resolved = copy.deepcopy(dict(config))
    models = resolved.get("models")
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError(
            "--target-llm override requires models to contain exactly one model template."
        )
    model = copy.deepcopy(dict(models[0]))
    model["name"] = name
    model["tokenizer_path"] = _replace_path_name(
        str(model["tokenizer_path"]),
        name,
    )
    resolved["models"] = [model]
    validate_config(resolved)
    return resolved


def override_guard_model(
    config: Mapping[str, Any],
    guard_model: str,
) -> dict[str, Any]:
    """Switch the Gen/Stream backend, local path, and limits using a command-line model name."""
    resolved = _apply_guard_model(
        config,
        guard_model,
        reset_model_limit=True,
    )
    validate_config(resolved)
    return resolved


def configured_guard_model(config: Mapping[str, Any]) -> str:
    name = str(config["guard"].get("model_name", ""))
    _guard_profile(name)
    return name


def guard_backend_kind(config: Mapping[str, Any]) -> str:
    return str(_guard_profile(configured_guard_model(config))["backend"])


def guard_slug(config: Mapping[str, Any]) -> str:
    return str(_guard_profile(configured_guard_model(config))["slug"])


def configured_target_llm(config: Mapping[str, Any]) -> str:
    models = config["models"]
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError("The current run must configure exactly one target model.")
    return validate_name(str(models[0]["name"]), field="models[].name")


def selected_model(config: Mapping[str, Any]) -> dict[str, Any]:
    configured_target_llm(config)
    return copy.deepcopy(dict(config["models"][0]))


def source_evaluation_dir(config: Mapping[str, Any]) -> Path:
    return resolve_repo_path(config["paths"]["source_evaluation_dir"])


def input_root(config: Mapping[str, Any]) -> Path:
    return resolve_local_path(config["paths"]["input_root"])


def run_root(config: Mapping[str, Any]) -> Path:
    return resolve_local_path(config["paths"]["run_root"])


def artifact_relative_dir(config: Mapping[str, Any]) -> Path:
    """Return stable relative artifact paths uniquely determined by the target LLM and guard model."""
    target_model = configured_target_llm(config)
    guard_model = validate_name(
        configured_guard_model(config),
        field="guard.model_name",
    )
    return Path(target_model) / guard_model


def input_dir(config: Mapping[str, Any]) -> Path:
    return require_local_write(
        input_root(config) / artifact_relative_dir(config)
    )


def run_dir(config: Mapping[str, Any]) -> Path:
    return require_local_write(
        run_root(config) / artifact_relative_dir(config)
    )


def target_tokenizer_reference(model: Mapping[str, Any]) -> str:
    return str(resolve_repo_path(str(model["tokenizer_path"])))


def guard_reference(config: Mapping[str, Any]) -> str:
    return str(resolve_repo_path(str(config["guard"]["model_path"])))


def classifier_reference(config: Mapping[str, Any]) -> str:
    return str(resolve_repo_path(str(config["judgment"]["model_path"])))
