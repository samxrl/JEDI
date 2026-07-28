from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import resolve_repo_path, sha256_text, stable_json_hash


def resolve_model_reference(value: str | Path) -> str:
    """Resolve local relative paths from the repository; otherwise preserve the Hugging Face repository name."""
    raw = str(value)
    candidate = resolve_repo_path(raw)
    return str(candidate) if candidate.exists() else raw


def is_qwen3_tokenizer(tokenizer: Any) -> bool:
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    model_input_names = " ".join(getattr(tokenizer, "model_input_names", []))
    return "qwen3" in name or "qwen3" in model_input_names.lower()


def apply_chat_template_compat(
    tokenizer: Any,
    conversation: list[dict[str, str]],
    **kwargs: Any,
) -> Any:
    """Match JEDI's main evaluation by explicitly disabling thinking for Qwen3."""
    if is_qwen3_tokenizer(tokenizer):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(conversation, **kwargs)


def format_user_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    add_generation_prompt: bool,
) -> str:
    return apply_chat_template_compat(
        tokenizer,
        [{"role": "user", "content": str(prompt)}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def load_model_and_tokenizer(
    model_config: Mapping[str, Any],
    *,
    device: str | None = None,
) -> tuple[Any, Any]:
    kwargs = dict(model_config.get("kwargs", {}))
    dtype = kwargs.get("torch_dtype")
    if isinstance(dtype, str):
        if dtype == "auto":
            pass
        elif hasattr(torch, dtype):
            kwargs["torch_dtype"] = getattr(torch, dtype)
        else:
            raise ValueError(f"Invalid torch_dtype: {dtype}")
    if "device_map" not in kwargs and (device or "cuda") == "cuda":
        kwargs["device_map"] = "auto"

    reference = resolve_model_reference(model_config["path"])
    model = AutoModelForCausalLM.from_pretrained(reference, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        reference,
        trust_remote_code=bool(kwargs.get("trust_remote_code", True)),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if "device_map" not in kwargs and (device or "cuda") == "cuda":
        model.to("cuda")
    model.eval()
    return model, tokenizer


def model_input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def tokenizer_template_hash(tokenizer: Any) -> str:
    template = str(getattr(tokenizer, "chat_template", "") or "")
    return sha256_text(template)


def model_metadata(model: Any, tokenizer: Any, configured_name: str) -> dict[str, Any]:
    config_dict = model.config.to_dict()
    return {
        "configured_name": configured_name,
        "model_type": str(getattr(model.config, "model_type", "")),
        "num_hidden_layers": int(getattr(model.config, "num_hidden_layers")),
        "hidden_size": int(getattr(model.config, "hidden_size")),
        "config_sha256": stable_json_hash(config_dict),
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "chat_template_sha256": tokenizer_template_hash(tokenizer),
        "qwen3_thinking_disabled": bool(is_qwen3_tokenizer(tokenizer)),
    }
