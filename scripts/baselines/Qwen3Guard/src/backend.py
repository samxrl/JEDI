from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Protocol

from .config import guard_backend_kind, guard_reference
from .schema import GuardBatchResult, GuardDecision


_SAFETY_LABELS = {
    "safety: safe": "safe",
    "safety: unsafe": "unsafe",
    "safety: controversial": "controversial",
}
_STREAM_SAFETY_LABELS = {
    "safe": "safe",
    "unsafe": "unsafe",
    "controversial": "controversial",
}


class GuardBackend(Protocol):
    def classify_batch(
        self,
        conversations: Sequence[tuple[str, str]],
    ) -> GuardBatchResult:
        ...

    def close(self) -> None:
        ...


def parse_qwen3guard_output(raw_output: str) -> GuardDecision:
    """Strictly parse Qwen3Guard-Gen's three-line response-moderation format."""
    lines = [
        line.strip()
        for line in str(raw_output).splitlines()
        if line.strip()
    ]
    if not lines:
        return GuardDecision(
            raw_output=str(raw_output),
            is_safe=False,
            verdict="unknown",
            categories=[],
            parse_status="empty",
        )
    verdict = _SAFETY_LABELS.get(lines[0].lower())
    if verdict is None:
        return GuardDecision(
            raw_output=str(raw_output),
            is_safe=False,
            verdict="unknown",
            categories=[],
            parse_status="unrecognized_first_line",
        )

    categories: list[str] = []
    categories_found = False
    refusal: str | None = None
    for line in lines[1:]:
        lowered = line.lower()
        if lowered.startswith("categories:"):
            categories_found = True
            payload = line.split(":", 1)[1].strip()
            if payload.lower() != "none":
                for category in payload.split(","):
                    normalized = category.strip()
                    if normalized and normalized not in categories:
                        categories.append(normalized)
        elif lowered == "refusal: yes":
            refusal = "yes"
        elif lowered == "refusal: no":
            refusal = "no"

    if categories_found and refusal is not None:
        parse_status = "parsed"
    elif not categories_found and refusal is None:
        parse_status = "parsed_missing_categories_and_refusal"
    elif not categories_found:
        parse_status = "parsed_missing_categories"
    else:
        parse_status = "parsed_missing_refusal"
    return GuardDecision(
        raw_output=str(raw_output),
        is_safe=verdict == "safe",
        verdict=verdict,
        categories=categories,
        parse_status=parse_status,
        refusal=refusal,
    )


def _last_result_value(result: Mapping[str, Any], key: str) -> Any:
    value = result.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return value[-1] if value else None


def _optional_probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_qwen3guard_stream_result(
    result: Mapping[str, Any],
) -> GuardDecision:
    """Parse the final-token result returned by official ``stream_moderate_from_ids``."""
    raw_output = json.dumps(
        dict(result),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    risk_level = _last_result_value(result, "risk_level")
    verdict = _STREAM_SAFETY_LABELS.get(
        str(risk_level or "").strip().lower()
    )
    if verdict is None:
        return GuardDecision(
            raw_output=raw_output,
            is_safe=False,
            verdict="unknown",
            categories=[],
            parse_status="stream_unrecognized_risk_level",
        )
    category = str(
        _last_result_value(result, "category") or ""
    ).strip()
    categories = (
        [category]
        if verdict != "safe" and category and category.lower() != "none"
        else []
    )
    return GuardDecision(
        raw_output=raw_output,
        is_safe=verdict == "safe",
        verdict=verdict,
        categories=categories,
        parse_status="stream_parsed",
        risk_probability=_optional_probability(
            _last_result_value(result, "risk_prob")
        ),
        category_probability=_optional_probability(
            _last_result_value(result, "category_prob")
        ),
        output_tokens=0,
    )


def find_user_turn_end(
    token_ids: Sequence[int],
    *,
    im_start_id: int,
    user_id: int,
    im_end_id: int,
) -> int:
    """Locate the final ``<|im_start|>user ... <|im_end|>`` as in the official example."""
    values = [int(value) for value in token_ids]
    last_start = next(
        (
            index
            for index in range(len(values) - 2, -1, -1)
            if values[index : index + 2] == [im_start_id, user_id]
        ),
        None,
    )
    if last_start is None:
        raise ValueError("No user turn was found in the Qwen3Guard-Stream template.")
    for index in range(last_start + 2, len(values)):
        if values[index] == im_end_id:
            return index
    raise ValueError("The user turn in the Qwen3Guard-Stream template is not closed.")


def _torch_dtype(torch: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    try:
        return getattr(torch, value)
    except AttributeError as exc:
        raise ValueError(f"Invalid guard.dtype: {value}") from exc


def _model_dtype_kwargs(
    transformers: Any,
    torch: Any,
    value: str,
) -> dict[str, Any]:
    """Support the torch_dtype argument in 4.55 and the dtype argument in 4.56+."""
    try:
        major, minor = (
            int(part)
            for part in str(transformers.__version__).split(".")[:2]
        )
    except (TypeError, ValueError):
        major, minor = 4, 56
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    return {key: _torch_dtype(torch, value)}


def _model_input_device(model: Any) -> Any:
    embedding = model.get_input_embeddings()
    device = getattr(getattr(embedding, "weight", None), "device", None)
    if device is not None and str(device) != "meta":
        return device
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RuntimeError("Unable to determine the Qwen3Guard model's input device.")


def _sync_cuda(torch: Any, device: Any) -> None:
    if getattr(device, "type", str(device).split(":")[0]) == "cuda":
        torch.cuda.synchronize(device)


def _output_token_count(
    ids: Sequence[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> int:
    count = 0
    for token_id in ids:
        value = int(token_id)
        if pad_token_id is not None and value == pad_token_id:
            break
        count += 1
        if eos_token_id is not None and value == eos_token_id:
            break
    return count


class Qwen3GuardGenBackend:
    """Generative response-moderation wrapper for Qwen3Guard-Gen-8B."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        guard = config["guard"]
        reference = guard_reference(config)
        local_only = bool(guard.get("local_files_only", True))
        if local_only and not Path(reference).exists():
            raise FileNotFoundError(f"Missing local Qwen3Guard weights: {reference}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            reference,
            trust_remote_code=bool(guard.get("trust_remote_code", True)),
            local_files_only=local_only,
            padding_side="left",
        )
        configured_pad_id = guard.get("pad_token_id")
        if configured_pad_id is not None:
            self.tokenizer.pad_token_id = int(configured_pad_id)
        elif self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            **_model_dtype_kwargs(
                transformers,
                torch,
                str(guard.get("dtype", "bfloat16")),
            ),
            "trust_remote_code": bool(guard.get("trust_remote_code", True)),
            "local_files_only": local_only,
        }
        device_map = guard.get("device_map")
        if device_map:
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            reference,
            **model_kwargs,
        )
        if not device_map:
            self.model.to(str(guard.get("device", "cuda")))
        self.model.eval()

        self.torch = torch
        self.device = _model_input_device(self.model)
        self.max_input_tokens = int(guard.get("max_input_tokens", 32576))
        self.max_new_tokens = int(guard.get("max_new_tokens", 128))
        self.retry_max_new_tokens = int(
            guard.get("retry_max_new_tokens", 192)
        )
        self.unknown_policy = str(
            guard.get("unknown_policy", "retry_then_fail_closed")
        )
        self.controversial_policy = str(
            guard.get("controversial_policy", "block")
        )

    def _generate_once(
        self,
        conversations: Sequence[tuple[str, str]],
        *,
        max_new_tokens: int,
    ) -> tuple[list[str], list[int], list[int], float]:
        chats = [
            [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(prefix)},
            ]
            for prompt, prefix in conversations
        ]
        rendered = [
            self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
            )
            for chat in chats
        ]
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_width = int(inputs["input_ids"].shape[1])
        if input_width > self.max_input_tokens:
            raise ValueError(
                "Qwen3Guard templated input exceeds guard.max_input_tokens: "
                f"{input_width} > {self.max_input_tokens}."
            )
        input_lengths = [
            int(value) for value in inputs["attention_mask"].sum(dim=1).tolist()
        ]
        inputs = inputs.to(self.device)

        _sync_cuda(self.torch, self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        _sync_cuda(self.torch, self.device)
        generate_ms = (time.perf_counter() - started) * 1000.0

        suffix_ids = generated[:, input_width:].detach().cpu()
        raw_outputs = self.tokenizer.batch_decode(
            suffix_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_lengths = [
            _output_token_count(
                row,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            for row in suffix_ids.tolist()
        ]
        return raw_outputs, input_lengths, output_lengths, generate_ms

    def _apply_controversial_policy(
        self,
        parsed: GuardDecision,
    ) -> GuardDecision:
        if parsed.verdict != "controversial":
            return parsed
        return replace(
            parsed,
            is_safe=self.controversial_policy == "allow",
            parse_status=(
                f"{parsed.parse_status}_controversial_"
                f"{self.controversial_policy}"
            ),
        )

    def _unknown_decision(
        self,
        parsed: GuardDecision,
        *,
        status_prefix: str,
    ) -> GuardDecision:
        if self.unknown_policy == "raise":
            raise ValueError(
                f"Unable to parse Qwen3Guard output: {parsed.raw_output!r}"
            )
        fail_open = self.unknown_policy == "fail_open"
        return GuardDecision(
            raw_output=parsed.raw_output,
            is_safe=fail_open,
            verdict="safe" if fail_open else "unsafe",
            categories=[],
            parse_status=(
                f"{status_prefix}_fail_open"
                if fail_open
                else f"{status_prefix}_fail_closed"
            ),
            refusal=parsed.refusal,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            error=(
                "Qwen3Guard did not return a strict first line of "
                "Safety: Safe/Unsafe/Controversial."
            ),
        )

    def classify_batch(
        self,
        conversations: Sequence[tuple[str, str]],
    ) -> GuardBatchResult:
        if not conversations:
            return GuardBatchResult([], 0.0, 0.0, 0)
        e2e_started = time.perf_counter()
        raw, input_lengths, output_lengths, generate_ms = self._generate_once(
            conversations,
            max_new_tokens=self.max_new_tokens,
        )
        decisions: list[GuardDecision] = []
        unknown_indices: list[int] = []
        for index, (text, input_tokens, output_tokens) in enumerate(
            zip(raw, input_lengths, output_lengths)
        ):
            parsed = parse_qwen3guard_output(text)
            parsed = replace(
                parsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            parsed = self._apply_controversial_policy(parsed)
            decisions.append(parsed)
            if parsed.verdict == "unknown":
                unknown_indices.append(index)

        retry_count = 0
        if (
            unknown_indices
            and self.unknown_policy == "retry_then_fail_closed"
        ):
            retry_count = len(unknown_indices)
            retry_conversations = [
                conversations[index] for index in unknown_indices
            ]
            retry_raw, retry_inputs, retry_outputs, retry_ms = (
                self._generate_once(
                    retry_conversations,
                    max_new_tokens=self.retry_max_new_tokens,
                )
            )
            generate_ms += retry_ms
            for original_index, text, input_tokens, output_tokens in zip(
                unknown_indices,
                retry_raw,
                retry_inputs,
                retry_outputs,
            ):
                retried = parse_qwen3guard_output(text)
                retried = replace(
                    retried,
                    parse_status=(
                        "parsed_after_retry"
                        if retried.verdict != "unknown"
                        else retried.parse_status
                    ),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                retried = self._apply_controversial_policy(retried)
                decisions[original_index] = (
                    retried
                    if retried.verdict != "unknown"
                    else self._unknown_decision(
                        retried,
                        status_prefix="unknown_after_retry",
                    )
                )
        elif unknown_indices:
            for index in unknown_indices:
                decisions[index] = self._unknown_decision(
                    decisions[index],
                    status_prefix="unknown",
                )

        e2e_ms = (time.perf_counter() - e2e_started) * 1000.0
        return GuardBatchResult(
            decisions=decisions,
            generate_ms=generate_ms,
            e2e_ms=e2e_ms,
            batch_size=len(conversations),
            retry_count=retry_count,
        )

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        torch = getattr(self, "torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen3GuardStreamBackend:
    """Qwen3Guard-Stream-8B backend that replays cumulative prefixes through the official state interface."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        guard = config["guard"]
        reference = guard_reference(config)
        local_only = bool(guard.get("local_files_only", True))
        if local_only and not Path(reference).exists():
            raise FileNotFoundError(
                f"Missing local Qwen3Guard-Stream weights: {reference}"
            )
        trust_remote_code = bool(guard.get("trust_remote_code", True))
        if not trust_remote_code:
            raise ValueError(
                "Qwen3Guard-Stream requires trust_remote_code=true."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            reference,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        model_kwargs: dict[str, Any] = {
            **_model_dtype_kwargs(
                transformers,
                torch,
                str(guard.get("dtype", "bfloat16")),
            ),
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        device_map = guard.get("device_map")
        if device_map:
            model_kwargs["device_map"] = device_map
        self.model = AutoModel.from_pretrained(reference, **model_kwargs)
        if not device_map:
            self.model.to(str(guard.get("device", "cuda")))
        self.model.eval()
        for method_name in ("stream_moderate_from_ids", "close_stream"):
            if not hasattr(self.model, method_name):
                raise TypeError(
                    "The loaded model does not provide the official Qwen3Guard-Stream interface: "
                    f"{method_name}"
                )

        self.torch = torch
        self.device = _model_input_device(self.model)
        self.max_input_tokens = int(guard.get("max_input_tokens", 8192))
        self.unknown_policy = str(
            guard.get("unknown_policy", "retry_then_fail_closed")
        )
        self.controversial_policy = str(
            guard.get("controversial_policy", "block")
        )
        self.im_start_id = int(
            self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        )
        self.user_id = int(
            self.tokenizer.convert_tokens_to_ids("user")
        )
        self.im_end_id = int(
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    def _render_token_ids(self, prompt: str, prefix: str) -> Any:
        messages = [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(prefix)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        encoded = self.tokenizer(text, return_tensors="pt")
        token_ids = encoded.input_ids[0]
        if int(token_ids.shape[0]) > self.max_input_tokens:
            raise ValueError(
                "Qwen3Guard-Stream templated input exceeds "
                f"guard.max_input_tokens: {int(token_ids.shape[0])} > "
                f"{self.max_input_tokens}."
            )
        return token_ids

    def _moderate_once(
        self,
        prompt: str,
        prefix: str,
    ) -> tuple[GuardDecision, float]:
        token_ids = self._render_token_ids(prompt, prefix)
        user_end_index = find_user_turn_end(
            token_ids.tolist(),
            im_start_id=self.im_start_id,
            user_id=self.user_id,
            im_end_id=self.im_end_id,
        )
        stream_state = None
        final_result: Mapping[str, Any] | None = None
        _sync_cuda(self.torch, self.device)
        started = time.perf_counter()
        try:
            _, stream_state = self.model.stream_moderate_from_ids(
                token_ids[: user_end_index + 1],
                role="user",
                stream_state=None,
            )
            for index in range(user_end_index + 1, len(token_ids)):
                final_result, stream_state = (
                    self.model.stream_moderate_from_ids(
                        token_ids[index],
                        role="assistant",
                        stream_state=stream_state,
                    )
                )
        finally:
            self.model.close_stream(stream_state)
        _sync_cuda(self.torch, self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        if final_result is None:
            raise RuntimeError(
                "The Qwen3Guard-Stream template contains no assistant tokens to moderate."
            )
        decision = parse_qwen3guard_stream_result(final_result)
        return (
            replace(
                decision,
                input_tokens=int(token_ids.shape[0]),
                output_tokens=0,
            ),
            inference_ms,
        )

    def _apply_controversial_policy(
        self,
        parsed: GuardDecision,
    ) -> GuardDecision:
        if parsed.verdict != "controversial":
            return parsed
        return replace(
            parsed,
            is_safe=self.controversial_policy == "allow",
            parse_status=(
                f"{parsed.parse_status}_controversial_"
                f"{self.controversial_policy}"
            ),
        )

    def _unknown_decision(
        self,
        parsed: GuardDecision,
        *,
        status_prefix: str,
    ) -> GuardDecision:
        if self.unknown_policy == "raise":
            raise ValueError(
                f"Unable to parse Qwen3Guard-Stream result: {parsed.raw_output!r}"
            )
        fail_open = self.unknown_policy == "fail_open"
        return replace(
            parsed,
            is_safe=fail_open,
            verdict="safe" if fail_open else "unsafe",
            parse_status=(
                f"{status_prefix}_fail_open"
                if fail_open
                else f"{status_prefix}_fail_closed"
            ),
            error="Qwen3Guard-Stream did not return a valid risk_level.",
        )

    def classify_batch(
        self,
        conversations: Sequence[tuple[str, str]],
    ) -> GuardBatchResult:
        if not conversations:
            return GuardBatchResult([], 0.0, 0.0, 0)
        if len(conversations) != 1:
            raise ValueError(
                "The official Qwen3Guard-Stream state interface supports single-stream scoring only."
            )
        e2e_started = time.perf_counter()
        prompt, prefix = conversations[0]
        parsed, inference_ms = self._moderate_once(prompt, prefix)
        retry_count = 0
        if (
            parsed.verdict == "unknown"
            and self.unknown_policy == "retry_then_fail_closed"
        ):
            retry_count = 1
            retried, retry_ms = self._moderate_once(prompt, prefix)
            inference_ms += retry_ms
            parsed = (
                replace(retried, parse_status="stream_parsed_after_retry")
                if retried.verdict != "unknown"
                else self._unknown_decision(
                    retried,
                    status_prefix="stream_unknown_after_retry",
                )
            )
        elif parsed.verdict == "unknown":
            parsed = self._unknown_decision(
                parsed,
                status_prefix="stream_unknown",
            )
        parsed = self._apply_controversial_policy(parsed)
        e2e_ms = (time.perf_counter() - e2e_started) * 1000.0
        return GuardBatchResult(
            decisions=[parsed],
            generate_ms=inference_ms,
            e2e_ms=e2e_ms,
            batch_size=1,
            retry_count=retry_count,
        )

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        torch = getattr(self, "torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_guard_backend(config: Mapping[str, Any]) -> GuardBackend:
    kind = guard_backend_kind(config)
    if kind == "gen":
        return Qwen3GuardGenBackend(config)
    if kind == "stream":
        return Qwen3GuardStreamBackend(config)
    raise ValueError(f"Unknown Qwen3Guard backend: {kind}")


# Preserve the legacy class name so existing local calls continue to work in Gen mode.
Qwen3GuardBackend = Qwen3GuardGenBackend


class ScriptedBackend:
    """Deterministic Qwen3Guard backend for CPU-only semantic tests."""

    def __init__(
        self,
        responder: Callable[[str, str], str],
        *,
        per_batch_ms: float = 0.0,
        controversial_policy: str = "block",
    ) -> None:
        self.responder = responder
        self.per_batch_ms = float(per_batch_ms)
        self.controversial_policy = controversial_policy

    def classify_batch(
        self,
        conversations: Sequence[tuple[str, str]],
    ) -> GuardBatchResult:
        decisions: list[GuardDecision] = []
        for prompt, prefix in conversations:
            parsed = parse_qwen3guard_output(self.responder(prompt, prefix))
            if parsed.verdict == "controversial":
                parsed = replace(
                    parsed,
                    is_safe=self.controversial_policy == "allow",
                )
            decisions.append(parsed)
        return GuardBatchResult(
            decisions=decisions,
            generate_ms=self.per_batch_ms,
            e2e_ms=self.per_batch_ms,
            batch_size=len(conversations),
        )

    def close(self) -> None:
        return None
