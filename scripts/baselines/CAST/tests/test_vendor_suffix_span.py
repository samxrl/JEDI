from __future__ import annotations

from activation_steering.steering_vector import (
    _expand_aligned_suffixes,
    _hidden_to_numpy,
    _suffix_token_count,
    _suffix_token_indices,
)
import numpy as np
import torch


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[tuple[int, int]]]:
        assert add_special_tokens
        assert return_offsets_mapping
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(range(len(text) + int(add_special_tokens)))


def test_suffixes_follow_positive_negative_sample_order() -> None:
    inputs = [f"sample-{index}" for index in range(8)]
    suffixes = [("AA", "B"), ("C", "DD")]
    assert _expand_aligned_suffixes(inputs, suffixes) == [
        "AA",
        "B",
        "AA",
        "B",
        "C",
        "DD",
        "C",
        "DD",
    ]


def test_exact_suffix_span_uses_current_suffix() -> None:
    tokenizer = CharacterTokenizer()
    assert _suffix_token_count(tokenizer, "promptXYZ", "XYZ") == 3
    assert _suffix_token_count(tokenizer, "promptQ", "Q") == 1
    assert _suffix_token_indices(tokenizer, "promptXYZ", "XYZ") == [6, 7, 8]


def test_bfloat_hidden_states_are_converted_for_sklearn() -> None:
    value = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    converted = _hidden_to_numpy(value)
    assert converted.dtype == np.float32
    assert converted.tolist() == [1.0, 2.0]
