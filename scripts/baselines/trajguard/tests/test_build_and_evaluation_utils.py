# -*- coding: utf-8 -*-
"""Lightweight tests for TrajGuard data isolation, MVD, and checkpoint recovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from scripts.baselines.trajguard.build_artifacts import (
    drop_evaluation_overlaps,
    estimate_mvd,
    extract_hidden_batch,
    normalize_text,
    text_hash,
    validate_data_isolation,
)
from scripts.baselines.trajguard.run_evaluation import (
    build_terminal_metric_lines,
    calculate_trajguard_diagnostics,
    prepare_detailed_output,
    split_resume_rows,
)


class DataIsolationTests(unittest.TestCase):
    def test_normalization_is_nfkc_casefold_and_whitespace_stable(self) -> None:
        self.assertEqual(normalize_text("  Ａ  B\n"), "a b")
        self.assertEqual(text_hash("Prompt"), text_hash(" prompt "))

    def test_overlap_between_reference_and_evaluation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_data_isolation(
                {
                    "benign_reference": ["Same prompt"],
                    "malicious_reference": ["different"],
                    "evaluation": [" same   PROMPT "],
                },
                strict_internal_disjoint=False,
            )

    def test_mvd_estimator_is_deterministic_for_fixed_rng(self) -> None:
        values = np.zeros((4, 2), dtype=np.float32)
        mean = np.zeros(2, dtype=np.float32)
        precision = np.eye(2, dtype=np.float32)
        first = estimate_mvd(
            values,
            mean,
            precision,
            0.5,
            rng=np.random.default_rng(7),
            trials=8,
            search_steps=10,
            r_max=1.0,
        )
        second = estimate_mvd(
            values,
            mean,
            precision,
            0.5,
            rng=np.random.default_rng(7),
            trials=8,
            search_steps=10,
            r_max=1.0,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first, 0.6, places=6)

    def test_small_evaluation_overlap_can_be_removed_and_audited(self) -> None:
        splits = {
            "benign_reference": ["keep", "duplicate"],
            "malicious_reference": ["malicious"],
            "layer_selection": ["selection"],
            "benign_validation": ["validation"],
            "evaluation": [" DUPLICATE "],
        }
        excluded = drop_evaluation_overlaps(splits, max_fraction=0.5)
        self.assertEqual(splits["benign_reference"], ["keep"])
        self.assertEqual(
            excluded["benign_reference"],
            [text_hash("duplicate")],
        )
        validate_data_isolation(
            splits,
            strict_internal_disjoint=True,
        )

    def test_large_evaluation_overlap_is_not_silently_removed(self) -> None:
        splits = {
            "benign_reference": ["duplicate"],
            "malicious_reference": ["malicious"],
            "layer_selection": ["selection"],
            "benign_validation": ["validation"],
            "evaluation": ["duplicate"],
        }
        with self.assertRaises(ValueError):
            drop_evaluation_overlaps(splits, max_fraction=0.01)


class AdditiveBlock(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, hidden: Tensor) -> tuple[Tensor]:
        return (hidden + self.increment,)


class OfflineBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AdditiveBlock(1.0), AdditiveBlock(2.0)])


class OfflineHookModel(nn.Module):
    """Multiply by 10 at the end to verify extraction does not mistakenly read final-norm output."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 2)
        self.model = OfflineBackbone()
        with torch.no_grad():
            for token_id in range(8):
                self.embedding.weight[token_id].fill_(float(token_id))

    def forward(self, input_ids: Tensor, **_kwargs: Any) -> Any:
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        hidden = hidden * 10.0
        return SimpleNamespace(logits=hidden)


class OfflineTokenizer:
    name_or_path = "offline-dummy"

    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: Any) -> str:
        return messages[0]["content"]

    def __call__(self, _texts: list[str], **_kwargs: Any) -> dict[str, Tensor]:
        return {
            "input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }


class OfflineExtractionTests(unittest.TestCase):
    def test_offline_extraction_uses_block_hooks_not_final_norm(self) -> None:
        features = extract_hidden_batch(
            OfflineHookModel(),
            OfflineTokenizer(),
            ["first", "second"],
            context_tokens=2,
            max_length=16,
        )
        np.testing.assert_allclose(
            features[0],
            np.asarray([[2.5, 2.5], [5.5, 5.5]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            features[1],
            np.asarray([[4.5, 4.5], [7.5, 7.5]], dtype=np.float32),
        )


class EvaluationUtilityTests(unittest.TestCase):
    def test_terminal_metric_lines_report_weighted_fpr_and_dsr(self) -> None:
        lines = build_terminal_metric_lines(
            "test-model",
            {
                "benign_a": {"total": 10, "harmful_count": 1},
                "benign_b": {"total": 20, "false_positive_count": 4},
            },
            {
                "attack_a": {"total": 10, "harmful_count": 2},
                "attack_b": {"total": 20, "harmful_count": 8},
            },
        )
        report = "\n".join(lines)
        self.assertIn("Benign Dataset FPR", report)
        self.assertIn("Overall: 16.67% (5/30)", report)
        self.assertIn("Safety Dataset DSR", report)
        self.assertIn("Overall: 66.67% (20/30; ASR=33.33%)", report)

    def test_negative_trigger_is_serialized_as_nullable_integer(self) -> None:
        detailed = prepare_detailed_output(
            pd.DataFrame(
                {
                    "trigger_step": [-1, 2],
                    "trajguard_trigger_step": [-1, 2],
                    "trajguard_unsafe_trigger_step": [-1, 3],
                }
            )
        )
        self.assertTrue(pd.isna(detailed.loc[0, "trigger_step"]))
        self.assertEqual(detailed.loc[1, "trigger_step"], 2)
        self.assertEqual(detailed.loc[1, "trajguard_unsafe_trigger_step"], 3)
        self.assertEqual(str(detailed["trigger_step"].dtype), "Int64")

    def test_resume_requires_matching_artifact_and_runtime_hashes(self) -> None:
        source = pd.DataFrame({"prompt": ["p1", "p2"]})
        existing = pd.DataFrame(
            {
                "prompt": ["p1", "p2"],
                "assistant_output": ["a1", "a2"],
                "trajguard_artifact_sha256": ["artifact", "old"],
                "trajguard_runtime_config_sha256": ["runtime", "runtime"],
                "eval_split": ["safety", "safety"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.csv"
            existing.to_csv(path, index=False)
            completed, pending = split_resume_rows(
                source,
                path,
                artifact_sha256="artifact",
                runtime_config_sha256="runtime",
            )

        self.assertEqual(completed["prompt"].tolist(), ["p1"])
        self.assertEqual(pending["prompt"].tolist(), ["p2"])
        self.assertEqual(completed["eval_split"].tolist(), ["safety"])

    def test_trajguard_diagnostics_report_judge_exposure_and_latency(self) -> None:
        frame = pd.DataFrame(
            {
                "trajguard_judge_calls": [0, 2],
                "trajguard_unsafe": [False, True],
                "trajguard_trigger_step": [np.nan, 3],
                "trajguard_unsafe_trigger_step": [np.nan, 5],
                "trajguard_exposed_tokens": [0, 5],
                "trajguard_ttft_ms": [10.0, 20.0],
                "trajguard_generation_runtime_ms": [100.0, 200.0],
                "trajguard_runtime_ms": [2.0, 4.0],
                "trajguard_judge_runtime_ms": [0.0, 2.0],
                "trajguard_scored_steps": [4, 8],
            }
        )
        result = calculate_trajguard_diagnostics(frame)
        self.assertEqual(result["judge_call_sequence_rate"], 0.5)
        self.assertEqual(result["mean_judge_calls_per_sequence"], 1.0)
        self.assertEqual(result["unsafe_stop_rate"], 0.5)
        self.assertEqual(result["unsafe_trigger_step"]["median"], 5.0)
        self.assertEqual(
            result["zero_prefix_exposure_rate_among_unsafe_stops"],
            0.0,
        )
        self.assertAlmostEqual(
            result["mean_defense_ms_per_scored_step"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
