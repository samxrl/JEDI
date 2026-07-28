# -*- coding: utf-8 -*-
"""TrajGuard tensor, hook, and generation-adapter tests that require no real LLM."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from scripts.baselines.trajguard.core import (
    ARTIFACT_FILENAME,
    MultiLayerHookManager,
    StreamingGeometricSurveillance,
    TrajGuard,
    TrajGuardArtifacts,
    TrajGuardLogitsProcessor,
    TrajGuardRequestLog,
    truncated_mean,
)


def make_payload(
    *,
    threshold: float = 1.0,
    sustain_steps: int = 3,
    ewma_lambda: float = 0.0,
) -> dict[str, Any]:
    """Build minimal valid artifacts with two-dimensional identity PCA and identity precision."""
    layer_stats = {
        "pca_mean": torch.zeros(2),
        "pca_components": torch.eye(2),
        "benign_mean": torch.zeros(2),
        "benign_precision": torch.eye(2),
        "malicious_mean": torch.ones(2),
        "malicious_precision": torch.eye(2),
    }
    return {
        "schema_version": 1,
        "method": "trajguard_paper_consistent",
        "model": {
            "num_layers": 2,
            "hidden_size": 2,
            "chat_template_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855"
            ),
        },
        "parameters": {
            "context_tokens": 3,
            "pca_dim": 2,
            "window_size": 8,
            "trim_count": 1,
            "ewma_lambda": ewma_lambda,
            "sustain_steps": sustain_steps,
            "threshold": threshold,
        },
        "selected_layers": [0, 1],
        "layers": {
            "0": {key: value.clone() for key, value in layer_stats.items()},
            "1": {key: value.clone() for key, value in layer_stats.items()},
        },
    }


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)]
        )
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.eye(2))

    def forward(self, hidden: Tensor) -> Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TinyHookModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyBackbone()

    def forward(self, hidden: Tensor) -> Tensor:
        return self.model(hidden)


class DummyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None
    name_or_path = "dummy"

    def decode(self, token_ids: Any, **_kwargs: Any) -> str:
        values = torch.as_tensor(token_ids).flatten().tolist()
        return " ".join(str(int(value)) for value in values if int(value) != 2)

    def __call__(self, _text: str, **_kwargs: Any) -> dict[str, Tensor]:
        # Fixed refusal text is used only in the unsafe post-processing path.
        return {"input_ids": torch.tensor([[7, 8]], dtype=torch.long)}


class DummyGenerateModel(nn.Module):
    """Simulate two-step greedy generation and trigger decoder hooks at each step."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 2)
        self.model = TinyBackbone()
        self.config = SimpleNamespace(hidden_size=2)
        self.generation_config = SimpleNamespace(
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=False,
            eos_token_id=2,
        )

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.model(self.embedding(input_ids))

    def generate(
        self,
        *,
        input_ids: Tensor,
        generation_config: Any = None,
        logits_processor: Any = None,
        **_kwargs: Any,
    ) -> Tensor:
        sequence = input_ids.clone()
        for step in range(2):
            current = sequence if step == 0 else sequence[:, -1:]
            self(input_ids=current)
            scores = torch.zeros((1, 16), device=sequence.device)
            scores[:, 4] = 1.0
            if logits_processor is not None:
                scores = logits_processor(sequence, scores)
            next_token = torch.argmax(scores, dim=-1, keepdim=True)
            sequence = torch.cat([sequence, next_token], dim=1)
            if int(next_token.item()) == 2:
                break
        return sequence


class FixedCaptureHooks:
    """Return the same activations each time for logits-processor state tests."""

    def consume(self) -> dict[int, Tensor]:
        return {
            0: torch.tensor([[1.0, 1.0]]),
            1: torch.tensor([[1.0, 1.0]]),
        }


class SequenceJudge:
    def __init__(self, decisions: list[str]) -> None:
        self.decisions = list(decisions)

    def decide(self, _prompt: str, _response: str) -> tuple[str, dict[str, float]]:
        return self.decisions.pop(0), {}


class CoreTensorTests(unittest.TestCase):
    def test_mahalanobis_risk_matches_independent_formula(self) -> None:
        artifacts = TrajGuardArtifacts(make_payload())
        hidden = torch.tensor([1.0, 1.0])
        risk, d_benign, d_malicious = artifacts.score_hidden(0, hidden)
        self.assertAlmostEqual(d_benign, math.sqrt(2.0), places=6)
        self.assertAlmostEqual(d_malicious, 0.0, places=6)
        self.assertLess(abs(risk - (math.sqrt(2.0) - 0.0)), 1.0e-4)

    def test_truncated_mean_removes_each_extreme(self) -> None:
        self.assertEqual(truncated_mean([1.0, 2.0, 100.0], trim_count=1), 2.0)
        self.assertEqual(truncated_mean([1.0, 3.0], trim_count=1), 2.0)

    def test_streaming_state_triggers_after_three_consecutive_steps(self) -> None:
        artifacts = TrajGuardArtifacts(
            make_payload(threshold=1.0, sustain_steps=3, ewma_lambda=0.0)
        )
        monitor = StreamingGeometricSurveillance(artifacts)
        captured = {
            0: torch.tensor([[1.0, 1.0]]),
            1: torch.tensor([[1.0, 1.0]]),
        }
        updates = [monitor.update(captured) for _ in range(3)]
        self.assertFalse(updates[0].triggered)
        self.assertFalse(updates[1].triggered)
        self.assertTrue(updates[2].triggered)
        self.assertEqual(updates[2].consecutive_count, 3)

        monitor.reset(clear_windows=False)
        self.assertEqual(monitor.ewma_score, 0.0)
        self.assertEqual(monitor.consecutive_count, 0)
        self.assertEqual(len(monitor.risk_windows[0]), 3)

    def test_streaming_rejects_non_cached_multi_token_followup(self) -> None:
        artifacts = TrajGuardArtifacts(make_payload())
        monitor = StreamingGeometricSurveillance(artifacts)
        first = {
            0: torch.ones(3, 2),
            1: torch.ones(3, 2),
        }
        monitor.update(first)
        with self.assertRaises(RuntimeError):
            monitor.update(first)

    def test_artifact_round_trip_uses_single_runtime_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / ARTIFACT_FILENAME
            torch.save(make_payload(), artifact_path)
            loaded = TrajGuardArtifacts.load(Path(directory))
        self.assertEqual(loaded.selected_layers, [0, 1])
        self.assertEqual(loaded.window_size, 8)


class HookAndAdapterTests(unittest.TestCase):
    def test_multi_layer_hooks_capture_tail_and_are_removed(self) -> None:
        model = TinyHookModel()
        manager = MultiLayerHookManager(model, [0, 1], context_tokens=2)
        manager.attach()
        model(torch.ones(1, 3, 2))
        captured = manager.consume()
        self.assertEqual(tuple(captured[0].shape), (2, 2))
        self.assertEqual(tuple(captured[1].shape), (2, 2))

        with manager.pause():
            model(torch.ones(1, 1, 2))
        with self.assertRaises(RuntimeError):
            manager.consume()

        manager.remove()
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)

    def test_untriggered_guard_preserves_tokens_and_restores_generate(self) -> None:
        model = DummyGenerateModel()
        tokenizer = DummyTokenizer()
        artifacts = TrajGuardArtifacts(
            make_payload(threshold=1.0e6, sustain_steps=3)
        )
        guard = TrajGuard(artifacts, tokenizer, use_pair_judge=False)
        original_function = model.generate.__func__
        record: dict[str, Any] = {}

        with guard.attach(model):
            guard.prepare_request("harmless", record)
            result = model.generate(
                input_ids=torch.tensor([[1, 3]], dtype=torch.long),
                generation_config=model.generation_config,
            )

        self.assertEqual(result.tolist(), [[1, 3, 4, 4]])
        self.assertEqual(record["trajguard_trigger_step"], -1)
        self.assertFalse(record["trajguard_unsafe"])
        self.assertIs(model.generate.__func__, original_function)

    def test_triggered_replace_discards_prefix_and_logs_zero_exposure(self) -> None:
        model = DummyGenerateModel()
        tokenizer = DummyTokenizer()
        artifacts = TrajGuardArtifacts(
            make_payload(threshold=-1.0e6, sustain_steps=1)
        )
        guard = TrajGuard(
            artifacts,
            tokenizer,
            use_pair_judge=False,
            unsafe_action="replace",
        )
        record: dict[str, Any] = {}

        with guard.attach(model):
            guard.prepare_request("unsafe", record)
            result = model.generate(
                input_ids=torch.tensor([[1, 3]], dtype=torch.long),
                generation_config=model.generation_config,
            )

        self.assertEqual(result.tolist(), [[1, 3, 7, 8, 2]])
        self.assertEqual(record["trajguard_trigger_step"], 0)
        self.assertEqual(record["trajguard_exposed_tokens"], 0)
        self.assertTrue(record["trajguard_unsafe"])
        self.assertEqual(record["trajguard_judge_decision"], "UNSAFE_SGS")

    def test_safe_then_unsafe_records_the_final_stop_prefix(self) -> None:
        tokenizer = DummyTokenizer()
        artifacts = TrajGuardArtifacts(
            make_payload(threshold=-1.0e6, sustain_steps=1)
        )
        monitor = StreamingGeometricSurveillance(artifacts)
        request_log = TrajGuardRequestLog()
        processor = TrajGuardLogitsProcessor(
            monitor=monitor,
            hooks=FixedCaptureHooks(),  # type: ignore[arg-type]
            tokenizer=tokenizer,
            prompt_length=2,
            user_prompt="unsafe",
            request_log=request_log,
            eos_token_id=2,
            pair_judge=SequenceJudge(["SAFE", "UNSAFE"]),  # type: ignore[arg-type]
            use_pair_judge=True,
            monitor_only=False,
            fail_on_monitor_error=True,
        )
        scores = torch.zeros((1, 16))

        first = processor(torch.tensor([[1, 3]]), scores)
        second = processor(torch.tensor([[1, 3, 4]]), scores)

        self.assertIs(first, scores)
        self.assertTrue(torch.isneginf(second[0, 0]))
        self.assertEqual(request_log.trigger_step, 0)
        self.assertEqual(request_log.unsafe_trigger_step, 1)
        self.assertEqual(request_log.first_trigger_prefix, "")
        self.assertEqual(request_log.raw_prefix, "4")
        self.assertEqual(request_log.judge_decisions, ["SAFE", "UNSAFE"])

    def test_terminate_preserves_exact_generated_prefix_token_ids(self) -> None:
        model = DummyGenerateModel()
        tokenizer = DummyTokenizer()
        artifacts = TrajGuardArtifacts(
            make_payload(threshold=-1.0e6, sustain_steps=2)
        )
        guard = TrajGuard(
            artifacts,
            tokenizer,
            use_pair_judge=False,
            unsafe_action="terminate",
        )
        record: dict[str, Any] = {}

        with guard.attach(model):
            guard.prepare_request("unsafe", record)
            result = model.generate(
                input_ids=torch.tensor([[1, 3]], dtype=torch.long),
                generation_config=model.generation_config,
            )

        self.assertEqual(result.tolist(), [[1, 3, 4, 7, 8, 2]])
        self.assertEqual(record["trajguard_unsafe_trigger_step"], 1)
        self.assertEqual(record["trajguard_exposed_tokens"], 1)
        self.assertEqual(record["trajguard_raw_prefix"], "4")


if __name__ == "__main__":
    unittest.main()
