from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from activation_steering import SteeringVector
from activation_steering.leash_layer import LeashLayer

from src.runtime import CASTArtifacts, CASTGuard
from run_evaluation import FirstTokenTimer


class IdentityBlock(nn.Module):
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return hidden_states


class ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([IdentityBlock(), IdentityBlock()])


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model = ToyBackbone()
        self.config = SimpleNamespace(
            model_type="toy",
            num_hidden_layers=2,
            hidden_size=2,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    @property
    def device(self) -> torch.device:
        return self.anchor.device


def _artifacts() -> CASTArtifacts:
    behavior = SteeringVector(
        model_type="toy",
        directions={1: np.asarray([0.0, 1.0], dtype=np.float32)},
        explained_variances={1: 1.0},
    )
    condition = SteeringVector(
        model_type="toy",
        directions={0: np.asarray([1.0, 0.0], dtype=np.float32)},
        explained_variances={0: 1.0},
    )
    params = {
        "cast_source_commit": "test",
        "condition_layers": [0],
        "condition_threshold": 0.7,
        "condition_official_direction": "smaller",
        "condition_effective_operator": ">",
        "condition_comparison_mode": "mean",
        "behavior_layers": [1],
        "behavior_strength": 1.0,
        "apply_behavior_on_first_call": True,
        "use_ooi_preventive_normalization": False,
        "evaluation_batch_size": 1,
    }
    return CASTArtifacts(
        behavior_vector=behavior,
        condition_vector=condition,
        params=params,
        metadata={},
        artifact_hashes={"test": "test"},
    )


def _forward(model: ToyModel, hidden: torch.Tensor) -> torch.Tensor:
    value = hidden.clone()
    for layer in model.model.layers:
        value = layer(value)
    return value


def test_request_state_is_isolated_and_layers_are_restored() -> None:
    model = ToyModel()
    original_layers = list(model.model.layers)
    guard = CASTGuard(_artifacts(), enforce_model_binding=False)
    with guard.attach(
        model,
        tokenizer=SimpleNamespace(eos_token="</s>", pad_token=None),
    ):
        harmful_log: dict[str, object] = {}
        guard.prepare_request(harmful_log)
        harmful_output = _forward(
            model,
            torch.tensor([[[1.0, 0.1]]]),
        )
        guard.finish_request(harmful_log)
        assert harmful_log["cast_condition_met"] is True
        assert harmful_log["trigger_step"] == 0
        assert harmful_output[0, 0, 1].item() > 0.1

        benign_log: dict[str, object] = {}
        guard.prepare_request(benign_log)
        benign_output = _forward(
            model,
            torch.tensor([[[0.1, 1.0]]]),
        )
        guard.finish_request(benign_log)
        assert benign_log["cast_condition_met"] is False
        assert benign_log["trigger_step"] == -1
        assert torch.allclose(benign_output, torch.tensor([[[0.1, 1.0]]]))

    assert all(
        restored is original
        for restored, original in zip(model.model.layers, original_layers)
    )
    assert LeashLayer.condition_layers is None
    assert LeashLayer.behavior_layers is None


def test_request_reset_preserves_attached_layer_maps() -> None:
    LeashLayer.condition_layers = {0: True}
    LeashLayer.behavior_layers = {0: False}
    LeashLayer.condition_met[0] = True
    CASTGuard.reset_request_state()
    assert LeashLayer.condition_layers == {0: True}
    assert LeashLayer.behavior_layers == {0: False}
    assert not LeashLayer.condition_met[0]
    LeashLayer.reset_class()


def test_first_token_timer_ignores_prompt_put(monkeypatch) -> None:
    monkeypatch.setattr("run_evaluation.cuda_synchronize", lambda: None)
    timer = FirstTokenTimer(10.0)
    times = iter([10.025])
    monkeypatch.setattr("run_evaluation.time.perf_counter", lambda: next(times))
    timer.put(torch.tensor([[1, 2, 3]]))
    assert timer.ttft_ms is None
    timer.put(torch.tensor([4]))
    assert timer.ttft_ms == pytest.approx(25.0)
