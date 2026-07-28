from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping

import numpy as np
import yaml

from .calibration import effective_operator
from .modeling import model_metadata
from .paths import ensure_vendor_on_path, sha256_file, stable_json_hash

ensure_vendor_on_path()

from activation_steering import MalleableModel, SteeringVector  # noqa: E402
from activation_steering.config import GlobalConfig  # noqa: E402
from activation_steering.leash_layer import LeashLayer  # noqa: E402


BEHAVIOR_VECTOR_FILENAME = "behavior_vector.svec"
CONDITION_VECTOR_FILENAME = "condition_vector.svec"
PARAMS_FILENAME = "cast_params.yaml"
VECTOR_METADATA_FILENAME = "vector_metadata.json"


@dataclass(frozen=True)
class CASTArtifacts:
    behavior_vector: SteeringVector
    condition_vector: SteeringVector
    params: dict[str, Any]
    metadata: dict[str, Any]
    artifact_hashes: dict[str, str]

    @classmethod
    def load(cls, directory: str | Path) -> "CASTArtifacts":
        root = Path(directory).resolve()
        paths = {
            "behavior_vector": root / BEHAVIOR_VECTOR_FILENAME,
            "condition_vector": root / CONDITION_VECTOR_FILENAME,
            "params": root / PARAMS_FILENAME,
            "metadata": root / VECTOR_METADATA_FILENAME,
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"CAST artifacts are incomplete: {missing}")
        with paths["params"].open("r", encoding="utf-8") as handle:
            params = yaml.safe_load(handle)
        with paths["metadata"].open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(params, dict) or not isinstance(metadata, dict):
            raise TypeError("The top level of CAST params/metadata must be a mapping.")
        artifacts = cls(
            behavior_vector=SteeringVector.load(str(paths["behavior_vector"])),
            condition_vector=SteeringVector.load(str(paths["condition_vector"])),
            params=params,
            metadata=metadata,
            artifact_hashes={
                name: sha256_file(path)
                for name, path in paths.items()
            },
        )
        artifacts.validate_structure()
        return artifacts

    def validate_structure(self) -> None:
        condition_layers = [int(value) for value in self.params["condition_layers"]]
        behavior_layers = [int(value) for value in self.params["behavior_layers"]]
        for layer in condition_layers:
            if layer not in self.condition_vector.directions:
                raise KeyError(f"Condition vector is missing layer {layer}.")
        for layer in behavior_layers:
            if layer not in self.behavior_vector.directions:
                raise KeyError(f"Behavior vector is missing layer {layer}.")
        direction = str(self.params["condition_official_direction"])
        expected = effective_operator(direction)
        actual = str(self.params.get("condition_effective_operator", expected))
        if actual != expected:
            raise ValueError(
                f"Condition comparator mismatch: {direction} should map to {expected}, got {actual}."
            )
        if int(self.params.get("evaluation_batch_size", 1)) != 1:
            raise ValueError("CAST artifacts must declare evaluation_batch_size=1.")

    @property
    def fingerprint(self) -> str:
        return stable_json_hash(
            {
                "artifact_hashes": self.artifact_hashes,
                "params": self.params,
            }
        )


class CASTGuard:
    """Per-request state-isolation adapter based on the official MalleableModel."""

    def __init__(
        self,
        artifacts: CASTArtifacts,
        *,
        enforce_model_binding: bool = True,
    ) -> None:
        self.artifacts = artifacts
        self.enforce_model_binding = enforce_model_binding
        self._malleable: MalleableModel | None = None
        self._active = False
        self._request_started_at: float | None = None

    @classmethod
    def from_artifact_dir(
        cls,
        directory: str | Path,
        *,
        enforce_model_binding: bool = True,
    ) -> "CASTGuard":
        return cls(
            CASTArtifacts.load(directory),
            enforce_model_binding=enforce_model_binding,
        )

    @property
    def params(self) -> Mapping[str, Any]:
        return self.artifacts.params

    def _validate_model_binding(self, model: Any, tokenizer: Any) -> None:
        expected = self.artifacts.metadata.get("model")
        if not isinstance(expected, Mapping):
            raise KeyError("vector_metadata.json is missing model-binding information.")
        current = model_metadata(
            model,
            tokenizer,
            str(expected.get("configured_name", "")),
        )
        keys = (
            "model_type",
            "num_hidden_layers",
            "hidden_size",
            "config_sha256",
            "chat_template_sha256",
            "qwen3_thinking_disabled",
        )
        mismatches = {
            key: {"expected": expected.get(key), "actual": current.get(key)}
            for key in keys
            if expected.get(key) != current.get(key)
        }
        if mismatches:
            raise ValueError(f"CAST artifacts do not match the current model/tokenizer: {mismatches}")

    @staticmethod
    def reset_request_state() -> None:
        """Clear only request state; do not clear layer maps set by steer()."""
        LeashLayer.condition_met = defaultdict(bool)
        LeashLayer.forward_calls = defaultdict(int)
        LeashLayer.condition_similarities = defaultdict(
            lambda: defaultdict(float)
        )

    @contextmanager
    def attach(self, model: Any, tokenizer: Any) -> Iterator["CASTGuard"]:
        if self._active:
            raise RuntimeError("The same CASTGuard cannot be attached more than once.")
        if self.enforce_model_binding:
            self._validate_model_binding(model, tokenizer)

        for class_name in (
            "global",
            "LeashLayer",
            "MalleableModel",
            "SteeringVector",
            "SteeringDataset",
        ):
            GlobalConfig.set_verbose(False, class_name)

        malleable = MalleableModel(model=model, tokenizer=tokenizer)
        self._malleable = malleable
        self._active = True
        try:
            malleable.steer(
                behavior_vector=self.artifacts.behavior_vector,
                behavior_layer_ids=[
                    int(value) for value in self.params["behavior_layers"]
                ],
                behavior_vector_strength=float(self.params["behavior_strength"]),
                condition_vector=self.artifacts.condition_vector,
                condition_layer_ids=[
                    int(value) for value in self.params["condition_layers"]
                ],
                condition_vector_threshold=float(self.params["condition_threshold"]),
                condition_comparator_threshold_is=str(
                    self.params["condition_official_direction"]
                ),
                condition_threshold_comparison_mode=str(
                    self.params.get("condition_comparison_mode", "mean")
                ),
                use_explained_variance=False,
                use_ooi_preventive_normalization=bool(
                    self.params.get("use_ooi_preventive_normalization", False)
                ),
                apply_behavior_on_first_call=bool(
                    self.params.get("apply_behavior_on_first_call", True)
                ),
            )
            self.reset_request_state()
            yield self
        finally:
            try:
                malleable.unwrap()
            finally:
                LeashLayer.reset_class()
                self._malleable = None
                self._active = False
                self._request_started_at = None

    def prepare_request(self, log_record: MutableMapping[str, Any]) -> None:
        if not self._active:
            raise RuntimeError("CASTGuard must be used inside an attach context.")
        self.reset_request_state()
        self._request_started_at = time.perf_counter()
        params = self.params
        log_record.update(
            {
                "defense": "CAST",
                "condition": "guarded",
                "cast_source_commit": str(params["cast_source_commit"]),
                "cast_artifact_fingerprint": self.artifacts.fingerprint,
                "cast_condition_layer": int(params["condition_layers"][0]),
                "cast_condition_threshold": float(params["condition_threshold"]),
                "cast_condition_official_direction": str(
                    params["condition_official_direction"]
                ),
                "cast_condition_effective_operator": str(
                    params["condition_effective_operator"]
                ),
                "cast_behavior_layers": json.dumps(params["behavior_layers"]),
                "cast_behavior_strength": float(params["behavior_strength"]),
                "cast_behavior_vector_norm": float(
                    np.mean(
                        [
                            np.linalg.norm(
                                self.artifacts.behavior_vector.directions[int(layer)]
                            )
                            for layer in params["behavior_layers"]
                        ]
                    )
                ),
                "cast_condition_vector_norm": float(
                    np.linalg.norm(
                        self.artifacts.condition_vector.directions[
                            int(params["condition_layers"][0])
                        ]
                    )
                ),
            }
        )

    def finish_request(
        self,
        log_record: MutableMapping[str, Any],
        *,
        generation_error: str | None = None,
    ) -> None:
        condition_layers = [int(value) for value in self.params["condition_layers"]]
        layer_scores = {
            str(layer): float(LeashLayer.condition_similarities[0][layer])
            for layer in condition_layers
            if layer in LeashLayer.condition_similarities[0]
        }
        condition_met = bool(LeashLayer.condition_met[0])
        primary_score = layer_scores.get(str(condition_layers[0]))
        log_record.update(
            {
                "cast_condition_score": primary_score,
                "cast_condition_scores": json.dumps(layer_scores),
                "cast_condition_met": condition_met,
                "trigger_step": 0 if condition_met else -1,
                "generation_error": generation_error,
                "cast_request_runtime_ms": (
                    (time.perf_counter() - self._request_started_at) * 1000.0
                    if self._request_started_at is not None
                    else None
                ),
            }
        )
        self.reset_request_state()
        self._request_started_at = None
