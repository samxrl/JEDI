from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.evaluation import atomic_to_csv, export_run, summarize_run
from src.paths import QWEN3GUARD_ROOT


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "target_model": "model",
        "guard_model": "Qwen3Guard-Gen-8B",
        "eval_split": "safety",
        "attack_method": "GCG",
        "utility_dataset_name": None,
        "label": "no",
        "triggered": True,
        "trigger_guard_label": "unsafe",
        "trigger_token_end": 16,
        "released_token_count": 0,
        "zero_original_token_exposure": True,
        "num_guard_checks": 1,
        "guard_e2e_ms_total": 2.0,
        "blocking_guard_e2e_ms_total": 4.0,
        "guard_ms_until_trigger": 2.0,
        "guard_error": None,
        "assistant_output": "refusal",
        "prompt": "prompt",
        "instruction": "instruction",
    }
    row.update(overrides)
    return row


def test_export_and_summary_remain_inside_run_root() -> None:
    cache = QWEN3GUARD_ROOT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        config = {
            "paths": {"run_root": temporary},
            "models": [{"name": "model"}],
            "guard": {"model_name": "Qwen3Guard-Gen-8B"},
            "streaming": {"chunk_sizes": [16]},
            "evaluation": {
                "allow_unjudged_safety": False,
                "export_alpaca_eval": True,
                "export_xstest": True,
                "export_or_bench": True,
            },
        }
        run_root = (
            Path(temporary) / "model" / "Qwen3Guard-Gen-8B"
        )
        frame = pd.DataFrame(
            [
                _row(),
                _row(
                    eval_split="utility",
                    attack_method=None,
                    utility_dataset_name="alpaca_eval",
                    label=None,
                    triggered=False,
                    zero_original_token_exposure=False,
                    assistant_output="helpful answer",
                ),
                _row(
                    eval_split="utility",
                    attack_method=None,
                    utility_dataset_name="xstest",
                    label=None,
                    triggered=False,
                    zero_original_token_exposure=False,
                    assistant_output="compliant answer",
                    id=7,
                    type="safe",
                ),
                _row(
                    eval_split="utility",
                    attack_method=None,
                    utility_dataset_name="or_bench",
                    label=None,
                    triggered=True,
                    zero_original_token_exposure=True,
                    assistant_output="I'm sorry",
                ),
            ]
        )
        atomic_to_csv(
            frame,
            run_root / "judged" / "detailed_c16_judged.csv",
        )

        exported = export_run(config)
        summaries = summarize_run(config)

        assert len(exported) == 3
        assert all(run_root in path.parents for path in exported)
        assert any(
            "qwen3guard-gen" in path.name for path in exported
        )
        stream_config = {
            **config,
            "guard": {"model_name": "Qwen3Guard-Stream-8B"},
        }
        stream_root = (
            Path(temporary) / "model" / "Qwen3Guard-Stream-8B"
        )
        stream_frame = frame.copy()
        stream_frame["guard_model"] = "Qwen3Guard-Stream-8B"
        atomic_to_csv(
            stream_frame,
            stream_root / "judged" / "detailed_c16_judged.csv",
        )
        stream_exported = export_run(stream_config)
        assert any(
            "qwen3guard-stream" in path.name
            or "qwen3guard_stream" in path.name
            for path in stream_exported
        )
        safety = pd.read_csv(
            run_root / "summaries" / "safety_by_attack.csv",
            encoding="utf-8-sig",
        )
        assert safety.loc[0, "dsr"] == 1.0
        safety_by_model = pd.read_csv(
            run_root / "summaries" / "safety_by_model.csv",
            encoding="utf-8-sig",
        )
        assert safety_by_model.loc[0, "dsr"] == 1.0
        latency = pd.read_csv(
            run_root / "summaries" / "latency_and_triggers.csv",
            encoding="utf-8-sig",
        )
        assert latency.loc[0, "unsafe_trigger_count"] == 2
        assert len(summaries) == 4
