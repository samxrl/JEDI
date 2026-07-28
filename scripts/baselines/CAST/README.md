# CAST baseline for JEDI

This directory reproduces IBM's Conditional Activation Steering (CAST) in the
JEDI repository and integrates it with JEDI's current safety, utility, and
over-refusal evaluation protocol. The implementation is pinned to the official
`activation-steering` commit `52be60235ee309b46c49d6d5877f36e20c52e6ab`;
see [THIRD_PARTY.md](THIRD_PARTY.md) for the third-party source and local
compatibility patches.

## Method definition

CAST extracts two PCA directions during the offline stage:

- behavior vector: refusal prefixes (positive examples) and compliance prefixes
  (negative examples) under the same benign instruction, using
  `pca_pairwise + suffix-only`;
- condition vector: harmful prompts (positive examples) and benign prompts
  (negative examples), using `pca_pairwise + all-token mean`.

The condition layer, threshold, and comparator are selected by F1 score only on
the independent `condition_calibration` split. During the online stage, the
condition is evaluated once at prompt prefill. If it matches, the refusal
behavior vector is injected at a fixed strength of 1.5 into a fixed set of
layers and remains active throughout subsequent decoding.

This implementation intentionally does not use JEDI's whitening, CUSUM,
Gram-Schmidt decoupling, per-token risk updates, dynamic beta, or intervention
vector. It is therefore a prompt-conditioned activation-steering baseline, not
a simplified variant of JEDI.

## Data isolation and zero contamination

By default, 100 harmful prompts and 100 JustEval benign prompts are split with a
fixed seed as follows:

| split | harmful | benign | purpose |
| --- | ---: | ---: | --- |
| `vector_train` | 60 | 60 | condition vector; benign examples also train the behavior vector |
| `condition_calibration` | 20 | 20 | condition layer/threshold/comparator |
| `steering_validation` | 20 | 20 | validate the fixed steering parameters later if needed |

`prepare_data.py` applies Unicode NFKC and whitespace normalization, then checks
all candidates for SHA-256 overlap with JEDI's jailbreak, AlpacaEval, XSTest,
and OR-Bench prompts. Overlapping candidates are removed first and written to
`dropped_evaluation_overlaps.jsonl`; processing fails if any overlap remains in
the three sampled splits.

All writable paths are checked by path guards and must remain under this
directory. Hugging Face, Torch, Matplotlib, temporary files, Python caches,
offline data, vectors, checkpoints, and final results are written to this
directory's `cache/` and `runs/` subdirectories. JEDI's `data/raw/`, model
directories, main configuration, and existing evaluation results are read-only.

## Files

- `prepare_data.py`: builds prompt-level splits, copies prefix pairs, and
  generates the overlap audit;
- `extract_vectors.py`: extracts behavior/condition `.svec` files with the
  pinned official source;
- `calibrate_condition.py`: searches for the condition layer, threshold, and
  comparator;
- `src/runtime.py`: isolates per-request state, wraps/restores the model, and
  records runtime diagnostics;
- `src/progress.py`: provides unified step progress for all four stages and
  tqdm bars for long loops;
- `run_evaluation.py`: handles generation, checkpointing, HarmBench
  classification, and JEDI-format results;
- `run.py`: provides a unified entry point for the four stages above;
- `tests/`: covers paths, PCA/thresholds, suffix handling, and runtime behavior
  without loading a large model.

## Usage

First run the read-only or low-cost checks:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py prepare `
  --config scripts/baselines/CAST/configs/cast_config.yaml --dry-run

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py extract `
  --config scripts/baselines/CAST/configs/cast_config.yaml --dry-run

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py calibrate `
  --config scripts/baselines/CAST/configs/cast_config.yaml --dry-run

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py evaluate `
  --config scripts/baselines/CAST/configs/cast_config.yaml --dry-run
```

Build the artifacts:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py prepare `
  --config scripts/baselines/CAST/configs/cast_config.yaml

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py extract `
  --config scripts/baselines/CAST/configs/cast_config.yaml

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py calibrate `
  --config scripts/baselines/CAST/configs/cast_config.yaml
```

Run the unified evaluation:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py evaluate `
  --config scripts/baselines/CAST/configs/cast_config.yaml
```

### Specify the target LLM from the command line

Every stage and the unified entry point accept `--llm-name <model-name>`. When
provided, it overrides `model.name` in the configuration and replaces the final
path component after the last `/` or `\` in `model.path` with that name. For
example:

```powershell
$llm = 'qwen2_5_7b'

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py prepare `
  --config scripts/baselines/CAST/configs/cast_config.yaml --llm-name $llm

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py extract `
  --config scripts/baselines/CAST/configs/cast_config.yaml --llm-name $llm

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py calibrate `
  --config scripts/baselines/CAST/configs/cast_config.yaml --llm-name $llm

& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/CAST/run.py evaluate `
  --config scripts/baselines/CAST/configs/cast_config.yaml --llm-name $llm
```

If the original path is `../../../models/vicuna_7b_v1_5`, the argument above
produces `../../../models/qwen2_5_7b`. The `harmful_calibration_file`, model
loading `kwargs`, and classifier configuration still come from YAML; confirm
that they are appropriate for the target model before running.

All four stages display dynamic progress bars on terminal `stderr`, while the
final JSON is still written separately to `stdout`:

- `prepare`: data loading, evaluation-set scanning, deduplicated splitting,
  overlap auditing, and writing;
- `extract`: model loading, data construction, extraction of both vectors, and
  atomic saving; hidden-state batches and per-layer PCA have their own internal
  progress bars;
- `calibrate`: artifact loading, model loading, per-prompt condition scoring,
  threshold search, and saving;
- `evaluate`: artifact/model loading, generation by dataset or attack,
  HarmBench classification, and metric aggregation.

This work did not run vector extraction or the full evaluation. Complete a
single-model, small-sample smoke test for the GPU stages above before running
the formal six-model experiment.

## Artifacts and results

All content for each model is stored directly under:

```text
runs/<model>/
├── data/
│   ├── calibration_splits.jsonl
│   ├── prefix_pairs.json
│   ├── evaluation_manifest.jsonl
│   ├── dropped_evaluation_overlaps.jsonl
│   ├── overlap_report.json
│   └── data_manifest.json
├── artifacts/
│   ├── behavior_vector.svec
│   ├── condition_vector.svec
│   ├── vector_metadata.json
│   ├── condition_point.json
│   └── cast_params.yaml
└── results/
    ├── *_cast_evaluation_detailed_attack_*.csv
    ├── *_cast_evaluation_detailed_utility_*.csv
    ├── *-alpaca_eval-CAST.json
    ├── *_xstest_CAST.csv
    └── all_metrics.json
```

Each result additionally records the condition score/decision, official
comparator name and actual operator, fixed behavior layers/strength, vector
norm, trigger position (0 for a match, otherwise empty), streamer-measured TTFT,
generation time, artifact fingerprint, and configuration fingerprint.

After evaluation, the terminal prints refusal-keyword FPR for each benign
dataset, HarmBench DSR/ASR for each attack type, and sample-count-weighted
overall FPR, overall DSR, and condition trigger rate. The same metrics and
counts are written to `results/all_metrics.json`.

## Implementation boundaries

- Only the decoder-only Hugging Face models, greedy decoding, and single beam
  required by the current main experiment are supported.
- The official `LeashLayer` has class-level state and only processes batch row
  0, so batch size is forced to 1 both offline and online.
- Qwen3 follows JEDI's `enable_thinking=False` setting.
- `trigger_step=0` means the condition matched during prefill; it does not imply
  that the defense ultimately succeeded. Both condition trigger rate and
  HarmBench DSR must be reported.
- This evaluation still calls ordinary `model.generate()` and does not claim a
  completed integration with a real streaming interface.
