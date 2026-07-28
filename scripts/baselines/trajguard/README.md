# TrajGuard baseline for JEDI

This directory implements a TrajGuard reproduction for JEDI's evaluation
framework that follows the final paper formulas. It reuses JEDI's models,
attack data, HarmBench classifier, and result format, adding only TrajGuard's
offline geometric artifacts, online SGS, and conditional self-judging.

The method follows the [ACL 2026 Findings paper](https://aclanthology.org/2026.findings-acl.655/)
and the [authors' official repository](https://github.com/neuron-insight-lab/Trajguard).
Formal experiments should also record `git rev-parse HEAD` for the official
data repository to pin the data version.

## Implementation protocol

During the offline stage, for each decoder block:

1. Average representations from the final three valid tokens of the
   chat-templated prompt.
2. Fit 64-dimensional Incremental PCA using benign and malicious reference
   examples.
3. Fit benign/malicious Gaussian regions separately in projected space using
   Ledoit-Wolf.
4. Estimate MVD on an independent jailbreak set and select the eight layers
   with the smallest MVD.
5. Generate normally on an independent benign validation set and use the 99.5th
   percentile of all per-token streaming scores as the threshold.

Online risk is `d_benign - d_malicious`. Each layer first computes a truncated
mean over an eight-step window, the results are averaged across eight layers,
and an EWMA with historical weight 0.8 smooths the score. Triggering requires
three consecutive steps above the threshold. After a trigger, the same target
model compares sequence log probabilities for `" Yes"` and `" No"` by default.
A SAFE decision resets only the EWMA and consecutive counter. An UNSAFE decision
stops generation and replaces or terminates the response according to the
configuration.

Unlike the official experiment script, this implementation does not load a
second scorer model or re-encode and repeat prefill after every token. Online
scores read decoder-block outputs directly from the target model's current
cached decoding forward pass. Offline extraction uses the same block hooks,
avoiding offline/online coordinate differences caused by final-layer
normalization.

## Files

- `core.py`: artifact validation, multi-layer hooks, SGS, PAIR-Judge, and a
  Guard-style `model.generate` adapter.
- `build_artifacts.py`: data isolation, PCA, Gaussian regions, MVD layer
  selection, and streaming-threshold calibration.
- `run_evaluation.py`: integrates JEDI data loading, HarmBench decisions,
  FPR/ASR, and checkpointing.
- `artifact_config.yaml` / `evaluation_config.yaml`: formal reproduction
  configurations.
- `*_smoke.yaml`: low-cost integration configurations using data already in the
  repository; not for the paper's main table.
- `tests/`: tensor, hook, state-machine, hash, and recovery tests that do not
  load a large model.

## Prepare formal data

The formal configuration reads only data from the official TrajGuard repository
and does not depend on its runtime code. The official repository is pinned as a
Git submodule; initialize it after cloning JEDI:

```powershell
git submodule update --init scripts/baselines/trajguard/official/Trajguard
```

First check only paths, fields, and prompt-hash isolation across all splits:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/trajguard/build_artifacts.py `
  --config scripts/baselines/trajguard/artifact_config.yaml --dry-run
```

The formal configuration may automatically remove exact evaluation overlaps up
to 2% of a split and writes the removed hashes to `data_manifest.json`.
Exceeding that limit still terminates the run. If the dry run reports many
overlaps, resplit the data rather than disabling isolation checks in the formal
configuration.

## Build artifacts

Formal configuration:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/trajguard/build_artifacts.py `
  --config scripts/baselines/trajguard/artifact_config.yaml
```

Append `--llm-name <name>` to temporarily override the configured model name
and local model directory. For example, `--llm-name vicuna_7b_v1_5` replaces
the directory after the final `/` (or `\` on Windows) in the model path with
`vicuna_7b_v1_5`, so the path need not contain `models/`. The argument accepts
only a single directory name.

Low-cost integration configuration:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/trajguard/build_artifacts.py `
  --config scripts/baselines/trajguard/artifact_config_smoke.yaml
```

The output directory `scripts/baselines/trajguard/artifacts/<model>/` contains:

- `trajguard_artifacts.pt`: PCA and Gaussian tensors required at runtime;
- `selected_layers.json`: per-layer MVD, AUROC, Fisher ratio, and risk means;
- `calibration_report.json`: threshold and token/sequence trigger rates;
- four kinds of `*_hashes.json` plus `data_manifest.json`: data-isolation
  evidence;
- `artifact_config_resolved.yaml` and `trajguard_params.yaml`: auditable
  parameters.

Artifacts are bound to the model layer count, hidden size, and tokenizer
chat-template hash. Rebuild and validate them after changing the model version,
template, or quantization method.

## Run evaluation

Formal configuration:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/trajguard/run_evaluation.py `
  --config scripts/baselines/trajguard/evaluation_config.yaml
```

The evaluation command also supports `--llm-name <name>`. When specified, only
that target model runs, using the loading settings from the first model entry in
the configuration.

Smoke configuration:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' scripts/baselines/trajguard/run_evaluation.py `
  --config scripts/baselines/trajguard/evaluation_config_smoke.yaml
```

Each record additionally stores the first SGS trigger token, final UNSAFE stop
token, maximum streaming score, judge call/decision history, first/final
generated prefixes, exposed-token count, SGS/judge/total defense time, TTFT,
total generation time, trigger count, and monitoring errors. Checkpoint
recovery validates both artifact SHA-256 and the SHA-256 of
TrajGuard/generation/data/target-model configurations, preventing changed
parameters from reusing old results incorrectly. Each summary also aggregates
judge call rate, UNSAFE stop rate, first/final trigger steps, median/P95 prefix
exposure before unsafe stop, zero-prefix-exposure rate, and TTFT/runtime
diagnostics. After all evaluations complete, the terminal prints per-dataset
and weighted-overall FPR for benign data, plus per-attack and weighted-overall
DSR ($1-\mathrm{ASR}$), showing both numerators and denominators. Formal
evaluation details, checkpoints, and summaries are stored under
`scripts/baselines/trajguard/evaluations/<model>/`; run-script logs are stored
under `scripts/baselines/trajguard/logs/`. The code rejects any artifact or
evaluation output path outside the baseline directory to avoid contaminating
JEDI's shared `data/` directory.

## Two variants and output actions

- `use_pair_judge: false`: TrajGuard-SGS, which refuses immediately after a
  geometric trigger.
- `use_pair_judge: true`: TrajGuard, combining SGS with conditional
  self-judging.
- `unsafe_action: replace`: discard the generated prefix and return a fixed
  refusal; recommended for the main DSR table.
- `unsafe_action: terminate`: retain the prefix and append a refusal; used for
  real streaming-leakage analysis.

`replace` assumes the server buffers the response before making a decision. If
tokens have already been streamed to the client, report actual prefix exposure
using `trajguard_raw_prefix`, `trajguard_trigger_step`, and
`trajguard_exposed_tokens`; do not interpret it as zero leakage.

## Current boundaries

The online path is intentionally limited to decoder-only Hugging Face models,
`batch_size=1`, greedy decoding, `num_beams=1`, and plain tensor returns. This
covers JEDI's current main-experiment configuration without introducing
beam-search complexity that the rebuttal evaluation does not need.

Before formally running all six models, complete a smoke test on Vicuna-7B,
then validate layer separation, benign sequence-level trigger rate, judge call
rate, and harmful-prefix leakage on a second 7B model.
