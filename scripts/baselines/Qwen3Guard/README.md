# Periodic Qwen3Guard Gen/Stream 16/32 for JEDI

This directory implements an external safety-classifier baseline paired with
JEDI's current evaluation results. It reads only saved `condition=baseline`
responses under `data/evaluations/<target_llm>/`. It does not regenerate target
model outputs or modify JEDI's `src/`, root configuration, or original
evaluation results.

Two official 8B guard models are supported:

- [`Qwen/Qwen3Guard-Gen-8B`](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B);
- [`Qwen/Qwen3Guard-Stream-8B`](https://huggingface.co/Qwen/Qwen3Guard-Stream-8B).

In a paper or rebuttal, use the names `Qwen3Guard-Gen-16/32` and
`Qwen3Guard-Stream-16/32 (our checkpoint replay adaptation)`, respectively.
Both are offline checkpoint replays over deterministic undefended outputs and
do not represent integration with a real online generation service.

## Shared chunk-buffer semantics

The semantic context for every check is:

```text
user:      original jailbreak/benign prompt
assistant: cumulative prefix from the start of the response through the current checkpoint
```

Target-model tokens first enter a buffer. The current chunk is released only
after the cumulative prefix satisfies the release policy. The default is
`guard.controversial_policy: block`, so both `Unsafe` and `Controversial` are
blocked, while the original three-level severity is preserved.

If chunk $j$ of size $C$ triggers for the first time:

$$
\operatorname{trigger\_end}=\min(jC,T)
$$

$$
\operatorname{released\_token\_count}=(j-1)C
$$

The current unreleased chunk and all later content are discarded. The already
released prefix remains visible, followed by a fixed refusal. A final remainder
shorter than $C$ tokens must also be checked. Non-triggered responses reuse the
source string verbatim to prevent tokenizer round trips from changing spaces.
An empty response that encodes to zero target-tokenizer tokens has no content to
release, so no checkpoint is created; it is recorded as having zero guard
checks and preserved unchanged.

The 16-token check grid also covers all regular 32-token checkpoints. The
implementation scores shared prefixes once, then materializes both chunk sizes
separately. The 32-token result ignores labels at positions such as 16, 48, and
80 that are not part of its own grid.

## Two official backends

### Qwen3Guard-Gen

The Gen backend uses `AutoModelForCausalLM` and the model's own chat template,
with a generation limit of 128 tokens. The canonical output is:

```text
Safety: Safe | Unsafe | Controversial
Categories: <category list> | None
Refusal: Yes | No
```

The parser strictly matches the first nonempty `Safety:` line and saves the
category and refusal attributes. By default, a parse failure is retried once;
if it still fails, the system fails closed.

### Qwen3Guard-Stream

The Stream backend strictly follows Qwen's official interface:

1. Load the dedicated classification head with
   `AutoModel(..., trust_remote_code=True)`.
2. Use
   `apply_chat_template(add_generation_prompt=False, enable_thinking=False)`.
3. First pass the complete user turn to
   `stream_moderate_from_ids(..., role="user", stream_state=None)`.
4. Then pass assistant tokens one by one while reusing the returned
   `stream_state`.
5. Use `risk_level/risk_prob` and `category/category_prob` from the final
   assistant token, then call `close_stream`.

This baseline performs output-side defense only: user moderation initializes
the official state but does not terminate early based on the user label. For a
target model with a non-Qwen3 tokenizer, checkpoint text is re-encoded into the
Stream model's Qwen3 vocabulary. To preserve the existing cumulative-checkpoint
experiment definition, every checkpoint reruns the official streaming-state
interface from the user turn. This is not a real online integration that
retains KV/state across target-model generation calls.

## Guard-model selection

In the configuration file, change only:

```yaml
guard:
  model_name: "Qwen3Guard-Stream-8B"
```

Alternatively, use this argument with any command:

```powershell
--guard-model Qwen3Guard-Stream-8B
```

Allowed values are `Qwen3Guard-Gen-8B` and `Qwen3Guard-Stream-8B`. The program
automatically switches:

- the Gen/Stream backend;
- `model_id`;
- the final directory of `model_path`;
- the official context length (Gen 32768, Stream 8192);
- Stream's single-stream `batch_size=1`;
- the result condition: `qwen3guard_gen_c16/c32` or
  `qwen3guard_stream_c16/c32`.

The default read-only model paths resolve to:

```text
R:\models\Qwen3Guard-Gen-8B
R:\models\Qwen3Guard-Stream-8B
```

Artifact paths are determined automatically by the target LLM and guard model.
Switching guard models naturally selects a different directory. `--run-id` is
no longer used or accepted.

## Write isolation and target-model override

All runtime writes pass through path guards and may only enter this directory:

```text
scripts/baselines/Qwen3Guard/
├── inputs/<target_llm>/<guard_model>/  # source hashes, target token IDs, checkpoint plan
├── runs/<target_llm>/<guard_model>/    # prefix labels, materialized results, HarmBench results, summaries
└── cache/             # Hugging Face, Torch, CUDA, and temporary caches
```

For example, evaluating Vicuna-7B with the Stream guard always writes to:

```text
inputs/vicuna_7b_v1_5/Qwen3Guard-Stream-8B/
runs/vicuna_7b_v1_5/Qwen3Guard-Stream-8B/
```

Repeated runs with the same target LLM, guard model, and configuration always
use the same path and resume through manifests and hashes. If that stable
directory already contains an incompatible configuration, the program reports
an explicit error rather than silently mixing or overwriting existing prefix
labels.

JEDI's `data/evaluations/` is read-only, and its SHA-256 hashes are checked at
every stage. The guard model, target tokenizer, and HarmBench weights are all
read-only inputs.

`prepare` reads only JEDI main-experiment attack files listed in
`source.attack_methods` and enabled utility files. For example, whitelisted
`GCG` maps to `<model>_evaluation_detailed_attack_GCG.csv`; adaptive-attack
artifacts such as `adaptive_gcg` and `adaptive_pair` are excluded before CSV
loading. Selected main-experiment files must still contain
`condition=baseline`, or processing fails strictly.

Every command requires `--target-llm <name>`. It overrides the target-model name
in the configuration and replaces the directory name after the final `/` or
`\` in `tokenizer_path`. JEDI's main evaluation script
`scripts/run_evaluation.py` supports an optional argument of the same name to
override `llm_name` and the final directory in `llm_config.path`.

## Execution stages

```powershell
$python = 'R:\SARC\venv\Scripts\python.exe'
$root = 'scripts/baselines/Qwen3Guard'
$target = 'vicuna_7b_v1_5'
$guard = 'Qwen3Guard-Stream-8B'

& $python "$root/run.py" prepare `
  --config "$root/configs/periodic_qwen3guard.yaml" `
  --target-llm $target --guard-model $guard

& $python "$root/run.py" score-prefixes `
  --config "$root/configs/periodic_qwen3guard.yaml" `
  --target-llm $target --guard-model $guard

& $python "$root/run.py" materialize `
  --config "$root/configs/periodic_qwen3guard.yaml" `
  --target-llm $target --guard-model $guard `
  --chunk-sizes 16 32

& $python "$root/run_evaluation.py" all `
  --config "$root/configs/periodic_qwen3guard.yaml" `
  --target-llm $target --guard-model $guard `
  --chunk-sizes 16 32

& $python "$root/run.py" validate `
  --config "$root/configs/periodic_qwen3guard.yaml" `
  --target-llm $target --guard-model $guard
```

To validate input adaptation only, append `--max-samples-per-file 2` to
`prepare`. `score-prefixes` uses per-batch partial JSONL files. Rerunning with
the same target LLM, guard model, and configuration skips completed
checkpoints. `execution.batch_size` and `--batch-size` affect only throughput
and timing statistics, not label-semantic identity. Therefore, after overriding
the batch size on the command line during scoring, there is no need to pass it
again to `materialize`. New manifests record both the configured and observed
batch sizes and can read legacy `config.resolved.yaml` files to reuse completed
scores automatically.

## Evaluation outputs

- `raw/prefix_scores.jsonl.gz`: severity, release decision, categories,
  probabilities, and classification time;
- `defended/detailed_c16.csv`, `detailed_c32.csv`: user-visible responses;
- `judged/`: changed safety responses reclassified with JEDI's HarmBench
  template;
- `exports/`: AlpacaEval JSON, eight-column XSTest CSV, and OR-Bench CSV;
- `summaries/`: DSR/ASR, three-level trigger distribution, released-token
  counts, and detector-only latency.

`Refusal` is provided only by the Gen model; Stream outputs risk and category
probabilities. An unchanged response reuses its HarmBench label only when the
response hash matches and the original label is `yes/no`.

`score-prefixes` displays one global progress bar covering all checkpoints to
be scored. Resumed runs initialize it with the number of existing scores. Set
`execution.show_progress: false` to disable it.

## External metric aggregation for Gen results

`evaluate_gen_metrics.py` reads only completed `Qwen3Guard-Gen-8B` run
artifacts and writes the statistical section to
`scripts/baselines/baseline_results_summary.md`. AlpacaEval uses
`text_davinci_003` outputs from `data/raw/alpaca_eval.json` as reference
answers and invokes a DeepSeek OpenAI-compatible endpoint for concurrent
pairwise judging. The API key may be supplied only temporarily through an
environment variable or command-line argument and is never written to a file.
Per-item results for each model/chunk size are written to that run's
`runs/<target>/Qwen3Guard-Gen-8B/external_evaluations/`; reruns skip existing
judgments whose hashes match.

```powershell
$env:DEEPSEEK_API_KEY = '<temporary API key>'
& 'R:\SARC\venv\Scripts\python.exe' `
  scripts/baselines/Qwen3Guard/evaluate_gen_metrics.py --workers 32
```

If an AlpacaEval run contains fewer than 805 artifacts, the script computes
metrics from the available count and displays `n` in the summary table.
XSTest/OR-Bench independently sample 20% for each model and chunk using stable
SHA-256 ordering. The former reports Compliance Rate as the non-refusal-keyword
rate on sampled safe prompts; the latter reports FPR as the refusal-keyword
rate. Both are reproducible keyword proxies consistent with existing baselines
in this repository and cannot replace formal human or specialized-classifier
labels.

AlpacaEval timing columns use the average undefended times recorded in
`evaluate_gen_metrics.py`:
`TTFT = undefended TTFT + amortized e2e time of the first guard check`, and
`total generation time = undefended per-response generation time + cumulative guard e2e time`.
This is a synthetic time for offline replay. It does not measure the additional
time required for the target model to generate the first 16/32-token buffered
chunk and therefore is not a strict TTFR.

Recorded time covers guard-classification calls only and excludes the time the
target model spends generating its first $C$ tokens. It therefore cannot be
called complete TTFT or TTFR.

Run the CPU-only semantic tests with:

```powershell
& 'R:\SARC\venv\Scripts\python.exe' -m pytest `
  scripts/baselines/Qwen3Guard/tests -q
```

Qwen officially requires `transformers>=4.55.0` for Stream; this directory's
`requirements.txt` is configured accordingly.
