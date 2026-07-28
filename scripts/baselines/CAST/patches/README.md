# Local compatibility patches

## 1. Per-sample suffix span

At upstream commit `52be602…`, `batched_get_hiddens(..., "suffix-only")` uses
the token count of `suffixes[0][0]` for every sample. This mishandles all of the
following:

- different refusal-prefix lengths;
- the length difference between negative compliance and positive refusal
  examples;
- tokens merged by the tokenizer across the prompt/suffix boundary.

The local patch expands suffixes according to `SteeringDataset`'s suffix-major
ordering and preferentially uses a fast tokenizer's `offset_mapping` to select
tokens that overlap the actual suffix character interval. The slow-tokenizer
fallback uses the length difference and includes one extra boundary token. This
change only implements the original definition of suffix-only aggregation; it
does not change positive/negative examples, PCA, or direction selection.

Corresponding test:

```text
tests/test_vendor_suffix_span.py
```

## 2. Local log directory

Optional upstream file logs are written to the current working directory by
default. `config.py` now reads `CAST_ACTIVATION_STEERING_LOG_DIR`; every entry
point sets it to
`scripts/baselines/CAST/cache/activation_steering_logs` before importing the
upstream package.

Corresponding test:

```text
tests/test_paths.py
```

## 3. Optional rich output

The official package uses `rich` for terminal logs and progress bars, but this
optional display dependency is not installed in JEDI's current virtual
environment. When `rich` is unavailable, `config.py`, `console.py`, `utils.py`,
and `malleable_model.py` use JEDI's existing `tqdm` installation to display
progress for long loops. They fall back to plain `print`/iteration only if
`tqdm` is also unavailable. The numeric paths for PCA, hidden states, condition
scores, and steering remain unchanged. This avoids resolving the official
`pyproject.toml`, which could downgrade JEDI's Torch/Transformers versions.

## 4. sklearn-compatible activation dtype

JEDI's production model configuration uses `bfloat16`, while
NumPy/scikit-learn cannot directly accept bfloat16 ndarrays from some PyTorch
versions. `steering_vector.py` explicitly converts aggregated hidden states to
float32 when moving them to the CPU. This gives PCA a stable input precision
without changing the model-forward dtype.

Corresponding test:

```text
tests/test_vendor_suffix_span.py
```

## 5. Python 3.9 deferred annotations

The pinned upstream source uses union types such as `Tensor | None` and
`MalleableModel | PreTrainedModel` in `utils.py`, `malleable_model.py`, and
`steering_vector.py`. Python 3.9 evaluates these expressions during import and
raises `TypeError`. The local patch adds
`from __future__ import annotations` to the top of all three files. It only
defers annotation evaluation and does not change any model, vector, or steering
runtime logic.

Corresponding test:

```text
tests/test_vendor_python39.py
```

## 6. JSON-safe and atomic vector serialization

Upstream `SteeringVector.save()` passes the `numpy.float32` explained variance
returned by scikit-learn PCA directly to `json.dump`, which raises
`TypeError: Object of type float32 is not JSON serializable` after vector
extraction has completed. The local patch first converts explained variance to
a Python `float`. The adapter's `save_vector()` writes a temporary `.svec` in
the same directory and atomically replaces the target file on success; on
failure, it deletes the temporary file and preserves the original target.

Corresponding test:

```text
tests/test_vector_serialization.py
```

## Explicitly preserved official behavior

- Condition-comparator names retain their upstream semantics: `smaller`
  actually evaluates `score > threshold`, while `larger` actually evaluates
  `score < threshold`.
- Vector extraction reads `hidden_states[layer_id + 1]`, while `LeashLayer`
  evaluates/injects at the input to decoder block `layer_id`. Artifacts record
  this official layer convention; the baseline does not silently correct it.
- The condition is evaluated only on the first prefill forward pass of each
  request.
- Batch size is fixed at 1, and class-level state is reset independently before
  and after every request.
