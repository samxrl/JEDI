# Third-party source

This directory vendors the following pinned version of IBM's official
[`activation-steering`](https://github.com/IBM/activation-steering):

```text
commit: 52be60235ee309b46c49d6d5877f36e20c52e6ab
date:   2025-08-29
license: Apache License 2.0
```

The source is located in `third_party/activation-steering/`; the original
license is preserved as `LICENSE` in the same directory, and the original
README is stored as `UPSTREAM_README.md`. This baseline does not run
`pip install -e`. Instead, it places the pinned source directory first on
`sys.path` to prevent the officially declared Torch 2.3.1 / Transformers
4.41.2 dependencies from overriding JEDI's current environment.

There are only six categories of local changes, each documented at the top of
the modified file or next to the affected code:

1. `steering_vector.py`: expands each positive/negative example's own suffix
   and calculates the actual token span from character offsets, fixing the
   upstream behavior that uses the length of `suffixes[0][0]` for every sample.
2. `config.py`: allows optional upstream logs to be redirected to CAST's local
   cache through `CAST_ACTIVATION_STEERING_LOG_DIR`.
3. `config.py`, `console.py`, `utils.py`, and `malleable_model.py`: makes
   `rich`, which is used only for terminal display, optional. When it is
   unavailable, `tqdm` progress bars are used; plain output is used only if
   `tqdm` is also unavailable.
4. `steering_vector.py`: explicitly converts aggregated fp16/bf16 hidden states
   to float32 NumPy arrays for use by scikit-learn PCA.
5. `utils.py`, `malleable_model.py`, and `steering_vector.py`: enables deferred
   annotations so Python 3.9 does not evaluate Python 3.10 union syntax such as
   `Tensor | None` during import.
6. `steering_vector.py`: converts the PCA explained-variance NumPy scalar to a
   native `float` before writing JSON. The adapter also atomically replaces the
   final vector file using a temporary `.svec`.

The algorithm retains the official `pca_pairwise` method, one-time condition
evaluation at prefill, fixed behavior vector, and comparator semantics. See
`patches/README.md` for change details and validation.
