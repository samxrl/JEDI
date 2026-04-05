# Probe Baseline

This directory contains a lightweight probe-based baseline for safety and utility evaluation.

## Files

- `run_evaluation.py`: trains or loads a single-layer probe, runs generation, and writes evaluation outputs.
- `evaluation_config.yaml`: configuration for model paths, datasets, generation settings, and probe thresholding.
- `data/`: saved probe artifacts, such as `probe.pt` and `probe_summary.json`, organized by model name.
- `evaluations/`: generated evaluation results, including detailed CSV files and summary JSON files, organized by model name.

## Usage

Run the baseline from the project root:

```bash
python scripts/baselines/probe/run_evaluation.py --config scripts/baselines/probe/evaluation_config.yaml
```

The script reuses dataset loading, classification, and metric logic from the main evaluation pipeline.
