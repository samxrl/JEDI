# -*- coding: utf-8 -*-
"""
probe baseline evaluation script.

Goals:
1. Reproduce a defense baseline based on a single-layer activation probe (probe-style) without modifying the existing project files.
2. Reuse the project's existing evaluation data loading, classification, and statistics logic.
3. Produce evaluation outputs similar to scripts/run_evaluation.py (utility + safety), without comparing against JEDI.
"""

import argparse
import contextlib
import gc
import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import GenerationConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = BASE_DIR / 'src'
SCRIPTS_DIR = BASE_DIR / 'scripts'

for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse helper utilities from the main evaluation script to avoid duplicating
# evaluation details:
# - dataset loading
# - model loading
# - harmfulness classification
# - metric aggregation and part of the formatted export logic
main_eval = importlib.import_module('run_evaluation')

load_model_and_tokenizer = main_eval.load_model_and_tokenizer
load_utility_dataset = main_eval.load_utility_dataset
load_safety_dataset = main_eval.load_safety_dataset
run_classification = main_eval.run_classification
calculate_metrics = main_eval.calculate_metrics
should_compute_keyword_fpr = main_eval.should_compute_keyword_fpr
annotate_keyword_false_positives = main_eval.annotate_keyword_false_positives
calculate_keyword_fpr_metrics = main_eval.calculate_keyword_fpr_metrics
build_generation_config = main_eval.build_generation_config
get_override_max_new_tokens = main_eval.get_override_max_new_tokens
apply_chat_template_compat = main_eval.apply_chat_template_compat

# Dynamically import LLAMA2_CLS_PROMPT to stay aligned with the main
# evaluation script.
judge_script_path = SCRIPTS_DIR / '02.5_judge_harmfulness.py'
spec = importlib.util.spec_from_file_location('judge_script', judge_script_path)
judge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge_module)
LLAMA2_CLS_PROMPT = judge_module.LLAMA2_CLS_PROMPT


@dataclass
class probeProbe:
    """
    Trained probe artifact.
    - layer_id: which hidden-state layer to use
    - threshold: refusal trigger threshold, based on a benign quantile
    - clf/mean/std: linear classifier and normalization parameters
    """
    layer_id: int
    threshold: float
    clf: LogisticRegression
    mean: np.ndarray
    std: np.ndarray

    def score(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.mean) / self.std
        return self.clf.predict_proba(z)[:, 1]


def _load_activation_file(path: Path) -> Optional[dict]:
    """Load an activation file; return None if it does not exist."""
    if not path.exists():
        logger.warning(f"Activation file not found: {path}")
        return None
    return torch.load(path, map_location='cpu')


def _find_subsequence(main_list: List[int], sub_list: List[int]) -> int:
    if not sub_list:
        return -1
    main_len = len(main_list)
    sub_len = len(sub_list)
    for i in range(main_len - sub_len + 1):
        if main_list[i:i + sub_len] == sub_list:
            return i
    return -1


def _extract_early_window_first_token(data: dict, layer: int) -> Optional[np.ndarray]:
    layer_data = data.get(layer)
    if not isinstance(layer_data, dict):
        return None
    tensor = layer_data.get('early_window')
    if tensor is None or tensor.numel() == 0:
        return None
    if tensor.dim() == 3:
        tensor = tensor[:, 0, :]
    elif tensor.dim() != 2:
        return None
    # NumPy does not support direct conversion from torch.bfloat16.
    # Cast in PyTorch first, then convert to NumPy.
    return tensor.detach().to(dtype=torch.float32, device='cpu').numpy()


def train_probe_probe(config: dict) -> probeProbe:
    """
    Train a single-layer probe-style detector:
    1) load compliance / benign activations
    2) train logistic regression on each candidate layer
    3) select the best layer based on training AUC
    4) determine the threshold from a benign quantile
    """
    llm_name = config['llm_name']
    activations_root = BASE_DIR / config['activations_dir'] / llm_name

    compliance_path = activations_root / f"{llm_name}_compliance_activations.pt"
    benign_path = activations_root / f"{llm_name}_benign_activations.pt"

    compliance_data = _load_activation_file(compliance_path)
    benign_data = _load_activation_file(benign_path)

    if compliance_data is None or benign_data is None:
        raise FileNotFoundError(
            "Missing activation files required to train the probe baseline. "
            "Please complete the scripts/02_extract_activations.py and "
            "scripts/03_extract_vectors.py pipeline first."
        )

    common_layers = sorted(set(compliance_data.keys()) & set(benign_data.keys()))
    if not common_layers:
        raise ValueError("No shared layers were found in compliance/benign activations for probe training.")

    best: Optional[probeProbe] = None
    best_auc = -1.0

    for layer in common_layers:
        # Build binary classification training data for the current layer:
        # compliance=1, benign=0.
        x_pos = _extract_early_window_first_token(compliance_data, layer)
        x_neg = _extract_early_window_first_token(benign_data, layer)
        if x_pos is None or x_neg is None:
            continue

        y = np.concatenate([
            np.ones(len(x_pos), dtype=np.int64),
            np.zeros(len(x_neg), dtype=np.int64),
        ])
        x = np.concatenate([x_pos, x_neg], axis=0)
        if len(x) < 10:
            continue

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        x_norm = (x - mean) / std

        clf = LogisticRegression(max_iter=500, class_weight='balanced', solver='liblinear')
        clf.fit(x_norm, y)
        probs = clf.predict_proba(x_norm)[:, 1]
        auc = roc_auc_score(y, probs)

        # The threshold is determined only by the benign sample distribution
        # to control false triggers.
        benign_probs = probs[y == 0]
        threshold = float(np.quantile(benign_probs, config['probe_config'].get('threshold_quantile', 0.95)))

        logger.info(f"Layer {layer}: train AUC={auc:.4f}, threshold={threshold:.4f}")

        if auc > best_auc:
            best_auc = auc
            best = probeProbe(
                layer_id=int(layer),
                threshold=threshold,
                clf=clf,
                mean=mean,
                std=std,
            )

    if best is None:
        raise ValueError("Failed to train a usable probe baseline. Please inspect the activation file contents.")

    logger.info(f"Selected layer {best.layer_id} for the probe baseline (train AUC={best_auc:.4f})")
    return best


def save_probe_artifacts(probe: probeProbe, model_name: str) -> Path:
    """
    Save the trained probe parameters to scripts/baselines/probe/data/<model_name>.
    Saved content includes:
    - threshold, layer id, and feature extraction configuration
    - normalization parameters mean/std
    - linear classifier weights and bias
    """
    artifact_dir = BASE_DIR / 'scripts' / 'baselines' / 'probe' / 'data' / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        'model_name': model_name,
        'layer_id': probe.layer_id,
        'threshold': probe.threshold,
        'feature_source': 'train_early_window_first_token__online_prompt_last_token',
        'mean': probe.mean,
        'std': probe.std,
        'clf': {
            'class_name': probe.clf.__class__.__name__,
            'params': probe.clf.get_params(),
            'classes_': probe.clf.classes_,
            'coef_': probe.clf.coef_,
            'intercept_': probe.clf.intercept_,
            'n_features_in_': getattr(probe.clf, 'n_features_in_', None),
        },
    }

    artifact_path = artifact_dir / 'probe.pt'
    torch.save(artifact, artifact_path)

    summary_path = artifact_dir / 'probe_summary.json'
    summary = {
        'model_name': model_name,
        'layer_id': probe.layer_id,
        'threshold': probe.threshold,
        'feature_source': 'train_early_window_first_token__online_prompt_last_token',
        'artifact_path': str(artifact_path),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved probe artifact: {artifact_path}")
    logger.info(f"Saved probe summary: {summary_path}")
    return artifact_dir


def load_probe_artifacts(model_name: str) -> Optional[probeProbe]:
    """
    Load a saved probe from scripts/baselines/probe/data/<model_name>.
    It is considered reusable only when both probe.pt and probe_summary.json exist.
    """
    artifact_dir = BASE_DIR / 'scripts' / 'baselines' / 'probe' / 'data' / model_name
    artifact_path = artifact_dir / 'probe.pt'
    summary_path = artifact_dir / 'probe_summary.json'

    if not artifact_path.exists() or not summary_path.exists():
        return None

    payload = torch.load(artifact_path, map_location='cpu')
    if payload.get('feature_source') != 'train_early_window_first_token__online_prompt_last_token':
        logger.info(f"Ignored legacy probe artifact due to mismatched feature source: {artifact_path}")
        return None
    clf_payload = payload['clf']

    clf = LogisticRegression(**clf_payload['params'])
    clf.classes_ = np.asarray(clf_payload['classes_'])
    clf.coef_ = np.asarray(clf_payload['coef_'])
    clf.intercept_ = np.asarray(clf_payload['intercept_'])
    if clf_payload.get('n_features_in_') is not None:
        clf.n_features_in_ = int(clf_payload['n_features_in_'])

    probe = probeProbe(
        layer_id=int(payload['layer_id']),
        threshold=float(payload['threshold']),
        clf=clf,
        mean=np.asarray(payload['mean']),
        std=np.asarray(payload['std']),
    )
    logger.info(f"Loaded probe artifact: {artifact_path}")
    return probe


def get_or_train_probe(config: dict) -> probeProbe:
    model_name = config['llm_name']
    probe = load_probe_artifacts(model_name)
    if probe is not None:
        return probe

    logger.info(f"No reusable probe artifact found. Training and calibrating a new probe for {model_name}")
    probe = train_probe_probe(config)
    save_probe_artifacts(probe, model_name)
    return probe


@torch.no_grad()
def _prompt_is_harmful(model, tokenizer, prompt: str, probe: probeProbe) -> bool:
    """
    Run probe detection on a single prompt.
    Note: detection happens before generation; harmful prompts are refused directly.
    """
    conv = [{"role": "user", "content": prompt}]
    text = apply_chat_template_compat(tokenizer, conv, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors='pt', padding=False, truncation=True).to(model.device)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = inputs['input_ids'][0].detach().cpu().tolist()
    prompt_start_idx = _find_subsequence(input_ids, prompt_ids)
    if prompt_start_idx == -1:
        raise ValueError("Could not locate the early_window span corresponding to the user prompt in the model input.")
    prompt_end_idx = prompt_start_idx + len(prompt_ids) - 1

    outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden = outputs.hidden_states[probe.layer_id + 1]  # hidden_states[0] is the embedding output
    feat = hidden[:, prompt_end_idx, :].detach().float().cpu().numpy()
    score = probe.score(feat)[0]
    return bool(score >= probe.threshold)


@torch.no_grad()
def run_probe_generation(
    model,
    tokenizer,
    prompts: List[str],
    gen_config: GenerationConfig,
    batch_size: int,
    probe: probeProbe,
    refusal_message: str,
) -> Tuple[List[str], List[int]]:
    """
    Baseline generation entry point:
    - run prompt-level detection first
    - return refusal_message directly if the threshold is hit (trigger_step=0)
    - otherwise continue with normal generation
    """
    outputs: List[str] = []
    trigger_steps: List[int] = []

    # Keep batch-wise iteration to stay close to the project's original
    # generation behavior, while running detection per sample.
    for i in tqdm(range(0, len(prompts), batch_size), desc='Running probe generation'):
        batch_prompts = prompts[i:i + batch_size]

        safe_prompts: List[str] = []
        safe_indices: List[int] = []
        batch_outputs: List[Optional[str]] = [None] * len(batch_prompts)
        batch_triggers: List[int] = [-1] * len(batch_prompts)

        for j, prompt in enumerate(batch_prompts):
            try:
                flagged = _prompt_is_harmful(model, tokenizer, prompt, probe)
            except Exception as e:
                logger.warning(f"Probe detection failed; falling back to normal generation. error={e}")
                flagged = False

            if flagged:
                batch_outputs[j] = refusal_message
                batch_triggers[j] = 0
            else:
                safe_prompts.append(prompt)
                safe_indices.append(j)

        if safe_prompts:
            conversations = [[{"role": "user", "content": p}] for p in safe_prompts]
            input_texts = [
                apply_chat_template_compat(tokenizer, conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]
            inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).to(model.device)
            generated_ids = model.generate(**inputs, generation_config=gen_config)

            for local_idx, ids in enumerate(generated_ids):
                input_len = len(inputs['input_ids'][local_idx])
                out_text = tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
                dst = safe_indices[local_idx]
                batch_outputs[dst] = out_text

        outputs.extend([x if x is not None else '' for x in batch_outputs])
        trigger_steps.extend(batch_triggers)

    return outputs, trigger_steps


def save_alpaca_format(sub_df: pd.DataFrame, output_path: Path, generator_name: str) -> None:
    records = []
    for _, row in sub_df.iterrows():
        records.append({
            'dataset': row.get('dataset', 'alpaca_eval'),
            'instruction': row.get('instruction', row.get('prompt', '')),
            'output': row.get('assistant_output', ''),
            'generator': generator_name,
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def save_xstest_format(sub_df: pd.DataFrame, output_path: Path) -> None:
    target_cols = ['id', 'type', 'prompt', 'completion', 'annotation_1', 'annotation_2', 'agreement', 'final_label']
    out = pd.DataFrame()
    out['id'] = sub_df['id'] if 'id' in sub_df.columns else range(1, len(sub_df) + 1)
    out['type'] = sub_df['type'] if 'type' in sub_df.columns else 'N/A'
    out['prompt'] = sub_df['prompt']
    out['completion'] = sub_df['assistant_output']
    for c in ['annotation_1', 'annotation_2', 'agreement', 'final_label']:
        out[c] = None
    out = out[target_cols]
    out.to_csv(output_path, index=False, encoding='utf-8-sig')


def main():
    """
    Main workflow, following the style of the project's primary evaluation script:
    A. load config + train probe
    B. run utility/safety generation
    C. run the harmfulness classifier on safety results
    D. save the full table, per-dataset/per-attack details, and summaries
    """
    parser = argparse.ArgumentParser(description='Run probe baseline evaluation.')
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/baselines/probe/evaluation_config.yaml',
        help='Path to the config file, relative to the project root.',
    )
    args = parser.parse_args()

    config_path = BASE_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    llm_name = config['llm_name']
    output_dir = BASE_DIR / config['output_dir'] / llm_name
    data_dir = BASE_DIR / config['data_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    run_utility = config.get('run_utility_evaluation', True)
    run_safety = config.get('run_safety_evaluation', True)

    probe = get_or_train_probe(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm_config = config['llm_config']
    model, tokenizer = load_model_and_tokenizer(
        model_name=llm_config['path'],
        model_kwargs=llm_config.get('kwargs', {}),
        device=device,
    )

    gen_kwargs = config.get('generation_kwargs', {})
    gen_config = GenerationConfig(
        max_new_tokens=gen_kwargs.get('max_new_tokens', 256),
        do_sample=gen_kwargs.get('do_sample', False),
        **{k: v for k, v in gen_kwargs.items() if k not in {'max_new_tokens', 'do_sample'}},
    )
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id
    batch_size = config.get('batch_size', 1)

    all_results: List[pd.DataFrame] = []
    refusal_message = config.get('probe_config', {}).get('refusal_message',
                                                         "I'm sorry, I can't help with that request.")

    if run_utility:
        # Utility: evaluate each dataset separately and append results to all_results.
        utility_cfg = config['utility_dataset_config']
        df_u = load_utility_dataset(utility_cfg, data_dir, utility_cfg.get('sample_size', 0))
        if not df_u.empty:
            for dataset_name in df_u['utility_dataset_name'].unique():
                sub = df_u[df_u['utility_dataset_name'] == dataset_name].reset_index(drop=True)
                prompts = sub['prompt'].tolist()
                local_gen_cfg = build_generation_config(gen_config, get_override_max_new_tokens(sub))
                out, trig = run_probe_generation(model, tokenizer, prompts, local_gen_cfg, batch_size, probe, refusal_message)
                sub['assistant_output'] = out
                sub['trigger_step'] = trig
                sub['condition'] = 'probe'
                sub['eval_split'] = 'utility'
                all_results.append(sub)

    if run_safety:
        # Safety: evaluate each attack method separately and append results to all_results.
        safety_cfg = config['safety_dataset_config']
        df_s = load_safety_dataset(safety_cfg, data_dir, safety_cfg.get('sample_size', 0))
        if not df_s.empty:
            for method in df_s['attack_method'].unique():
                sub = df_s[df_s['attack_method'] == method].reset_index(drop=True)
                prompts = sub['prompt'].tolist()
                local_gen_cfg = build_generation_config(gen_config, get_override_max_new_tokens(sub))
                out, trig = run_probe_generation(model, tokenizer, prompts, local_gen_cfg, batch_size, probe, refusal_message)
                sub['assistant_output'] = out
                sub['trigger_step'] = trig
                sub['condition'] = 'probe'
                sub['eval_split'] = 'safety'
                all_results.append(sub)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_results:
        logger.warning('No evaluation results were generated to save.')
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df['trigger_step'] = final_df['trigger_step'].apply(lambda x: pd.NA if x == -1 else x).astype('Int64')

    # Harmfulness classification is only needed for safety samples.
    mask_safety = final_df['eval_split'] == 'safety'
    final_df['label'] = pd.NA

    if mask_safety.any():
        # Only safety samples need harmfulness classification, matching the
        # main evaluation script.
        cls_cfg = config['classifier_config']
        classifier_model, classifier_tokenizer = load_model_and_tokenizer(
            model_name=cls_cfg['path'],
            model_kwargs=cls_cfg.get('kwargs', {}),
            device=device,
        )
        labels = run_classification(
            classifier_model,
            classifier_tokenizer,
            final_df[mask_safety].copy(),
            config.get('classifier_batch_size', 1),
        )
        final_df.loc[mask_safety, 'label'] = labels

        del classifier_model, classifier_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Full detailed results table.
    all_csv = output_dir / f"{llm_name}_probe_evaluation_all_results.csv"
    final_df.to_csv(all_csv, index=False, encoding='utf-8-sig')
    logger.info(f"Saved detailed results: {all_csv}")

    # Utility: output one CSV + summary JSON for each dataset.
    utility_df = final_df[final_df['eval_split'] == 'utility']
    for dataset_name in utility_df.get('utility_dataset_name', pd.Series(dtype=str)).dropna().unique():
        if dataset_name == 'N/A':
            continue
        sub = utility_df[utility_df['utility_dataset_name'] == dataset_name].copy()
        if should_compute_keyword_fpr(dataset_name, sub):
            sub = annotate_keyword_false_positives(sub)
            metrics = calculate_keyword_fpr_metrics(sub, condition='guarded')
        else:
            metrics = calculate_metrics(sub.assign(label='no'), condition='guarded')

        detail_path = output_dir / f"{llm_name}_probe_evaluation_detailed_utility_{dataset_name}.csv"
        sub.to_csv(detail_path, index=False, encoding='utf-8-sig')

        summary = {f'utility_fpr_{dataset_name}': {'probe': metrics}}
        summary_path = output_dir / f"{llm_name}_probe_evaluation_summary_utility_{dataset_name}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(
            f"Utility (FPR) for {dataset_name} - "
            f"probe: {metrics['rate']:.4f}, "
            f"TriggerRate: {metrics.get('trigger_rate', 0.0):.4f}"
        )

        if dataset_name == 'alpaca_eval':
            # Reuse the main script ecosystem: export in Alpaca Eval standard format.
            save_alpaca_format(
                sub,
                output_dir / f"{llm_name}-alpaca_eval-probe.json",
                f"{llm_name}-probe",
            )

        if 'xstest' in str(dataset_name).lower():
            # Reuse the main script ecosystem: export in xstest standard format.
            save_xstest_format(sub, output_dir / f"{llm_name}_xstest_probe.csv")

    # Safety: output one CSV + summary JSON for each attack method.
    safety_df = final_df[final_df['eval_split'] == 'safety']
    for method in safety_df.get('attack_method', pd.Series(dtype=str)).dropna().unique():
        if method == 'N/A':
            continue
        sub = safety_df[safety_df['attack_method'] == method].copy()
        metrics = calculate_metrics(sub, condition='guarded')

        detail_path = output_dir / f"{llm_name}_probe_evaluation_detailed_attack_{method}.csv"
        sub.to_csv(detail_path, index=False, encoding='utf-8-sig')

        summary = {f'safety_asr_attack_{method}': {'probe': metrics}}
        summary_path = output_dir / f"{llm_name}_probe_evaluation_summary_attack_{method}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(
            f"Safety (ASR) for {method} - "
            f"probe: {metrics['rate']:.4f}, "
            f"TriggerRate: {metrics.get('trigger_rate', 0.0):.4f}"
        )

    logger.info('Probe baseline evaluation completed.')


if __name__ == '__main__':
    main()
