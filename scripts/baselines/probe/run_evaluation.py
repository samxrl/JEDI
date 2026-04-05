# -*- coding: utf-8 -*-
"""
probe baseline evaluation script.

目标：
1. 在不修改现有项目文件的前提下，复现一个基于单层激活探针（probe 风格）的防御基线；
2. 复用项目已有评估数据加载、分类与统计逻辑；
3. 输出与 scripts/run_evaluation.py 同类评估结果（utility + safety），但不与 JEDI 对比。
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

# 复用主评估脚本的工具函数，避免重复维护评估细节：
# - 数据集加载
# - 模型加载
# - harmfulness 分类
# - 指标统计与部分格式化导出逻辑
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

# 动态导入 LLAMA2_CLS_PROMPT（与主评估脚本保持一致）
judge_script_path = SCRIPTS_DIR / '02.5_judge_harmfulness.py'
spec = importlib.util.spec_from_file_location('judge_script', judge_script_path)
judge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge_module)
LLAMA2_CLS_PROMPT = judge_module.LLAMA2_CLS_PROMPT


@dataclass
class probeProbe:
    """
    训练完成后的探针对象。
    - layer_id: 使用哪一层隐藏状态
    - threshold: 触发拒答阈值（基于 benign 分位数）
    - clf/mean/std: 线性分类器与标准化参数
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
    """加载激活文件；不存在时返回 None（由上层统一报错）。"""
    if not path.exists():
        logger.warning(f"激活文件不存在: {path}")
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
    训练 probe 风格单层 probe：
    1) 读取 compliance / benign 激活
    2) 对每个候选层训练 logistic regression
    3) 依据训练 AUC 选择最佳层
    4) 用 benign 分位数确定阈值
    """
    llm_name = config['llm_name']
    activations_root = BASE_DIR / config['activations_dir'] / llm_name

    compliance_path = activations_root / f"{llm_name}_compliance_activations.pt"
    benign_path = activations_root / f"{llm_name}_benign_activations.pt"

    compliance_data = _load_activation_file(compliance_path)
    benign_data = _load_activation_file(benign_path)

    if compliance_data is None or benign_data is None:
        raise FileNotFoundError(
            "缺少训练 probe probe 所需激活文件。"
            "请先完成 scripts/02_extract_activations.py 与 scripts/03_extract_vectors.py 相关流程。"
        )

    common_layers = sorted(set(compliance_data.keys()) & set(benign_data.keys()))
    if not common_layers:
        raise ValueError("在 compliance/benign 激活中没有共同层可用于训练 probe。")

    best: Optional[probeProbe] = None
    best_auc = -1.0

    for layer in common_layers:
        # 为当前层构建二分类训练数据：compliance=1, benign=0
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

        # 阈值仅由 benign 样本分布决定（控制误触发）
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
        raise ValueError("未能训练出可用的 probe probe（请检查激活文件内容）。")

    logger.info(f"选择层 {best.layer_id} 作为 probe baseline probe（train AUC={best_auc:.4f}）")
    return best


def save_probe_artifacts(probe: probeProbe, model_name: str) -> Path:
    """
    将训练后的 probe 参数保存到 scripts/baselines/probe/data/<model_name>。
    保存内容包含：
    - 阈值、层号、特征提取配置
    - 标准化参数 mean/std
    - 线性分类器权重与偏置
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

    logger.info(f"已保存 probe 工件: {artifact_path}")
    logger.info(f"已保存 probe 摘要: {summary_path}")
    return artifact_dir


def load_probe_artifacts(model_name: str) -> Optional[probeProbe]:
    """
    从 scripts/baselines/probe/data/<model_name> 加载已保存的 probe。
    仅当 probe.pt 和 probe_summary.json 同时存在时才视为可复用。
    """
    artifact_dir = BASE_DIR / 'scripts' / 'baselines' / 'probe' / 'data' / model_name
    artifact_path = artifact_dir / 'probe.pt'
    summary_path = artifact_dir / 'probe_summary.json'

    if not artifact_path.exists() or not summary_path.exists():
        return None

    payload = torch.load(artifact_path, map_location='cpu')
    if payload.get('feature_source') != 'train_early_window_first_token__online_prompt_last_token':
        logger.info(f"已忽略旧版 probe 工件（特征来源不匹配）: {artifact_path}")
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
    logger.info(f"已加载 probe 工件: {artifact_path}")
    return probe


def get_or_train_probe(config: dict) -> probeProbe:
    model_name = config['llm_name']
    probe = load_probe_artifacts(model_name)
    if probe is not None:
        return probe

    logger.info(f"未找到可复用的 probe 工件，开始为 {model_name} 训练并校准新 probe")
    probe = train_probe_probe(config)
    save_probe_artifacts(probe, model_name)
    return probe


@torch.no_grad()
def _prompt_is_harmful(model, tokenizer, prompt: str, probe: probeProbe) -> bool:
    """
    对单条 prompt 进行探针检测。
    注意：这里在“生成前”检测；若判为有害将直接拒答。
    """
    conv = [{"role": "user", "content": prompt}]
    text = apply_chat_template_compat(tokenizer, conv, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors='pt', padding=False, truncation=True).to(model.device)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = inputs['input_ids'][0].detach().cpu().tolist()
    prompt_start_idx = _find_subsequence(input_ids, prompt_ids)
    if prompt_start_idx == -1:
        raise ValueError("无法在模型输入中定位用户 prompt 对应的 early_window 区间。")
    prompt_end_idx = prompt_start_idx + len(prompt_ids) - 1

    outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden = outputs.hidden_states[probe.layer_id + 1]  # hidden_states[0] 为 embedding
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
    baseline 生成入口：
    - 先做 prompt 级检测
    - 命中阈值则直接返回 refusal_message（trigger_step=0）
    - 未命中则执行常规生成
    """
    outputs: List[str] = []
    trigger_steps: List[int] = []

    # 为了尽量复用项目生成行为，仍按 batch 遍历；检测在样本级进行。
    for i in tqdm(range(0, len(prompts), batch_size), desc='运行 probe 生成'):
        batch_prompts = prompts[i:i + batch_size]

        safe_prompts: List[str] = []
        safe_indices: List[int] = []
        batch_outputs: List[Optional[str]] = [None] * len(batch_prompts)
        batch_triggers: List[int] = [-1] * len(batch_prompts)

        for j, prompt in enumerate(batch_prompts):
            try:
                flagged = _prompt_is_harmful(model, tokenizer, prompt, probe)
            except Exception as e:
                logger.warning(f"探针检测失败，降级为正常生成。error={e}")
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
    主流程（与项目主评估脚本保持同风格）：
    A. 读取配置 + 训练 probe
    B. 运行 utility/safety 生成
    C. 对 safety 结果跑 harmfulness 分类器
    D. 保存总表、分数据集/分攻击方法明细与汇总
    """
    parser = argparse.ArgumentParser(description='运行 probe baseline 评估。')
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/baselines/probe/evaluation_config.yaml',
        help='配置文件路径（相对项目根目录）。',
    )
    args = parser.parse_args()

    config_path = BASE_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

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
        # Utility：按数据集逐个评估并保存到 all_results
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
        # Safety：按攻击方法逐个评估并保存到 all_results
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
        logger.warning('没有可保存的评估结果。')
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df['trigger_step'] = final_df['trigger_step'].apply(lambda x: pd.NA if x == -1 else x).astype('Int64')

    # 仅 safety 需要 harmfulness 分类
    mask_safety = final_df['eval_split'] == 'safety'
    final_df['label'] = pd.NA

    if mask_safety.any():
        # 仅 safety 样本需要 harmfulness 判别（与主评估脚本一致）
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

    # 详细结果总表
    all_csv = output_dir / f"{llm_name}_probe_evaluation_all_results.csv"
    final_df.to_csv(all_csv, index=False, encoding='utf-8-sig')
    logger.info(f"已保存详细结果: {all_csv}")

    # Utility: 每个数据集输出 CSV + summary JSON
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
            f"可用性 (FPR) for {dataset_name} - "
            f"probe: {metrics['rate']:.4f}, "
            f"TriggerRate: {metrics.get('trigger_rate', 0.0):.4f}"
        )

        if dataset_name == 'alpaca_eval':
            # 复用主脚本生态：导出 Alpaca Eval 标准格式
            save_alpaca_format(
                sub,
                output_dir / f"{llm_name}-alpaca_eval-probe.json",
                f"{llm_name}-probe",
            )

        if 'xstest' in str(dataset_name).lower():
            # 复用主脚本生态：导出 xstest 标准格式
            save_xstest_format(sub, output_dir / f"{llm_name}_xstest_probe.csv")

    # Safety: 每种攻击输出 CSV + summary JSON
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
            f"安全性 (ASR) for {method} - "
            f"probe: {metrics['rate']:.4f}, "
            f"TriggerRate: {metrics.get('trigger_rate', 0.0):.4f}"
        )

    logger.info('probe baseline 评估完成。')


if __name__ == '__main__':
    main()
