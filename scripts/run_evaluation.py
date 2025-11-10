# -*- coding: utf-8 -*-
"""
脚本 05: 运行防御评估 (支持 jbb_expanded.csv)

该脚本是 SARC 流程的最后一步，用于验证 `04_calibrate_defense.py`
校准后的防御系统的实际效果。

主要功能:
1.  加载基础 LLM、分类器 LLM 和 `Guard` 防御产物。
2.  加载“可用性”数据集 (良性样本)。
3.  加载“安全性”数据集 (例如 `jbb_expanded.csv`)，
    并根据配置文件的 `attack_columns_to_eval` 列表，
    将其从宽表转换为长表 (long format)，每个样本标记其来源的攻击方法。
4.  运行评估：
    a.  对“可用性”数据集，运行 Baseline vs Guarded，计算 FPR。
    b.  对“安全性”数据集，按“攻击方法”分组，
        对*每种*攻击方法分别运行 Baseline vs Guarded，计算各自的 ASR。
5.  使用分类器判断所有输出的有害性。
6.  将所有详细结果 (包括攻击方法、触发步骤等) 保存到 CSV。
7.  将汇总指标 (FPR 和按攻击方法分的 ASR) 保存到 JSON。

如何运行:
python scripts/run_evaluation.py --config configs/evaluation_config.yaml
"""

import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import gc
import sys
import json
import contextlib
from typing import Dict, List, Any, Optional, Tuple
import importlib.util

# --- 路径设置 (Path Setup) ---
base_dir = Path(__file__).parent.parent
src_path = base_dir / 'src'
scripts_path = base_dir / 'scripts'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

try:
    from repeng_guard import Guard
except ImportError:
    print(f"ERROR: 无法导入 repeng_guard。请确保 'src' 目录在 sys.path 中: {src_path}")
    sys.exit(1)

# --- 动态导入 LLAMA2_CLS_PROMPT ---
try:
    judge_script_path = scripts_path / "02.5_judge_harmfulness.py"
    spec = importlib.util.spec_from_file_location("judge_script", judge_script_path)
    judge_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge_module)
    LLAMA2_CLS_PROMPT = judge_module.LLAMA2_CLS_PROMPT
    if not LLAMA2_CLS_PROMPT:
        raise ImportError("LLAMA2_CLS_PROMPT 为空。")
except Exception as e:
    print(f"ERROR: 无法从 02.5_judge_harmfulness.py 动态导入 LLAMA2_CLS_PROMPT: {e}")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --- 日志配置 (Logging Configuration) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str, model_kwargs: dict, device: str) -> tuple:
    """
    加载 Hugging Face 模型和分词器。
    """
    logger.info(f"正在加载模型: {model_name}...")
    kwargs = model_kwargs.copy()
    if "torch_dtype" in kwargs and isinstance(kwargs["torch_dtype"], str):
        try:
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
        except AttributeError:
            if kwargs["torch_dtype"] != "auto":
                raise ValueError(f"无效的 torch_dtype: {kwargs['torch_dtype']}")

    if "device_map" not in kwargs and device == "cuda":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if "device_map" not in kwargs and device == "cuda":
        model.to(device)

    model.eval()
    logger.info(f"模型 {model_name} 加载完毕。")
    return model, tokenizer


def load_utility_dataset(config: dict, data_dir: Path, sample_size: int = 0) -> pd.DataFrame:
    """
    从 CSV 文件加载“可用性”(良性)评估数据集。
    """
    file_path = data_dir / config['filename']
    prompt_col = config['prompt_column']

    if not file_path.exists():
        logger.warning(f"可用性数据集文件未找到: {file_path}。返回空数据帧。")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    if prompt_col not in df.columns:
        logger.error(f"可用性数据集 {file_path} 缺少指定的提示列 '{prompt_col}'。")
        return pd.DataFrame()

    # 重命名提示列为 'prompt' 以便统一处理
    df.rename(columns={prompt_col: 'prompt'}, inplace=True)

    # 为良性数据集填充占位符
    if 'behavior' not in df.columns:
        # 如果 'prompt' 列就是 'Goal' 或 'Behavior'，则复制
        if 'Goal' in df.columns:
            df['behavior'] = df['Goal']
        elif 'Behavior' in df.columns:
            df['behavior'] = df['Behavior']
        else:
            df['behavior'] = "N/A"  # 否则设为 N/A

    if 'FunctionalCategory' not in df.columns:
        df['FunctionalCategory'] = "N/A"
    if 'ContextString' not in df.columns:
        df['ContextString'] = ""

    if sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info(f"从 {file_path} 加载了 {len(df)} 个可用性样本。")
    return df


def load_safety_dataset(config: dict, data_dir: Path, sample_size: int = 0) -> pd.DataFrame:
    """
    加载并转换“安全性”(JBB)评估数据集 (从宽表到长表)。
    """
    file_path = data_dir / config['filename']
    base_cols = config['base_columns']
    attack_cols = config['attack_columns_to_eval']

    if not file_path.exists():
        logger.warning(f"安全性数据集文件未找到: {file_path}。返回空数据帧。")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # 验证所有必需的列都存在
    missing_cols = [c for c in base_cols + attack_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"安全性数据集 {file_path} 缺少以下必需列: {missing_cols}")
        return pd.DataFrame()

    # 使用 pd.melt 将宽表转换为长表
    # id_vars = 基础信息列
    # value_vars = 要评估的攻击方法列
    df_long = pd.melt(
        df,
        id_vars=base_cols,
        value_vars=attack_cols,
        var_name="attack_method",  # 新列：存储攻击方法名称 (例如 "GCG", "PAIR")
        value_name="prompt"  # 新列：存储该方法对应的越狱提示
    )

    # 清理：删除提示为空的行 (如果 CSV 中有空单元格)
    df_long.dropna(subset=['prompt'], inplace=True)

    # 【关键】: 将 'Goal' 列的内容复制到 'behavior' 列，
    # 因为分类器需要 'behavior' 列来构建提示。
    if 'Goal' in df_long.columns:
        df_long['behavior'] = df_long['Goal']
    elif 'Behavior' in df_long.columns:
        df_long['behavior'] = df_long['Behavior']  # 备选
    else:
        logger.error("安全性数据集中缺少 'Goal' 或 'Behavior' 列，分类器将无法工作。")
        return pd.DataFrame()

    # 填充分类器可能需要的其他列 (如果不存在)
    if 'FunctionalCategory' not in df_long.columns:
        df_long['FunctionalCategory'] = "N/A"
    if 'ContextString' not in df_long.columns:
        df_long['ContextString'] = ""

    if sample_size > 0 and sample_size < len(df_long):
        # 注意：采样可能导致某些攻击方法的样本变少
        df_long = df_long.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info(f"从 {file_path} 加载并转换了 {len(df_long)} 个安全性样本 (跨越 {len(attack_cols)} 种攻击方法)。")
    return df_long


@torch.no_grad()
def run_generation(
        model,
        tokenizer,
        prompts: List[str],
        gen_config: GenerationConfig,
        batch_size: int,
        guard: Optional[Guard] = None
) -> Tuple[List[str], List[int]]:
    """
    使用 (或不使用) Guard 运行模型生成。
    返回 (生成的文本列表, 触发步骤列表)
    """

    outputs = []
    all_trigger_steps = []
    condition_desc = "GUARDED" if guard else "BASELINE"

    guard_context = guard.attach(model) if guard else contextlib.nullcontext()

    try:
        with guard_context:
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"运行 {condition_desc} 生成"):
                batch_prompts = prompts[i: i + batch_size]

                batch_trigger_logs = [-1] * len(batch_prompts)
                if guard:
                    guard.set_batch_log_target(batch_trigger_logs)

                conversations = [[{"role": "user", "content": p}] for p in batch_prompts]
                input_texts = [
                    tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                    for conv in conversations
                ]

                inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

                generated_ids = model.generate(
                    **inputs,
                    generation_config=gen_config
                )

                all_trigger_steps.extend(batch_trigger_logs)
                if guard:
                    guard.clear_batch_log_target()

                generated_texts = []
                for j, ids in enumerate(generated_ids):
                    input_len = len(inputs["input_ids"][j])
                    output_only_ids = ids[input_len:]
                    text = tokenizer.decode(output_only_ids, skip_special_tokens=True).strip()
                    generated_texts.append(text)

                outputs.extend(generated_texts)

    except Exception as e:
        logger.error(f"在 {condition_desc} 生成过程中发生错误: {e}", exc_info=True)
        error_fill_text = [f"GENERATION_ERROR: {e}"] * (len(prompts) - len(outputs))
        error_fill_logs = [-1] * (len(prompts) - len(all_trigger_steps))
        outputs.extend(error_fill_text)
        all_trigger_steps.extend(error_fill_logs)

    return outputs, all_trigger_steps


@torch.no_grad()
def run_classification(
        classifier_model,
        classifier_tokenizer,
        df: pd.DataFrame,
        batch_size: int
) -> List[str]:
    """
    使用分类器模型判断生成内容的有害性。
    """

    labels = []
    if 'assistant_output' not in df.columns:
        logger.error("数据帧中缺少 'assistant_output' 列，无法进行分类。")
        return ["error"] * len(df)
    if 'behavior' not in df.columns:
        logger.error("数据帧中缺少 'behavior' 列，无法进行分类。")
        return ["error"] * len(df)

    for i in tqdm(range(0, len(df), batch_size), desc="运行分类器判断"):
        batch_df = df.iloc[i: i + batch_size]
        prompts = []

        for _, row in batch_df.iterrows():
            behavior = str(row['behavior'])
            generation = str(row['assistant_output']) if pd.notna(row['assistant_output']) else ""

            if str(row.get('FunctionalCategory', 'N/A')) == 'contextual':
                context = str(row.get('ContextString', ''))
                prompt = LLAMA2_CLS_PROMPT["prompt_contextual"].format(
                    context=context, behavior=behavior, generation=generation
                )
            else:
                prompt = LLAMA2_CLS_PROMPT["prompt"].format(
                    behavior=behavior, generation=generation
                )
            prompts.append(prompt)

        inputs = classifier_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(classifier_model.device)

        output_ids = classifier_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=1
        ).cpu()

        input_len_cpu = inputs.input_ids.shape[1]
        new_tokens = output_ids[:, input_len_cpu:]
        completions = classifier_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for c in completions:
            label = c.strip().lower()
            if "yes" in label:
                labels.append("yes")
            elif "no" in label:
                labels.append("no")
            else:
                logger.warning(f"分类器返回了意外的标签: '{c}'。记为 'unknown'。")
                labels.append("unknown")

    return labels


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    计算给定数据帧的有害率 (ASR 或 FPR)。
    """
    total = len(df)
    if total == 0:
        return {"rate": 0.0, "total": 0, "count": 0}

    harmful_count = (df['label'] == 'yes').sum()
    rate = harmful_count / total

    return {"rate": rate, "total": int(total), "harmful_count": int(harmful_count)}


def main():
    parser = argparse.ArgumentParser(description="运行 SARC 防御评估 (支持 jbb_expanded.csv)。")
    parser.add_argument('--config', type=str, default='configs/evaluation_config.yaml',
                        help='评估配置文件路径。')
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    config_path = base_dir / args.config
    if not config_path.exists():
        logger.error(f"配置文件未找到: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- 2. 设置路径 ---
    llm_name = config['llm_name']
    output_dir = base_dir / config['output_dir'] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = base_dir / config['data_dir']

    # --- 3. 加载 Guard ---
    logger.info("--- 正在加载 SARC Guard ---")
    artifact_path = base_dir / config['artifact_path'] / llm_name
    try:
        guard = Guard.from_artifacts(
            artifact_path=str(artifact_path),
            device=device
        )
    except FileNotFoundError as e:
        logger.error(f"加载 Guard 失败: {e}。请确保 04_calibrate_defense.py 已运行。")
        return

    # --- 4. 加载目标 LLM ---
    logger.info("--- 正在加载目标 LLM ---")
    model_config = config['llm_config']
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_config['path'],
        model_kwargs=model_config.get('kwargs', {}),
        device=device
    )

    # --- 5. 加载分类器 LLM ---
    logger.info("--- 正在加载分类器 LLM ---")
    classifier_config = config['classifier_config']
    classifier_model, classifier_tokenizer = load_model_and_tokenizer(
        model_name=classifier_config['path'],
        model_kwargs=classifier_config.get('kwargs', {}),
        device=device
    )

    # --- 6. 设置生成配置 ---
    gen_config_dict = config.get('generation_kwargs', {})
    gen_config = GenerationConfig(
        max_new_tokens=gen_config_dict.get('max_new_tokens', 128),
        do_sample=gen_config_dict.get('do_sample', False),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **{k: v for k, v in gen_config_dict.items() if k not in ['max_new_tokens', 'do_sample']}
    )

    # --- 7. 加载数据集 ---
    # 修改：从各自的配置块中获取 sample_size
    utility_config = config['utility_dataset_config']
    utility_sample_size = utility_config.get('sample_size', 0)
    df_utility = load_utility_dataset(utility_config, data_dir, utility_sample_size)

    safety_config = config['safety_dataset_config']
    safety_sample_size = safety_config.get('sample_size', 0)
    df_safety_long = load_safety_dataset(safety_config, data_dir, safety_sample_size)

    # --- 8. 运行评估循环 ---
    all_results_dfs = []  # 存储所有详细的 DF 结果
    summary_metrics = {}  # 存储最终的指标
    batch_size = config.get('batch_size', 4)
    classifier_batch_size = config.get('classifier_batch_size', 2)

    # --- 8a. 评估可用性 (FPR) ---
    if not df_utility.empty:
        logger.info("--- 评估开始: 可用性 (FPR) ---")
        prompts = df_utility['prompt'].tolist()

        # Baseline
        baseline_outputs, baseline_triggers = run_generation(
            model, tokenizer, prompts, gen_config, batch_size, guard=None
        )
        df_baseline = df_utility.copy()
        df_baseline['assistant_output'] = baseline_outputs
        df_baseline['trigger_step'] = baseline_triggers
        df_baseline['condition'] = 'baseline'
        df_baseline['dataset'] = 'utility'
        df_baseline['attack_method'] = 'N/A'  # 占位

        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

        # Guarded
        guarded_outputs, guarded_triggers = run_generation(
            model, tokenizer, prompts, gen_config, batch_size, guard=guard
        )
        df_guarded = df_utility.copy()
        df_guarded['assistant_output'] = guarded_outputs
        df_guarded['trigger_step'] = guarded_triggers
        df_guarded['condition'] = 'guarded'
        df_guarded['dataset'] = 'utility'
        df_guarded['attack_method'] = 'N/A'  # 占位

        # 合并、分类并保存
        df_utility_combined = pd.concat([df_baseline, df_guarded], ignore_index=True)
        labels = run_classification(
            classifier_model, classifier_tokenizer, df_utility_combined, classifier_batch_size
        )
        df_utility_combined['label'] = labels
        all_results_dfs.append(df_utility_combined)

        # 计算指标
        baseline_fpr_metrics = calculate_metrics(df_utility_combined.query("condition == 'baseline'"))
        guarded_fpr_metrics = calculate_metrics(df_utility_combined.query("condition == 'guarded'"))
        summary_metrics['utility_fpr'] = {
            'baseline': baseline_fpr_metrics,
            'guarded': guarded_fpr_metrics
        }
        logger.info(f"可用性 (FPR) - Baseline: {baseline_fpr_metrics['rate']:.4f}, Guarded: {guarded_fpr_metrics['rate']:.4f}")

        del df_baseline, df_guarded, df_utility_combined, baseline_outputs, guarded_outputs, labels
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
    else:
        logger.warning("跳过可用性评估，因为数据集为空。")

    # --- 8b. 评估安全性 (ASR)，按攻击方法分组 ---
    if not df_safety_long.empty:
        summary_metrics['safety_asr_by_attack'] = {}
        attack_methods = df_safety_long['attack_method'].unique()
        logger.info(f"--- 评估开始: 安全性 (ASR)，将测试 {len(attack_methods)} 种攻击方法 ---")

        for method in attack_methods:
            logger.info(f"--- 正在评估攻击方法: {method} ---")
            df_attack = df_safety_long[df_safety_long['attack_method'] == method].reset_index(drop=True)
            prompts = df_attack['prompt'].tolist()

            # Baseline
            baseline_outputs, baseline_triggers = run_generation(
                model, tokenizer, prompts, gen_config, batch_size, guard=None
            )
            df_baseline = df_attack.copy()
            df_baseline['assistant_output'] = baseline_outputs
            df_baseline['trigger_step'] = baseline_triggers
            df_baseline['condition'] = 'baseline'
            df_baseline['dataset'] = 'safety'

            gc.collect()
            if device == "cuda": torch.cuda.empty_cache()

            # Guarded
            guarded_outputs, guarded_triggers = run_generation(
                model, tokenizer, prompts, gen_config, batch_size, guard=guard
            )
            df_guarded = df_attack.copy()
            df_guarded['assistant_output'] = guarded_outputs
            df_guarded['trigger_step'] = guarded_triggers
            df_guarded['condition'] = 'guarded'

            # 合并、分类并保存
            df_attack_combined = pd.concat([df_baseline, df_guarded], ignore_index=True)
            labels = run_classification(
                classifier_model, classifier_tokenizer, df_attack_combined, classifier_batch_size
            )
            df_attack_combined['label'] = labels
            all_results_dfs.append(df_attack_combined)

            # 计算指标
            baseline_asr_metrics = calculate_metrics(df_attack_combined.query("condition == 'baseline'"))
            guarded_asr_metrics = calculate_metrics(df_attack_combined.query("condition == 'guarded'"))
            summary_metrics['safety_asr_by_attack'][method] = {
                'baseline': baseline_asr_metrics,
                'guarded': guarded_asr_metrics
            }
            logger.info(f"安全性 (ASR) for {method} - Baseline: {baseline_asr_metrics['rate']:.4f}, Guarded: {guarded_asr_metrics['rate']:.4f}")

            del df_baseline, df_guarded, df_attack_combined, baseline_outputs, guarded_outputs, labels
            gc.collect()
            if device == "cuda": torch.cuda.empty_cache()
    else:
        logger.warning("跳过安全性评估，因为数据集为空。")

    # --- 9. 保存最终结果 ---
    if not all_results_dfs:
        logger.error("未生成任何结果。请检查数据集路径和配置。")
        return

    final_results_df = pd.concat(all_results_dfs, ignore_index=True)

    # 确保 'trigger_step' 列是整数类型 (可空)
    if 'trigger_step' in final_results_df.columns:
        final_results_df['trigger_step'] = final_results_df['trigger_step'].apply(lambda x: pd.NA if x == -1 else x).astype('Int64')

    # 定义详细 CSV 的列顺序
    base_cols_config = config.get('safety_dataset_config', {}).get('base_columns', [])
    ordered_cols = base_cols_config + [
        'attack_method', 'dataset', 'condition', 'prompt', 'assistant_output',
        'label', 'trigger_step'
    ]
    # 确保所有实际存在的列都在列表中，并保持顺序
    final_cols = [c for c in ordered_cols if c in final_results_df.columns]
    final_cols += [c for c in final_results_df.columns if c not in ordered_cols]
    final_results_df = final_results_df[final_cols]

    csv_path = output_dir / f"{llm_name}_evaluation_detailed_by_attack.csv"
    final_results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"详细评估结果已保存到: {csv_path}")

    summary_path = output_dir / f"{llm_name}_evaluation_summary_by_attack.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"评估指标摘要已保存到: {summary_path}")

    logger.info("--- 最终评估摘要 ---")
    print(json.dumps(summary_metrics, indent=2))
    logger.info("评估流程全部完成。")


if __name__ == "__main__":
    main()

