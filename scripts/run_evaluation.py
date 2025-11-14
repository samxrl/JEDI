# -*- coding: utf-8 -*-
"""
脚本 05: 运行防御评估 (支持 jbb_expanded.csv)

该脚本是 SARC 流程的最后一步，用于验证 `04_calibrate_defense.py`
校准后的防御系统的实际效果。

** [!] 此版本已根据内存优化请求进行修改 **
流程被分为三个阶段，以确保被测 LLM 和分类器 LLM 不会同时占用显存：
1.  **阶段 1 (生成)**: 加载被测 LLM，运行所有生成 (baseline + guarded)，
    保存结果，然后释放被测 LLM。
2.  **阶段 2 (分类)**: 加载分类器 LLM，对所有生成结果进行判断，
    保存标签，然后释放分类器 LLM。
3.  **阶段 3 (报告)**: 计算所有指标并保存到文件。

** [!] 此版本已修改，支持加载和评估多个良性数据集。 **
1.  `load_utility_dataset` 现在从配置中加载一个数据集列表。
2.  实现了基于配额的等额采样逻辑。
3.  评估和保存阶段现在会为每个良性数据集分别生成报告。
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


def load_utility_dataset(config: dict, data_dir: Path, total_sample_size: int = 0) -> pd.DataFrame:
    """
    [!! 已修改 !!]
    从 CSV 文件加载一个或多个“可用性”(良性)评估数据集。
    支持基于配额的等额采样。
    """
    logger.info("正在加载可用性 (Utility) 数据集...")
    dataset_list = config.get('datasets')
    if not dataset_list or not isinstance(dataset_list, list):
        logger.warning("在配置中未找到 'utility_dataset_config.datasets' 列表。返回空数据帧。")
        return pd.DataFrame()

    all_loaded_dfs = []
    for dataset_config in dataset_list:
        name = dataset_config.get('name')
        filename = dataset_config.get('filename')
        prompt_col = dataset_config.get('prompt_column')

        if not name or not filename or not prompt_col:
            logger.warning(f"跳过一个无效的良性数据集条目 (缺少 name, filename, 或 prompt_column): {dataset_config}")
            continue

        file_path = data_dir / filename
        if not file_path.exists():
            logger.warning(f"可用性数据集文件未找到: {file_path}。跳过。")
            continue

        try:
            df = pd.read_csv(file_path)
            if prompt_col not in df.columns:
                logger.error(f"可用性数据集 {file_path} 缺少列 '{prompt_col}'。跳过。")
                continue

            df.rename(columns={prompt_col: 'prompt'}, inplace=True)
            # [!] 关键：添加数据集名称
            df['utility_dataset_name'] = name

            # 为良性数据集填充占位符
            if 'behavior' not in df.columns:
                if 'Goal' in df.columns:
                    df['behavior'] = df['Goal']
                elif 'Behavior' in df.columns:
                    df['behavior'] = df['Behavior']
                else:
                    df['behavior'] = "N/A"
            if 'FunctionalCategory' not in df.columns:
                df['FunctionalCategory'] = "N/A"
            if 'ContextString' not in df.columns:
                df['ContextString'] = ""
            df['attack_method'] = 'N/A'  # 占位

            all_loaded_dfs.append(df)
            logger.info(f"已从 {file_path} 加载 {len(df)} 个样本 (标记为 '{name}')。")

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}", exc_info=True)

    if not all_loaded_dfs:
        logger.warning("未成功加载任何可用性数据集。")
        return pd.DataFrame()

    # --- 采样逻辑 ---
    if total_sample_size <= 0:
        logger.info(f"sample_size <= 0, 正在合并所有 {len(all_loaded_dfs)} 个数据集 (总计 {sum(len(df) for df in all_loaded_dfs)} 个样本)。")
        return pd.concat(all_loaded_dfs, ignore_index=True)

    logger.info(f"正在从 {len(all_loaded_dfs)} 个数据集中进行等额采样，目标总共 {total_sample_size} 个样本...")

    num_datasets = len(all_loaded_dfs)
    base_quota = total_sample_size // num_datasets
    remainder = total_sample_size % num_datasets
    # 基础配额，并将余数分配给前几个数据集
    quotas = [base_quota + 1 if i < remainder else base_quota for i in range(num_datasets)]

    sampled_dfs = []
    total_shortfall = 0
    donors = []  # (index, extra_capacity)

    # 第一遍：尝试满足配额
    logger.debug(f"初始配额: {quotas}")
    for i, df in enumerate(all_loaded_dfs):
        quota = quotas[i]
        if len(df) >= quota:
            sampled_dfs.append(df.sample(n=quota, random_state=42))
            extra = len(df) - quota
            if extra > 0:
                donors.append((i, extra))
        else:  # 样本数不足
            sampled_dfs.append(df)  # 全部采样
            total_shortfall += (quota - len(df))
            logger.info(f"数据集 {i} 样本不足 (需要 {quota}, 只有 {len(df)})。缺口: {quota - len(df)}")

    # 第二遍：重新分配缺口
    if total_shortfall > 0 and donors:
        logger.info(f"出现 {total_shortfall} 个样本的缺口，正在从其他数据集中补充...")

        # 汇总所有可用的额外样本
        all_extra_samples = []
        for donor_idx, extra_cap in donors:
            donor_df = all_loaded_dfs[donor_idx]
            # 找到未被采样的索引
            sampled_indices = sampled_dfs[donor_idx].index
            available_indices = donor_df.index.difference(sampled_indices)
            # 添加可用的额外行
            all_extra_samples.append(donor_df.loc[available_indices])

        if all_extra_samples:
            # 合并所有可用的额外样本
            extra_df = pd.concat(all_extra_samples, ignore_index=True)
            # 从这个池中随机采样以弥补缺口
            take_n = min(total_shortfall, len(extra_df))
            if take_n > 0:
                final_extra_samples = extra_df.sample(n=take_n, random_state=42)
                sampled_dfs.append(final_extra_samples)
                logger.info(f"已成功补充 {take_n} 个额外样本。")

    final_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
    logger.info(f"成功加载并采样了 {len(final_df)} 个可用性样本。")
    return final_df


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

    # [!] 添加占位符，以便与良性数据集合并
    df_long['utility_dataset_name'] = 'N/A'

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


def calculate_metrics(df: pd.DataFrame, condition: str) -> Dict[str, Any]:
    """
    计算给定数据帧的有害率 (ASR 或 FPR)。
    如果 condition 为 'guarded'，还会计算触发率。
    """
    total = len(df)
    if total == 0:
        metrics = {"rate": 0.0, "total": 0, "harmful_count": 0}
        # [!] 只有 guarded 才添加触发字段
        if condition == 'guarded':
            metrics["trigger_rate"] = 0.0
            metrics["trigger_count"] = 0
        return metrics

    # 1. 计算有害率 (ASR / FPR)
    harmful_count = (df['label'] == 'yes').sum()
    rate = harmful_count / total if total > 0 else 0.0

    metrics = {
        "rate": rate,
        "total": int(total),
        "harmful_count": int(harmful_count),
    }

    # 2. (仅)为 guarded 计算触发率
    if condition == 'guarded':
        trigger_count = 0
        trigger_rate = 0.0
        if 'trigger_step' in df.columns:
            # .notna() 会捕获所有非 <NA> 的值 (即 trigger_step >= 0)
            # 因为我们在阶段 1 中已经将 -1 转换为了 <NA>
            trigger_count = (df['trigger_step'].notna()).sum()
            trigger_rate = trigger_count / total if total > 0 else 0.0

        metrics["trigger_rate"] = trigger_rate
        metrics["trigger_count"] = int(trigger_count)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="运行 SARC 防御评估 (支持 jbb_expanded.csv 和多个良性数据集)。")
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

    # --- 3. 设置生成配置 ---
    gen_config_dict = config.get('generation_kwargs', {})
    gen_config = GenerationConfig(
        max_new_tokens=gen_config_dict.get('max_new_tokens', 128),
        do_sample=gen_config_dict.get('do_sample', False),
        **{k: v for k, v in gen_config_dict.items() if k not in ['max_new_tokens', 'do_sample']}
    )
    batch_size = config.get('batch_size', 4)

    all_results_dfs = []  # 存储所有详细的 DF 结果

    # ---
    # --- 阶段 1: 生成
    # ---
    logger.info("--- 阶段 1: 开始文本生成 ---")
    try:
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
        # 确保 gen_config 包含 pad_token_id (在模型加载后设置)
        gen_config.pad_token_id = tokenizer.pad_token_id
        gen_config.eos_token_id = tokenizer.eos_token_id

        # --- 7. 加载数据集 ---
        # [!] 修改：加载一个或多个良性数据集
        utility_config = config['utility_dataset_config']
        utility_sample_size = utility_config.get('sample_size', 0)  # 这是总采样大小
        df_utility_all = load_utility_dataset(utility_config, data_dir, utility_sample_size)

        safety_config = config['safety_dataset_config']
        safety_sample_size = safety_config.get('sample_size', 0)
        df_safety_long = load_safety_dataset(safety_config, data_dir, safety_sample_size)

        # --- 8a. 评估可用性 (FPR) ---
        # [!] 修改：按良性数据集名称循环
        if not df_utility_all.empty:
            utility_dataset_names = df_utility_all['utility_dataset_name'].unique()
            logger.info(f"--- 评估开始: 可用性 (FPR)，将测试 {len(utility_dataset_names)} 个良性数据集 ---")

            for dataset_name in utility_dataset_names:
                logger.info(f"--- 正在评估良性数据集: {dataset_name} ---")
                df_utility_subset = df_utility_all[df_utility_all['utility_dataset_name'] == dataset_name].reset_index(drop=True)
                prompts = df_utility_subset['prompt'].tolist()

                if not prompts:
                    logger.warning(f"良性数据集 {dataset_name} 没有可运行的提示，跳过。")
                    continue

                # Baseline
                baseline_outputs, baseline_triggers = run_generation(
                    model, tokenizer, prompts, gen_config, batch_size, guard=None
                )
                df_baseline = df_utility_subset.copy()
                df_baseline['assistant_output'] = baseline_outputs
                df_baseline['trigger_step'] = baseline_triggers
                df_baseline['condition'] = 'baseline'
                df_baseline['dataset'] = 'utility'
                # 'attack_method' 和 'utility_dataset_name' 已在加载时设置
                all_results_dfs.append(df_baseline)

                del baseline_outputs, baseline_triggers, df_baseline
                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()

                # Guarded
                guarded_outputs, guarded_triggers = run_generation(
                    model, tokenizer, prompts, gen_config, batch_size, guard=guard
                )
                df_guarded = df_utility_subset.copy()
                df_guarded['assistant_output'] = guarded_outputs
                df_guarded['trigger_step'] = guarded_triggers
                df_guarded['condition'] = 'guarded'
                df_guarded['dataset'] = 'utility'
                all_results_dfs.append(df_guarded)

                del guarded_outputs, guarded_triggers, df_guarded
                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()
        else:
            logger.warning("跳过可用性评估，因为数据集为空。")

        # --- 8b. 评估安全性 (ASR)，按攻击方法分组 ---
        if not df_safety_long.empty:
            attack_methods = df_safety_long['attack_method'].unique()
            logger.info(f"--- 评估开始: 安全性 (ASR)，将测试 {len(attack_methods)} 种攻击方法 ---")

            for method in attack_methods:
                logger.info(f"--- 正在评估攻击方法: {method} ---")
                df_attack = df_safety_long[df_safety_long['attack_method'] == method].reset_index(drop=True)
                prompts = df_attack['prompt'].tolist()

                if not prompts:
                    logger.warning(f"方法 {method} 没有可运行的提示，跳过。")
                    continue

                # Baseline
                baseline_outputs, baseline_triggers = run_generation(
                    model, tokenizer, prompts, gen_config, batch_size, guard=None
                )
                df_baseline = df_attack.copy()
                df_baseline['assistant_output'] = baseline_outputs
                df_baseline['trigger_step'] = baseline_triggers
                df_baseline['condition'] = 'baseline'
                df_baseline['dataset'] = 'safety'
                all_results_dfs.append(df_baseline)

                del df_baseline, baseline_outputs, baseline_triggers
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
                df_guarded['dataset'] = 'safety'
                all_results_dfs.append(df_guarded)

                del df_guarded, guarded_outputs, guarded_triggers
                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()
        else:
            logger.warning("跳过安全性评估，因为数据集为空。")

    finally:
        # --- 关键步骤: 释放被测 LLM ---
        if 'model' in locals():
            model_name = model.config.name_or_path if hasattr(model, 'config') and hasattr(model.config, 'name_or_path') else "被测 LLM"
            logger.info(f"正在从内存中释放模型: {model_name}...")
            del model
            if 'tokenizer' in locals():
                del tokenizer
            if 'guard' in locals():
                del guard
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("模型内存已释放。")

    logger.info("--- 阶段 1: 文本生成全部完成 ---")

    if not all_results_dfs:
        logger.error("未生成任何结果。请检查数据集路径和配置。")
        return

    # ---
    # --- 阶段 2: 分类
    # ---
    logger.info("--- 阶段 2: 开始有害性判断 ---")

    final_results_df = pd.concat(all_results_dfs, ignore_index=True)
    del all_results_dfs  # 释放列表内存

    interim_csv_path = output_dir / f"{llm_name}_evaluation_INTERIM_all_results.csv"

    # --- 在保存中间文件和最终文件之前，格式化 trigger_step ---
    if 'trigger_step' in final_results_df.columns:
        final_results_df['trigger_step'] = final_results_df['trigger_step'].apply(lambda x: pd.NA if x == -1 else x).astype('Int64')

    # --- 整理列并保存 *第一次* (无 label) 到 *临时* 路径 ---
    try:
        logger.info(f"正在保存中间生成结果 (无标签) 到: {interim_csv_path}")

        base_cols_config = config.get('safety_dataset_config', {}).get('base_columns', [])
        ordered_cols = base_cols_config + [
            'attack_method', 'utility_dataset_name', 'dataset', 'condition', 'prompt', 'assistant_output', 'trigger_step'
        ]
        current_cols = final_results_df.columns
        final_ordered_cols = [c for c in ordered_cols if c in current_cols]
        extra_cols = [c for c in current_cols if c not in final_ordered_cols]
        final_ordered_cols.extend(extra_cols)

        if 'label' not in final_ordered_cols:
            final_ordered_cols.append('label')
            final_results_df['label'] = pd.NA

        final_results_df = final_results_df[final_ordered_cols]
        final_results_df.to_csv(interim_csv_path, index=False, encoding='utf-8-sig')
        logger.info("中间生成结果CSV 保存成功。")
    except Exception as e:
        logger.warning(f"保存 CSV 文件失败: {e}", exc_info=True)

    try:
        # --- 5. 加载分类器 LLM ---
        logger.info("--- 正在加载分类器 LLM ---")
        classifier_config = config['classifier_config']
        classifier_batch_size = config.get('classifier_batch_size', 2)
        classifier_model, classifier_tokenizer = load_model_and_tokenizer(
            model_name=classifier_config['path'],
            model_kwargs=classifier_config.get('kwargs', {}),
            device=device
        )

        # --- 运行分类 ---
        labels = run_classification(
            classifier_model, classifier_tokenizer, final_results_df, classifier_batch_size
        )
        final_results_df['label'] = labels

    finally:
        # --- 关键步骤: 释放分类器 LLM ---
        if 'classifier_model' in locals():
            model_name = classifier_model.config.name_or_path if hasattr(classifier_model, 'config') and hasattr(classifier_model.config,
                                                                                                                 'name_or_path') else "分类器 LLM"
            logger.info(f"正在从内存中释放模型: {model_name}...")
            del classifier_model
            if 'classifier_tokenizer' in locals():
                del classifier_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("模型内存已释放。")

    logger.info("--- 阶段 3: 计算指标并保存拆分报告 ---")

    # --- [!] 修改：计算并保存可用性指标 (FPR)，按数据集名称循环 ---
    df_utility_results = final_results_df[final_results_df['dataset'] == 'utility']
    if not df_utility_results.empty:
        # 从结果中获取所有唯一的良性数据集名称
        utility_dataset_names = df_utility_results['utility_dataset_name'].unique()
        logger.info(f"正在为 {len(utility_dataset_names)} 个良性数据集计算 FPR 指标...")

        for dataset_name in utility_dataset_names:
            if pd.isna(dataset_name) or dataset_name == 'N/A': continue

            # 筛选出当前循环的良性数据集
            df_utility_subset = df_utility_results[df_utility_results['utility_dataset_name'] == dataset_name]
            if df_utility_subset.empty: continue

            # 分别计算 baseline 和 guarded 的指标
            baseline_fpr_metrics = calculate_metrics(df_utility_subset.query("condition == 'baseline'"), condition='baseline')
            guarded_fpr_metrics = calculate_metrics(df_utility_subset.query("condition == 'guarded'"), condition='guarded')

            # [!] 为这个特定的数据集创建独立的 summary
            utility_summary = {
                f'utility_fpr_{dataset_name}': {
                    'baseline': baseline_fpr_metrics,
                    'guarded': guarded_fpr_metrics
                }
            }
            logger.info(f"可用性 (FPR) for {dataset_name} - Baseline: {baseline_fpr_metrics['rate']:.4f}, Guarded: {guarded_fpr_metrics['rate']:.4f}")

            # [!] 为这个特定的数据集保存独立的文件
            try:
                utility_csv_path = output_dir / f"{llm_name}_evaluation_detailed_utility_{dataset_name}.csv"
                df_utility_subset.to_csv(utility_csv_path, index=False, encoding='utf-8-sig')

                utility_json_path = output_dir / f"{llm_name}_evaluation_summary_utility_{dataset_name}.json"
                with open(utility_json_path, 'w', encoding='utf-8') as f:
                    json.dump(utility_summary, f, indent=2, default=str)

                logger.info(f"已保存可用性 (Utility) '{dataset_name}' 结果到: {utility_csv_path.name} 和 {utility_json_path.name}")
            except Exception as e:
                logger.error(f"保存可用性 (Utility) '{dataset_name}' 结果文件时失败: {e}", exc_info=True)
    else:
        logger.info("未找到可用性结果，跳过 FPR 计算和保存。")

    # --- 计算并保存安全性指标 (ASR) ---
    df_safety_results = final_results_df[final_results_df['dataset'] == 'safety']
    if not df_safety_results.empty:
        attack_methods = df_safety_results['attack_method'].unique()
        logger.info(f"正在为 {len(attack_methods)} 种攻击方法计算 ASR 指标...")

        for method in attack_methods:
            if pd.isna(method) or method == 'N/A': continue

            df_attack = df_safety_results[df_safety_results['attack_method'] == method]
            if df_attack.empty: continue

            baseline_asr_metrics = calculate_metrics(df_attack.query("condition == 'baseline'"), condition='baseline')
            guarded_asr_metrics = calculate_metrics(df_attack.query("condition == 'guarded'"), condition='guarded')

            attack_summary = {
                f'safety_asr_attack_{method}': {
                    'baseline': baseline_asr_metrics,
                    'guarded': guarded_asr_metrics
                }
            }
            logger.info(f"安全性 (ASR) for {method} - Baseline: {baseline_asr_metrics['rate']:.4f}, Guarded: {guarded_asr_metrics['rate']:.4f}")

            try:
                attack_csv_path = output_dir / f"{llm_name}_evaluation_detailed_attack_{method}.csv"
                df_attack.to_csv(attack_csv_path, index=False, encoding='utf-8-sig')

                attack_json_path = output_dir / f"{llm_name}_evaluation_summary_attack_{method}.json"
                with open(attack_json_path, 'w', encoding='utf-8') as f:
                    json.dump(attack_summary, f, indent=2, default=str)

                logger.info(f"已保存攻击方法 '{method}' 结果到: {attack_csv_path.name} 和 {attack_json_path.name}")
            except Exception as e:
                logger.error(f"保存攻击方法 '{method}' 结果文件时失败: {e}", exc_info=True)
    else:
        logger.info("未找到安全性结果，跳过 ASR 计算和保存。")

    logger.info("评估流程全部完成。")

    # --- [!] 新增: 自动删除临时文件 ---
    try:
        if 'interim_csv_path' in locals() and interim_csv_path.exists():
            interim_csv_path.unlink()
            logger.info(f"已清理临时文件: {interim_csv_path.name}")
    except Exception as e:
        logger.warning(f"清理临时文件 {interim_csv_path.name} 时失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()