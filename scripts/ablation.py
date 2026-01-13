# -*- coding: utf-8 -*-
"""
脚本 05: 运行防御评估 (支持 jbb_expanded.csv, alpaca_eval.json 和 xstest_prompts.csv)

该脚本是 JEDI 流程的最后一步，用于验证 `04_calibrate_defense.py`
校准后的防御系统的实际效果。

** [!] 此版本已根据内存优化请求进行修改 **
流程被分为三个阶段，以确保被测 LLM 和分类器 LLM 不会同时占用显存：
1.  **阶段 1 (生成)**: 加载被测 LLM，运行所有生成 (baseline + guarded)，
    保存结果，然后释放被测 LLM。
2.  **阶段 2 (分类)**: 加载分类器 LLM，对所有生成结果进行判断，
    保存标签，然后释放分类器 LLM。
3.  **阶段 3 (报告)**: 计算所有指标并保存到文件。

** [!] 此版本已修改，支持加载和评估多个良性数据集。 **
1.  `load_utility_dataset` 现在从配置中加载一个数据集列表，支持 CSV 和 JSON。
2.  实现了基于配额的等额采样逻辑。
3.  评估和保存阶段现在会为每个良性数据集分别生成报告。
4.  **新增**: 特别支持 `alpaca_eval` 格式输出。
5.  **新增**: 特别支持 `xstest` (xstest_prompts.csv) 格式输出 (8列 CSV)。

** [!] 此版本已修改，支持通过配置控制是否运行可用性/安全性评测。 **
1.  新增 `run_utility_evaluation` 和 `run_safety_evaluation` 配置项。
2.  根据配置项条件性地加载数据集和执行评测。

** [!] 修改：在测试 utility 数据集时，添加了耗时统计。 **
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
import time  # [!] 新增：导入 time 模块用于计时
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
    from JEDI_guard import Guard
except ImportError:
    print(f"ERROR: 无法导入 JEDI_guard。请确保 'src' 目录在 sys.path 中: {src_path}")
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
    从 CSV 或 JSON 文件加载一个或多个“可用性”(良性)评估数据集。
    支持基于配额的等额采样。
    支持 alpaca_eval.json 格式。
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
        # prompt_column 对于 json 可能是 instruction，对于 csv 可能是列名
        prompt_col = dataset_config.get('prompt_column', 'prompt')
        max_new_tokens = dataset_config.get('max_new_tokens')

        if not name or not filename:
            logger.warning(f"跳过一个无效的良性数据集条目 (缺少 name 或 filename): {dataset_config}")
            continue

        file_path = data_dir / filename
        if not file_path.exists():
            logger.warning(f"可用性数据集文件未找到: {file_path}。跳过。")
            continue

        try:
            # [!] 新增：支持 JSON 格式 (特别是 alpaca_eval)
            if filename.lower().endswith('.json'):
                logger.info(f"检测到 JSON 文件: {filename}，正在加载...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)

                # 如果是 alpaca_eval 常见格式，instruction 实际上是 prompt
                if 'instruction' in df.columns and 'prompt' not in df.columns:
                    logger.info("检测到 'instruction' 列，将其映射为 'prompt'。")
                    df['prompt'] = df['instruction']
                elif prompt_col in df.columns and prompt_col != 'prompt':
                    df.rename(columns={prompt_col: 'prompt'}, inplace=True)

                # 确保 alpaca_eval 所需的字段存在（如果原文件有，pd.DataFrame会自动保留）
                if 'dataset' not in df.columns:
                    df['dataset'] = name  # 使用配置名称作为 dataset 字段的默认值

            else:
                # CSV 格式处理 (包括 XSTest, OR-Bench 等)
                # 注意：pd.read_csv 会保留所有列，包括 xstest 的 'id' 和 'type'
                df = pd.read_csv(file_path)
                if prompt_col not in df.columns:
                    logger.error(f"可用性数据集 {file_path} 缺少列 '{prompt_col}'。跳过。")
                    continue
                df.rename(columns={prompt_col: 'prompt'}, inplace=True)

            # [!] 关键：添加数据集名称
            df['utility_dataset_name'] = name
            df['max_new_tokens'] = max_new_tokens

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
    # [!] 修改：预先检查攻击列配置
    # 如果 attack_columns_to_eval 为 None 或空，直接跳过加载，防止后续报错
    attack_cols = config.get('attack_columns_to_eval')
    if not attack_cols:
        logger.info("配置中 'attack_columns_to_eval' 为空或 None，跳过安全性数据集加载。")
        return pd.DataFrame()

    file_path = data_dir / config['filename']
    base_cols = config['base_columns']
    max_new_tokens = config.get('max_new_tokens')
    # attack_cols = config['attack_columns_to_eval'] # 上方已获取

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
    df_long['max_new_tokens'] = max_new_tokens

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
    # [!] 注意：如果未运行分类，label 列将为 pd.NA 或 'skipped'
    # 这种情况下，harmful_count 将为 0，rate 为 0.0
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


def get_override_max_new_tokens(df: pd.DataFrame) -> Optional[int]:
    """
    从数据集中解析 max_new_tokens 覆盖值。
    """
    if 'max_new_tokens' not in df.columns:
        return None
    values = df['max_new_tokens'].dropna().unique()
    if len(values) == 0:
        return None
    if len(values) > 1:
        logger.warning(f"检测到多个 max_new_tokens 值: {values}。将使用第一个值。")
    try:
        return int(values[0])
    except (TypeError, ValueError):
        logger.warning(f"max_new_tokens 值无效: {values[0]}，将使用默认配置。")
        return None


def build_generation_config(base_config: GenerationConfig, max_new_tokens_override: Optional[int]) -> GenerationConfig:
    """
    基于全局 GenerationConfig 构建带有可选 max_new_tokens 覆盖的配置。
    """
    if max_new_tokens_override is None:
        return base_config
    config_dict = base_config.to_dict()
    config_dict["max_new_tokens"] = max_new_tokens_override
    return GenerationConfig(**config_dict)


def format_ablation_tag(prefix: str, value: float) -> str:
    """
    将消融实验参数格式化为安全的目录/文件标签。
    """
    formatted = f"{value:g}"
    return f"{prefix}_{formatted.replace('.', 'p')}"


def normalize_ablation_list(name: str, values: Any) -> List[float]:
    """
    规范化消融实验参数列表为浮点数列表。
    """
    if values is None:
        logger.warning(f"未在配置中找到 {name}，将跳过该消融实验列表。")
        return []
    if not isinstance(values, list):
        logger.warning(f"{name} 应为列表，但收到: {type(values)}。将跳过该消融实验列表。")
        return []
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        logger.warning(f"{name} 列表包含无法转换为浮点数的值: {values}。将跳过该消融实验列表。")
        return []


def run_full_evaluation(
    guard: Guard,
    model,
    tokenizer,
    df_utility_all: pd.DataFrame,
    df_safety_long: pd.DataFrame,
    output_dir: Path,
    config: dict,
    gen_config: GenerationConfig,
    batch_size: int,
    run_utility: bool,
    run_safety: bool,
    device: str,
    llm_name: str,
    current_alpha: float,
    current_beta: float
) -> None:
    """
    运行一次完整的评测流程，并将结果保存到指定目录。
    """
    all_results_dfs = []  # 存储所有详细的 DF 结果

    # ---
    # --- 阶段 1: 生成
    # ---
    logger.info("--- 阶段 1: 开始文本生成 ---")
    # --- 8a. 评估可用性 (FPR) ---
    if not df_utility_all.empty:
        utility_dataset_names = df_utility_all['utility_dataset_name'].unique()
        logger.info(f"--- 评估开始: 可用性 (FPR)，将测试 {len(utility_dataset_names)} 个良性数据集 ---")

        for dataset_name in utility_dataset_names:
            logger.info(f"--- 正在评估良性数据集: {dataset_name} ---")
            df_utility_subset = df_utility_all[df_utility_all['utility_dataset_name'] == dataset_name].reset_index(drop=True)
            prompts = df_utility_subset['prompt'].tolist()
            num_samples = len(prompts)
            dataset_max_new_tokens = get_override_max_new_tokens(df_utility_subset)
            dataset_gen_config = build_generation_config(gen_config, dataset_max_new_tokens)

            if not prompts:
                logger.warning(f"良性数据集 {dataset_name} 没有可运行的提示，跳过。")
                continue

            # Guarded
            start_time = time.time()
            guarded_outputs, guarded_triggers = run_generation(
                model, tokenizer, prompts, dataset_gen_config, batch_size, guard=guard
            )
            end_time = time.time()
            total_duration = end_time - start_time
            avg_duration = total_duration / num_samples if num_samples > 0 else 0
            logger.info(f"[Utility - {dataset_name}] Guarded 生成统计: 总耗时 {total_duration:.2f}s, 单样本平均耗时 {avg_duration:.4f}s")

            df_guarded = df_utility_subset.copy()
            df_guarded['assistant_output'] = guarded_outputs
            df_guarded['trigger_step'] = guarded_triggers
            df_guarded['condition'] = 'guarded'
            df_guarded['eval_split'] = 'utility'
            all_results_dfs.append(df_guarded)

            del guarded_outputs, guarded_triggers, df_guarded
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Baseline
            start_time = time.time()
            baseline_outputs, baseline_triggers = run_generation(
                model, tokenizer, prompts, dataset_gen_config, batch_size, guard=None
            )
            end_time = time.time()
            total_duration = end_time - start_time
            avg_duration = total_duration / num_samples if num_samples > 0 else 0
            logger.info(f"[Utility - {dataset_name}] Baseline 生成统计: 总耗时 {total_duration:.2f}s, 单样本平均耗时 {avg_duration:.4f}s")

            df_baseline = df_utility_subset.copy()
            df_baseline['assistant_output'] = baseline_outputs
            df_baseline['trigger_step'] = baseline_triggers
            df_baseline['condition'] = 'baseline'
            df_baseline['eval_split'] = 'utility'
            all_results_dfs.append(df_baseline)

            del baseline_outputs, baseline_triggers, df_baseline
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    else:
        if run_utility:
            logger.warning("跳过可用性评估，因为数据集为空。")
        else:
            logger.info("根据配置，跳过可用性评估。")

    # --- 8b. 评估安全性 (ASR)，按攻击方法分组 ---
    if not df_safety_long.empty:
        attack_methods = df_safety_long['attack_method'].unique()
        logger.info(f"--- 评估开始: 安全性 (ASR)，将测试 {len(attack_methods)} 种攻击方法 ---")

        for method in attack_methods:
            logger.info(f"--- 正在评估攻击方法: {method} ---")
            df_attack = df_safety_long[df_safety_long['attack_method'] == method].reset_index(drop=True)
            prompts = df_attack['prompt'].tolist()
            dataset_max_new_tokens = get_override_max_new_tokens(df_attack)
            dataset_gen_config = build_generation_config(gen_config, dataset_max_new_tokens)

            if not prompts:
                logger.warning(f"方法 {method} 没有可运行的提示，跳过。")
                continue

            # Baseline
            baseline_outputs, baseline_triggers = run_generation(
                model, tokenizer, prompts, dataset_gen_config, batch_size, guard=None
            )
            df_baseline = df_attack.copy()
            df_baseline['assistant_output'] = baseline_outputs
            df_baseline['trigger_step'] = baseline_triggers
            df_baseline['condition'] = 'baseline'
            df_baseline['eval_split'] = 'safety'
            all_results_dfs.append(df_baseline)

            del df_baseline, baseline_outputs, baseline_triggers
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Guarded
            guarded_outputs, guarded_triggers = run_generation(
                model, tokenizer, prompts, dataset_gen_config, batch_size, guard=guard
            )
            df_guarded = df_attack.copy()
            df_guarded['assistant_output'] = guarded_outputs
            df_guarded['trigger_step'] = guarded_triggers
            df_guarded['condition'] = 'guarded'
            df_guarded['eval_split'] = 'safety'
            all_results_dfs.append(df_guarded)

            del df_guarded, guarded_outputs, guarded_triggers
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    else:
        if run_safety:
            logger.warning("跳过安全性评估，因为数据集为空。")
        else:
            logger.info("根据配置，跳过安全性评估。")

    logger.info("--- 阶段 1: 文本生成全部完成 ---")

    if not all_results_dfs:
        logger.error("未生成任何结果。请检查数据集路径和配置。")
        return

    # ---
    # --- 阶段 2: 分类
    # ---
    logger.info("--- 阶段 2: 开始有害性判断 ---")

    final_results_df = pd.concat(all_results_dfs, ignore_index=True)
    del all_results_dfs

    final_results_df['ablation_alpha'] = current_alpha
    final_results_df['ablation_beta'] = current_beta

    interim_csv_path = output_dir / f"{llm_name}_evaluation_INTERIM_all_results.csv"

    if 'trigger_step' in final_results_df.columns:
        final_results_df['trigger_step'] = final_results_df['trigger_step'].apply(lambda x: pd.NA if x == -1 else x).astype('Int64')

    try:
        logger.info(f"正在保存中间生成结果 (无标签) 到: {interim_csv_path}")

        base_cols_config = config.get('safety_dataset_config', {}).get('base_columns', [])
        ordered_cols = base_cols_config + [
            'attack_method', 'utility_dataset_name', 'eval_split', 'dataset', 'condition',
            'prompt', 'assistant_output', 'trigger_step', 'ablation_alpha', 'ablation_beta'
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
        mask_safety = final_results_df['eval_split'] == 'safety'
        df_to_classify = final_results_df[mask_safety].copy()

        if not df_to_classify.empty:
            logger.info(f"正在对 {len(df_to_classify)} 条 Safety 样本进行有害性判断 (跳过 Utility 样本)...")

            logger.info("--- 正在加载分类器 LLM ---")
            classifier_config = config['classifier_config']
            classifier_batch_size = config.get('classifier_batch_size', 2)
            classifier_model, classifier_tokenizer = load_model_and_tokenizer(
                model_name=classifier_config['path'],
                model_kwargs=classifier_config.get('kwargs', {}),
                device=device
            )

            labels = run_classification(
                classifier_model, classifier_tokenizer, df_to_classify, classifier_batch_size
            )
            df_to_classify['label'] = labels

            final_results_df.update(df_to_classify)
            logger.info("Safety 样本分类完成。")
        else:
            logger.info("没有需要分类的 Safety 样本 (或未启用 Safety 评测)。")

    finally:
        if 'classifier_model' in locals():
            model_name = classifier_model.config.name_or_path if hasattr(classifier_model, 'config') and hasattr(
                classifier_model.config, 'name_or_path') else "分类器 LLM"
            logger.info(f"正在从内存中释放模型: {model_name}...")
            del classifier_model
            if 'classifier_tokenizer' in locals():
                del classifier_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("模型内存已释放。")

    logger.info("--- 阶段 3: 计算指标并保存拆分报告 ---")

    df_utility_results = final_results_df[final_results_df['eval_split'] == 'utility']
    if not df_utility_results.empty:
        utility_dataset_names = df_utility_results['utility_dataset_name'].unique()
        logger.info(f"正在为 {len(utility_dataset_names)} 个良性数据集生成结果 (跳过 FPR 指标计算)...")

        for dataset_name in utility_dataset_names:
            if pd.isna(dataset_name) or dataset_name == 'N/A':
                continue

            df_utility_subset = df_utility_results[df_utility_results['utility_dataset_name'] == dataset_name]
            if df_utility_subset.empty:
                continue

            logger.info(f"正在保存可用性 (Utility) '{dataset_name}' 的结果文件 (CSV)...")

            try:
                utility_csv_path = output_dir / f"{llm_name}_evaluation_detailed_utility_{dataset_name}.csv"
                df_utility_subset.to_csv(utility_csv_path, index=False, encoding='utf-8-sig')

                logger.info(f"已保存可用性 (Utility) '{dataset_name}' 详细 CSV 到: {utility_csv_path.name}")

                if dataset_name == "alpaca_eval":
                    logger.info("检测到 alpaca_eval 数据集，正在生成专用的评估 JSON 文件...")

                    def save_alpaca_format(sub_df, out_filename, generator_name):
                        records = []
                        for _, row in sub_df.iterrows():
                            record = {
                                "dataset": row.get("dataset", "alpaca_eval"),
                                "instruction": row.get("instruction", row.get("prompt", "")),
                                "output": row.get("assistant_output", ""),
                                "generator": generator_name
                            }
                            records.append(record)

                        out_path = output_dir / out_filename
                        with open(out_path, 'w', encoding='utf-8') as f:
                            json.dump(records, f, indent=2, ensure_ascii=False)
                        logger.info(f"Alpaca Eval 格式结果已保存到: {out_path.name}")

                    df_base = df_utility_subset[df_utility_subset['condition'] == 'baseline']
                    if not df_base.empty:
                        save_alpaca_format(df_base, f"{llm_name}-alpaca_eval-baseline.json", f"{llm_name}-baseline")

                    df_guard = df_utility_subset[df_utility_subset['condition'] == 'guarded']
                    if not df_guard.empty:
                        save_alpaca_format(df_guard, f"{llm_name}-alpaca_eval-JEDI.json", f"{llm_name}-JEDI")

                if "xstest" in str(dataset_name).lower():
                    logger.info(f"检测到 xstest 数据集 ('{dataset_name}')，正在生成专用的评估 CSV 文件...")

                    def save_xstest_format(sub_df, out_filename):
                        target_cols = ['id', 'type', 'prompt', 'completion', 'annotation_1', 'annotation_2', 'agreement', 'final_label']
                        xstest_out = pd.DataFrame()

                        if 'id' in sub_df.columns:
                            xstest_out['id'] = sub_df['id']
                        else:
                            xstest_out['id'] = range(1, len(sub_df) + 1)

                        if 'type' in sub_df.columns:
                            xstest_out['type'] = sub_df['type']
                        else:
                            xstest_out['type'] = 'N/A'

                        xstest_out['prompt'] = sub_df['prompt']
                        xstest_out['completion'] = sub_df['assistant_output']

                        for col in ['annotation_1', 'annotation_2', 'agreement', 'final_label']:
                            xstest_out[col] = None

                        xstest_out = xstest_out[target_cols]

                        out_path = output_dir / out_filename
                        xstest_out.to_csv(out_path, index=False, encoding='utf-8-sig')
                        logger.info(f"XSTest 格式结果已保存到: {out_path.name}")

                    df_base = df_utility_subset[df_utility_subset['condition'] == 'baseline']
                    if not df_base.empty:
                        save_xstest_format(df_base, f"{llm_name}_xstest_baseline.csv")

                    df_guard = df_utility_subset[df_utility_subset['condition'] == 'guarded']
                    if not df_guard.empty:
                        save_xstest_format(df_guard, f"{llm_name}_xstest_guarded.csv")

            except Exception as e:
                logger.error(f"保存可用性 (Utility) '{dataset_name}' 结果文件时失败: {e}", exc_info=True)
    else:
        logger.info("未找到可用性结果。")

    df_safety_results = final_results_df[final_results_df['eval_split'] == 'safety']
    if not df_safety_results.empty:
        attack_methods = df_safety_results['attack_method'].unique()
        logger.info(f"正在为 {len(attack_methods)} 种攻击方法计算 ASR 指标...")

        for method in attack_methods:
            if pd.isna(method) or method == 'N/A':
                continue

            df_attack = df_safety_results[df_safety_results['attack_method'] == method]
            if df_attack.empty:
                continue

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

    try:
        if interim_csv_path.exists():
            interim_csv_path.unlink()
            logger.info(f"已清理临时文件: {interim_csv_path.name}")
    except Exception as e:
        logger.warning(f"清理临时文件 {interim_csv_path.name} 时失败: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="运行 JEDI 防御评估 (支持 jbb_expanded.csv 和 alpaca_eval.json)。")
    parser.add_argument('--config', type=str, default='configs/ablation_config.yaml',
                        help='评估配置文件路径。')
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    config_path = base_dir / args.config
    if not config_path.exists():
        logger.error(f"配置文件未找到: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- [新增] 加载流程控制标志 ---
    run_utility = config.get('run_utility_evaluation', True)
    run_safety = config.get('run_safety_evaluation', True)
    logger.info(f"评测流程配置: run_utility_evaluation={run_utility}, run_safety_evaluation={run_safety}")

    if not run_utility and not run_safety:
        logger.info("run_utility_evaluation 和 run_safety_evaluation 均为 false，没有评测任务要执行。正在退出。")
        return

    # --- 2. 设置路径 ---
    llm_name = config['llm_name']
    output_base_dir = base_dir / config['output_dir'] / llm_name
    output_base_dir.mkdir(parents=True, exist_ok=True)
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

    try:
        # --- 3. 加载 Guard ---
        logger.info("--- 正在加载 JEDI Guard ---")
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
        # [!] 修改：根据 run_utility 标志条件性加载
        df_utility_all = pd.DataFrame()  # [!] 初始化为空
        if run_utility:
            logger.info("正在加载“可用性”数据集...")
            utility_config = config['utility_dataset_config']
            utility_sample_size = utility_config.get('sample_size', 0)  # 这是总采样大小
            df_utility_all = load_utility_dataset(utility_config, data_dir, utility_sample_size)
        else:
            logger.info("根据配置，跳过加载“可用性”数据集。")

        # [!] 修改：根据 run_safety 标志条件性加载
        df_safety_long = pd.DataFrame()  # [!] 初始化为空
        if run_safety:
            logger.info("正在加载“安全性”数据集...")
            safety_config = config['safety_dataset_config']
            safety_sample_size = safety_config.get('sample_size', 0)
            df_safety_long = load_safety_dataset(safety_config, data_dir, safety_sample_size)
        else:
            logger.info("根据配置，跳过加载“安全性”数据集。")
        base_alpha = guard.alpha
        base_beta = config.get('base_beta', guard.base_beta)
        alpha_list = normalize_ablation_list('alpha_list', config.get('alpha_list'))
        bata_list = normalize_ablation_list('bata_list', config.get('bata_list'))

        if not alpha_list and not bata_list:
            logger.error("未提供任何消融实验参数列表 (alpha_list 或 bata_list)。")
            return

        ablation_settings = []
        for alpha_offset in alpha_list:
            current_alpha = base_alpha + alpha_offset
            ablation_settings.append({
                'name': format_ablation_tag('alpha', current_alpha),
                'alpha': current_alpha,
                'beta': base_beta
            })
        for beta in bata_list:
            ablation_settings.append({
                'name': format_ablation_tag('beta', beta),
                'alpha': base_alpha,
                'beta': beta
            })

        for setting in ablation_settings:
            guard.alpha = setting['alpha']
            guard.base_beta = setting['beta']
            output_dir = output_base_dir / setting['name']
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"=== 开始消融实验: {setting['name']} (alpha={setting['alpha']}, beta={setting['beta']}) ==="
            )
            run_full_evaluation(
                guard=guard,
                model=model,
                tokenizer=tokenizer,
                df_utility_all=df_utility_all,
                df_safety_long=df_safety_long,
                output_dir=output_dir,
                config=config,
                gen_config=gen_config,
                batch_size=batch_size,
                run_utility=run_utility,
                run_safety=run_safety,
                device=device,
                llm_name=llm_name,
                current_alpha=setting['alpha'],
                current_beta=setting['beta']
            )

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


if __name__ == "__main__":
    main()
