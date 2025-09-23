# -*- coding: utf-8 -*-
"""
脚本 02: 提取模型激活

该脚本遵循“方法流程.md”文档中的第二阶段，负责从语言模型中提取隐藏状态（激活）。

主要功能:
1.  加载由 `01_prepare_datasets.py` 生成的结构化数据集（满足、拒绝和良性样本）。
2.  按 'prompt' 对数据进行分组，确保每个批次内的样本共享相同的前序提示以优化性能。
3.  对于每个批次，从头构建完整的对话模板并将其输入模型。
4.  根据“早期窗口”(W_early, 对应于前缀) 和“内容窗口” (W_cont, 对应于新生成的内容) 对隐藏状态进行切片和聚合。
5.  将聚合后的激活张量保存到文件，同时将模型的自然语言输出保存到单独的 CSV 文件。

如何运行:
python scripts/02_extract_activations.py --config configs/extraction_config.yaml
"""
import argparse
import json
import logging
import torch
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc
from typing import List

# 配置基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_subsequence(main_list: List[int], sub_list: List[int]) -> int:
    """
    在主列表中查找子列表的起始索引。

    Args:
        main_list (List[int]): 主 token ID 列表。
        sub_list (List[int]): 要查找的子 token ID 列表。

    Returns:
        int: 子列表在主列表中的起始索引，如果未找到则返回 -1。
    """
    main_len = len(main_list)
    sub_len = len(sub_list)
    for i in range(main_len - sub_len + 1):
        if main_list[i:i + sub_len] == sub_list:
            return i
    return -1


def get_model_and_tokenizer(model_name: str, model_kwargs: dict, device: str):
    """
    加载指定的 Hugging Face 模型和分词器。

    Args:
        model_name (str): 要从 Hugging Face Hub 加载的模型的名称。
        model_kwargs (dict): 模型加载的关键字参数 (例如, torch_dtype, load_in_8bit)。
        device (str): 如果不使用 8-bit 加载，要将模型移动到的设备。

    Returns:
        tuple: 一个包含已加载模型、分词器和模型配置的元组。
    """
    logging.info(f"正在加载模型 '{model_name}'...")
    # 复制关键字参数以避免修改原始配置字典
    kwargs = model_kwargs.copy()

    # 安全地评估 torch_dtype 字符串, 将其从字符串转换为 torch.dtype 对象
    if "torch_dtype" in kwargs and isinstance(kwargs["torch_dtype"], str):
        try:
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
        except AttributeError:
            # 如果是 "auto"，则保持为 "auto"，由 transformers 自动处理
            if kwargs["torch_dtype"] != "auto":
                raise ValueError(f"无效的 torch_dtype: {kwargs['torch_dtype']}")

    # 检查是否以 8-bit 模式加载
    load_in_8bit = kwargs.get("load_in_8bit", False)
    if load_in_8bit:
        logging.info("正在以 8-bit 模式加载模型。")
        # 8-bit 加载通常与 device_map="auto" 配合使用效果最好
        if "device_map" not in kwargs:
            kwargs["device_map"] = "auto"
        # 以 8-bit 量化加载模型
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        # 用于全精度加载的原始逻辑
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)

    model.eval()  # 将模型设置为评估模式
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 如果分词器没有 pad token, 将其设置为 eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    logging.info("模型和分词器加载成功。")
    return model, tokenizer, config


def aggregate_activations(activations: torch.Tensor, method: str) -> torch.Tensor:
    """
    在序列长度维度上聚合隐藏状态。

    Args:
        activations (torch.Tensor): 形状为 (batch_size, seq_len, hidden_dim) 的张量。
        method (str): 聚合方法，'mean' (平均) 或 'last' (取最后一个)。

    Returns:
        torch.Tensor: 聚合后的形状为 (batch_size, hidden_dim) 的张量。
    """
    if method == "mean":
        return activations.mean(dim=1)
    elif method == "last":
        return activations[:, -1, :]
    else:
        raise ValueError(f"未知的聚合方法: {method}")


def process_batch(
        batch_df, model, tokenizer, generation_config, extraction_config, device
):
    """
    处理一个批次的数据以提取激活。
    此版本采用最稳健的两步法，避免 KV 缓存传递的复杂性：
    1. 使用 model.generate() 获取完整的输出 token 序列。
    2. 对完整的序列进行一次 model() 前向传播，以可靠地获取所有隐藏状态。
    """
    current_batch_size = len(batch_df)
    conversations = [json.loads(conv_str) for conv_str in batch_df['conversation']]

    # 检查批次中的数据类型 (benign vs jailbreak)
    is_benign_batch = 'prefix' not in batch_df.columns

    if is_benign_batch:
        # 对于良性数据，直接使用标准模板让模型自由生成
        full_input_texts = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    else:
        # --- 新策略：应用完整的对话模板，然后移除EOS标记 ---
        full_input_texts = []
        for conv in conversations:
            text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            if tokenizer.eos_token:
                text = text.rstrip()  # 移除末尾空白
                text = text[:-len(tokenizer.eos_token)]
            full_input_texts.append(text)

    # 对最终的文本输入进行分词
    inputs = tokenizer(full_input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        # --- 步骤 1: 生成完整的 token 序列 ---
        generated_sequences = model.generate(
            **inputs,
            max_new_tokens=generation_config['max_new_tokens'],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )

        # --- 解码和索引查找 ---
        assistant_outputs = []
        prefix_start_indices = [-1] * current_batch_size  # 存储索引以供重用

        if is_benign_batch:
            generated_tokens_only = generated_sequences[:, prompt_len:]
            assistant_outputs = tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)
        else:
            for i in range(current_batch_size):
                prefix_text = conversations[i][1]['content']
                prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

                # 在输入的 token ID 中搜索前缀的 token ID
                prompt_ids_list = inputs['input_ids'][i].tolist()
                start_idx = find_subsequence(prompt_ids_list, prefix_ids)

                if start_idx != -1:
                    prefix_start_indices[i] = start_idx
                    # 从找到的精确位置开始解码，得到 "前缀 + 生成内容"
                    tokens_to_decode = generated_sequences[i, start_idx:]
                    output_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True)
                    assistant_outputs.append(output_text)
                else:
                    # 如果未找到前缀 token（不太可能发生，但作为保障），则回退
                    logging.warning(f"无法为样本 {i} 定位前缀 token。回退到基于长度的解码。")
                    generated_tokens_only = generated_sequences[i, prompt_len:]
                    decoded_new_part = tokenizer.decode(generated_tokens_only, skip_special_tokens=True)
                    assistant_outputs.append(prefix_text + decoded_new_part)

        # --- 步骤 2: 对完整序列进行一次前向传播，以获取所有隐藏状态 ---
        full_outputs = model(generated_sequences, output_hidden_states=True)
        all_hidden_states = [h.cpu() for h in full_outputs.hidden_states]

    batch_activations = {layer: {'early_window': [], 'content_window': []} for layer in extraction_config['layers']}

    # --- 提取激活 ---
    for layer_idx in extraction_config['layers']:
        layer_hidden_states = all_hidden_states[layer_idx]

        # 提取内容窗口 (W_cont) 的激活 (无变化)
        content_window_states = layer_hidden_states[:, prompt_len:, :]
        if content_window_states.shape[1] > 0:
            agg_content = aggregate_activations(content_window_states, extraction_config['aggregation'])
            for i in range(current_batch_size):
                batch_activations[layer_idx]['content_window'].append(agg_content[i].unsqueeze(0).cpu())

        if is_benign_batch:
            continue

        # 提取早期窗口 (W_early) 的激活
        for i in range(current_batch_size):
            prefix_start_idx = prefix_start_indices[i]
            if prefix_start_idx == -1:
                continue  # 如果之前未找到，则跳过

            prefix_text = conversations[i][1]['content']
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_ids)

            if prefix_len > 0:
                prefix_end_idx = prefix_start_idx + prefix_len
                if prefix_start_idx < prefix_end_idx and prefix_end_idx <= prompt_len:
                    # 从当前样本的层隐藏状态中切片
                    early_window_slice = layer_hidden_states[i, prefix_start_idx:prefix_end_idx, :].unsqueeze(0)
                    agg_early = aggregate_activations(early_window_slice, extraction_config['aggregation'])
                    batch_activations[layer_idx]['early_window'].append(agg_early.cpu())
                else:
                    logging.warning(
                        f"样本 {i} 的前缀索引 [{prefix_start_idx}:{prefix_end_idx}] (通过 token 搜索得到) 对于 prompt_len {prompt_len} 无效。")

    return batch_activations, assistant_outputs


def main():
    parser = argparse.ArgumentParser(
        description="根据配置文件从语言模型中提取隐藏状态激活。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config', type=str, default='../configs/extraction_config.yaml',
                        help='提取阶段的 YAML 配置文件路径。\n默认: ../configs/extraction_config.yaml')
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    config_path = Path(args.config)
    if not config_path.is_file():
        logging.error(f"配置文件未找到: {config_path}")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- 2. 设置路径和加载模型 ---
    base_dir = Path(__file__).parent.parent
    processed_data_dir = base_dir / config['processed_data_dir']

    # 从配置中获取模型名称，并创建特定于模型的输出目录
    llm_name = config['model_name'].split('/')[-1]
    output_dir = base_dir / config['output_dir'] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"所有激活和输出文件将保存到: {output_dir}")

    dataset_llm_name = config['dataset_llm_name']

    model, tokenizer, model_config_details = get_model_and_tokenizer(config['model_name'], config.get('model_kwargs', {}),
                                                                     config['processing']['device'])

    # 验证请求的层是否存在于模型中
    num_hidden_layers = getattr(model_config_details, 'num_hidden_layers', float('inf'))
    requested_layers = config['extraction']['layers']
    valid_layers = [l for l in requested_layers if 0 <= l < num_hidden_layers]
    if not valid_layers:
        logging.error(f"请求的层 {requested_layers} 无效 (模型层数: {num_hidden_layers})。正在中止。")
        return
    config['extraction']['layers'] = valid_layers
    logging.info(f"将从以下层提取激活: {valid_layers}")

    # 从配置中获取 sample_size
    sample_size = config.get('processing', {}).get('sample_size', 0)

    # 定义要处理的数据集
    datasets_to_process = {
        "compliance": processed_data_dir / f"{dataset_llm_name}_jailbreak_compliance.csv",
        "refusal": processed_data_dir / f"{dataset_llm_name}_jailbreak_refusal.csv",
        "benign": processed_data_dir / "benign_prompts.csv",
    }

    # --- 3. 遍历并处理每个数据集 ---
    for name, path in datasets_to_process.items():

        if not path.is_file():
            logging.warning(f"未找到 '{name}' 的数据集文件，路径: {path}，正在跳过。")
            continue

        logging.info(f"--- 正在处理数据集: {name} ---")
        df = pd.read_csv(path)

        # 如果在配置中指定了 sample_size，则对数据集进行随机采样
        if sample_size > 0 and sample_size < len(df):
            logging.info(f"数据集 '{name}' 包含 {len(df)} 个样本。正在根据配置随机采样 {sample_size} 个样本...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        total_activations = {layer: {'early_window': [], 'content_window': []} for layer in config['extraction']['layers']}
        all_outputs_for_csv = []  # (新) 为当前数据集初始化一个列表来存储输出
        batch_size = config['processing']['batch_size']

        # 判断是否应该使用基于 prompt 分组的优化
        use_grouping = 'prompt' in df.columns and name in ["compliance", "refusal"]

        # 按 'prompt' 分组，如果不需要分组则创建一个包含所有数据的伪分组
        grouped = df.groupby('prompt') if use_grouping else [('all', df)]
        progress_bar = tqdm(total=len(df), desc=f"正在提取 {name}")

        # --- 4. 按分组处理 ---
        for prompt, group_df in grouped:
            # --- 5. 在分组内按批次处理 ---
            for i in range(0, len(group_df), batch_size):
                batch_df = group_df.iloc[i:i + batch_size]

                # 调用核心处理函数，现在返回两个值
                batch_activations, assistant_outputs = process_batch(
                    batch_df, model, tokenizer,
                    config['generation'], config['extraction'], config['processing']['device']
                )

                # 准备要写入 CSV 的数据
                for idx, assistant_text in enumerate(assistant_outputs):
                    original_row = batch_df.iloc[idx]
                    output_record = {
                        'prompt': original_row['prompt'],
                        'assistant_output': assistant_text
                    }
                    if 'prefix' in original_row:
                        output_record['prefix'] = original_row['prefix']
                    all_outputs_for_csv.append(output_record)

                # 收集当前批次的结果
                for layer_idx, windows in batch_activations.items():
                    if windows['early_window']: total_activations[layer_idx]['early_window'].extend(windows['early_window'])
                    if windows['content_window']: total_activations[layer_idx]['content_window'].extend(windows['content_window'])

                progress_bar.update(len(batch_df))

                # 清理缓存和垃圾回收
                del batch_activations, assistant_outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        progress_bar.close()

        # --- 6. 聚合结果并保存 ---
        # 保存激活 .pt 文件
        final_tensors = {}
        for layer_idx, windows in total_activations.items():
            final_tensors[layer_idx] = {}
            if windows['early_window']: final_tensors[layer_idx]['early_window'] = torch.cat(windows['early_window'], dim=0)
            if windows['content_window']: final_tensors[layer_idx]['content_window'] = torch.cat(windows['content_window'], dim=0)

        output_path = output_dir / f"{llm_name}_{name}_activations.pt"
        torch.save(final_tensors, output_path)
        logging.info(f"成功将 '{name}' 的激活保存到 {output_path}")

        # 保存自然语言输出 .csv 文件
        if all_outputs_for_csv:
            output_df = pd.DataFrame(all_outputs_for_csv)
            # [修改] 添加 'label' 列并初始化
            output_df['label'] = None

            # 重新排序以获得更清晰的输出，并包含 'label' 列
            if 'prefix' in output_df.columns:
                output_df = output_df[['prompt', 'prefix', 'assistant_output', 'label']]
            else:
                output_df = output_df[['prompt', 'assistant_output', 'label']]

            output_csv_path = output_dir / f"{llm_name}_{name}_outputs.csv"
            output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logging.info(f"成功将 '{name}' 的自然语言输出保存到 {output_csv_path}")

    logging.info("所有数据集的激活提取和输出保存过程已完成。")


if __name__ == "__main__":
    main()
