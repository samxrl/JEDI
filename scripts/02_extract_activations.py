# -*- coding: utf-8 -*-
"""
脚本 02: 提取模型激活

该脚本遵循“方法流程.md”文档中的第二阶段，负责从语言模型中提取隐藏状态（激活）。
此版本已更新，以支持双白化流程，并同时保存两种类型的激活：

主要功能:
1.  加载由 `01_prepare_datasets.py` 生成的结构化数据集 (A1-满足, A2-拒绝, B1-良性满足)。
2.  按 'prompt' 对数据进行分组，以优化性能。
3.  对于每个批次，从头构建完整的对话模板并将其输入模型。
4.  为所有样本类型 (A1, A2, B1) 提取并保存两类激活窗口：
    - “早期窗口” (early_window): 对应于前缀部分的隐藏态。
    - “内容窗口” (content_window): 对应于新生成内容部分的隐藏态。
5.  **修改点**: 对于每个窗口，同时保存两种形式的激活：
    - `aggregated`: 根据配置（如 'mean' 或 'last'）聚合后的单个向量。
    - `per_token`: 窗口内每个 token 的原始隐藏状态序列。
6.  将包含上述结构的激活字典保存到 .pt 文件，并将模型的自然语言输出保存到单独的 CSV 文件。

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # 设置为左填充以避免生成时的告警
    )
    # 如果分词器没有 pad token, 将其设置为 eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # （可选）确保模型配置与分词器同步
    model.config.pad_token_id = tokenizer.pad_token_id

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
    此版本返回一个包含 'aggregated' 和 'per_token' 两种激活的字典。
    """
    current_batch_size = len(batch_df)
    conversations = [json.loads(conv_str) for conv_str in batch_df['conversation']]

    # --- 统一构建输入文本 ---
    full_input_texts = []
    for conv in conversations:
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        if tokenizer.eos_token:
            text = text.rstrip()
            if text.endswith(tokenizer.eos_token):
                text = text[:-len(tokenizer.eos_token)]
        full_input_texts.append(text)

    inputs = tokenizer(full_input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    padded_prompt_len = inputs['input_ids'].shape[1]

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
        prefix_start_indices = [-1] * current_batch_size

        for i in range(current_batch_size):
            prefix_text = conversations[i][1]['content']
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            prompt_ids_list = inputs['input_ids'][i].tolist()
            start_idx = find_subsequence(prompt_ids_list, prefix_ids)
            if start_idx != -1:
                prefix_start_indices[i] = start_idx
                tokens_to_decode = generated_sequences[i, start_idx:]
                output_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True)
                assistant_outputs.append(output_text)
            else:
                logging.warning(f"无法为样本 {i} 定位前缀 token。回退到基于长度的解码。")
                decoded_new_part = tokenizer.decode(generated_sequences[i, padded_prompt_len:], skip_special_tokens=True)
                assistant_outputs.append(prefix_text + decoded_new_part)

        # --- 步骤 2: 对完整序列进行一次前向传播，以获取所有隐藏状态 ---
        full_outputs = model(generated_sequences, output_hidden_states=True)
        all_hidden_states = [h.cpu() for h in full_outputs.hidden_states]

    # --- MODIFICATION START: 初始化新的激活数据结构 ---
    batch_activations = {
        layer: {
            'early_window': {'aggregated': [], 'per_token': []},
            'content_window': {'aggregated': [], 'per_token': []}
        } for layer in extraction_config['layers']
    }
    # --- MODIFICATION END ---

    # --- 提取激活 ---
    for layer_idx in extraction_config['layers']:
        layer_hidden_states = all_hidden_states[layer_idx]

        # --- MODIFICATION START: 提取并存储两种类型的 'content_window' 激活 ---
        content_window_states = layer_hidden_states[:, padded_prompt_len:, :]
        if content_window_states.shape[1] > 0:
            # 存储逐 token 激活 (整个批次)
            batch_activations[layer_idx]['content_window']['per_token'].append(content_window_states.cpu())
            # 存储聚合后激活 (整个批次)
            agg_content_batch = aggregate_activations(content_window_states, extraction_config['aggregation'])
            batch_activations[layer_idx]['content_window']['aggregated'].append(agg_content_batch.cpu())
        # --- MODIFICATION END ---

        for i in range(current_batch_size):
            prefix_start_idx = prefix_start_indices[i]
            if prefix_start_idx == -1:
                continue

            prefix_text = conversations[i][1]['content']
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_ids)

            if prefix_len > 0:
                prefix_end_idx = prefix_start_idx + prefix_len
                early_window_slice = layer_hidden_states[i, prefix_start_idx:prefix_end_idx, :].unsqueeze(0)

                # --- MODIFICATION START: 提取并存储两种类型的 'early_window' 激活 ---
                # 存储逐 token 激活 (单个样本)
                batch_activations[layer_idx]['early_window']['per_token'].append(early_window_slice.squeeze(0).cpu())
                # 存储聚合后激活 (单个样本)
                agg_early = aggregate_activations(early_window_slice, extraction_config['aggregation'])
                batch_activations[layer_idx]['early_window']['aggregated'].append(agg_early.cpu())
                # --- MODIFICATION END ---

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
    llm_name = config['model_name'].split('/')[-1]
    output_dir = base_dir / config['output_dir'] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"所有激活和输出文件将保存到: {output_dir}")
    dataset_llm_name = config['dataset_llm_name']
    model, tokenizer, model_config_details = get_model_and_tokenizer(config['model_name'], config.get('model_kwargs', {}),
                                                                     config['processing']['device'])
    num_hidden_layers = getattr(model_config_details, 'num_hidden_layers', float('inf'))
    requested_layers = config['extraction']['layers']
    valid_layers = [l for l in requested_layers if 0 <= l < num_hidden_layers]
    if not valid_layers:
        logging.error(f"请求的层 {requested_layers} 无效 (模型层数: {num_hidden_layers})。正在中止。")
        return
    config['extraction']['layers'] = valid_layers
    logging.info(f"将从以下层提取激活: {valid_layers}")
    sample_size = config.get('processing', {}).get('sample_size', 0)
    datasets_to_process = {
        "compliance": processed_data_dir / f"{dataset_llm_name}_jailbreak_compliance.csv",
        "refusal": processed_data_dir / f"{dataset_llm_name}_jailbreak_refusal.csv",
        "benign": processed_data_dir / f"{dataset_llm_name}_benign_compliance.csv",
    }

    # --- 3. 遍历并处理每个数据集 ---
    for name, path in datasets_to_process.items():
        if not path.is_file():
            logging.warning(f"未找到 '{name}' 的数据集文件，路径: {path}，正在跳过。")
            continue

        logging.info(f"--- 正在处理数据集: {name} ---")
        df = pd.read_csv(path)

        if sample_size > 0 and sample_size < len(df):
            logging.info(f"数据集 '{name}' 包含 {len(df)} 个样本。正在根据配置随机采样 {sample_size} 个样本...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # --- MODIFICATION START: 初始化新的总激活数据结构 ---
        total_activations = {
            layer: {
                'early_window': {'aggregated': [], 'per_token': []},
                'content_window': {'aggregated': [], 'per_token': []}
            } for layer in config['extraction']['layers']
        }
        # --- MODIFICATION END ---

        all_outputs_for_csv = []
        batch_size = config['processing']['batch_size']
        use_grouping = 'prompt' in df.columns
        grouped = df.groupby('prompt') if use_grouping else [('all', df)]
        progress_bar = tqdm(total=len(df), desc=f"正在提取 {name}")

        # --- 4. 按分组处理 ---
        for prompt, group_df in grouped:
            # --- 5. 在分组内按批次处理 ---
            for i in range(0, len(group_df), batch_size):
                batch_df = group_df.iloc[i:i + batch_size]
                batch_activations, assistant_outputs = process_batch(
                    batch_df, model, tokenizer,
                    config['generation'], config['extraction'], config['processing']['device']
                )

                for idx, assistant_text in enumerate(assistant_outputs):
                    original_row = batch_df.iloc[idx]
                    output_record = {'prompt': original_row['prompt'], 'assistant_output': assistant_text}
                    for col in ['prefix', 'behavior', 'FunctionalCategory', 'ContextString', 'source']:
                        if col in original_row:
                            output_record[col] = original_row[col]
                    all_outputs_for_csv.append(output_record)

                # --- MODIFICATION START: 收集批次结果到总激活字典 ---
                for layer_idx, windows in batch_activations.items():
                    # early_window 的结果是列表，直接扩展
                    if windows['early_window']['aggregated']:
                        total_activations[layer_idx]['early_window']['aggregated'].extend(windows['early_window']['aggregated'])
                    if windows['early_window']['per_token']:
                        total_activations[layer_idx]['early_window']['per_token'].extend(windows['early_window']['per_token'])
                    # content_window 的结果是批次张量，直接追加
                    if windows['content_window']['aggregated']:
                        total_activations[layer_idx]['content_window']['aggregated'].extend(windows['content_window']['aggregated'])
                    if windows['content_window']['per_token']:
                        total_activations[layer_idx]['content_window']['per_token'].extend(windows['content_window']['per_token'])
                # --- MODIFICATION END ---

                progress_bar.update(len(batch_df))
                del batch_activations, assistant_outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        progress_bar.close()

        # --- 6. 聚合结果并保存 ---
        # --- MODIFICATION START: 构建最终要保存的嵌套字典 ---
        final_tensors = {}
        for layer_idx, windows in total_activations.items():
            final_tensors[layer_idx] = {}
            # -- 处理 early_window --
            final_tensors[layer_idx]['early_window'] = {}
            if windows['early_window']['aggregated']:
                final_tensors[layer_idx]['early_window']['aggregated'] = torch.cat(windows['early_window']['aggregated'], dim=0)
            # per_token 激活由于长度不一，作为张量列表保存
            if windows['early_window']['per_token']:
                final_tensors[layer_idx]['early_window']['per_token'] = windows['early_window']['per_token']

            # -- 处理 content_window --
            final_tensors[layer_idx]['content_window'] = {}
            if windows['content_window']['aggregated']:
                final_tensors[layer_idx]['content_window']['aggregated'] = torch.cat(windows['content_window']['aggregated'], dim=0)

            # FIX: The assumption that all content windows have the same length across batches is incorrect,
            # as generation can stop early (e.g., EOS token). This was causing the RuntimeError.
            # We now save it as a list of tensors, just like with the early_window.
            # First, we flatten the list of batch tensors into a single list of sample-level tensors.
            if windows['content_window']['per_token']:
                flat_per_token_list = [
                    sample_tensor.clone()  # clone to ensure memory is contiguous
                    for batch_tensor in windows['content_window']['per_token']
                    for sample_tensor in torch.unbind(batch_tensor, dim=0)
                ]
                final_tensors[layer_idx]['content_window']['per_token'] = flat_per_token_list
        # --- MODIFICATION END ---

        output_path = output_dir / f"{llm_name}_{name}_activations.pt"
        torch.save(final_tensors, output_path)
        logging.info(f"成功将 '{name}' 的激活保存到 {output_path}")

        # 保存自然语言输出 .csv 文件
        if all_outputs_for_csv:
            output_df = pd.DataFrame(all_outputs_for_csv)
            if name == "benign":
                cols_order = ['prompt', 'prefix', 'assistant_output', 'source']
            else:
                output_df['label'] = None
                cols_order = ['prompt', 'prefix', 'behavior', 'FunctionalCategory', 'ContextString', 'assistant_output', 'label']
            final_cols = [col for col in cols_order if col in output_df.columns]
            output_df = output_df[final_cols]
            output_csv_path = output_dir / f"{llm_name}_{name}_outputs.csv"
            output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logging.info(f"成功将 '{name}' 的自然语言输出保存到 {output_csv_path}")

    logging.info("所有数据集的激活提取和输出保存过程已完成。")


if __name__ == "__main__":
    main()

