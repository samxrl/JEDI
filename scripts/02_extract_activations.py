# -*- coding: utf-8 -*-
"""
脚本 02: 提取模型激活

该脚本遵循“方法流程.md”文档中的第二阶段，负责从语言模型中提取隐藏状态（激活）。
此版本已更新，以支持双白化流程，并为节省磁盘空间，仅保存聚合后的激活。

主要功能:
1.  加载由 `01_prepare_datasets.py` 生成的结构化数据集 (A1-满足, A2-拒绝, B1-良性满足)。
2.  按 'prompt' 对数据进行分组，以优化性能。
3.  对于每个批次，从头构建完整的对话模板并将其输入模型。
4.  为所有样本类型 (A1, A2, B1) 提取并保存两类激活窗口：
    - “早期窗口” (early_window): 对应于前缀部分的隐藏态。
    - “内容窗口” (content_window): 对应于新生成内容部分的隐藏态。
5.  **修改点**: 对于每个窗口，仅保存根据配置（如 'mean' 或 'last'）聚合后的单个向量。
    不再保存逐 token 的原始隐藏状态序列，以大幅减少存储占用。
6.  将聚合后的激活张量直接保存在字典中 (格式: {layer: {window: tensor}})，并将模型的自然语言输出保存到单独的 CSV 文件。
7.  **内存优化**: 采用分块保存策略，每处理10个批次就将数据写入临时文件并清空内存，
    最后将所有临时块合并，以处理大规模数据集。

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
from typing import List, Dict, Any

# 配置基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def is_qwen3_tokenizer(tokenizer) -> bool:
    """判断当前 tokenizer 是否属于 Qwen3 系列。"""
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
    return "qwen3" in tokenizer_name


def apply_chat_template_compat(tokenizer, conversation, **kwargs):
    """对 Qwen3 系列关闭 thinking 模式，其余模型保持原行为。"""
    if is_qwen3_tokenizer(tokenizer):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(conversation, **kwargs)


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


def aggregate_activations(activations: torch.Tensor, method: str, max_tokens: int = 0) -> torch.Tensor:
    """
    在序列长度维度上聚合隐藏状态。

    Args:
        activations (torch.Tensor): 形状为 (batch_size, seq_len, hidden_dim) 的张量。
        method (str): 聚合方法，'mean' (平均) 或 'last' (取最后一个)。
        max_tokens (int, optional): 如果 method 为 'mean'，则仅使用前 max_tokens 个 token 计算平均值。
                                    默认为 0，表示使用所有 token。

    Returns:
        torch.Tensor: 聚合后的形状为 (batch_size, hidden_dim) 的张量。
    """
    if method == "mean":
        # 如果指定了 max_tokens 且当前序列长度超过该值，则截断
        if max_tokens > 0 and activations.shape[1] > max_tokens:
            activations = activations[:, :max_tokens, :]
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
    此版本通过删除临时变量来加强内存管理。
    """
    current_batch_size = len(batch_df)
    conversations = [json.loads(conv_str) for conv_str in batch_df['conversation']]

    # --- 统一构建输入文本 ---
    full_input_texts = []
    for conv in conversations:
        text = apply_chat_template_compat(
            tokenizer, conv, tokenize=False, add_generation_prompt=False
        )
        if tokenizer.eos_token:
            text = text.rstrip()
            if text.endswith(tokenizer.eos_token):
                text = text[:-len(tokenizer.eos_token)]
        full_input_texts.append(text)

    inputs = tokenizer(full_input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    del full_input_texts  # 删除已使用的列表
    padded_prompt_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        # --- 步骤 1: 生成完整的 token 序列 ---
        generated_sequences = model.generate(
            **inputs,
            max_new_tokens=generation_config['max_new_tokens'],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )
        # `inputs` 在 generate 后不再需要
        del inputs

        # --- 解码和索引查找 ---
        assistant_outputs = []
        prefix_start_indices = [-1] * current_batch_size

        for i in range(current_batch_size):
            prefix_text = conversations[i][1]['content']
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            prompt_ids_list = generated_sequences[i, :padded_prompt_len].tolist()
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

        del prefix_ids, prompt_ids_list  # 删除循环中的临时变量

        # --- 步骤 2: 对完整序列进行一次前向传播，以获取所有隐藏状态 ---
        full_outputs = model(generated_sequences, output_hidden_states=True)
        all_hidden_states = [h.cpu() for h in full_outputs.hidden_states]
        # `full_outputs` 包含计算图和原始张量，是主要内存消耗者
        del full_outputs

    # --- MODIFICATION START: 初始化更简单的数据结构 ---
    batch_activations = {
        layer: {
            'early_window': [],
            'content_window': []
        } for layer in extraction_config['layers']
    }
    # --- MODIFICATION END ---

    # 从配置中获取 mean_k 参数 (如果存在)，默认为 0 (不截断)
    mean_k = extraction_config.get('mean_k', 0)

    # --- 提取激活 ---
    for layer_idx in extraction_config['layers']:
        layer_hidden_states = all_hidden_states[layer_idx]

        # --- MODIFICATION START: 仅提取并存储 'content_window' 的聚合激活 ---
        content_window_states = layer_hidden_states[:, padded_prompt_len:, :]
        if content_window_states.shape[1] > 0:
            #  关键修改: 仅对 content_window 应用 mean_k 截断
            agg_content_batch = aggregate_activations(
                content_window_states,
                extraction_config['aggregation'],
                max_tokens=mean_k
            )
            batch_activations[layer_idx]['content_window'].append(agg_content_batch.cpu())
            del agg_content_batch
        del content_window_states
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

                # --- MODIFICATION START: 仅提取并存储 'early_window' 的聚合激活 ---
                # 对于 early_window (前缀)，我们通常不使用 k 截断，保留 max_tokens=0
                agg_early = aggregate_activations(
                    early_window_slice,
                    extraction_config['aggregation'],
                    max_tokens=0
                )
                batch_activations[layer_idx]['early_window'].append(agg_early.cpu())
                del early_window_slice, agg_early
                # --- MODIFICATION END ---

        del layer_hidden_states

    # 函数返回前进行最终清理
    del all_hidden_states
    del generated_sequences
    gc.collect()

    return batch_activations, assistant_outputs


def aggregate_and_format_chunk(activations_dict: Dict) -> Dict:
    """
    对一个块（chunk）内的激活数据进行聚合和格式化，使其可以被保存。
    """
    chunk_tensors = {}
    for layer_idx, windows in activations_dict.items():
        chunk_tensors[layer_idx] = {}
        if windows['early_window']:
            chunk_tensors[layer_idx]['early_window'] = torch.cat(windows['early_window'], dim=0)
        if windows['content_window']:
            chunk_tensors[layer_idx]['content_window'] = torch.cat(windows['content_window'], dim=0)
    return chunk_tensors


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
    llm_name = config['dataset_llm_name']
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

        # 定义最终和临时文件路径
        final_pt_path = output_dir / f"{llm_name}_{name}_activations.pt"
        final_csv_path = output_dir / f"{llm_name}_{name}_outputs.csv"

        # 清理上一次运行可能留下的临时文件
        for temp_file in output_dir.glob(f"{llm_name}_{name}_activations_chunk_*.pt"):
            temp_file.unlink()
        if final_csv_path.exists():
            final_csv_path.unlink()

        # 初始化用于存储当前块数据的容器
        def get_empty_activations_dict():
            return {
                layer: {
                    'early_window': [],
                    'content_window': []
                } for layer in config['extraction']['layers']
            }

        chunk_activations = get_empty_activations_dict()
        chunk_outputs_for_csv = []
        chunk_index = 0
        batches_in_chunk = 0

        batch_size = config['processing']['batch_size']
        use_grouping = 'prompt' in df.columns
        grouped = df.groupby('prompt') if use_grouping else [('all', df)]
        progress_bar = tqdm(total=len(df), desc=f"正在提取 {name}")

        # --- 4. 按分组和批次处理 ---
        processed_rows = 0
        for prompt, group_df in grouped:
            for i in range(0, len(group_df), batch_size):
                batch_df = group_df.iloc[i:i + batch_size]
                batch_activations, assistant_outputs = process_batch(
                    batch_df, model, tokenizer,
                    config['generation'], config['extraction'], config['processing']['device']
                )

                # 累积当前块的结果
                for idx, assistant_text in enumerate(assistant_outputs):
                    original_row = batch_df.iloc[idx]
                    output_record = {'prompt': original_row['prompt'], 'assistant_output': assistant_text}
                    for col in ['prefix', 'behavior', 'FunctionalCategory', 'ContextString', 'source']:
                        if col in original_row:
                            output_record[col] = original_row[col]
                    chunk_outputs_for_csv.append(output_record)

                for layer_idx, windows in batch_activations.items():
                    if windows['early_window']:
                        chunk_activations[layer_idx]['early_window'].extend(windows['early_window'])
                    if windows['content_window']:
                        chunk_activations[layer_idx]['content_window'].extend(windows['content_window'])

                del batch_activations, assistant_outputs
                batches_in_chunk += 1
                processed_rows += len(batch_df)
                progress_bar.update(len(batch_df))

                # --- 5. 检查是否需要保存块 ---
                is_last_batch_of_dataset = processed_rows == len(df)
                if (batches_in_chunk >= 20 or is_last_batch_of_dataset) and chunk_outputs_for_csv:
                    logging.info(f"已处理 {batches_in_chunk} 个批次, 正在保存块 {chunk_index}...")

                    # 格式化并保存激活块
                    chunk_tensors = aggregate_and_format_chunk(chunk_activations)
                    chunk_pt_path = output_dir / f"{llm_name}_{name}_activations_chunk_{chunk_index}.pt"
                    torch.save(chunk_tensors, chunk_pt_path)

                    # 保存输出块到CSV
                    output_df = pd.DataFrame(chunk_outputs_for_csv)
                    if name == "benign":
                        cols_order = ['prompt', 'prefix', 'assistant_output', 'source']
                    else:
                        output_df['label'] = None
                        cols_order = ['prompt', 'prefix', 'behavior', 'FunctionalCategory', 'ContextString', 'assistant_output', 'label']
                    final_cols = [col for col in cols_order if col in output_df.columns]
                    output_df = output_df[final_cols]

                    # 只有第一个块需要写入表头
                    is_first_chunk = (chunk_index == 0)
                    output_df.to_csv(final_csv_path, mode='a', index=False, header=is_first_chunk, encoding='utf-8-sig')

                    # 重置容器以释放内存
                    chunk_activations = get_empty_activations_dict()
                    chunk_outputs_for_csv = []
                    batches_in_chunk = 0
                    chunk_index += 1

                    del chunk_tensors, output_df
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        progress_bar.close()

        # --- 6. 合并所有激活块 ---
        logging.info(f"正在合并 {chunk_index} 个激活块以创建最终文件...")
        final_activations = get_empty_activations_dict()

        for i in range(chunk_index):
            chunk_path = output_dir / f"{llm_name}_{name}_activations_chunk_{i}.pt"
            if not chunk_path.exists():
                logging.warning(f"未找到块文件 {chunk_path}，跳过。")
                continue

            chunk_data = torch.load(chunk_path, map_location='cpu')
            for layer_idx, windows in chunk_data.items():
                if 'early_window' in windows:
                    final_activations[layer_idx]['early_window'].append(windows['early_window'])
                if 'content_window' in windows:
                    final_activations[layer_idx]['content_window'].append(windows['content_window'])

            del chunk_data
            chunk_path.unlink()  # 删除已合并的块文件

        # 最后一次聚合所有块
        final_tensors_to_save = aggregate_and_format_chunk(final_activations)
        torch.save(final_tensors_to_save, final_pt_path)
        logging.info(f"成功将 '{name}' 的激活合并并保存到 {final_pt_path}")

        # 清理内存
        del df, final_activations, final_tensors_to_save
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info("所有数据集的激活提取和输出保存过程已完成。")


if __name__ == "__main__":
    main()
