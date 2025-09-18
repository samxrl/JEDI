# -*- coding: utf-8 -*-
"""
脚本 02: 提取模型激活

该脚本遵循“方法流程.md”文档中的第二阶段，负责从语言模型中提取隐藏状态（激活）。

主要功能:
1.  加载由 `01_prepare_datasets.py` 生成的结构化数据集（满足、拒绝和良性样本）。
2.  按 'prompt' 对数据进行分组。对于每个组：
    a. 仅计算一次共享提示的输入嵌入（Input Embeddings）并缓存。
    b. 将此缓存重用于该组内所有后续批次，以最大化效率，显著减少计算开销。
3.  加载指定的预训练语言模型（例如，Vicuna, LLaMA）。
4.  根据“早期窗口”(W_early, 对应于前缀) 和“内容窗口” (W_cont, 对应于新生成的内容) 对隐藏状态进行切片和聚合。
5.  将聚合后的激活张量保存到文件。

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

# 配置基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_model_and_tokenizer(model_name: str, model_kwargs: dict, device: str):
    """
    加载指定的 Hugging Face 模型和分词器。

    Args:
        model_name (str): 要从 Hugging Face Hub 加载的模型的名称。
        model_kwargs (dict): 模型加载的关键字参数 (例如, torch_dtype)。
        device (str): 要将模型移动到的设备。

    Returns:
        tuple: 一个包含已加载模型、分词器和模型配置的元组。
    """
    logging.info(f"正在加载模型 '{model_name}'...")
    # 安全地评估 torch_dtype 字符串, 将其从字符串转换为 torch.dtype 对象
    if "torch_dtype" in model_kwargs and isinstance(model_kwargs["torch_dtype"], str):
        try:
            model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])
        except AttributeError:
            # 如果是 "auto"，则保持为 "auto"，由 transformers 自动处理
            if model_kwargs["torch_dtype"] != "auto":
                raise ValueError(f"无效的 torch_dtype: {model_kwargs['torch_dtype']}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()  # 将模型设置为评估模式
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 如果分词器没有 pad token, 将其设置为 eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name)

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
        batch_df, model, tokenizer, generation_config, extraction_config, device,
        prompt_embeds=None, prompt_len=0
):
    """
    处理一个批次的数据以提取激活，可以选择性地重用一个预先计算好的提示嵌入。

    Args:
        batch_df (pd.DataFrame): 包含当前批次数据的 DataFrame。
        model: 已加载的语言模型。
        tokenizer: 已加载的分词器。
        generation_config (dict): 生成参数配置。
        extraction_config (dict): 激活提取配置。
        device (str): 计算设备。
        prompt_embeds (torch.Tensor, optional): 预计算的提示嵌入。默认为 None。
        prompt_len (int, optional): 提示的 token 长度。默认为 0。

    Returns:
        dict: 包含提取出的激活的字典。
    """
    current_batch_size = len(batch_df)
    final_assistant_texts = []  # 用于存储处理后的、纯净的助手回合文本

    if prompt_embeds is not None:
        # --- 路径 A: 重用缓存的提示嵌入 ---
        # 这种方式可以避免对每个样本中相同的 prompt 部分进行重复的嵌入计算。
        prefixes = batch_df['prefix'].tolist()

        # 使用“虚拟用户回合”技巧来精确地剥离出助手回合的模板部分。
        # 这样可以避免 `apply_chat_template` 自动添加多余的系统提示或BOS token。
        dummy_user_turn = [{"role": "user", "content": "DUMMY"}]
        templated_dummy_user_turn = tokenizer.apply_chat_template(dummy_user_turn, tokenize=False, add_generation_prompt=False)

        for p in prefixes:
            full_turn = dummy_user_turn + [{"role": "assistant", "content": p}]
            templated_full_turn = tokenizer.apply_chat_template(full_turn, tokenize=False, add_generation_prompt=False)
            assistant_text = templated_full_turn.removeprefix(templated_dummy_user_turn)
            # 移除可能由模板添加的结束符，确保我们只处理前缀本身
            if tokenizer.eos_token and assistant_text.endswith(tokenizer.eos_token):
                assistant_text = assistant_text.removesuffix(tokenizer.eos_token)
            final_assistant_texts.append(assistant_text)

        # 对处理后的纯净前缀文本进行分词
        prefix_inputs = tokenizer(final_assistant_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        if prefix_inputs.input_ids.shape[1] == 0:
            logging.warning("处理后的前缀输入为空，跳过此批次。")
            return {layer: {'early_window': [], 'content_window': []} for layer in extraction_config['layers']}

        # 获取前缀的词嵌入
        with torch.no_grad():
            prefix_embeds = model.get_input_embeddings()(prefix_inputs.input_ids)

        # 将缓存的提示嵌入扩展到当前批次大小，并与前缀嵌入拼接
        expanded_prompt_embeds = prompt_embeds.expand(current_batch_size, -1, -1)
        combined_embeds = torch.cat([expanded_prompt_embeds, prefix_embeds], dim=1)

        # 创建对应的注意力掩码
        prompt_attention_mask = torch.ones(current_batch_size, prompt_len, device=device, dtype=torch.long)
        combined_attention_mask = torch.cat([prompt_attention_mask, prefix_inputs.attention_mask], dim=1)

        # 准备传递给 model.generate 的参数字典
        generate_args = {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_attention_mask,
        }

    else:
        # --- 路径 B: 无缓存 ---
        # 用于处理良性样本，或每个提示组的第一个批次。
        conversations = [json.loads(conv_str) for conv_str in batch_df['conversation']]
        templated_inputs = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(templated_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        generate_args = inputs

    # 使用 model.generate 一次性完成生成和隐藏状态的提取
    with torch.no_grad():
        outputs = model.generate(
            **generate_args,
            max_new_tokens=generation_config['max_new_tokens'],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    # `prompt_hidden_states` 包含了输入部分（提示+前缀）在所有层的隐藏状态
    early_window_hidden_states = outputs.prompt_hidden_states
    # `hidden_states` 包含了新生成部分在所有层的隐藏状态
    generated_hidden_states_raw = outputs.hidden_states

    # --- 通用的激活提取逻辑 ---

    # 将生成步骤中分散的隐藏状态堆叠起来，方便后续处理
    if generated_hidden_states_raw:
        num_gen_steps = len(generated_hidden_states_raw)
        num_layers = len(generated_hidden_states_raw[0])
        generated_hidden_states_stacked = [
            torch.cat([generated_hidden_states_raw[t][l] for t in range(num_gen_steps)], dim=1)
            for l in range(num_layers)
        ]
    else:
        generated_hidden_states_stacked = None

    batch_activations = {layer: {'early_window': [], 'content_window': []} for layer in extraction_config['layers']}
    prefixes = batch_df['prefix'].tolist() if 'prefix' in batch_df.columns else [None] * len(batch_df)

    # 逐个样本处理，精确切分出早期窗口和内容窗口的激活
    for i in range(current_batch_size):
        prefix_text = prefixes[i]

        # 根据是否使用缓存，计算前缀的 token 长度和起始索引
        if prompt_embeds is not None:  # 缓存路径
            prefix_input_ids = tokenizer(final_assistant_texts[i], add_special_tokens=False)['input_ids']
            prefix_len = len(prefix_input_ids)
            prefix_start_idx = prompt_len  # 前缀紧跟在提示后面
        else:  # 非缓存路径
            prefix_len = len(tokenizer.encode(prefix_text, add_special_tokens=False)) if prefix_text else 0
            hs_len = early_window_hidden_states[0][i].shape[0] if early_window_hidden_states else 0
            prefix_start_idx = hs_len - prefix_len  # 前缀是输入部分的最后几个 token

        # 遍历需要提取的层
        for layer_idx in extraction_config['layers']:
            # 提取早期窗口 (W_early) 的激活
            if prefix_len > 0 and early_window_hidden_states:
                hs_sample = early_window_hidden_states[layer_idx][i]
                if prefix_start_idx >= 0:
                    prefix_end_idx = prefix_start_idx + prefix_len
                    early_window_states = hs_sample[prefix_start_idx:prefix_end_idx].unsqueeze(0)
                    agg_early = aggregate_activations(early_window_states, extraction_config['aggregation'])
                    batch_activations[layer_idx]['early_window'].append(agg_early.cpu())

            # 提取内容窗口 (W_cont) 的激活
            if generated_hidden_states_stacked:
                gen_hs_sample = generated_hidden_states_stacked[layer_idx][i]
                content_window_states = gen_hs_sample.unsqueeze(0)
                if content_window_states.shape[1] > 0:
                    agg_content = aggregate_activations(content_window_states, extraction_config['aggregation'])
                    batch_activations[layer_idx]['content_window'].append(agg_content.cpu())

    return batch_activations


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
    output_dir = base_dir / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_name = config['dataset_llm_name']

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

    # 定义要处理的数据集
    datasets_to_process = {
        "compliance": processed_data_dir / f"{llm_name}_jailbreak_compliance.csv",
        "refusal": processed_data_dir / f"{llm_name}_jailbreak_refusal.csv",
        "benign": processed_data_dir / "benign_prompts.csv",
    }

    # --- 3. 遍历并处理每个数据集 ---
    for name, path in datasets_to_process.items():
        if not path.is_file():
            logging.warning(f"未找到 '{name}' 的数据集文件，路径: {path}，正在跳过。")
            continue

        logging.info(f"--- 正在处理数据集: {name} ---")
        df = pd.read_csv(path)
        total_activations = {layer: {'early_window': [], 'content_window': []} for layer in config['extraction']['layers']}
        batch_size = config['processing']['batch_size']

        # 判断是否应该使用基于 prompt 分组的嵌入缓存优化
        use_grouping_cache = 'prompt' in df.columns and name in ["compliance", "refusal"]

        # 按 'prompt' 分组，如果不需要缓存则创建一个包含所有数据的伪分组
        grouped = df.groupby('prompt') if use_grouping_cache else [('all', df)]
        progress_bar = tqdm(total=len(df), desc=f"正在提取 {name}")

        # --- 4. 按分组处理 ---
        for prompt, group_df in grouped:
            prompt_embeds = None
            prompt_len = 0

            # --- 5. 在分组内按批次处理 ---
            for i in range(0, len(group_df), batch_size):
                batch_df = group_df.iloc[i:i + batch_size]

                # 如果启用缓存且这是该组的第一个批次，则计算并缓存提示嵌入
                if use_grouping_cache and prompt_embeds is None:
                    user_conversation = [{"role": "user", "content": prompt}]
                    prompt_template = tokenizer.apply_chat_template(user_conversation, tokenize=False, add_generation_prompt=False)
                    prompt_inputs = tokenizer(prompt_template, return_tensors="pt").to(config['processing']['device'])
                    with torch.no_grad():
                        # 计算一次性的提示嵌入和长度
                        prompt_embeds = model.get_input_embeddings()(prompt_inputs.input_ids)
                        prompt_len = prompt_inputs.input_ids.shape[1]

                # 调用核心处理函数
                batch_activations = process_batch(
                    batch_df, model, tokenizer,
                    config['generation'], config['extraction'], config['processing']['device'],
                    prompt_embeds=prompt_embeds if use_grouping_cache else None,
                    prompt_len=prompt_len if use_grouping_cache else 0,
                )

                # 收集当前批次的结果
                for layer_idx, windows in batch_activations.items():
                    if windows['early_window']: total_activations[layer_idx]['early_window'].extend(windows['early_window'])
                    if windows['content_window']: total_activations[layer_idx]['content_window'].extend(windows['content_window'])

                progress_bar.update(len(batch_df))

        progress_bar.close()

        # --- 6. 聚合结果并保存 ---
        final_tensors = {}
        for layer_idx, windows in total_activations.items():
            final_tensors[layer_idx] = {}
            if windows['early_window']: final_tensors[layer_idx]['early_window'] = torch.cat(windows['early_window'], dim=0)
            if windows['content_window']: final_tensors[layer_idx]['content_window'] = torch.cat(windows['content_window'], dim=0)

        output_path = output_dir / f"{llm_name}_{name}_activations.pt"
        torch.save(final_tensors, output_path)
        logging.info(f"成功将 '{name}' 的激活保存到 {output_path}")

    logging.info("所有数据集的激活提取过程已完成。")


if __name__ == "__main__":
    main()

