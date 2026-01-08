# -*- coding: utf-8 -*-
"""
脚本 01: 数据集准备与切片

该脚本遵循“方法流程.md”文档中的第一阶段，用于准备训练表征向量所需的数据集。

主要功能:
1.  加载原始的越狱提示数据集（例如，JSONL 格式）。
2.  根据配置文件中定义的“满足”（compliance）和“拒绝”（refusal）前缀列表。
3.  为每个越狱提示，生成两种类型的对话样本：
    - A1 (满足型): 用户提示 + 模型满足的前缀。
    - A2 (拒绝型): 用户提示 + 模型拒绝的前缀。
4.  处理良性（benign）数据集，将其与“满足”前缀配对，生成 B1 (良性-满足) 样本。
    - 支持 CSV 格式（读取 'prompt' 或 'Goal' 列）。
    - 支持 JSONL 格式（特定支持 justeval 结构，筛选 source_id!=alpaca_eval 且 category==regular）。
5.  将所有生成的对话样本以结构化的 CSV 格式保存到处理后的数据目录中。
    - 每个样本包含原始提示、所用前缀、类别以及符合模型输入的对话结构（对话结构将作为JSON字符串存储）。


如何运行:
python scripts/01_prepare_datasets.py --config configs/data_prep_config.yaml
"""

import json
import argparse
import csv
from pathlib import Path
import yaml
from tqdm import tqdm
import pandas as pd
import random


def create_paired_samples(
        prompt: str,
        behavior: str,
        functional_category: str,
        context_string: str,
        compliance_prefixes: list,
        refusal_prefixes: list,
        compliance_category: str = "compliance",
        refusal_category: str = "refusal"
) -> tuple[list, list]:
    """
    为给定的提示词创建成对的样本。

    Args:
        prompt (str): 用户的原始提示。
        behavior (str): 行为描述。
        functional_category (str): 功能类别。
        context_string (str): 上下文信息字符串。
        compliance_prefixes (list): 表示模型满足意图的前缀列表。
        refusal_prefixes (list): 表示模型拒绝意图的前缀列表。
        compliance_category (str): 满足型样本的类别标签。
        refusal_category (str): 拒绝型样本的类别标签。

    Returns:
        tuple[list, list]: 一个元组，包含两个列表：(满足样本列表, 拒绝样本列表)。
                           每个样本都是一个包含对话历史的字典。
    """
    compliance_samples = []
    refusal_samples = []

    # --- 创建满足样本 ---
    for prefix in compliance_prefixes:
        # 构建符合模型对话模板的输入结构
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix}
        ]
        compliance_samples.append({
            "prompt": prompt,
            "prefix": prefix,
            "category": compliance_category,
            "conversation": conversation,
            "behavior": behavior,
            "FunctionalCategory": functional_category,
            "ContextString": context_string,
        })

    # --- 创建拒绝样本 ---
    for prefix in refusal_prefixes:
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix}
        ]
        refusal_samples.append({
            "prompt": prompt,
            "prefix": prefix,
            "category": refusal_category,
            "conversation": conversation,
            "behavior": behavior,
            "FunctionalCategory": functional_category,
            "ContextString": context_string,
        })

    return compliance_samples, refusal_samples


def process_dataset(config: dict):
    """
    根据配置文件处理所有指定的数据集。

    Args:
        config (dict): 从 YAML 文件加载的配置字典。
    """
    config_path = Path(config['__config_path__'])
    base_dir = config_path.parent.parent
    output_dir = base_dir / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"所有处理后的文件将保存在: {output_dir.resolve()}")

    # --- 统一加载前缀列表 ---
    prefixes_config = config.get('prefixes', {})
    source_file = prefixes_config.get('source_file')
    compliance_prefixes = []
    refusal_prefixes = []

    if source_file:
        source_file_path = base_dir / source_file
        print(f"从文件加载前缀: {source_file_path.resolve()}")
        try:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                prefix_data = json.load(f)

            compliance_key = prefixes_config.get('compliance_key')
            refusal_key = prefixes_config.get('refusal_key')

            if not compliance_key or not refusal_key:
                raise ValueError("当指定 'source_file' 时，配置文件中必须同时提供 'compliance_key' 和 'refusal_key'。")

            compliance_prefixes = prefix_data.get(compliance_key, [])
            refusal_prefixes = prefix_data.get(refusal_key, [])
            print(f"加载了 {len(compliance_prefixes)} 条满足型前缀和 {len(refusal_prefixes)} 条拒绝型前缀。")

        except FileNotFoundError:
            print(f"ERROR: 前缀源文件未找到: {source_file_path.resolve()}")
            return
        except (json.JSONDecodeError, KeyError) as e:
            print(f"ERROR: 解析前缀文件 {source_file_path.resolve()} 或查找指定的键时出错: {e}")
            return
    else:
        # 如果未指定源文件，则尝试从配置中直接读取（旧版兼容）
        print("从配置文件直接加载前缀列表...")
        compliance_prefixes = prefixes_config.get('compliance', [])
        refusal_prefixes = prefixes_config.get('refusal', [])
        print(f"加载了 {len(compliance_prefixes)} 条满足型前缀和 {len(refusal_prefixes)} 条拒绝型前缀。")

    # --- 处理越狱数据集 ---
    if 'jailbreak_dataset' in config:
        if not compliance_prefixes or not refusal_prefixes:
            raise ValueError("处理越狱数据集时，必须成功加载 'compliance' 和 'refusal' 前缀列表。")

        print("开始处理越狱数据集...")
        jb_config = config['jailbreak_dataset']
        input_path = base_dir / jb_config['input_file']

        compliance_output_path = output_dir / f"{config['llm_name']}_jailbreak_compliance.csv"
        refusal_output_path = output_dir / f"{config['llm_name']}_jailbreak_refusal.csv"

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records = data["records"]
        except FileNotFoundError:
            print(f"ERROR: 输入文件未找到: {input_path.resolve()}")
            return
        except (json.JSONDecodeError, KeyError) as e:
            print(f"ERROR: 解析文件 {input_path.resolve()} 时出错: {e}. 请确保文件是有效的 JSON 格式，且'records'列表下每项都有 'prompt' 键。")
            return

        print(f"从 {input_path.resolve()} 加载了 {len(records)} 条越狱提示。")

        with open(compliance_output_path, 'w', encoding='utf-8-sig', newline='') as f_comply, \
                open(refusal_output_path, 'w', encoding='utf-8-sig', newline='') as f_refuse:

            comply_writer = csv.writer(f_comply)
            refuse_writer = csv.writer(f_refuse)

            header = ["prompt", "prefix", "category", "conversation", "behavior", "FunctionalCategory", "ContextString"]
            comply_writer.writerow(header)
            refuse_writer.writerow(header)

            for record in tqdm(records, desc="处理越狱提示中"):
                prompt = record["prompt"]
                behavior = record["behavior"]
                functional_category = record["FunctionalCategory"]
                context_string = record["ContextString"] if functional_category == 'contextual' else ""

                compliance_samples, refusal_samples = create_paired_samples(
                    prompt,
                    behavior,
                    functional_category,
                    context_string,
                    compliance_prefixes,
                    refusal_prefixes
                )
                for sample in compliance_samples:
                    conversation_str = json.dumps(sample['conversation'], ensure_ascii=False)
                    row = [sample['prompt'], sample['prefix'], sample['category'], conversation_str, sample['behavior'], sample['FunctionalCategory'],
                           sample['ContextString']]
                    comply_writer.writerow(row)
                for sample in refusal_samples:
                    conversation_str = json.dumps(sample['conversation'], ensure_ascii=False)
                    row = [sample['prompt'], sample['prefix'], sample['category'], conversation_str, sample['behavior'], sample['FunctionalCategory'],
                           sample['ContextString']]
                    refuse_writer.writerow(row)

        print(f"成功将满足型 (A1) 样本写入: {compliance_output_path.resolve()}")
        print(f"成功将拒绝型 (A2) 样本写入: {refusal_output_path.resolve()}")

    # --- 处理良性数据集 ---
    if 'benign_dataset' in config and config['benign_dataset'].get('input_files') is not None:
        if not compliance_prefixes:
            raise ValueError("处理良性数据集时，必须成功加载 'compliance' 前缀列表。")

        print("开始处理良性数据集...")
        benign_config = config['benign_dataset']
        input_files = benign_config['input_files']
        sample_size = benign_config.get('sample_size', 100)
        output_path = output_dir / f"{config['llm_name']}_benign_compliance.csv"

        all_prompts = []
        for file_info in input_files:
            file_path = base_dir / Path(file_info)
            prompts_from_file = []

            try:
                # 检查文件后缀，区分处理逻辑
                if file_path.suffix.lower() == '.jsonl':
                    # 处理 JSONL 格式 (专门针对 justeval 类型的结构)
                    print(f"检测到 JSONL 文件，按 JustEval 结构处理: {file_path.resolve()}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    candidates = []
                    for line in lines:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            # 筛选条件:
                            # 1. source_id 不包含 'alpaca_eval'
                            # 2. category 为 'regular'
                            source_id = str(item.get("source_id", ""))
                            category = str(item.get("category", ""))

                            if "alpaca_eval" not in source_id and category == "regular":
                                # 提取 prompt，优先使用 'instruction'，其次 'prompt'
                                p_text = item.get("instruction") or item.get("prompt")
                                if p_text:
                                    candidates.append(str(p_text))
                        except json.JSONDecodeError:
                            print(f"WARNING: 跳过无法解析的 JSON 行 in {file_path.name}")
                            continue

                    # 对筛选后的数据进行采样
                    if candidates:
                        num_candidates = len(candidates)
                        actual_sample_size = min(sample_size, num_candidates)
                        sampled_list = random.sample(candidates, actual_sample_size)
                        for p in sampled_list:
                            prompts_from_file.append((p, file_path.name))
                        print(f"从 {file_path.resolve()} 筛选出 {num_candidates} 条有效样本，并采样了 {len(prompts_from_file)} 条。")
                    else:
                        print(f"WARNING: 在 {file_path.name} 中未找到符合条件的样本 (source_id!=alpaca_eval, category==regular)。")

                else:
                    # 默认处理 CSV 格式 (原有逻辑)
                    df = pd.read_csv(file_path)
                    prompt_column = 'prompt'
                    if prompt_column not in df.columns:
                        if 'Goal' in df.columns:
                            prompt_column = 'Goal'
                        else:
                            raise ValueError(f"在 {file_path} 中找不到合适的提示列 ('prompt' or 'Goal')。")

                    num_rows = len(df)
                    if num_rows > 0:
                        actual_sample_size = min(sample_size, num_rows)
                        sampled_df = df.sample(n=actual_sample_size, random_state=42)
                        for _, row in sampled_df.iterrows():
                            prompt = row[prompt_column]
                            if pd.notna(prompt):
                                # 保存 prompt 和其来源文件名
                                prompts_from_file.append((str(prompt), file_path.name))
                    print(f"从 {file_path.resolve()} (CSV) 成功采样 {len(prompts_from_file)} 条良性提示。")

                all_prompts.extend(prompts_from_file)

            except FileNotFoundError:
                print(f"ERROR: 输入文件未找到: {file_path.resolve()}")
            except Exception as e:
                print(f"ERROR: 处理文件 {file_path.resolve()} 时出错: {e}")

        if all_prompts:
            # 再次打乱所有来源的样本
            random.shuffle(all_prompts)
            print(f"总共收集 {len(all_prompts)} 条良性提示进行处理。")

            with open(output_path, 'w', encoding='utf-8-sig', newline='') as f_out:
                header = ["prompt", "prefix", "category", "conversation", "source"]
                writer = csv.writer(f_out)
                writer.writerow(header)

                total_samples_written = 0
                for prompt, source in tqdm(all_prompts, desc="处理良性提示中"):
                    compliance_samples, _ = create_paired_samples(
                        prompt=prompt,
                        behavior="",
                        functional_category="",
                        context_string="",
                        compliance_prefixes=compliance_prefixes,
                        refusal_prefixes=[],  # 良性样本不需要与拒绝前缀配对
                        compliance_category="benign_compliance"
                    )

                    for sample in compliance_samples:
                        conversation_str = json.dumps(sample['conversation'], ensure_ascii=False)
                        row = [sample['prompt'], sample['prefix'], sample['category'], conversation_str, source]
                        writer.writerow(row)
                        total_samples_written += 1

            print(f"成功将 {total_samples_written} 条良性满足型 (B1) 样本写入: {output_path.resolve()}")


def main():
    """
    主函数，用于解析命令行参数并启动数据集处理流程。
    """
    parser = argparse.ArgumentParser(
        description="根据配置文件准备用于表征工程的数据集。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/data_prep_config.yaml',
        help='指定数据准备阶段的 YAML 配置文件路径。\n默认: ../configs/data_prep_config.yaml'
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: 配置文件不存在: {config_path.resolve()}")
        return

    print(f"加载配置文件: {config_path.resolve()}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        config['__config_path__'] = str(config_path.resolve())  # 将配置文件路径注入，方便计算相对路径

    process_dataset(config)

    print("数据集准备流程全部完成。")


if __name__ == "__main__":
    main()