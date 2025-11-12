# -*- coding: utf-8 -*-
"""
脚本 03: 提取表征向量并计算 Token 分数

该脚本遵循“方法流程.md”文档中的第二和第三阶段，并为第四阶段准备数据。

主要功能:
1.  **向量提取**:
    - 加载由 `02_extract_activations.py` 提取的聚合后激活 (格式: {layer: {window: tensor}})。
    - 加载由 `02.5_judge_harmfulness.py` 生成的有害性标签。
    - 过滤样本，仅保留有害的 "compliance" (A1) 和无害的 "refusal" (A2) 样本。
    - 为“早期窗口”和“内容窗口”分别计算白化/中心化变换。
    - 提取干预向量 `v_l` (拒绝行为) 和条件向量 `c_l` (有害内容)。
    - (可选) 对向量进行去耦合和可视化。
    - 保存变换矩阵和提取的向量。
2.  **分数计算 (优化版)**:
    - 在向量提取后，加载基础语言模型。
    - **高效地**重新处理所有过滤后的样本 (A1, A2, B1)，利用已有的 `assistant_output`。
    - 将 `prompt` 和 `assistant_output` 构建为完整对话，并执行一次前向传播。
    - 对每个样本，获取其 `assistant_output` 部分每个 token 的隐藏状态。
    - 使用已提取的条件向量 `c_l` 和变换，计算逐 token 的原始分数 `s_t`。
    - 将所有样本的逐 token 分数序列按数据集类型分别保存到 .pt 文件，以供 `04_calibrate_defense.py` 使用。

如何运行:
python scripts/03_extract_vectors.py --config configs/PCA_config.yaml
"""

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_filter_data(data_dir: Path, llm_name: str, dataset_name: str) -> tuple[dict, pd.DataFrame]:
    """
    加载激活张量和对应的带标签的 CSV 文件，并根据标签进行过滤。
    此版本适配已简化的激活数据结构 {layer: {window: tensor}}。
    """
    activations_path = data_dir / f"{llm_name}_{dataset_name}_activations.pt"
    outputs_path = data_dir / f"{llm_name}_{dataset_name}_outputs.csv"

    if not activations_path.exists() or not outputs_path.exists():
        logging.warning(f"未找到 {dataset_name} 的激活或输出文件，跳过加载。路径: {activations_path}")
        return None, None

    logging.info(f"正在加载 {dataset_name} 数据...")
    activations = torch.load(activations_path, map_location='cpu')
    df = pd.read_csv(outputs_path)

    if dataset_name == "benign":
        # 对于良性数据，不需要过滤，直接返回加载的激活和df
        return activations, df

    # --- 对 compliance 和 refusal 数据进行过滤 ---
    if dataset_name == "compliance":
        target_label = "yes"
    elif dataset_name == "refusal":
        target_label = "no"
    else:
        # 意外情况，作为安全保护
        return activations, df

    initial_count = len(df)
    df.dropna(subset=['label'], inplace=True)
    valid_indices = df.index[df['label'] == target_label].tolist()

    df_filtered = df.loc[valid_indices].reset_index(drop=True)

    activations_filtered = {}
    for layer, windows in activations.items():
        activations_filtered[layer] = {}
        for window_name, tensor in windows.items():
            if tensor is None:
                continue

            if tensor.shape[0] != initial_count:
                logging.warning(
                    f"在 {dataset_name} (L{layer}, {window_name}) 中，激活数量 ({tensor.shape[0]}) 与CSV行数 ({initial_count}) 不匹配。跳过此张量。")
                continue

            activations_filtered[layer][window_name] = tensor[valid_indices]

    logging.info(f"对于 {dataset_name}，从 {initial_count} 个样本中过滤出 {len(df_filtered)} 个标签为 '{target_label}' 的样本。")
    return activations_filtered, df_filtered


def calculate_dual_transforms(benign_activations: dict, whitening_config: dict) -> tuple[dict, dict]:
    """
    根据良性激活计算用于“行为”(early_window) 和“语义”(content_window) 的两套变换。
    变换包括均值（用于中心化）和可选的白化矩阵。
    """
    whitening_enabled = whitening_config.get('enabled', True)
    epsilon = whitening_config.get('epsilon', 1e-4)

    if whitening_enabled:
        logging.info("正在计算双白化变换 (early/content)...")
    else:
        logging.info("白化被禁用。仅计算用于中心化的均值...")

    transforms_early = {}
    transforms_cont = {}

    for layer in tqdm(benign_activations.keys(), desc="计算变换矩阵"):
        for window_name in ['early_window', 'content_window']:
            if window_name in benign_activations[layer] and benign_activations[layer][window_name].shape[0] > 1:
                H_benign = benign_activations[layer][window_name].to(torch.float32).cuda()
                mu = H_benign.mean(dim=0)
                W = None
                W_inv = None

                if whitening_enabled:
                    H_centered = H_benign - mu
                    cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1)
                    cov_reg = cov + torch.eye(cov.shape[0]).cuda() * epsilon
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov_reg)

                    if (eigenvalues < 0).any():
                        logging.warning(f"在第 {layer} 层 ({window_name}) 发现负特征值。最小特征值: {eigenvalues.min().item()}")
                        eigenvalues = torch.clamp(eigenvalues, min=1e-6)

                    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues))
                    W = D_inv_sqrt @ eigenvectors.T

                    D_sqrt = torch.diag(torch.sqrt(eigenvalues))
                    W_inv = eigenvectors @ D_sqrt

                transform_tuple = (
                    W.cpu() if W is not None else None,
                    mu.cpu(),
                    W_inv.cpu() if W_inv is not None else None
                )
                if window_name == 'early_window':
                    transforms_early[layer] = transform_tuple
                else:
                    transforms_cont[layer] = transform_tuple
            else:
                logging.warning(f"第 {layer} 层缺少 '{window_name}' 的良性激活，无法计算该窗口的变换。")

    return transforms_early, transforms_cont


def apply_transform(activations: torch.Tensor, transform: tuple) -> torch.Tensor:
    """
    将变换（中心化和可选的白化）应用于给定的激活张量。
    支持 2D (N, D) 或 3D (B, N, D) 张量。
    """
    W, mu, _ = transform
    # 确保在同一设备上操作
    device = activations.device
    mu_device = mu.to(device)

    centered = activations - mu_device

    if W is not None:
        W_device = W.to(device)
        # 使用 einsum 以支持 2D 和 3D
        if activations.dim() == 2:
            transformed = torch.einsum('nd,cd->nc', centered, W_device)
        elif activations.dim() == 3:
            transformed = torch.einsum('bnd,cd->bnc', centered, W_device)
        else:
            raise ValueError(f"不支持的激活维度: {activations.dim()}")
        return transformed
    else:  # 仅中心化
        return centered


def get_diff_vector(pos_activations: torch.Tensor, neg_activations: torch.Tensor) -> torch.Tensor:
    """
    通过直接计算均值差并归一化来提取方向向量。
    """
    mean_diff = pos_activations.mean(dim=0) - neg_activations.mean(dim=0)
    norm = torch.norm(mean_diff)
    if norm == 0:
        logging.warning("均值差分向量的范数为零，无法归一化。返回零向量。")
        return torch.zeros_like(mean_diff)
    vector = mean_diff / norm
    return vector


def plot_pca_visualizations(
        llm_name: str,
        window_name: str,
        layers: list,
        activations: dict,
        vectors: dict,
        output_dir: Path,
        whitening_enabled: bool,
        sample_size: int = 200
):
    """
    对指定窗口的激活进行PCA降维并生成可视化图表。
    """
    logging.info(f"正在为 '{window_name}' 生成PCA可视化图...")
    sns.set_theme(style="whitegrid", palette="deep")

    all_colors = {
        'Benign': '#2ca02c',
        'Refusal': '#1f77b4',
        'Compliance': '#d62728',
    }
    vec_colors = {
        'c_vector': '#9467bd',
        'v_vector': '#ff7f0e'
    }

    n_layers = len(layers)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axes = axes.flatten()

    for i, layer in enumerate(tqdm(layers, desc=f"绘制 {window_name}")):
        ax = axes[i]

        z_compliance = activations[layer].get('compliance')
        z_refusal = activations[layer].get('refusal')
        z_benign = activations[layer].get('benign')

        data_to_plot = {}
        plot_colors = {}

        if window_name == 'early_window':
            if z_compliance is not None and len(z_compliance) > 0: data_to_plot['Compliance'] = z_compliance
            if z_refusal is not None and len(z_refusal) > 0: data_to_plot['Refusal'] = z_refusal
        elif window_name == 'content_window':
            if z_compliance is not None and len(z_compliance) > 0: data_to_plot['Compliance'] = z_compliance
            if z_benign is not None and len(z_benign) > 0: data_to_plot['Benign'] = z_benign

        if len(data_to_plot) < 2:
            logging.warning(f"在第 {layer} 层 ('{window_name}') 数据不足，无法生成图表。")
            ax.text(0.5, 0.5, f'Layer {layer}\nInsufficient Data', ha='center', va='center', fontsize=9)
            ax.set_xticks([]);
            ax.set_yticks([])
            continue

        min_samples = min(len(v) for v in data_to_plot.values())
        plot_sample_size = min(sample_size, min_samples) if sample_size > 0 else min_samples
        sampled_data = {k: v[torch.randperm(v.size(0))[:plot_sample_size]] for k, v in data_to_plot.items()}
        all_preprocessed = torch.cat(list(sampled_data.values()), dim=0)

        pca = PCA(n_components=2)
        pca.fit(all_preprocessed.numpy())

        proj_dfs = [pd.DataFrame(pca.transform(t.numpy()), columns=['PC1', 'PC2']).assign(Category=c) for c, t in sampled_data.items()]
        proj_df = pd.concat(proj_dfs, ignore_index=True)

        current_plot_colors = {k: all_colors[k] for k in data_to_plot.keys()}
        sns.scatterplot(data=proj_df, x='PC1', y='PC2', hue='Category', palette=current_plot_colors, ax=ax, alpha=0.7, s=20)
        ax.get_legend().remove()

        v_l = vectors[layer].get('v')
        c_l = vectors[layer].get('c')
        axis_width = ax.get_xlim()[1] - ax.get_xlim()[0]
        arrow_length = axis_width * 0.3

        if v_l is not None and torch.norm(v_l) > 0:
            proj_v = (v_l.numpy().reshape(1, -1)) @ pca.components_.T
            proj_v_norm = np.linalg.norm(proj_v)
            if proj_v_norm > 1e-9:
                scaled_proj_v = (proj_v / proj_v_norm) * arrow_length
                ax.quiver(0, 0, scaled_proj_v[0, 0], scaled_proj_v[0, 1], color=vec_colors['v_vector'], scale=1, scale_units='xy', angles='xy',
                          width=0.01, label=r'$v_l$')
        if c_l is not None and torch.norm(c_l) > 0:
            proj_c = (c_l.numpy().reshape(1, -1)) @ pca.components_.T
            proj_c_norm = np.linalg.norm(proj_c)
            if proj_c_norm > 1e-9:
                scaled_proj_c = (proj_c / proj_c_norm) * arrow_length
                ax.quiver(0, 0, scaled_proj_c[0, 0], scaled_proj_c[0, 1], color=vec_colors['c_vector'], scale=1, scale_units='xy', angles='xy',
                          width=0.01, label=r'$c_l$')

        ax.set_title(f"Layer {layer}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    first_valid_ax = next((ax for ax in axes if ax.has_data()), None)
    if first_valid_ax:
        handles, labels = first_valid_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.01))

    title_prefix = "Whitened" if whitening_enabled else "Centered"
    fig.suptitle(f'PCA of {title_prefix} Activations ({window_name}) for {llm_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_path = output_dir / f"pca_visualization_{window_name}.png"
    plt.savefig(output_path, dpi=300)
    logging.info(f"PCA 可视化图已保存到: {output_path}")
    plt.close(fig)


def find_subsequence(main_list: List[int], sub_list: List[int]) -> int:
    """
    在主列表中查找子列表的起始索引。
    """
    main_len = len(main_list)
    sub_len = len(sub_list)
    for i in range(main_len - sub_len + 1):
        if main_list[i:i + sub_len] == sub_list:
            return i
    return -1


def get_model_and_tokenizer(model_path: str, model_kwargs: dict, device: str):
    """ 加载 Hugging Face 模型和分词器。"""
    logging.info(f"正在加载模型 '{model_path}' 用于分数计算...")
    kwargs = model_kwargs.copy()
    if "torch_dtype" in kwargs and isinstance(kwargs["torch_dtype"], str):
        try:
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
        except AttributeError:
            if kwargs["torch_dtype"] != "auto":
                raise ValueError(f"无效的 torch_dtype: {kwargs['torch_dtype']}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    logging.info("模型和分词器加载成功。")
    return model, tokenizer


def calculate_and_save_token_scores(
    datasets_to_score: Dict[str, pd.DataFrame],
    model,
    tokenizer,
    transforms: Dict,
    condition_vectors: Dict,
    config: Dict,
    output_dir
):
    """
    高效地重新处理样本，计算并保存逐 token 的分数，不再使用 model.generate()。
    此版本使用子序列搜索来精确定位 assistant_output 的位置。
    """
    llm_name = config['llm_name']
    device = config['processing']['device']
    batch_size = config.get('processing', {}).get('batch_size', 4)

    layers = sorted(condition_vectors.keys())
    transforms_cont = transforms['content_window']

    for name, df in datasets_to_score.items():
        if df is None or df.empty:
            logging.info(f"数据集 '{name}' 为空，跳过分数计算。")
            continue

        logging.info(f"--- 正在为 '{name}' ({len(df)} 个样本) 计算逐 token 分数 ---")

        all_scores = {layer: [] for layer in layers}

        for i in tqdm(range(0, len(df), batch_size), desc=f"正在为 {name} 评分"):
            batch_df = df.iloc[i:i + batch_size]

            full_input_texts = []
            output_token_sequences = []

            for idx, row in batch_df.iterrows():
                prompt = str(row['prompt']) if pd.notna(row['prompt']) else ""
                assistant_output = str(row['assistant_output']) if pd.notna(row['assistant_output']) else ""
                assistant_output = assistant_output.rstrip()

                # 1. 构建完整的对话历史
                full_conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_output}
                ]
                full_text = tokenizer.apply_chat_template(full_conversation, tokenize=False, add_generation_prompt=False)
                full_input_texts.append(full_text)

                # 2. 单独对 assistant_output 分词，用于后续搜索
                output_ids = tokenizer(assistant_output, add_special_tokens=False).input_ids
                output_token_sequences.append(output_ids)

            # 3. 批量分词完整对话
            inputs = tokenizer(full_input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

            # 4. 查找每个 assistant_output 在完整序列中的起始位置
            output_start_indices = []
            for batch_idx in range(len(batch_df)):
                full_ids_list = inputs.input_ids[batch_idx].tolist()
                output_ids_list = output_token_sequences[batch_idx]

                start_idx = find_subsequence(full_ids_list, output_ids_list)
                output_start_indices.append(start_idx)


            # 5. 执行一次前向传播
            with torch.no_grad():
                full_outputs = model(**inputs, output_hidden_states=True)
                all_hidden_states = full_outputs.hidden_states

            # 6. 逐层、逐样本计算分数
            for layer_idx in layers:
                if layer_idx not in transforms_cont or layer_idx not in condition_vectors:
                    continue

                layer_hidden_states = all_hidden_states[layer_idx].to(torch.float32)
                transform = transforms_cont[layer_idx]
                c_vector = condition_vectors[layer_idx].to(device, dtype=torch.float32)

                for batch_idx in range(len(batch_df)):
                    start_idx = output_start_indices[batch_idx]
                    output_len = len(output_token_sequences[batch_idx])

                    if start_idx == -1:
                        # 如果未找到或输出为空，则分数也为空
                        if output_len > 0:
                            logging.warning(f"无法为样本 {df.index[i + batch_idx]} 定位 assistant_output token 序列。")
                        scores = torch.tensor([], dtype=torch.float32)
                    else:
                        end_idx = start_idx + output_len
                        # 切片操作会自动处理边界，无需手动 min
                        output_hidden_states = layer_hidden_states[batch_idx, start_idx:end_idx, :]

                        if output_hidden_states.shape[0] > 0:
                            transformed_activations = apply_transform(output_hidden_states, transform)
                            scores = torch.einsum('sd,d->s', transformed_activations, c_vector)
                        else:
                            scores = torch.tensor([], dtype=torch.float32)

                    all_scores[layer_idx].append(scores.cpu())

            # 7. 清理内存
            del full_outputs, all_hidden_states, inputs
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 8. 保存分数
        scores_save_path = output_dir / f"{llm_name}_{name}_token_scores.pt"
        torch.save(all_scores, scores_save_path)
        logging.info(f"已将 '{name}' 的逐 token 分数保存到 {scores_save_path}")


def main():
    parser = argparse.ArgumentParser(description="从模型激活中提取干预向量和条件向量。")
    parser.add_argument('--config', type=str, default='../configs/PCA_config.yaml',
                        help='向量提取阶段的 YAML 配置文件路径。')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        logging.error(f"配置文件未找到: {config_path}")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_dir = Path(__file__).parent.parent
    llm_name = config['llm_name']
    data_dir = base_dir / config['data_dir'] / llm_name
    output_dir = base_dir / config['output_dir'] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"所有产物将保存到: {output_dir}")

    compliance_activations, compliance_df = load_and_filter_data(data_dir, llm_name, "compliance")
    refusal_activations, refusal_df = load_and_filter_data(data_dir, llm_name, "refusal")
    benign_activations, benign_df = load_and_filter_data(data_dir, llm_name, "benign")

    if not compliance_activations or not refusal_activations or not benign_activations:
        logging.error("缺少必要的数据集，无法继续。请先运行 02 和 02.5 脚本。")
        return

    whitening_config = config.get('whitening', {})
    whitening_enabled = whitening_config.get('enabled', True)
    transforms_early, transforms_cont = calculate_dual_transforms(
        benign_activations,
        whitening_config
    )

    all_transforms = {
        'early_window': transforms_early,
        'content_window': transforms_cont,
    }
    torch.save(all_transforms, output_dir / "transforms.pt")
    logging.info(f"变换矩阵 (均值/白化) 已保存到 {output_dir / 'transforms.pt'}")

    intervention_vectors, condition_vectors = {}, {}
    layers = sorted(list(benign_activations.keys()))

    preprocessed_activations_for_plot = {
        'early_window': {layer: {} for layer in layers},
        'content_window': {layer: {} for layer in layers}
    }
    all_vectors_for_plot = {layer: {} for layer in layers}

    log_balancing_info = True

    for layer in tqdm(layers, desc="提取向量"):
        transform_early = transforms_early.get(layer)
        transform_cont = transforms_cont.get(layer)

        if not transform_early or not transform_cont:
            logging.warning(f"第 {layer} 层缺少变换矩阵，跳过向量提取。")
            continue

        H_refusal_early = refusal_activations.get(layer, {}).get('early_window')
        H_compliance_early = compliance_activations.get(layer, {}).get('early_window')
        H_benign_early = benign_activations.get(layer, {}).get('early_window')

        if H_refusal_early is not None and H_compliance_early is not None and H_benign_early is not None:
            z_refusal_early = apply_transform(H_refusal_early.cuda(), transform_early).cpu()
            z_compliance_early = apply_transform(H_compliance_early.cuda(), transform_early).cpu()
            z_benign_early = apply_transform(H_benign_early.cuda(), transform_early).cpu()

            preprocessed_activations_for_plot['early_window'][layer]['refusal'] = z_refusal_early
            preprocessed_activations_for_plot['early_window'][layer]['compliance'] = z_compliance_early
            preprocessed_activations_for_plot['early_window'][layer]['benign'] = z_benign_early

            if len(z_refusal_early) > 0 and len(z_compliance_early) > 0:
                n_refusal, n_compliance = len(z_refusal_early), len(z_compliance_early)
                min_samples_v = min(n_refusal, n_compliance)
                if log_balancing_info:
                    logging.info(f"为提取 v_l 平衡样本: refusal({n_refusal}) vs compliance({n_compliance}). 将使用 {min_samples_v} 个样本。")
                indices_refusal = torch.randperm(n_refusal)[:min_samples_v]
                indices_compliance = torch.randperm(n_compliance)[:min_samples_v]
                v = get_diff_vector(z_refusal_early[indices_refusal], z_compliance_early[indices_compliance])
                if (v @ z_refusal_early.mean(0)) <= (v @ z_compliance_early.mean(0)): v = -v
                v_norm = torch.norm(v)
                intervention_vectors[layer] = v / v_norm if v_norm > 0 else v
                all_vectors_for_plot[layer]['v'] = intervention_vectors[layer]

        H_compliance_cont = compliance_activations.get(layer, {}).get('content_window')
        H_benign_cont = benign_activations.get(layer, {}).get('content_window')
        H_refusal_cont = refusal_activations.get(layer, {}).get('content_window')

        if H_compliance_cont is not None and H_benign_cont is not None and H_refusal_cont is not None:
            z_compliance_cont = apply_transform(H_compliance_cont.cuda(), transform_cont).cpu()
            z_benign_cont = apply_transform(H_benign_cont.cuda(), transform_cont).cpu()
            z_refusal_cont = apply_transform(H_refusal_cont.cuda(), transform_cont).cpu()

            preprocessed_activations_for_plot['content_window'][layer]['compliance'] = z_compliance_cont
            preprocessed_activations_for_plot['content_window'][layer]['benign'] = z_benign_cont
            preprocessed_activations_for_plot['content_window'][layer]['refusal'] = z_refusal_cont

            if len(z_compliance_cont) > 0 and len(z_benign_cont) > 0:
                n_compliance, n_benign = len(z_compliance_cont), len(z_benign_cont)
                min_samples_c = min(n_compliance, n_benign)
                if log_balancing_info:
                    logging.info(f"为提取 c_l 平衡样本: compliance({n_compliance}) vs benign({n_benign}). 将使用 {min_samples_c} 个样本。")
                indices_compliance = torch.randperm(n_compliance)[:min_samples_c]
                indices_benign = torch.randperm(n_benign)[:min_samples_c]
                c = get_diff_vector(z_compliance_cont[indices_compliance], z_benign_cont[indices_benign])
                if (c @ z_compliance_cont.mean(0)) <= (c @ z_benign_cont.mean(0)): c = -c
                c_norm = torch.norm(c)
                condition_vectors[layer] = c / c_norm if c_norm > 0 else c
                all_vectors_for_plot[layer]['c'] = condition_vectors[layer]

        if log_balancing_info: log_balancing_info = False

    if config.get('decoupling', {}).get('enabled', True):
        logging.info("正在对向量进行去耦合处理...")
        for layer in layers:
            if layer in intervention_vectors and layer in condition_vectors:
                v_l, c_l = intervention_vectors[layer], condition_vectors[layer]
                if torch.norm(v_l) > 0 and torch.norm(c_l) > 0:
                    v_l_decoupled = v_l - (c_l @ v_l) * c_l
                    norm_v = torch.norm(v_l_decoupled)
                    if norm_v > 0: intervention_vectors[layer] = v_l_decoupled / norm_v
                    c_l_decoupled = c_l - (v_l @ c_l) * v_l
                    norm_c = torch.norm(c_l_decoupled)
                    if norm_c > 0: condition_vectors[layer] = c_l_decoupled / norm_c
                all_vectors_for_plot[layer]['v'] = intervention_vectors[layer]
                all_vectors_for_plot[layer]['c'] = condition_vectors[layer]

    torch.save(intervention_vectors, output_dir / "intervention_vectors.pt")
    logging.info(f"干预向量已保存到 {output_dir / 'intervention_vectors.pt'}")
    torch.save(condition_vectors, output_dir / "condition_vectors.pt")
    logging.info(f"条件向量已保存到 {output_dir / 'condition_vectors.pt'}")

    vis_config = config.get('visualization', {})
    if vis_config.get('enabled', False):
        plot_pca_visualizations(
            llm_name=llm_name, window_name='early_window', layers=layers,
            activations=preprocessed_activations_for_plot['early_window'],
            vectors=all_vectors_for_plot, output_dir=output_dir,
            whitening_enabled=whitening_enabled, sample_size=vis_config.get('sample_size', 200)
        )
        plot_pca_visualizations(
            llm_name=llm_name, window_name='content_window', layers=layers,
            activations=preprocessed_activations_for_plot['content_window'],
            vectors=all_vectors_for_plot, output_dir=output_dir,
            whitening_enabled=whitening_enabled, sample_size=vis_config.get('sample_size', 200)
        )

    logging.info("向量提取流程完成。")

    # --- 新增：计算并保存逐 token 分数 ---
    logging.info("--- 开始计算逐 token 分数 ---")

    # 检查并设置模型加载所需配置
    if 'model_path' not in config:
        config['model_path'] = f"../../../models/{config['llm_name']}"
        logging.warning(f"在配置中未找到 'model_path'。推断路径为: {config['model_path']}")
    if 'model_kwargs' not in config:
        config['model_kwargs'] = {"torch_dtype": "bfloat16", "trust_remote_code": True}
        logging.warning("在配置中未找到 'model_kwargs'。使用默认值。")

    # 加载模型
    model, tokenizer = get_model_and_tokenizer(
        config['model_path'],
        config['model_kwargs'],
        config.get('processing', {}).get('device', 'cuda')
    )

    datasets_to_score = {
        "compliance": compliance_df,
        "refusal": refusal_df,
        "benign": benign_df,
    }

    calculate_and_save_token_scores(
        datasets_to_score=datasets_to_score,
        model=model,
        tokenizer=tokenizer,
        transforms=all_transforms,
        condition_vectors=condition_vectors,
        config=config,
        output_dir=output_dir
    )

    logging.info("所有流程全部完成。")


if __name__ == "__main__":
    main()

