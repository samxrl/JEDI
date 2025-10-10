# -*- coding: utf-8 -*-
"""
脚本 03: 提取表征向量 (双白化版)

该脚本遵循“方法流程.md”文档中的第二和第三阶段，负责计算和提取
干预向量 (intervention vectors) 和条件向量 (condition vectors)。
此版本已更新，以兼容包含聚合后 (`aggregated`) 和逐 token (`per_token`)
激活的新数据格式，并仅使用聚合后的激活进行向量提取。

主要功能:
1.  加载由 `02_extract_activations.py` 提取的激活，以及由 `02.5_judge_harmfulness.py`
    生成的有害性标签。
2.  **过滤样本**:
    - 对于 "compliance" (A1) 数据，仅保留被标记为有害 (`label=='yes'`) 的样本。
    - 对于 "refusal" (A2) 数据，仅保留被标记为无害 (`label=='no'`) 的样本。
3.  **平衡样本**: 在计算差分向量前，通过随机下采样确保正负样本数量一致。
4.  **双白化流程**:
    - 使用良性 (B1) 数据集的聚合后激活，为“早期窗口”和“内容窗口”分别计算变换（中心化/白化）。
5.  **提取干预向量 `v_l` (拒绝行为)**:
    - 对平衡后的 A1 和 A2 样本在“早期窗口”的聚合后激活应用**行为变换**，并计算均值差。
6.  **提取条件向量 `c_l` (有害内容)**:
    - 对平衡后的 A1 和 B1 样本在“内容窗口”的聚合后激活应用**语义变换**，并计算均值差。
7.  **(可选) 向量去耦合** 和 **可视化**。
8.  将最终的变换矩阵、干预向量和条件向量保存到产物目录。

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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_filter_data(data_dir: Path, llm_name: str, dataset_name: str) -> tuple[dict, pd.DataFrame]:
    """
    加载激活张量和对应的带标签的 CSV 文件，并根据标签进行过滤。
    此版本适配新的数据结构，仅提取 'aggregated' 激活。
    """
    activations_path = data_dir / f"{llm_name}_{dataset_name}_activations.pt"
    outputs_path = data_dir / f"{llm_name}_{dataset_name}_outputs.csv"

    if not activations_path.exists() or not outputs_path.exists():
        logging.warning(f"未找到 {dataset_name} 的激活或输出文件，跳过加载。路径: {activations_path}")
        return None, None

    logging.info(f"正在加载 {dataset_name} 数据...")
    activations = torch.load(activations_path, map_location='cpu')
    df = pd.read_csv(outputs_path)

    if dataset_name == "compliance":
        target_label = "yes"
    elif dataset_name == "refusal":
        target_label = "no"
    else:  # benign
        # 对于良性数据，我们不需要过滤，但仍需提取 aggregated 部分
        aggregated_activations = {}
        for layer, windows in activations.items():
            aggregated_activations[layer] = {}
            for window_name, data in windows.items():
                if 'aggregated' in data:
                    aggregated_activations[layer][window_name] = data['aggregated']
        return aggregated_activations, df

    # --- 对 compliance 和 refusal 数据进行过滤 ---
    initial_count = len(df)
    df.dropna(subset=['label'], inplace=True)
    valid_indices = df.index[df['label'] == target_label].tolist()

    df_filtered = df.loc[valid_indices].reset_index(drop=True)

    activations_filtered = {}
    for layer, windows in activations.items():
        activations_filtered[layer] = {}
        for window_name, data in windows.items():
            # 仅提取 'aggregated' 张量进行处理
            tensor = data.get('aggregated')
            if tensor is None:
                logging.warning(f"在 L{layer}/{window_name} 中未找到 'aggregated' 键，跳过。")
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

                if window_name == 'early_window':
                    transforms_early[layer] = (W.cpu() if W is not None else None, mu.cpu())
                else:
                    transforms_cont[layer] = (W.cpu() if W is not None else None, mu.cpu())
            else:
                logging.warning(f"第 {layer} 层缺少 '{window_name}' 的良性激活，无法计算该窗口的变换。")

    return transforms_early, transforms_cont


def apply_transform(activations: torch.Tensor, transform: tuple) -> torch.Tensor:
    """
    将变换（中心化和可选的白化）应用于给定的激活张量。
    """
    W, mu = transform
    activations_cuda = activations.to(torch.float32).cuda()
    mu_cuda = mu.to(torch.float32).cuda()
    centered = activations_cuda - mu_cuda

    if W is not None:
        W_cuda = W.to(torch.float32).cuda()
        transformed = centered @ W_cuda.T
        return transformed.cpu()
    else:  # 仅中心化
        return centered.cpu()


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

    compliance_activations, _ = load_and_filter_data(data_dir, llm_name, "compliance")
    refusal_activations, _ = load_and_filter_data(data_dir, llm_name, "refusal")
    benign_activations, _ = load_and_filter_data(data_dir, llm_name, "benign")

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

    # --- 在循环外只记录一次平衡信息 ---
    log_balancing_info = True

    for layer in tqdm(layers, desc="提取向量"):
        transform_early = transforms_early.get(layer)
        transform_cont = transforms_cont.get(layer)

        if not transform_early or not transform_cont:
            logging.warning(f"第 {layer} 层缺少变换矩阵，跳过向量提取。")
            continue

        # --- 提取干预向量 v_l (拒绝行为) ---
        H_refusal_early = refusal_activations.get(layer, {}).get('early_window')
        H_compliance_early = compliance_activations.get(layer, {}).get('early_window')
        H_benign_early = benign_activations.get(layer, {}).get('early_window')

        if H_refusal_early is not None and H_compliance_early is not None and H_benign_early is not None:
            z_refusal_early = apply_transform(H_refusal_early, transform_early)
            z_compliance_early = apply_transform(H_compliance_early, transform_early)
            z_benign_early = apply_transform(H_benign_early, transform_early)

            preprocessed_activations_for_plot['early_window'][layer]['refusal'] = z_refusal_early
            preprocessed_activations_for_plot['early_window'][layer]['compliance'] = z_compliance_early
            preprocessed_activations_for_plot['early_window'][layer]['benign'] = z_benign_early

            if len(z_refusal_early) > 0 and len(z_compliance_early) > 0:
                # 平衡样本以提取 v_l
                n_refusal = len(z_refusal_early)
                n_compliance = len(z_compliance_early)
                min_samples_v = min(n_refusal, n_compliance)

                if log_balancing_info:
                    logging.info(f"为提取 v_l 平衡样本: refusal({n_refusal}) vs compliance({n_compliance}). 将使用 {min_samples_v} 个样本。")

                indices_refusal = torch.randperm(n_refusal)[:min_samples_v]
                indices_compliance = torch.randperm(n_compliance)[:min_samples_v]

                z_refusal_balanced = z_refusal_early[indices_refusal]
                z_compliance_balanced = z_compliance_early[indices_compliance]

                v = get_diff_vector(z_refusal_balanced, z_compliance_balanced)
                if (v @ z_refusal_balanced.mean(0)) <= (v @ z_compliance_balanced.mean(0)): v = -v
                v_norm = torch.norm(v)
                intervention_vectors[layer] = v / v_norm if v_norm > 0 else v
                all_vectors_for_plot[layer]['v'] = intervention_vectors[layer]

        # --- 提取条件向量 c_l (有害内容) ---
        H_compliance_cont = compliance_activations.get(layer, {}).get('content_window')
        H_benign_cont = benign_activations.get(layer, {}).get('content_window')
        H_refusal_cont = refusal_activations.get(layer, {}).get('content_window')

        if H_compliance_cont is not None and H_benign_cont is not None and H_refusal_cont is not None:
            z_compliance_cont = apply_transform(H_compliance_cont, transform_cont)
            z_benign_cont = apply_transform(H_benign_cont, transform_cont)
            z_refusal_cont = apply_transform(H_refusal_cont, transform_cont)

            preprocessed_activations_for_plot['content_window'][layer]['compliance'] = z_compliance_cont
            preprocessed_activations_for_plot['content_window'][layer]['benign'] = z_benign_cont
            preprocessed_activations_for_plot['content_window'][layer]['refusal'] = z_refusal_cont

            if len(z_compliance_cont) > 0 and len(z_benign_cont) > 0:
                # 平衡样本以提取 c_l
                n_compliance = len(z_compliance_cont)
                n_benign = len(z_benign_cont)
                min_samples_c = min(n_compliance, n_benign)

                if log_balancing_info:
                    logging.info(f"为提取 c_l 平衡样本: compliance({n_compliance}) vs benign({n_benign}). 将使用 {min_samples_c} 个样本。")

                indices_compliance = torch.randperm(n_compliance)[:min_samples_c]
                indices_benign = torch.randperm(n_benign)[:min_samples_c]

                z_compliance_balanced = z_compliance_cont[indices_compliance]
                z_benign_balanced = z_benign_cont[indices_benign]

                c = get_diff_vector(z_compliance_balanced, z_benign_balanced)
                if (c @ z_compliance_balanced.mean(0)) <= (c @ z_benign_balanced.mean(0)): c = -c
                c_norm = torch.norm(c)
                condition_vectors[layer] = c / c_norm if c_norm > 0 else c
                all_vectors_for_plot[layer]['c'] = condition_vectors[layer]

        # 确保只记录一次
        if log_balancing_info:
            log_balancing_info = False

    if config.get('decoupling', {}).get('enabled', True):
        logging.info("正在对向量进行去耦合处理...")
        for layer in layers:
            if layer in intervention_vectors and layer in condition_vectors:
                v_l = intervention_vectors[layer]
                c_l = condition_vectors[layer]
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

    logging.info("向量提取流程全部完成。")


if __name__ == "__main__":
    main()
