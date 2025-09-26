# -*- coding: utf-8 -*-
"""
脚本 05: 可视化 PCA 处理后的向量

该脚本用于将 `03_extract_vectors.py` 提取的激活和向量进行可视化。

主要功能:
1.  加载“满足”(compliance)、“拒绝”(refusal) 和“良性”(benign) 数据集的
    内容窗口 (`content_window`) 激活。
2.  加载白化矩阵以及提取出的干预向量 (v_l) 和条件向量 (c_l)。
3.  对每个指定层级的激活应用白化变换。
4.  将三类样本的白化后激活合并，并使用 PCA 将其降维到二维空间。
5.  为每个层级生成一个散点图，用不同颜色展示三类样本在二维空间中的分布。
6.  在图上用箭头标出干预向量和条件向量在该二维空间中的投影方向。
7.  将所有图合并为一个网格图并保存为图像文件。

如何运行:
# 绘制默认的几个层
python scripts/05_plot_pca_vectors.py --llm_name "vicuna_7b_v1_5"

# 绘制指定的层
python scripts/05_plot_pca_vectors.py --llm_name "vicuna_7b_v1_5" --layers_to_plot 8 12 16 20 24 28 30 31

# 绘制所有可用的层
python scripts/05_plot_pca_vectors.py --llm_name "vicuna_7b_v1_5" --plot_all_layers
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_filter_data(data_dir: Path, llm_name: str, dataset_name: str) -> tuple[dict, pd.DataFrame]:
    """
    加载激活张量和对应的带标签的 CSV 文件，并根据标签进行过滤。
    此函数与 03_extract_vectors.py 中的版本保持一致。
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
        return activations, df

    initial_count = len(df)
    df.dropna(subset=['label'], inplace=True)
    valid_indices = df.index[df['label'] == target_label].tolist()

    df_filtered = df.loc[valid_indices].reset_index(drop=True)

    activations_filtered = {}
    for layer, windows in activations.items():
        activations_filtered[layer] = {}
        for window_name, tensor in windows.items():
            if tensor.shape[0] != initial_count:
                logging.warning(
                    f"在 {dataset_name} (L{layer}, {window_name}) 中，激活数量 ({tensor.shape[0]}) 与CSV行数 ({initial_count}) 不匹配。跳过此张量。")
                continue
            activations_filtered[layer][window_name] = tensor[valid_indices]

    logging.info(f"对于 {dataset_name}，从 {initial_count} 个样本中过滤出 {len(df_filtered)} 个标签为 '{target_label}' 的样本。")
    return activations_filtered, df_filtered


def apply_whitening(activations: torch.Tensor, transform: tuple) -> torch.Tensor:
    """
    将白化变换应用于给定的激活张量。
    此函数与 03_extract_vectors.py 中的版本保持一致。
    """
    W, mu = transform
    # 将所有张量移至 CPU 进行计算
    activations, W, mu = activations.cpu(), W.cpu(), mu.cpu()
    return (activations - mu) @ W.T


def main():
    parser = argparse.ArgumentParser(
        description="可视化 PCA 处理后的隐藏激活分布。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--llm_name', type=str, default='vicuna_7b_v1_5', help='必须与 extraction_config.yaml 中的 `model_name` 的基本名称匹配。')
    parser.add_argument('--layers_to_plot', type=int, nargs='+', default=[8, 16, 24, 31],
                        help='要绘制的层索引列表。如果提供了 --plot_all_layers，此参数将被忽略。默认值: [8, 16, 24, 31]。')
    parser.add_argument('--plot_all_layers', action='store_true', default=True, help='如果指定，则绘制所有可用的层，忽略 --layers_to_plot。默认为否。')
    parser.add_argument('--data_dir', type=str, default='data/activations', help='包含激活、向量和白化矩阵的目录。')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='保存生成的可视化图像的目录。')
    parser.add_argument('--plot_filename', type=str, default='pca_activation_distribution.png', help='输出图像的文件名。')
    parser.add_argument('--sample_size', type=int, default=200, help='每个类别绘制的最大点数。设为 0 表示绘制所有点。')

    args = parser.parse_args()

    # --- 1. 设置参数和路径 ---
    sns.set_theme(style="whitegrid", palette="deep")

    colors = {
        'Benign': '#2ca02c',  # tab:green
        'Refusal': '#1f77b4',  # tab:blue
        'Compliance (Harmful)': '#d62728',  # tab:red
    }
    vec_colors = {
        'c_vector': '#9467bd',  # tab:purple
        'v_vector': '#ff7f0e'  # tab:orange
    }

    base_dir = Path(__file__).parent.parent
    llm_name = args.llm_name
    data_dir = base_dir / args.data_dir / llm_name
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"可视化图像将保存到: {output_dir}")

    # --- 2. 加载所需数据 ---
    logging.info("开始加载激活、向量和白化矩阵...")
    compliance_activations, _ = load_and_filter_data(data_dir, llm_name, "compliance")
    refusal_activations, _ = load_and_filter_data(data_dir, llm_name, "refusal")
    benign_activations, _ = load_and_filter_data(data_dir, llm_name, "benign")

    if not all([compliance_activations, refusal_activations, benign_activations]):
        logging.error("缺少必要的数据集激活文件，无法继续。")
        return

    try:
        whitening_transforms = torch.load(data_dir / "whitening_matrices.pt", map_location='cpu')
        intervention_vectors = torch.load(data_dir / "intervention_vectors.pt", map_location='cpu')
        condition_vectors = torch.load(data_dir / "condition_vectors.pt", map_location='cpu')
    except FileNotFoundError as e:
        logging.error(f"加载向量或白化矩阵失败: {e}。请确保已成功运行 03_extract_vectors.py。")
        return

    # --- 3. 设置绘图 ---
    if args.plot_all_layers:
        layers_to_plot = sorted(list(whitening_transforms.keys()))
        logging.info(f"检测到 --plot_all_layers。将绘制所有 {len(layers_to_plot)} 个可用层。")
    else:
        layers_to_plot = args.layers_to_plot

    n_layers = len(layers_to_plot)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axes = axes.flatten()

    sample_size = args.sample_size

    # --- 4. 循环处理并绘制每一层 ---
    for i, layer in enumerate(tqdm(layers_to_plot, desc="正在为每一层生成图像")):
        ax = axes[i]

        if layer not in whitening_transforms:
            logging.warning(f"第 {layer} 层没有白化矩阵，跳过绘图。")
            ax.text(0.5, 0.5, f'Layer {layer}\nNo Data', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        transform = whitening_transforms[layer]

        # 提取并白化激活
        H_benign = benign_activations[layer]['content_window']
        H_refusal = refusal_activations[layer]['content_window']
        H_compliance = compliance_activations[layer]['content_window']

        z_benign = apply_whitening(H_benign, transform)
        z_refusal = apply_whitening(H_refusal, transform)
        z_compliance = apply_whitening(H_compliance, transform)

        # --- 平衡并采样数据 ---
        min_available_samples = min(len(z_benign), len(z_refusal), len(z_compliance))

        if sample_size > 0:
            # 如果设置了采样大小，则取用户指定值和最小可用样本数之间的较小者
            final_sample_count = min(sample_size, min_available_samples)
        else:
            # 如果 sample_size 为 0 (表示全部绘制)，则使用最小可用样本数以保证平衡
            final_sample_count = min_available_samples

        if i == 0:  # 只在处理第一层时打印一次日志
            logging.info(f"为确保各类别点数相同，将为每个类别绘制 {final_sample_count} 个点。")

        # 使用确定的数量进行采样
        z_benign = z_benign[torch.randperm(z_benign.size(0))[:final_sample_count]]
        z_refusal = z_refusal[torch.randperm(z_refusal.size(0))[:final_sample_count]]
        z_compliance = z_compliance[torch.randperm(z_compliance.size(0))[:final_sample_count]]

        # 合并数据并运行 PCA
        all_whitened = torch.cat([z_benign, z_refusal, z_compliance], dim=0)
        pca = PCA(n_components=2)
        pca.fit(all_whitened.numpy())

        # 变换数据到二维空间
        proj_benign = pca.transform(z_benign.numpy())
        proj_refusal = pca.transform(z_refusal.numpy())
        proj_compliance = pca.transform(z_compliance.numpy())

        # 为 seaborn 创建 DataFrame
        data_benign = pd.DataFrame(proj_benign, columns=['PC1', 'PC2'])
        data_benign['Category'] = 'Benign'
        data_refusal = pd.DataFrame(proj_refusal, columns=['PC1', 'PC2'])
        data_refusal['Category'] = 'Refusal'
        data_compliance = pd.DataFrame(proj_compliance, columns=['PC1', 'PC2'])
        data_compliance['Category'] = 'Compliance (Harmful)'
        # 重新排序以控制绘制顺序：先绘制有害和拒绝，最后绘制良性，使其在顶层
        plot_df = pd.concat([data_refusal, data_compliance, data_benign], ignore_index=True)

        # 使用 seaborn 绘制散点图
        sns.scatterplot(
            data=plot_df,
            x='PC1',
            y='PC2',
            hue='Category',
            # 明确指定 hue_order 以确保图例顺序正确
            hue_order=['Benign', 'Refusal', 'Compliance (Harmful)'],
            palette=colors,
            ax=ax,
            alpha=0.7,
            s=20,
            edgecolor='w',
            linewidth=0.5
        )
        # 移除每个子图的独立图例
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # 变换并绘制 c_l 和 v_l 向量
        v_l = intervention_vectors.get(layer)
        c_l = condition_vectors.get(layer)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        arrow_scale = np.mean([np.abs(xlim).sum(), np.abs(ylim).sum()]) * 0.2

        if c_l is not None and torch.norm(c_l) > 0:
            proj_c = pca.transform(c_l.numpy().reshape(1, -1))
            ax.quiver(0, 0, proj_c[0, 0] * arrow_scale, proj_c[0, 1] * arrow_scale, color=vec_colors.get('c_vector'), scale=1, scale_units='xy',
                      angles='xy', width=0.01, label=r'$c_l$ (Harmful Dir)')

        if v_l is not None and torch.norm(v_l) > 0:
            proj_v = pca.transform(v_l.numpy().reshape(1, -1))
            ax.quiver(0, 0, proj_v[0, 0] * arrow_scale, proj_v[0, 1] * arrow_scale, color=vec_colors.get('v_vector'), scale=1, scale_units='xy',
                      angles='xy', width=0.01, label=r'$v_l$ (Refusal Dir)')

        ax.set_title(f"Layer {layer}", fontsize=12)

    # --- 5. 清理并保存图像 ---
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 创建一个全局图例
    handles, labels = [], []
    # 从散点图获取图例项
    for cat, color in colors.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=color, markersize=10))
        labels.append(cat)
    # 从向量箭头获取图例项
    for cat, color in vec_colors.items():
        label_text = r'$c_l$ (Harmful Dir)' if cat == 'c_vector' else r'$v_l$ (Refusal Dir)'
        handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label_text))
        labels.append(label_text)

    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.01), frameon=True, fontsize=12)

    fig.suptitle(f'PCA of Hidden Activations for {llm_name}', fontsize=18, y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    output_path = output_dir / args.plot_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"可视化图像已成功保存到: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()


