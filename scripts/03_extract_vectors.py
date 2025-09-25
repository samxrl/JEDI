# -*- coding: utf-8 -*-
"""
脚本 03: 提取表征向量

该脚本遵循“方法流程.md”文档中的第二和第三阶段，负责计算和提取
干预向量 (intervention vectors) 和条件向量 (condition vectors)。

主要功能:
1.  加载由 `02_extract_activations.py` 提取的激活，以及由 `02.5_judge_harmfulness.py`
    生成的有害性标签。
2.  **过滤样本 (步骤 3.0)**:
    - 对于 "compliance" (A1) 数据，仅保留被标记为有害 (`label=='yes'`) 的样本。
    - 对于 "refusal" (A2) 数据，仅保留被标记为无害 (`label=='no'`) 的样本。
3.  **白化与中心化 (步骤 2.5)**:
    - 使用良性 (benign) 数据集的激活来计算每层的均值和协方差。
    - 生成并保存白化矩阵，用于后续的表征标准化。
4.  **提取干预向量 `v_l` (步骤 3.1)**:
    - 对 A1 和 A2 样本在“早期窗口”(`early_window`) 的激活进行差分。
    - 使用主成分分析 (PCA) 找到表示“拒绝行为”的主要方向。
5.  **提取条件向量 `c_l` (步骤 3.2)**:
    - 将 A1 样本在“内容窗口”(`content_window`) 的激活作为正类，
      将 A2 和良性样本的激活作为负类。
    - 使用对比性 PCA 找到区分“有害内容”与“无害内容”的主要方向。
6.  **向量去耦合 (步骤 3.3)**:
    - 对 `v_l` 和 `c_l` 进行正交化处理，以减少它们之间的相关性。
7.  将最终的白化矩阵、干预向量和条件向量保存到 `artifacts` 目录，
    以供在线防御系统 (`Guard`) 使用。

如何运行:
python scripts/03_extract_vectors.py --config configs/vector_extraction_config.yaml
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_filter_data(data_dir: Path, llm_name: str, dataset_name: str) -> tuple[dict, pd.DataFrame]:
    """
    加载激活张量和对应的带标签的 CSV 文件，并根据标签进行过滤。
    """
    activations_path = data_dir / f"{llm_name}_{dataset_name}_activations.pt"
    outputs_path = data_dir / f"{llm_name}_{dataset_name}_outputs.csv"

    if not activations_path.exists() or not outputs_path.exists():
        logging.warning(f"未找到 {dataset_name} 的激活或输出文件，跳过加载。路径: {activations_path}")
        return None, None

    logging.info(f"正在加载 {dataset_name} 数据...")
    activations = torch.load(activations_path)
    df = pd.read_csv(outputs_path)

    if dataset_name == "compliance":
        target_label = "yes"
    elif dataset_name == "refusal":
        target_label = "no"
    else:  # benign
        return activations, df

    # 过滤掉标签为空或不匹配的行
    initial_count = len(df)
    df.dropna(subset=['label'], inplace=True)
    valid_indices = df.index[df['label'] == target_label].tolist()

    df_filtered = df.loc[valid_indices].reset_index(drop=True)

    # 过滤激活张量
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


def calculate_whitening_transform(benign_activations: dict, epsilon: float) -> dict:
    """
    根据良性激活计算白化变换矩阵和均值。 (步骤 2.5)
    """
    logging.info("正在计算白化变换...")
    whitening_transforms = {}

    # 假设所有层都在 benign_activations 中
    for layer in tqdm(benign_activations.keys(), desc="计算白化矩阵"):
        # 我们只使用 content_window 来建立基线
        if 'content_window' in benign_activations[layer]:
            H_benign = benign_activations[layer]['content_window'].to(torch.float32).cuda()

            # 计算均值
            mu = H_benign.mean(dim=0)

            # 中心化
            H_centered = H_benign - mu

            # 计算协方差矩阵
            cov = (H_centered.T @ H_centered) / (H_centered.shape[0] - 1)

            # 添加 epsilon 以保证数值稳定性
            cov_reg = cov + torch.eye(cov.shape[0]).cuda() * epsilon

            # 计算特征值和特征向量
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_reg)

            # 检查是否有负的特征值
            if (eigenvalues < 0).any():
                logging.warning(f"在第 {layer} 层发现负特征值。最小特征值: {eigenvalues.min().item()}")
                eigenvalues = torch.clamp(eigenvalues, min=1e-6)

            # 计算白化矩阵 W = D^(-1/2) V^T
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues))
            W = D_inv_sqrt @ eigenvectors.T

            whitening_transforms[layer] = (W.cpu(), mu.cpu())

    return whitening_transforms


def apply_whitening(activations: torch.Tensor, transform: tuple) -> torch.Tensor:
    """
    将白化变换应用于给定的激活张量。
    """
    W, mu = transform
    return ((activations.cuda() - mu.cuda()) @ W.cuda().T).cpu()


def get_contrast_vector(pos_activations: torch.Tensor, neg_activations: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    使用对比性 PCA 提取条件向量 c_l (步骤 3.2)。
    """
    # 修复：确保有足够的样本运行 PCA
    if pos_activations.shape[0] < n_components:
        logging.warning(f"运行对比性 PCA 的正样本不足 ({pos_activations.shape[0]})。需要至少 {n_components} 个。返回零向量。")
        return torch.zeros(pos_activations.shape[1], dtype=torch.float32)

    # 计算负类激活的均值
    mu_neg = neg_activations.mean(dim=0)

    # 将正类激活与负类均值进行对比
    pos_centered = pos_activations - mu_neg

    # 在对比后的数据上执行 PCA
    pca = PCA(n_components=n_components)
    pca.fit(pos_centered.numpy())

    # 提取第一主成分
    vector = torch.tensor(pca.components_[0], dtype=torch.float32)

    return vector


def get_diff_vector(pos_activations: torch.Tensor, neg_activations: torch.Tensor) -> torch.Tensor:
    """
    通过直接计算均值差并归一化来提取干预向量 v_l (步骤 3.1)。
    这个版本移除了会产生警告的 PCA 调用。
    """
    # 计算两组激活的均值差
    mean_diff = pos_activations.mean(dim=0) - neg_activations.mean(dim=0)

    # 直接对差分向量进行归一化，得到方向
    norm = torch.norm(mean_diff)
    if norm == 0:
        logging.warning("均值差分向量的范数为零，无法归一化。返回零向量。")
        return torch.zeros_like(mean_diff)

    vector = mean_diff / norm
    return vector


def main():
    parser = argparse.ArgumentParser(description="从模型激活中提取干预向量和条件向量。")
    parser.add_argument('--config', type=str, default='../configs/PCA_config.yaml',
                        help='向量提取阶段的 YAML 配置文件路径。')
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    config_path = Path(args.config)
    if not config_path.is_file():
        logging.error(f"配置文件未找到: {config_path}")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- 2. 设置路径 ---
    base_dir = Path(__file__).parent.parent
    llm_name = config['llm_name']
    data_dir = base_dir / config['data_dir'] / llm_name
    output_dir = base_dir / config['output_dir'] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"所有产物将保存到: {output_dir}")

    # --- 3. 加载并过滤数据 ---
    compliance_activations, compliance_df = load_and_filter_data(data_dir, llm_name, "compliance")
    refusal_activations, refusal_df = load_and_filter_data(data_dir, llm_name, "refusal")
    benign_activations, _ = load_and_filter_data(data_dir, llm_name, "benign")

    if not compliance_activations or not refusal_activations or not benign_activations:
        logging.error("缺少必要的数据集，无法继续。请先运行 02 和 02.5 脚本。")
        return

    # --- 4. 计算白化变换 ---
    whitening_transforms = calculate_whitening_transform(
        benign_activations,
        config.get('whitening', {}).get('epsilon', 1e-4)
    )
    torch.save(whitening_transforms, output_dir / "whitening_matrices.pt")
    logging.info(f"白化矩阵已保存到 {output_dir / 'whitening_matrices.pt'}")

    intervention_vectors, condition_vectors = {}, {}
    layers = list(benign_activations.keys())

    for layer in tqdm(layers, desc="提取向量"):
        if layer not in whitening_transforms:
            logging.warning(f"第 {layer} 层没有白化矩阵，跳过。")
            continue

        transform = whitening_transforms[layer]

        # --- 5. 提取干预向量 (v_l) ---
        H_refusal_early = refusal_activations[layer]['early_window']
        H_compliance_early = compliance_activations[layer]['early_window']

        # 应用白化
        z_refusal_early = apply_whitening(H_refusal_early, transform)
        z_compliance_early = apply_whitening(H_compliance_early, transform)

        # 获取向量 (已更新为不使用 PCA 的版本)
        v = get_diff_vector(z_refusal_early, z_compliance_early)

        # 符号校准：确保 v 的方向“更像拒绝”
        if v @ z_refusal_early.mean(dim=0) < v @ z_compliance_early.mean(dim=0):
            v = -v
        intervention_vectors[layer] = v / torch.norm(v)  # 再次归一化以确保单位长度

        # --- 6. 提取条件向量 (c_l) ---
        H_compliance_cont = compliance_activations[layer]['content_window']
        H_refusal_cont = refusal_activations[layer]['content_window']
        H_benign_cont = benign_activations[layer]['content_window']

        # 应用白化
        z_compliance_cont = apply_whitening(H_compliance_cont, transform)
        z_refusal_cont = apply_whitening(H_refusal_cont, transform)
        z_benign_cont = apply_whitening(H_benign_cont, transform)

        # 构造正负类
        z_pos = z_compliance_cont
        z_neg = torch.cat([z_refusal_cont, z_benign_cont], dim=0)

        # 获取向量
        c = get_contrast_vector(z_pos, z_neg, config['pca']['n_components'])

        # 符号校准：确保 c 的方向“更像有害”
        if torch.norm(c) > 0 and (c @ z_pos.mean(dim=0) < c @ z_neg.mean(dim=0)):
            c = -c
        condition_vectors[layer] = c / torch.norm(c) if torch.norm(c) > 0 else c  # 归一化

    # --- 7. (可选) 向量去耦合 ---
    if config.get('decoupling', {}).get('enabled', True):
        logging.info("正在对向量进行去耦合处理...")
        for layer in layers:
            if layer in intervention_vectors and layer in condition_vectors:
                v_l = intervention_vectors[layer]
                c_l = condition_vectors[layer]

                # 确保向量非零
                if torch.norm(v_l) > 0 and torch.norm(c_l) > 0:
                    # v_l_new = v_l - (c_l.T @ v_l) * c_l
                    v_l_decoupled = v_l - (c_l @ v_l) * c_l
                    intervention_vectors[layer] = v_l_decoupled / torch.norm(v_l_decoupled)

                    # c_l_new = c_l - (v_l.T @ c_l) * v_l
                    c_l_decoupled = c_l - (v_l @ c_l) * v_l
                    condition_vectors[layer] = c_l_decoupled / torch.norm(c_l_decoupled)

    # --- 8. 保存最终向量 ---
    torch.save(intervention_vectors, output_dir / "intervention_vectors.pt")
    logging.info(f"干预向量已保存到 {output_dir / 'intervention_vectors.pt'}")

    torch.save(condition_vectors, output_dir / "condition_vectors.pt")
    logging.info(f"条件向量已保存到 {output_dir / 'condition_vectors.pt'}")

    logging.info("向量提取流程全部完成。")


if __name__ == "__main__":
    main()

