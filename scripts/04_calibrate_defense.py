# -*- coding: utf-8 -*-
"""
脚本 04: 校准防御参数

该脚本遵循“方法流程.md”文档中的第四阶段，负责计算单步风险分数、选择最优层，
并为 CUSUM 累积检测机制校准关键参数。

此版本专门设计用于处理在 `02_extract_activations.py` 中提取的、未经聚合的
逐 token 激活数据。

主要功能:
1.  **加载所需构件**:
    - 加载由 `03_extract_vectors.py` 生成的变换矩阵 (transforms.pt) 和条件向量 (condition_vectors.pt)。
    - 加载“有害满足型”(A1) 和“良性”(B1) 数据集的逐 token 激活数据。
2.  **计算逐 token 分数**:
    - 对于每个指定的层，使用其变换矩阵和条件向量 `c_l`，计算每个 token 激活的原始有害分数 `s_t`。
    - 这会为 A1 和 B1 数据集生成分数分布。
3.  **选择最优层**:
    - 通过计算区分 A1 和 B1 分数分布的 AUROC 和科恩 d 值，评估每个层的性能。
    - 选择综合性能最佳的层作为防御层。
4.  **校准分数阈值 (theta)**:
    - 对于最优层，使用其在 B1 (良性) 数据集上的分数分布，计算指定分位数 (如 95%) 作为分数阈值 `theta`。
    - 使用 ReLU 函数将原始分数 `s_t` 转换为非负的风险分数 `r_t = max(0, s_t - theta)`。
5.  **校准 CUSUM 参数 (kappa, h)**:
    - 在 `(kappa, h)` 参数网格上进行搜索。
    - 对每个参数对，在 B1 和 A1 数据集上模拟 CUSUM 过程，计算误报率 (FPR) 和平均检测延迟。
    - 找到在满足目标误报率约束下，具有最低检测延迟的最佳 `(kappa, h)` 对。
6.  **保存防御参数**:
    - 将所有校准得到的参数 (最优层、theta、kappa、h、以及对应的向量和变换) 保存到一个文件中，
      以供在线防御系统使用。

如何运行:
python scripts/04_calibrate_defense.py --config configs/PCA_config.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_filter_data(data_dir: Path, llm_name: str, dataset_name: str) -> tuple[dict, pd.DataFrame]:
    """
    加载逐 token 激活和对应的带标签 CSV，并根据标签过滤 "compliance" 数据集。
    假定激活文件保存的是一个字典，其中 'per_token' 键对应一个列表，
    列表中的每个元素是一个 (seq_len, hidden_dim) 的张量。
    """
    activations_path = data_dir / f"{llm_name}_{dataset_name}_activations.pt"
    outputs_path = data_dir / f"{llm_name}_{dataset_name}_outputs.csv"

    if not activations_path.exists() or not outputs_path.exists():
        logging.warning(f"未找到 {dataset_name} 的激活或输出文件，跳过。路径: {activations_path}")
        return None, None

    logging.info(f"正在加载 {dataset_name} 数据...")
    activations = torch.load(activations_path, map_location='cpu')
    df = pd.read_csv(outputs_path)
    initial_count = len(df)

    valid_indices = list(range(initial_count))
    df_filtered = df

    if dataset_name == "compliance":
        target_label = "yes"
        df.dropna(subset=['label'], inplace=True)
        valid_indices = df.index[df['label'] == target_label].tolist()
        df_filtered = df.loc[valid_indices].reset_index(drop=True)
        logging.info(f"对于 {dataset_name}，从 {initial_count} 个样本中过滤出 {len(df_filtered)} 个标签为 '{target_label}' 的样本。")

    activations_final = {}
    for layer, windows in activations.items():
        activations_final[layer] = {}
        for window_name, act_data in windows.items():
            if not isinstance(act_data, dict) or 'per_token' not in act_data:
                logging.warning(f"在 {dataset_name} (L{layer}, {window_name}) 激活数据格式不正确，缺少 'per_token' 键。跳过此窗口。")
                continue

            per_token_list = act_data['per_token']

            if not isinstance(per_token_list, list) or len(per_token_list) != initial_count:
                logging.warning(
                    f"在 {dataset_name} (L{layer}, {window_name}) 'per_token' 列表长度 ({len(per_token_list)}) 与 CSV 行数 ({initial_count}) 不匹配。跳过此窗口。")
                continue

            # 根据 valid_indices (对于 benign 是所有索引) 进行过滤
            activations_final[layer][window_name] = [per_token_list[i] for i in valid_indices]

    return activations_final, df_filtered


def apply_transform(activations: torch.Tensor, transform: tuple) -> torch.Tensor:
    """将变换（中心化和可选的白化）应用于激活张量。"""
    W, mu = transform
    activations = activations.to(torch.float32)
    mu = mu.to(torch.float32)
    centered = activations - mu
    if W is not None:
        W = W.to(torch.float32)
        return centered @ W.T
    return centered


def calculate_token_scores(
        activations_list: list, transform: tuple, vector: torch.Tensor
) -> tuple[list, np.ndarray]:
    """
    计算每个 token 的原始分数。

    Args:
        activations_list (list): 包含多个 (L, D) 张量的列表，每个代表一个样本。
        transform (tuple): (W, mu) 变换元组。
        vector (torch.Tensor): 条件向量 c_l, 形状为 (D)。

    Returns:
        tuple[list, np.ndarray]: (每个样本的分数序列列表, 所有 token 分数的扁平化 numpy 数组)
    """
    score_sequences = []
    vector = vector.to(torch.float32)
    for h in activations_list:
        if h.shape[0] == 0: continue
        z = apply_transform(h, transform)
        s = z @ vector
        score_sequences.append(s)

    flat_scores = np.concatenate([s.numpy() for s in score_sequences]) if score_sequences else np.array([])
    return score_sequences, flat_scores


def cohen_d(x, y):
    """计算两组数据的科恩 d 值。"""
    if len(x) < 2 or len(y) < 2: return 0.0
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    if pooled_std == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def find_best_layer(harmful_scores_by_layer, benign_scores_by_layer, layers):
    """根据 AUROC 和科恩 d 值选择最佳防御层。"""
    results = []
    for layer in layers:
        harmful_scores = harmful_scores_by_layer.get(layer)
        benign_scores = benign_scores_by_layer.get(layer)

        if harmful_scores is None or benign_scores is None or len(harmful_scores) == 0 or len(benign_scores) == 0:
            continue

        y_true = np.concatenate([np.ones_like(harmful_scores), np.zeros_like(benign_scores)])
        y_score = np.concatenate([harmful_scores, benign_scores])

        auroc = roc_auc_score(y_true, y_score)
        d = cohen_d(harmful_scores, benign_scores)
        results.append({'layer': layer, 'auroc': auroc, 'cohen_d': d})

    if not results:
        raise ValueError("没有任何层的分数可用于评估。")

    results_df = pd.DataFrame(results).sort_values(by=['auroc', 'cohen_d'], ascending=False)
    best_layer = results_df.iloc[0]

    logging.info("层评估结果:\n" + results_df.to_string())
    logging.info(f"*** 选定的最佳层: {best_layer['layer']} (AUROC={best_layer['auroc']:.4f}, Cohen's d={best_layer['cohen_d']:.4f}) ***")

    return int(best_layer['layer']), results_df


def simulate_cusum(score_sequences, theta, benign_r_mean, kappa, h):
    """模拟 CUSUM 过程以计算触发率和延迟。"""
    num_sequences = len(score_sequences)
    if num_sequences == 0: return 0.0, 0.0

    triggers = 0
    total_delay = 0

    for scores in score_sequences:
        r = torch.clamp(scores - theta, min=0)
        A = 0.0
        detected = False
        for t, r_t in enumerate(r):
            A = max(0, A + r_t.item() - benign_r_mean - kappa)
            if A > h:
                triggers += 1
                total_delay += (t + 1)
                detected = True
                break

    trigger_rate = triggers / num_sequences
    avg_delay = total_delay / triggers if triggers > 0 else float('inf')

    return trigger_rate, avg_delay


def calibrate_cusum(harmful_sequences, benign_sequences, theta, grid_config):
    """通过网格搜索校准 CUSUM 的 kappa 和 h 参数。"""
    logging.info(f"使用 theta={theta:.4f} 开始 CUSUM 参数网格搜索...")

    benign_r_flat = np.concatenate([torch.clamp(s - theta, min=0).numpy() for s in benign_sequences])
    benign_r_mean = benign_r_flat.mean()
    logging.info(f"良性风险分数 `r_t` 的均值 (用作 CUSUM 中的 mu_hat): {benign_r_mean:.4f}")

    kappas = np.arange(grid_config['k_min'], grid_config['k_max'] + grid_config['k_step'], grid_config['k_step'])
    hs = np.arange(grid_config['h_min'], grid_config['h_max'] + grid_config['h_step'], grid_config['h_step'])

    results = []

    pbar = tqdm(total=len(kappas) * len(hs), desc="CUSUM 网格搜索")
    for kappa in kappas:
        for h in hs:
            fpr, _ = simulate_cusum(benign_sequences, theta, benign_r_mean, kappa, h)
            tpr, delay = simulate_cusum(harmful_sequences, theta, benign_r_mean, kappa, h)
            results.append({'kappa': kappa, 'h': h, 'fpr': fpr, 'tpr': tpr, 'delay': delay})
            pbar.update(1)
    pbar.close()

    results_df = pd.DataFrame(results)

    # 找到满足 FPR 约束的最佳参数
    valid_params = results_df[results_df['fpr'] <= grid_config['target_fpr']]
    if valid_params.empty:
        logging.warning(f"没有参数组合满足 FPR <= {grid_config['target_fpr']} 的约束。")
        # 放宽约束，选择 FPR 最低的
        best_params = results_df.sort_values(by=['fpr', 'delay']).iloc[0]
    else:
        best_params = valid_params.sort_values(by='delay').iloc[0]

    logging.info("CUSUM 网格搜索结果摘要:\n" + results_df.to_string())
    logging.info(
        f"*** 最佳 CUSUM 参数 (FPR <= {grid_config['target_fpr']}): "
        f"kappa={best_params['kappa']:.4f}, h={best_params['h']:.4f} "
        f"-> (FPR={best_params['fpr']:.4f}, TPR={best_params['tpr']:.4f}, Delay={best_params['delay']:.2f}) ***"
    )

    return best_params.to_dict(), results_df


def main():
    parser = argparse.ArgumentParser(description="校准防御系统的参数。")
    parser.add_argument('--config', type=str, default='../configs/calibration_config.yaml', help='配置文件路径。')
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    config_path = Path(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- 2. 设置路径 ---
    base_dir = Path(__file__).parent.parent
    llm_name = config['llm_name']
    data_dir = base_dir / config['artifacts_dir'] / llm_name
    output_dir = base_dir / config['activations_dir'] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"防御参数将保存到: {output_dir}")

    # --- 3. 加载所需构件 ---
    try:
        transforms = torch.load(data_dir / "transforms.pt", map_location='cpu')
        condition_vectors = torch.load(data_dir / "condition_vectors.pt", map_location='cpu')
    except FileNotFoundError as e:
        logging.error(f"加载文件失败: {e}。请先运行 03_extract_vectors.py。")
        return

    # --- 4. 加载逐 token 激活数据 ---
    harmful_activations, _ = load_and_filter_data(data_dir, llm_name, "compliance")
    benign_activations, _ = load_and_filter_data(data_dir, llm_name, "benign")

    if not harmful_activations or not benign_activations:
        logging.error("缺少必要的激活文件，无法继续。")
        return

    layers = sorted(condition_vectors.keys())
    harmful_scores_by_layer, benign_scores_by_layer = {}, {}
    harmful_sequences_by_layer, benign_sequences_by_layer = {}, {}

    # --- 5. 计算所有层的逐 token 分数 ---
    for layer in tqdm(layers, desc="计算各层 Token 分数"):
        transform = transforms.get('content_window', {}).get(layer)
        c_vector = condition_vectors.get(layer)

        if transform is None or c_vector is None: continue

        harmful_act_list = harmful_activations.get(layer, {}).get('content_window', [])
        benign_act_list = benign_activations.get(layer, {}).get('content_window', [])

        if not harmful_act_list or not benign_act_list: continue

        h_seq, h_flat = calculate_token_scores(harmful_act_list, transform, c_vector)
        b_seq, b_flat = calculate_token_scores(benign_act_list, transform, c_vector)

        harmful_sequences_by_layer[layer] = h_seq
        harmful_scores_by_layer[layer] = h_flat
        benign_sequences_by_layer[layer] = b_seq
        benign_scores_by_layer[layer] = b_flat

    # --- 6. 选择最优层 ---
    best_layer_idx, _ = find_best_layer(harmful_scores_by_layer, benign_scores_by_layer, layers)

    # --- 7. 为最优层校准参数 ---

    # 计算 theta
    benign_scores_best_layer = benign_scores_by_layer[best_layer_idx]
    theta = np.quantile(benign_scores_best_layer, config['calibration_params']['theta_quantile'])

    # 校准 CUSUM
    best_cusum_params, _ = calibrate_cusum(
        harmful_sequences_by_layer[best_layer_idx],
        benign_sequences_by_layer[best_layer_idx],
        theta,
        config['calibration_params']
    )

    # --- 8. 保存最终的防御参数 ---
    defense_params = {
        'llm_name': llm_name,
        'best_layer': best_layer_idx,
        'theta': theta,
        'kappa': best_cusum_params['kappa'],
        'h': best_cusum_params['h'],
        'condition_vector': condition_vectors[best_layer_idx],
        'transform': transforms['content_window'][best_layer_idx],
    }

    save_path = output_dir / "defense_params.pt"
    torch.save(defense_params, save_path)
    logging.info(f"防御参数已成功校准并保存到: {save_path}")

    # (可选) 绘制最优层分数分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(benign_scores_best_layer, color='green', label='Benign (B1)', bins=100, stat='density', alpha=0.6)
    sns.histplot(harmful_scores_by_layer[best_layer_idx], color='red', label='Harmful (A1)', bins=100, stat='density', alpha=0.6)
    plt.axvline(theta, color='blue', linestyle='--', label=f'Theta (q={config["calibration_params"]["theta_quantile"]}) = {theta:.4f}')
    plt.title(f'Token Score Distribution for Best Layer ({best_layer_idx}) on {llm_name}')
    plt.xlabel('Raw Score (s_t)')
    plt.ylabel('Density')
    plt.legend()
    plot_path = output_dir / "best_layer_score_distribution.png"
    plt.savefig(plot_path)
    logging.info(f"最优层分数分布图已保存到: {plot_path}")


if __name__ == "__main__":
    main()

