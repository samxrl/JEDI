# -*- coding: utf-8 -*-
"""
脚本 04: 校准防御参数

该脚本遵循“方法流程.md”文档中的第四阶段，负责为在线防御系统校准关键参数。
此版本已更新，以使用 `02_extract_activations.py` 保存的逐 token 激活
来进行更精确的 CUSUM 参数扫描和校准。

主要功能:
1.  加载由 `03_extract_vectors.py` 预处理后的激活数据（聚合后）和条件向量 (c_l)。
2.  同时加载由 `02_extract_activations.py` 保存的逐 token 激活序列。
3.  **选择最佳防御层**:
    -   使用**聚合后**的激活计算 AUROC 和 Cohen's d，选择最佳层。
4.  **校准边界阈值 (theta)**:
    -   在最佳层上，使用**聚合后**的良性样本分数分布校准 theta。
5.  **计算初始基线 (initial_baseline)**:
    -   使用**逐 token** 的良性样本风险分数 `r_t` 计算均值作为 CUSUM 的初始基线。
6.  **扫描 CUSUM 参数 (h 和 k)**:
    -   使用**逐 token** 的激活序列计算真实的风险分数序列 `r_t`。
    -   在此真实序列上运行 CUSUM 模拟，以找到满足 FPR 并具有最低检测延迟的最优参数。
7.  **生成防御配置文件**:
    -   将所有校准结果保存到 `defense_config.json` 中，供 `Guard` 类加载。

如何运行:
python scripts/04_calibrate_defense.py --config configs/calibration_config.yaml
"""
import argparse
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import itertools

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_filter_data(data_dir: Path, llm_name: str, dataset_name: str) -> dict:
    """
    加载激活张量和对应的带标签的 CSV 文件，并根据标签进行过滤。
    此版本可处理包含 'aggregated' 和 'per_token' 的新数据结构。
    """
    activations_path = data_dir / f"{llm_name}_{dataset_name}_activations.pt"
    outputs_path = data_dir / f"{llm_name}_{dataset_name}_outputs.csv"

    if not activations_path.exists() or not outputs_path.exists():
        logging.warning(f"未找到 {dataset_name} 的激活或输出文件，跳过加载。")
        return None

    activations = torch.load(activations_path, map_location='cpu')
    df = pd.read_csv(outputs_path)

    if dataset_name == "compliance":
        target_label = "yes"
    elif dataset_name == "refusal":
        target_label = "no"
    else:  # benign
        return activations

    initial_count = len(df)
    df.dropna(subset=['label'], inplace=True)
    valid_indices = df.index[df['label'] == target_label].tolist()

    activations_filtered = {}
    for layer, windows in activations.items():
        activations_filtered[layer] = {}
        for window_name, data in windows.items():
            if 'aggregated' not in data or 'per_token' not in data:
                logging.warning(f"跳过 L{layer}/{window_name}，数据结构不完整。")
                continue

            if data['aggregated'].shape[0] != initial_count:
                continue

            activations_filtered[layer][window_name] = {
                'aggregated': data['aggregated'][valid_indices],
                'per_token': [data['per_token'][i] for i in valid_indices]
            }

    logging.info(f"对于 {dataset_name}，从 {initial_count} 个样本中过滤出 {len(valid_indices)} 个 '{target_label}' 样本。")
    return activations_filtered


def apply_transform(activations: torch.Tensor, transform: tuple) -> torch.Tensor:
    """将变换（中心化和可选的白化）应用于给定的激活张量。"""
    W, mu = transform
    activations_float = activations.to(torch.float32)
    mu_float = mu.to(torch.float32)
    if W is not None:
        return (activations_float - mu_float) @ W.to(torch.float32).T
    else:
        return activations_float - mu_float


def calculate_scores(activations: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """计算激活在指定向量上的投影分数。"""
    return activations @ vector.unsqueeze(1)


def calculate_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """计算两个独立样本的 Cohen's d。"""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return 0.0
    dof = nx + ny - 2
    if dof == 0: return 0.0
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_var = ((nx - 1) * var_x + (ny - 1) * var_y) / dof
    if pooled_var < 1e-9: return 0.0
    d = (mean_x - mean_y) / np.sqrt(pooled_var)
    return abs(d)


def select_best_layer(
        harmful_activations: dict,
        benign_activations: dict,
        condition_vectors: dict,
        transforms: dict
) -> tuple[int, dict]:
    """通过 AUROC 和 Cohen's d 选择最佳防御层。"""
    logging.info("正在通过 AUROC 和 Cohen's d 评估选择最佳防御层...")
    layer_metrics = {}
    layers = sorted(list(condition_vectors.keys()))

    for layer in tqdm(layers, desc="评估各层 AUROC 和 Cohen's d"):
        transform_cont = transforms['content_window'].get(layer)
        c_l = condition_vectors.get(layer)
        if transform_cont is None or c_l is None: continue

        H_harmful = harmful_activations.get(layer, {}).get('content_window', {}).get('aggregated')
        H_benign = benign_activations.get(layer, {}).get('content_window', {}).get('aggregated')
        if H_harmful is None or H_benign is None or len(H_harmful) == 0 or len(H_benign) == 0: continue

        z_harmful = apply_transform(H_harmful, transform_cont)
        z_benign = apply_transform(H_benign, transform_cont)

        scores_harmful = calculate_scores(z_harmful, c_l).squeeze().numpy()
        scores_benign = calculate_scores(z_benign, c_l).squeeze().numpy()

        y_true = np.concatenate([np.ones_like(scores_harmful), np.zeros_like(scores_benign)])
        y_scores = np.concatenate([scores_harmful, scores_benign])

        auroc = roc_auc_score(y_true, y_scores)
        cohens_d = calculate_cohens_d(scores_harmful, scores_benign)
        layer_metrics[layer] = {'auroc': auroc, 'cohens_d': cohens_d}

    if not layer_metrics:
        raise ValueError("无法为任何层计算评估指标。")

    sorted_layers = sorted(layer_metrics.items(), key=lambda item: (item[1]['auroc'], item[1]['cohens_d']), reverse=True)

    logging.info("所有层的 AUROC 和 Cohen's d 分数排序如下:")
    for layer, metrics in sorted_layers:
        logging.info(f"  - Layer {layer}: AUROC={metrics['auroc']:.4f}, Cohen's d={metrics['cohens_d']:.4f}")

    best_layer = sorted_layers[0][0]
    logging.info(f"评估完成。最佳防御层: {best_layer} (AUROC: {sorted_layers[0][1]['auroc']:.4f})")
    return best_layer, layer_metrics


def calibrate_theta(
        benign_activations_agg: torch.Tensor,
        condition_vector: torch.Tensor,
        transform: tuple,
        quantile: float
) -> float:
    """为最佳层校准边界阈值 theta。"""
    logging.info(f"正在使用聚合激活校准 theta (分位数: {quantile})...")
    z_benign = apply_transform(benign_activations_agg, transform)
    scores_benign = calculate_scores(z_benign, condition_vector).squeeze().numpy()
    theta = np.quantile(scores_benign, quantile)
    logging.info(f"Theta 校准完成。theta = {theta:.4f}")
    return float(theta)


def process_per_token_scores(
        per_token_activations: list,
        transform: tuple,
        vector: torch.Tensor,
        theta: float
) -> list:
    """使用逐 token 激活计算真实的风险分数序列 r_t。"""
    score_sequences = []
    for seq_tensor in per_token_activations:
        if seq_tensor.shape[0] == 0: continue
        z_sequence = apply_transform(seq_tensor, transform)
        s_sequence = calculate_scores(z_sequence, vector).squeeze()
        # 确保 s_sequence 至少是一维的
        if s_sequence.dim() == 0:
            s_sequence = s_sequence.unsqueeze(0)
        r_sequence = torch.clamp(s_sequence - theta, min=0).numpy()
        score_sequences.append(r_sequence)
    return score_sequences


def calculate_initial_baseline(benign_scores_sequences: list) -> float:
    """根据逐 token 的良性分数计算 CUSUM 的初始基线。"""
    logging.info("正在根据逐 token 的良性分数计算 CUSUM 初始基线...")
    # 过滤掉空的序列
    non_empty_sequences = [s for s in benign_scores_sequences if s.size > 0]
    if not non_empty_sequences:
        logging.warning("没有可用的良性分数来计算基线。默认返回 0.0。")
        return 0.0

    all_benign_scores = np.concatenate(non_empty_sequences)
    baseline = np.mean(all_benign_scores)
    logging.info(f"初始基线计算完成: initial_baseline = {baseline:.4f}")
    return float(baseline)


def run_cusum_simulation(scores: np.ndarray, h: float, k: float, baseline: float) -> tuple[bool, int]:
    """在单条分数序列上运行 CUSUM 模拟。"""
    s_t, m_t = 0.0, 0.0
    for t, r_t in enumerate(scores):
        s_t += (r_t - baseline - k)
        m_t = min(m_t, s_t)
        a_t = s_t - m_t
        if a_t > h:
            return True, t + 1
    return False, len(scores)


def find_optimal_cusum_params(
        harmful_scores_sequences: list,
        benign_scores_sequences: list,
        param_grid: dict,
        target_fpr: float,
        baseline: float
) -> dict:
    """通过网格搜索找到最优的 CUSUM 参数 h 和 k。"""
    logging.info("开始使用逐 token 分数序列扫描 CUSUM 参数...")
    h_values = np.arange(param_grid['h_min'], param_grid['h_max'], param_grid['h_step'])
    k_values = np.arange(param_grid['k_min'], param_grid['k_max'], param_grid['k_step'])
    valid_params = []

    param_combinations = list(itertools.product(h_values, k_values))
    for h, k in tqdm(param_combinations, desc="扫描 (h, k) 组合"):
        trigger_count = sum(1 for scores in benign_scores_sequences if run_cusum_simulation(scores, h, k, baseline)[0])
        fpr = trigger_count / len(benign_scores_sequences) if benign_scores_sequences else 0
        if fpr <= target_fpr:
            valid_params.append({'h': h, 'k': k, 'fpr': fpr})

    if not valid_params:
        logging.warning(f"没有找到满足目标 FPR <= {target_fpr} 的参数组合。")
        return None

    logging.info(f"找到 {len(valid_params)} 组满足 FPR 约束的 (h, k) 参数。")

    best_param = None
    min_avg_delay = float('inf')

    for params in tqdm(valid_params, desc="在有效参数中评估检测延迟"):
        delays = [delay for scores in harmful_scores_sequences for triggered, delay in
                  [run_cusum_simulation(scores, params['h'], params['k'], baseline)] if triggered]
        if delays:
            avg_delay = np.mean(delays)
            if avg_delay < min_avg_delay:
                min_avg_delay = avg_delay
                best_param = params
                best_param['avg_delay'] = avg_delay
                best_param['detection_rate'] = len(delays) / len(harmful_scores_sequences)

    if best_param:
        logging.info(f"最优 CUSUM 参数已找到: h={best_param['h']:.4f}, k={best_param['k']:.4f}")
        logging.info(f"  - 平均检测延迟: {best_param['avg_delay']:.2f} tokens")
    else:
        logging.warning("在满足 FPR 约束的参数中，未能找到任何能够检测出有害样本的组合。")

    return best_param


def main():
    parser = argparse.ArgumentParser(description="校准 SARC 防御系统的参数。")
    parser.add_argument('--config', type=str, default='../configs/calibration_config.yaml',
                        help='防御校准阶段的 YAML 配置文件路径。')
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_dir = Path(__file__).parent.parent
    llm_name = config['llm_name']
    artifacts_dir = base_dir / config['artifacts_dir'] / llm_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = base_dir / config['activations_dir'] / llm_name

    logging.info("正在加载预处理数据...")
    harmful_activations = load_and_filter_data(activations_dir, llm_name, "compliance")
    benign_activations = load_and_filter_data(activations_dir, llm_name, "benign")
    if harmful_activations is None or benign_activations is None:
        logging.error("缺少有害或良性激活数据，无法继续。")
        return

    try:
        condition_vectors = torch.load(artifacts_dir / "condition_vectors.pt", map_location='cpu')
        transforms = torch.load(artifacts_dir / "transforms.pt", map_location='cpu')
    except FileNotFoundError as e:
        logging.error(f"加载向量或变换失败: {e}。")
        return

    best_layer, all_layer_metrics = select_best_layer(harmful_activations, benign_activations, condition_vectors, transforms)

    best_layer_transform = transforms['content_window'][best_layer]
    best_layer_vector = condition_vectors[best_layer]
    benign_activations_agg = benign_activations[best_layer]['content_window']['aggregated']
    theta = calibrate_theta(benign_activations_agg, best_layer_vector, best_layer_transform,
                            config['calibration_params']['theta_quantile'])

    logging.info("正在使用逐 token 激活计算真实分数序列...")
    harmful_scores_sequences = process_per_token_scores(
        harmful_activations[best_layer]['content_window']['per_token'],
        best_layer_transform, best_layer_vector, theta
    )
    benign_scores_sequences = process_per_token_scores(
        benign_activations[best_layer]['content_window']['per_token'],
        best_layer_transform, best_layer_vector, theta
    )

    initial_baseline = calculate_initial_baseline(benign_scores_sequences)

    optimal_cusum_params = find_optimal_cusum_params(
        harmful_scores_sequences, benign_scores_sequences,
        config['cusum_param_grid'], config['calibration_params']['target_fpr'],
        initial_baseline
    )
    if optimal_cusum_params is None:
        logging.error("未能找到合适的 CUSUM 参数。校准失败。")
        return

    defense_config = {
        "llm_name": llm_name,
        "defense_layer": int(best_layer),
        "detection_vector_alias": "c_l",
        "intervention_vector_alias": "v_l",
        "transform_type": "whitening" if transforms['content_window'][best_layer][0] is not None else "centering",
        "scorer": {
            "type": "bounded_relu", "theta": float(theta),
            "upper_bound": None, "compression": None
        },
        "cusum_detector": {
            "h_threshold": float(optimal_cusum_params['h']),
            "k_tolerance": float(optimal_cusum_params['k']),
            "initial_baseline": initial_baseline,
            "baseline_update_rate": config['calibration_params'].get('baseline_update_rate', 0.01)
        },
        "calibration_metrics": {
            "best_layer_auroc": float(all_layer_metrics[best_layer]['auroc']),
            "best_layer_cohens_d": float(all_layer_metrics[best_layer]['cohens_d']),
            "theta_quantile": float(config['calibration_params']['theta_quantile']),
            "target_fpr": float(config['calibration_params']['target_fpr']),
            "final_fpr": float(optimal_cusum_params['fpr']),
            "final_avg_delay": float(optimal_cusum_params.get('avg_delay', -1.0)),
            "final_detection_rate": float(optimal_cusum_params.get('detection_rate', 0.0))
        }
    }

    output_path = artifacts_dir / "defense_config.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(defense_config, f, indent=2, ensure_ascii=False)

    logging.info(f"防御配置文件已成功保存到: {output_path}")
    logging.info("防御校准流程全部完成。")


if __name__ == "__main__":
    main()

