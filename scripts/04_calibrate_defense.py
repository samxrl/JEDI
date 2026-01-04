# -*- coding: utf-8 -*-
"""
脚本 04: 校准防御参数

该脚本遵循“方法流程.md”文档中的第四阶段，负责计算单步风险分数、选择最优层，
并为 CUSUM 累积检测机制校准关键参数。

此版本已更新，以直接使用由 `03_extract_vectors.py` 预先计算并保存的
逐 token 原始分数。

主要功能:
1.  **加载所需构件**:
    - 直接加载“有害满足型”(A1) 和“良性”(B1) 数据集的逐 token 分数文件 (`..._token_scores.pt`)。
    - 在最后阶段加载变换矩阵 (transforms.pt) 和条件向量 (condition_vectors.pt) 以保存最终防御配置。
2.  **选择最优层**:
    - 通过计算区分 A1 和 B1 分数分布的 AUROC 和科恩 d 值，评估每个层的性能。
    - 选择综合性能最佳的层作为防御层。
3.  **校准分数阈值 (theta)**:
    - 对于最优层，使用其在 B1 (良性) 数据集上的分数分布，计算指定分位数 (如 95%) 作为分数阈值 `theta`。
    - 使用 ReLU 函数将原始分数 `s_t` 转换为非负的风险分数 `r_t = max(0, s_t - theta)`。
4.  **校准 CUSUM 参数 (kappa, alpha)**:
    - 在 `(kappa, alpha)` 参数网格上进行搜索。
    - 对每个参数对，在 B1 和 A1 数据集上模拟 CUSUM 过程，计算误报率 (FPR) 和平均检测延迟。
    - 找到在满足目标误报率约束下，具有最低检测延迟的最佳 `(kappa, alpha)` 对。
    - **[!] 优化**: 使用多进程并行处理网格搜索，加快校准速度。
    - **[!] 修复**: 将数据转换为 Numpy 格式传递给子进程，解决 "Too many open files" 错误。
5.  **保存防御参数**:
    - 将所有校准得到的参数 (最优层、theta、kappa、alpha、以及对应的向量和变换) 保存到一个文件中，
      以供在线防御系统使用。
    - [!] 新增: 将网格搜索的详细结果保存为 CSV 表格。

如何运行:
python scripts/04_calibrate_defense.py --config configs/calibration_config.yaml
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
import functools
from multiprocessing import Pool, cpu_count

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_token_scores(data_dir: Path, llm_name: str, dataset_name: str) -> dict:
    """
    加载预先计算的逐 token 分数。
    对于 "compliance" 数据集，会加载对应的 CSV 并进行过滤，以确保样本数与分数文件一致。
    """
    scores_path = data_dir / f"{llm_name}_{dataset_name}_token_scores.pt"
    outputs_path = data_dir / f"{llm_name}_{dataset_name}_outputs.csv"

    if not scores_path.exists() or not outputs_path.exists():
        logging.warning(f"未找到 {dataset_name} 的分数或输出文件，跳过。路径: {scores_path}")
        return None

    logging.info(f"正在加载 {dataset_name} 分数...")
    scores_by_layer = torch.load(scores_path, map_location='cpu')
    df = pd.read_csv(outputs_path)

    # 检查分数文件是否为空
    if not scores_by_layer:
        logging.warning(f"分数文件 {scores_path} 为空。")
        return None

    first_layer = next(iter(scores_by_layer))
    num_scores = len(scores_by_layer[first_layer])

    # 对于 "benign"，分数文件和 CSV 的行数应一致
    if dataset_name == "benign":
        if num_scores != len(df):
            logging.error(f"对于 {dataset_name}，分数文件样本数 ({num_scores}) 与 CSV 文件行数 ({len(df)}) 不匹配。")
            return None
        return scores_by_layer

    # 对于 "compliance"，分数是基于过滤后的样本生成的，所以需要先过滤 CSV 再比较数量
    if dataset_name == "compliance":
        target_label = "yes"
        df.dropna(subset=['label'], inplace=True)
        num_filtered_rows = len(df[df['label'] == target_label])

        if num_scores != num_filtered_rows:
            logging.error(f"对于 {dataset_name}，分数文件样本数 ({num_scores}) 与过滤后的 CSV 文件行数 ({num_filtered_rows}) 不匹配。")
            return None

    # 如果验证通过，直接返回已加载的分数，因为它们已经对应于过滤后的样本
    return scores_by_layer


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


def simulate_cusum(score_sequences, theta, benign_r_mean, kappa, alpha):
    """
    模拟 CUSUM 过程以计算触发率和延迟。

    [!] 修改：支持 Numpy 数组输入，以避免多进程中的文件描述符耗尽问题。
    """
    num_sequences = len(score_sequences)
    if num_sequences == 0: return 0.0, 0.0

    triggers = 0
    total_delay = 0

    for scores in score_sequences:
        # [!] 兼容处理：检查是 Tensor 还是 Numpy
        if isinstance(scores, torch.Tensor):
            if scores.numel() == 0: continue
            scores_np = scores.numpy()
        else:
            # 假设是 Numpy array
            if scores.size == 0: continue
            scores_np = scores

        # [!] 使用 Numpy 进行计算 (np.maximum 替代 torch.clamp)
        r = np.maximum(scores_np - theta, 0)

        A = 0.0
        for t, r_t in enumerate(r):
            # r_t 此时是 numpy scalar，直接运算即可
            A = max(0, A + r_t - benign_r_mean - kappa)
            if A > alpha:
                triggers += 1
                total_delay += (t + 1)
                break

    trigger_rate = triggers / num_sequences if num_sequences > 0 else 0.0
    avg_delay = total_delay / triggers if triggers > 0 else float('inf')

    return trigger_rate, avg_delay


def _evaluate_single_grid_point(params, benign_sequences, harmful_sequences, theta, benign_r_mean):
    """
    辅助函数：计算单个网格点 (kappa, alpha) 的 FPR 和 TPR。
    必须位于顶层以便多进程 pickle。
    """
    kappa, alpha = params
    # 这里的 sequences 已经是 Numpy 数组列表（在 calibrate_cusum 中转换的）
    fpr, _ = simulate_cusum(benign_sequences, theta, benign_r_mean, kappa, alpha)
    tpr, delay = simulate_cusum(harmful_sequences, theta, benign_r_mean, kappa, alpha)
    return {'kappa': kappa, 'alpha': alpha, 'fpr': fpr, 'tpr': tpr, 'delay': delay}


def calibrate_cusum(harmful_sequences, benign_sequences, theta, grid_config):
    """通过网格搜索校准 CUSUM 的 kappa 和 alpha 参数 (使用多进程)。"""
    logging.info(f"使用 theta={theta:.4f} 开始 CUSUM 参数网格搜索...")

    # 计算良性均值 (此处仍可能处理 Tensor 列表，保持原有逻辑即可)
    benign_r_flat = np.concatenate([
        torch.clamp(s - theta, min=0).numpy() if isinstance(s, torch.Tensor) else np.maximum(s - theta, 0)
        for s in benign_sequences if (s.numel() > 0 if isinstance(s, torch.Tensor) else s.size > 0)
    ])

    if len(benign_r_flat) == 0:
        logging.warning("良性风险分数序列为空，无法计算均值。设为 0。")
        benign_r_mean = 0.0
    else:
        benign_r_mean = benign_r_flat.mean()
    logging.info(f"良性风险分数 `r_t` 的均值 (用作 CUSUM 中的 mu_hat): {benign_r_mean:.4f}")

    # 生成参数网格
    kappas = np.arange(grid_config['k_min'], grid_config['k_max'] + grid_config['k_step'], grid_config['k_step'])
    alphas = np.arange(grid_config['alpha_min'], grid_config['alpha_max'] + grid_config['alpha_step'], grid_config['alpha_step'])

    param_grid = [(k, alpha) for k in kappas for alpha in alphas]

    # 获取配置的进程数，默认为 4
    num_processes = grid_config.get('num_processes', 4)
    if num_processes <= 0:
        num_processes = cpu_count()

    logging.info(f"启动多进程网格搜索 (Processes: {num_processes}, Total Points: {len(param_grid)})...")

    # --- [!] 关键修改：转换为 Numpy 列表 ---
    # PyTorch Tensor 在多进程传递时会使用文件描述符（共享内存），大量小 Tensor 会耗尽句柄。
    # Numpy Array 使用 Pickle 序列化，无此问题。
    logging.info("正在将数据转换为 Numpy 格式以避免 'Too many open files' 错误...")
    benign_sequences_np = [s.numpy() if isinstance(s, torch.Tensor) else s for s in benign_sequences]
    harmful_sequences_np = [s.numpy() if isinstance(s, torch.Tensor) else s for s in harmful_sequences]

    # 使用 partial 固定数据参数，只变化 param_grid 中的 (kappa, alpha)
    worker_func = functools.partial(
        _evaluate_single_grid_point,
        benign_sequences=benign_sequences_np,  # 传入 Numpy 数据
        harmful_sequences=harmful_sequences_np,  # 传入 Numpy 数据
        theta=theta,
        benign_r_mean=benign_r_mean
    )

    results = []
    # 使用 multiprocessing.Pool 并行处理
    with Pool(processes=num_processes) as pool:
        # 使用 imap 可以在处理时显示进度条
        for res in tqdm(pool.imap(worker_func, param_grid), total=len(param_grid), desc="CUSUM Grid Search"):
            results.append(res)

    results_df = pd.DataFrame(results)

    # 找到满足 FPR 约束的最佳参数
    valid_params = results_df[results_df['fpr'] <= grid_config['target_fpr']]

    if valid_params.empty:
        logging.warning(f"没有参数组合满足 FPR <= {grid_config['target_fpr']} 的约束。将选择 FPR 最低的组合。")
        best_params = results_df.sort_values(by=['fpr', 'delay']).iloc[0]
    else:
        # [!] 修改：使用加权多目标优化选择最佳参数
        # Score = TPR + alpha * (1 / (Delay + 1))
        # alpha 由配置文件中的 delay_weight 指定，默认为 0.1
        alpha = grid_config.get('delay_weight', 0.1)

        valid_params = valid_params.copy()  # 避免 SettingWithCopyWarning
        valid_params['score'] = valid_params['tpr'] + alpha * (1.0 / (valid_params['delay'] + 1.0))

        best_params = valid_params.sort_values(by='score', ascending=False).iloc[0]

        logging.info(f"使用加权评分选择最佳参数 (alpha={alpha}, Score = TPR + alpha/(Delay+1))。")
        logging.info(f"在 {len(valid_params)} 个满足 FPR 约束的组合中，选出了得分最高的一组 (Score={best_params['score']:.4f})。")

    logging.info("CUSUM 网格搜索结果摘要:\n" + results_df.to_string(max_rows=20))
    logging.info(
        f"*** 最佳 CUSUM 参数 (FPR <= {grid_config['target_fpr']}): "
        f"kappa={best_params['kappa']:.4f}, alpha={best_params['alpha']:.4f} "
        f"-> (FPR={best_params['fpr']:.4f}, TPR={best_params['tpr']:.4f}, Delay={best_params['delay']:.2f}) ***"
    )

    return best_params.to_dict(), results_df, benign_r_mean


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
    data_dir = base_dir / config['activations_dir'] / llm_name
    output_dir = base_dir / config['activations_dir'] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"防御参数将保存到: {output_dir}")

    # --- 3. 加载预计算的逐 token 分数 ---
    harmful_sequences_by_layer = load_token_scores(data_dir, llm_name, "compliance")
    benign_sequences_by_layer = load_token_scores(data_dir, llm_name, "benign")

    if not harmful_sequences_by_layer or not benign_sequences_by_layer:
        logging.error("缺少必要的 token 分数文件，无法继续。请先运行 03_extract_vectors.py。")
        return

    layers = sorted(harmful_sequences_by_layer.keys())
    harmful_scores_by_layer, benign_scores_by_layer = {}, {}

    # --- 4. 扁平化分数用于层评估 ---
    for layer in layers:
        h_seq = harmful_sequences_by_layer.get(layer, [])
        b_seq = benign_sequences_by_layer.get(layer, [])

        if h_seq:
            harmful_scores_by_layer[layer] = np.concatenate([s.numpy() for s in h_seq if s.numel() > 0])
        if b_seq:
            benign_scores_by_layer[layer] = np.concatenate([s.numpy() for s in b_seq if s.numel() > 0])

    # --- 5. 选择最优层 ---
    best_layer_idx, _ = find_best_layer(harmful_scores_by_layer, benign_scores_by_layer, layers)

    # --- 6. 为最优层校准参数 ---
    # 计算 theta
    benign_scores_best_layer = benign_scores_by_layer[best_layer_idx]
    theta = np.quantile(benign_scores_best_layer, config['calibration_params']['theta_quantile'])

    # 校准 CUSUM (调用多进程版本)
    best_cusum_params, grid_search_results_df, benign_r_mean = calibrate_cusum(
        harmful_sequences_by_layer[best_layer_idx],
        benign_sequences_by_layer[best_layer_idx],
        theta,
        config['calibration_params']
    )

    # --- 7. 加载向量和变换，以保存最终的防御参数 ---
    try:
        transforms = torch.load(data_dir / "transforms.pt", map_location='cpu')
        condition_vectors = torch.load(data_dir / "condition_vectors.pt", map_location='cpu')
    except FileNotFoundError as e:
        logging.error(f"加载文件失败: {e}。请先运行 03_extract_vectors.py。")
        return

    defense_params = {
        'llm_name': llm_name,
        'best_layer': best_layer_idx,
        'theta': float(theta),
        'mu_hat': float(benign_r_mean),
        'kappa': best_cusum_params['kappa'],
        'alpha': best_cusum_params['alpha'],
    }

    # --- 8. 保存结果 ---

    # 8a. 保存 defense_params.yaml
    save_path = output_dir / "defense_params.yaml"
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(defense_params, f, indent=2, allow_unicode=True, sort_keys=False)
    logging.info(f"防御参数已成功校准并保存到: {save_path}")

    # 8b. [!] 保存网格搜索结果到 CSV
    grid_results_path = output_dir / "cusum_grid_search_results.csv"
    grid_search_results_df.to_csv(grid_results_path, index=False, encoding='utf-8-sig')
    logging.info(f"CUSUM 网格搜索详细结果已保存到: {grid_results_path}")

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