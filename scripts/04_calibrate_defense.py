# -*- coding: utf-8 -*-
"""Script 04: Calibrate defense parameters

This script follows the fourth stage in the "Methodflow.md" document and is responsible for calculating the single-step risk score, selecting the optimal layer,
and calibrating key parameters for the CUSUM accumulation detection mechanism.

This version has been updated to directly use the precomputed and saved by `03_extract_vectors.py`
Token-by-token raw score.

Main functions:
1. **Load required components**:
    - Directly load the token-by-token score files (`..._token_scores.pt`) for the "harmful" (A1) and "benign" (B1) datasets.
    - Load transformation matrices (transforms.pt) and condition vectors (condition_vectors.pt) in the final stage to save the final defense configuration.
2. **Select the optimal layer**:
    - Evaluate the performance of each layer by calculating the AUROC and Cohen's d values that distinguish the A1 and B1 score distributions.
    - Select the layer with the best overall performance as the defense layer.
3. **Calibration score threshold (theta)**:
    - For the optimal layer, calculate the specified quantile (e.g. 95%) as the score threshold `theta` using its score distribution on the B1 (benign) dataset.
    - Use the ReLU function to convert the original score `s_t` into a non-negative risk score `r_t = max(0, s_t - theta)`.
4. **Calibrate CUSUM parameters (kappa, alpha)**:
    - Search on `(kappa, alpha)` parameter grid.
    - For each parameter pair, simulate the CUSUM process on the B1 and A1 datasets and calculate the false positive rate (FPR) and average detection latency.
    - Find the best `(kappa, alpha)` pair with the lowest detection latency subject to the target false alarm rate constraint.
    - **Optimization**: Use multiple processes to process grid search in parallel to speed up calibration.
    - **Fix**: Convert data to Numpy format and pass it to the child process to resolve "Too many open files" errors.
5. **Save defense parameters**:
    - Save all calibrated parameters (optimal layer, theta, kappa, alpha, and corresponding vectors and transformations) to a file,
      For use by online defense systems.
    - New: Save detailed results of grid search as CSV table.

How to run:
python scripts/04_calibrate_defense.py --config configs/calibration_config.yaml"""

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

# Configuration log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_token_scores(data_dir: Path, llm_name: str, dataset_name: str) -> dict:
    """Loads precomputed token-by-token scores.
    For the "compliance" dataset, the corresponding CSV is loaded and filtered to ensure that the number of samples matches the score file.
    """
    scores_path = data_dir / f"{llm_name}_{dataset_name}_token_scores.pt"
    outputs_path = data_dir / f"{llm_name}_{dataset_name}_outputs.csv"

    if not scores_path.exists() or not outputs_path.exists():
        logging.warning(
            f"not found{dataset_name}The score or output file is skipped. path:{scores_path}"
        )
        return None

    logging.info(f"Loading{dataset_name}Fraction...")
    scores_by_layer = torch.load(scores_path, map_location="cpu")
    df = pd.read_csv(outputs_path)

    # Check if score file is empty
    if not scores_by_layer:
        logging.warning(f"score file{scores_path}is empty.")
        return None

    first_layer = next(iter(scores_by_layer))
    num_scores = len(scores_by_layer[first_layer])

    # For "benign", the scores file and CSV should have the same number of rows
    if dataset_name == "benign":
        if num_scores != len(df):
            logging.error(
                f"for{dataset_name}, the number of score file samples ({num_scores}) and the number of CSV file lines ({len(df)}) does not match."
            )
            return None
        return scores_by_layer

        # For "compliance", the scores are generated based on filtered samples, so the CSV needs to be filtered first before comparing the quantities
    if dataset_name == "compliance":
        target_label = "yes"
        df.dropna(subset=["label"], inplace=True)
        num_filtered_rows = len(df[df["label"] == target_label])

        if num_scores != num_filtered_rows:
            logging.error(
                f"for{dataset_name}, the number of score file samples ({num_scores}) and the number of rows in the filtered CSV file ({num_filtered_rows}) does not match."
            )
            return None

            # If the validation passes, return the loaded scores directly since they already correspond to the filtered samples
    return scores_by_layer


def cohen_d(x, y):
    """Calculate Cohen's d value for both sets of data."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def find_best_layer(harmful_scores_by_layer, benign_scores_by_layer, layers):
    """The best defense layer was selected based on AUROC and Cohen's d value."""
    results = []
    for layer in layers:
        harmful_scores = harmful_scores_by_layer.get(layer)
        benign_scores = benign_scores_by_layer.get(layer)

        if (
            harmful_scores is None
            or benign_scores is None
            or len(harmful_scores) == 0
            or len(benign_scores) == 0
        ):
            continue

        y_true = np.concatenate(
            [np.ones_like(harmful_scores), np.zeros_like(benign_scores)]
        )
        y_score = np.concatenate([harmful_scores, benign_scores])

        auroc = roc_auc_score(y_true, y_score)
        d = cohen_d(harmful_scores, benign_scores)
        results.append({"layer": layer, "auroc": auroc, "cohen_d": d})

    if not results:
        raise ValueError("No scores from any strata are available for evaluation.")

    results_df = pd.DataFrame(results).sort_values(
        by=["auroc", "cohen_d"], ascending=False
    )
    best_layer = results_df.iloc[0]

    logging.info("Layer evaluation results:\\n" + results_df.to_string())
    logging.info(
        f"*** Selected best layer:{best_layer['layer']} (AUROC={best_layer['auroc']:.4f}, Cohen's d={best_layer['cohen_d']:.4f}) ***"
    )

    return int(best_layer["layer"]), results_df


def simulate_cusum(score_sequences, theta, benign_r_mean, kappa, alpha):
    """Simulate the CUSUM process to calculate trigger rate and latency.

    Modification: Support Numpy array input to avoid file descriptor exhaustion issues in multiple processes.
    """
    num_sequences = len(score_sequences)
    if num_sequences == 0:
        return 0.0, 0.0

    triggers = 0
    total_delay = 0

    for scores in score_sequences:
        # Compatibility processing: check if it is Tensor or Numpy
        if isinstance(scores, torch.Tensor):
            if scores.numel() == 0:
                continue
            scores_np = scores.numpy()
        else:
            # Assume it is a Numpy array
            if scores.size == 0:
                continue
            scores_np = scores

            # Computation using Numpy (np.maximum replaces torch.clamp)
        r = np.maximum(scores_np - theta, 0)

        A = 0.0
        for t, r_t in enumerate(r):
            # r_t is numpy scalar at this time and can be calculated directly
            A = max(0, A + r_t - benign_r_mean - kappa)
            if A > alpha:
                triggers += 1
                total_delay += t + 1
                break

    trigger_rate = triggers / num_sequences if num_sequences > 0 else 0.0
    avg_delay = total_delay / triggers if triggers > 0 else float("inf")

    return trigger_rate, avg_delay


def _evaluate_single_grid_point(
    params, benign_sequences, harmful_sequences, theta, benign_r_mean
):
    """Helper function: Computes FPR and TPR for a single grid point (kappa, alpha).
    Must be at the top level for multi-process pickle."""
    kappa, alpha = params
    # The sequences here are already Numpy array lists (converted in calibrate_cusum)
    fpr, _ = simulate_cusum(benign_sequences, theta, benign_r_mean, kappa, alpha)
    tpr, delay = simulate_cusum(harmful_sequences, theta, benign_r_mean, kappa, alpha)
    return {"kappa": kappa, "alpha": alpha, "fpr": fpr, "tpr": tpr, "delay": delay}


def calibrate_cusum(harmful_sequences, benign_sequences, theta, grid_config):
    """Calibration of kappa and alpha parameters of CUSUM via grid search (using multiprocess)."""
    logging.info(f"Use theta={theta:.4f}Starting CUSUM parameter grid search...")

    # Calculate the benign mean (it is still possible to deal with Tensor lists here, just keep the original logic)
    benign_r_flat = np.concatenate(
        [
            (
                torch.clamp(s - theta, min=0).numpy()
                if isinstance(s, torch.Tensor)
                else np.maximum(s - theta, 0)
            )
            for s in benign_sequences
            if (s.numel() > 0 if isinstance(s, torch.Tensor) else s.size > 0)
        ]
    )

    if len(benign_r_flat) == 0:
        logging.warning(
            "The benign risk score series is empty and the mean cannot be calculated. Set to 0."
        )
        benign_r_mean = 0.0
    else:
        benign_r_mean = benign_r_flat.mean()
    logging.info(
        f"Mean of benign risk score `r_t` (used as mu_hat in CUSUM):{benign_r_mean:.4f}"
    )

    # Generate parametric grid
    kappas = np.arange(
        grid_config["k_min"],
        grid_config["k_max"] + grid_config["k_step"],
        grid_config["k_step"],
    )
    alphas = np.arange(
        grid_config["alpha_min"],
        grid_config["alpha_max"] + grid_config["alpha_step"],
        grid_config["alpha_step"],
    )

    param_grid = [(k, alpha) for k in kappas for alpha in alphas]

    # Get the configured number of processes, the default is 4
    num_processes = grid_config.get("num_processes", 4)
    if num_processes <= 0:
        num_processes = cpu_count()

    logging.info(
        f"Start a multi-process grid search (Processes:{num_processes}, Total Points: {len(param_grid)})..."
    )

    # --- Key modification: Convert to Numpy list ---
    # PyTorch Tensors use file descriptors (shared memory) when passed to multiple processes, and a large number of small Tensors can run out of handles.
    # Numpy Array uses Pickle serialization and does not have this problem.
    logging.info(
        "Converting data to Numpy format to avoid 'Too many open files' errors..."
    )
    benign_sequences_np = [
        s.numpy() if isinstance(s, torch.Tensor) else s for s in benign_sequences
    ]
    harmful_sequences_np = [
        s.numpy() if isinstance(s, torch.Tensor) else s for s in harmful_sequences
    ]

    # Use partial to fix data parameters and only change (kappa, alpha) in param_grid
    worker_func = functools.partial(
        _evaluate_single_grid_point,
        benign_sequences=benign_sequences_np,  # Pass in Numpy data
        harmful_sequences=harmful_sequences_np,  # Pass in Numpy data
        theta=theta,
        benign_r_mean=benign_r_mean,
    )

    results = []
    # Parallel processing using multiprocessing.Pool
    with Pool(processes=num_processes) as pool:
        # Use imap to display a progress bar while processing
        for res in tqdm(
            pool.imap(worker_func, param_grid),
            total=len(param_grid),
            desc="CUSUM Grid Search",
        ):
            results.append(res)

    results_df = pd.DataFrame(results)

    # Find optimal parameters that satisfy FPR constraints
    valid_params = results_df[results_df["fpr"] <= grid_config["target_fpr"]]

    if valid_params.empty:
        logging.warning(
            f"No parameter combination satisfies FPR <={grid_config['target_fpr']}constraints. The combination with the lowest FPR will be selected."
        )
        best_params = results_df.sort_values(by=["fpr", "delay"]).iloc[0]
    else:
        # Modification: Use weighted multi-objective optimization to select optimal parameters
        # Score = TPR + alpha * (1 / (Delay + 1))
        # alpha is specified by delay_weight in the configuration file and defaults to 0.1
        alpha = grid_config.get("delay_weight", 0.1)

        valid_params = valid_params.copy()  # Avoid SettingWithCopyWarning
        valid_params["score"] = valid_params["tpr"] + alpha * (
            1.0 / (valid_params["delay"] + 1.0)
        )

        best_params = valid_params.sort_values(by="score", ascending=False).iloc[0]

        logging.info(
            f"Use weighted scoring to select the best parameters (alpha={alpha}, Score = TPR + alpha/(Delay+1))。"
        )
        logging.info(
            f"exist{len(valid_params)}Among the combinations that satisfy the FPR constraint, the group with the highest score is selected (Score={best_params['score']:.4f})。"
        )

    logging.info(
        "CUSUM grid search result summary:\\n" + results_df.to_string(max_rows=20)
    )
    logging.info(
        f"*** Best CUSUM parameters (FPR <={grid_config['target_fpr']}): "
        f"kappa={best_params['kappa']:.4f}, alpha={best_params['alpha']:.4f} "
        f"-> (FPR={best_params['fpr']:.4f}, TPR={best_params['tpr']:.4f}, Delay={best_params['delay']:.2f}) ***"
    )

    return best_params.to_dict(), results_df, benign_r_mean


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the parameters of the defense system."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/calibration_config.yaml",
        help="Configuration file path.",
    )
    args = parser.parse_args()

    # --- 1. Load configuration ---
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

        # --- 2. Set path ---
    base_dir = Path(__file__).parent.parent
    llm_name = config["llm_name"]
    data_dir = base_dir / config["activations_dir"] / llm_name
    output_dir = base_dir / config["activations_dir"] / llm_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Defense parameters will be saved to:{output_dir}")

    # --- 3. Load precomputed token-by-token scores ---
    harmful_sequences_by_layer = load_token_scores(data_dir, llm_name, "compliance")
    benign_sequences_by_layer = load_token_scores(data_dir, llm_name, "benign")

    if not harmful_sequences_by_layer or not benign_sequences_by_layer:
        logging.error(
            "The necessary token score file is missing and cannot continue. Please run 03_extract_vectors.py first."
        )
        return

    layers = sorted(harmful_sequences_by_layer.keys())
    harmful_scores_by_layer, benign_scores_by_layer = {}, {}

    # --- 4. Flattened scores for layer evaluation ---
    for layer in layers:
        h_seq = harmful_sequences_by_layer.get(layer, [])
        b_seq = benign_sequences_by_layer.get(layer, [])

        if h_seq:
            harmful_scores_by_layer[layer] = np.concatenate(
                [s.numpy() for s in h_seq if s.numel() > 0]
            )
        if b_seq:
            benign_scores_by_layer[layer] = np.concatenate(
                [s.numpy() for s in b_seq if s.numel() > 0]
            )

            # --- 5. Select the optimal layer ---
    best_layer_idx, _ = find_best_layer(
        harmful_scores_by_layer, benign_scores_by_layer, layers
    )

    # --- 6. Calibrate parameters for optimal layer ---
    # Calculate theta
    benign_scores_best_layer = benign_scores_by_layer[best_layer_idx]
    theta = np.quantile(
        benign_scores_best_layer, config["calibration_params"]["theta_quantile"]
    )

    # Calibrate CUSUM (call multi-process version)
    best_cusum_params, grid_search_results_df, benign_r_mean = calibrate_cusum(
        harmful_sequences_by_layer[best_layer_idx],
        benign_sequences_by_layer[best_layer_idx],
        theta,
        config["calibration_params"],
    )

    # --- 7. Load vectors and transformations to save final defense parameters ---
    try:
        transforms = torch.load(data_dir / "transforms.pt", map_location="cpu")
        condition_vectors = torch.load(
            data_dir / "condition_vectors.pt", map_location="cpu"
        )
    except FileNotFoundError as e:
        logging.error(
            f"Failed to load file:{e}. Please run 03_extract_vectors.py first."
        )
        return

    defense_params = {
        "llm_name": llm_name,
        "best_layer": best_layer_idx,
        "theta": float(theta),
        "mu_hat": float(benign_r_mean),
        "kappa": best_cusum_params["kappa"],
        "alpha": best_cusum_params["alpha"],
    }

    # --- 8. Save results ---

    # 8a. Save defense_params.yaml
    save_path = output_dir / "defense_params.yaml"
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(defense_params, f, indent=2, allow_unicode=True, sort_keys=False)
    logging.info(
        f"Defense parameters have been successfully calibrated and saved to:{save_path}"
    )

    # 8b. Save grid search results to CSV
    grid_results_path = output_dir / "cusum_grid_search_results.csv"
    grid_search_results_df.to_csv(grid_results_path, index=False, encoding="utf-8-sig")
    logging.info(f"CUSUM grid search detailed results saved to:{grid_results_path}")

    # (Optional) Draw the optimal layer score distribution map
    plt.figure(figsize=(10, 6))
    sns.histplot(
        benign_scores_best_layer,
        color="green",
        label="Benign (B1)",
        bins=100,
        stat="density",
        alpha=0.6,
    )
    sns.histplot(
        harmful_scores_by_layer[best_layer_idx],
        color="red",
        label="Harmful (A1)",
        bins=100,
        stat="density",
        alpha=0.6,
    )
    plt.axvline(
        theta,
        color="blue",
        linestyle="--",
        label=f'Theta (q={config["calibration_params"]["theta_quantile"]}) = {theta:.4f}',
    )
    plt.title(
        f"Token Score Distribution for Best Layer ({best_layer_idx}) on {llm_name}"
    )
    plt.xlabel("Raw Score (s_t)")
    plt.ylabel("Density")
    plt.legend()
    plot_path = output_dir / "best_layer_score_distribution.png"
    plt.savefig(plot_path)
    logging.info(
        f"The optimal layer score distribution map has been saved to:{plot_path}"
    )


if __name__ == "__main__":
    main()
