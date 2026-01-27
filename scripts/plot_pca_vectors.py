# -*- coding: utf-8 -*-
"""Script 05: Visualizing PCA-processed vectors

This script is used to visualize activations and vectors extracted by `03_extract_vectors.py`.

Main functions:
1. Load the "compliance", "refusal" and "benign" datasets
    The content window (`content_window`) is activated.
2. Load the whitening matrix and the extracted intervention vector (v_l) and condition vector (c_l).
3. Apply a whitening transform to each specified level of activation.
4. Combine the whitened activations of the three types of samples and use PCA to reduce their dimensionality to a two-dimensional space.
5. Generate a scatter plot for each level, using different colors to show the distribution of the three types of samples in the two-dimensional space.
6. Use arrows on the diagram to mark the projection directions of the intervention vector and the condition vector in the two-dimensional space.
7. Merge all plots into a grid plot and save as an image file.

How to run:
# Draw the default layers
python scripts/05_plot_pca_vectors.py --llm_name "vicuna_7b_v1_5"

# Draw the specified layer
python scripts/05_plot_pca_vectors.py --llm_name "vicuna_7b_v1_5" --layers_to_plot 8 12 16 20 24 28 30 31

# Draw all available layers
python scripts/05_plot_pca_vectors.py --llm_name "vicuna_7b_v1_5" --plot_all_layers"""

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

# Configuration log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_filter_data(
    data_dir: Path, llm_name: str, dataset_name: str
) -> tuple[dict, pd.DataFrame]:
    """Load activation tensors and corresponding labeled CSV files and filter based on labels.
    This function is consistent with the version in 03_extract_vectors.py."""
    activations_path = data_dir / f"{llm_name}_{dataset_name}_activations.pt"
    outputs_path = data_dir / f"{llm_name}_{dataset_name}_outputs.csv"

    if not activations_path.exists() or not outputs_path.exists():
        logging.warning(
            f"not found{dataset_name}Activation or output file, skip loading. path:{activations_path}"
        )
        return None, None

    logging.info(f"Loading{dataset_name}data...")
    activations = torch.load(activations_path, map_location="cpu")
    df = pd.read_csv(outputs_path)

    if dataset_name == "compliance":
        target_label = "yes"
    elif dataset_name == "refusal":
        target_label = "no"
    else:  # benign
        return activations, df

    initial_count = len(df)
    df.dropna(subset=["label"], inplace=True)
    valid_indices = df.index[df["label"] == target_label].tolist()

    df_filtered = df.loc[valid_indices].reset_index(drop=True)

    activations_filtered = {}
    for layer, windows in activations.items():
        activations_filtered[layer] = {}
        for window_name, tensor in windows.items():
            if tensor.shape[0] != initial_count:
                logging.warning(
                    f"exist{dataset_name} (L{layer}, {window_name}), the number of activations ({tensor.shape[0]}) and the number of CSV rows ({initial_count}) does not match. Skip this tensor."
                )
                continue
            activations_filtered[layer][window_name] = tensor[valid_indices]

    logging.info(
        f"for{dataset_name},from{initial_count}filtered out from samples{len(df_filtered)}tagged '{target_label}' sample."
    )
    return activations_filtered, df_filtered


def apply_whitening(activations: torch.Tensor, transform: tuple) -> torch.Tensor:
    """Applies a whitening transform to the given activation tensor.
    This function is consistent with the version in 03_extract_vectors.py."""
    W, mu = transform
    # Move all tensors to CPU for computation
    activations, W, mu = activations.cpu(), W.cpu(), mu.cpu()
    return (activations - mu) @ W.T


def main():
    parser = argparse.ArgumentParser(
        description="Visualizing the hidden activation distribution after PCA.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="vicuna_7b_v1_5",
        help="Must match the base name of `model_name` in extraction_config.yaml.",
    )
    parser.add_argument(
        "--layers_to_plot",
        type=int,
        nargs="+",
        default=[8, 16, 24, 31],
        help="List of layer indices to draw. If --plot_all_layers is provided, this argument is ignored. Default value: [8, 16, 24, 31].",
    )
    parser.add_argument(
        "--plot_all_layers",
        action="store_true",
        default=True,
        help="If specified, plots all available layers, ignoring --layers_to_plot. Default is no.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/activations",
        help="Directory containing activations, vectors, and whitening matrices.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save generated visualization images.",
    )
    parser.add_argument(
        "--plot_filename",
        type=str,
        default="pca_activation_distribution.png",
        help="The filename of the output image.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Maximum number of points drawn per category. Set to 0 to plot all points.",
    )

    args = parser.parse_args()

    # --- 1. Set parameters and paths ---
    sns.set_theme(style="whitegrid", palette="deep")

    colors = {
        "Benign": "#2ca02c",  # tab:green
        "Refusal": "#1f77b4",  # tab:blue
        "Compliance (Harmful)": "#d62728",  # tab:red
    }
    vec_colors = {
        "c_vector": "#9467bd",  # tab:purple
        "v_vector": "#ff7f0e",  # tab:orange
    }

    base_dir = Path(__file__).parent.parent
    llm_name = args.llm_name
    data_dir = base_dir / args.data_dir / llm_name
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"The visualization will be saved to:{output_dir}")

    # --- 2. Load the required data ---
    logging.info("Starting loading activations, vectors and whitening matrices...")
    compliance_activations, _ = load_and_filter_data(data_dir, llm_name, "compliance")
    refusal_activations, _ = load_and_filter_data(data_dir, llm_name, "refusal")
    benign_activations, _ = load_and_filter_data(data_dir, llm_name, "benign")

    if not all([compliance_activations, refusal_activations, benign_activations]):
        logging.error(
            "The necessary dataset activation files are missing and cannot continue."
        )
        return

    try:
        whitening_transforms = torch.load(
            data_dir / "whitening_matrices.pt", map_location="cpu"
        )
        intervention_vectors = torch.load(
            data_dir / "intervention_vectors.pt", map_location="cpu"
        )
        condition_vectors = torch.load(
            data_dir / "condition_vectors.pt", map_location="cpu"
        )
    except FileNotFoundError as e:
        logging.error(
            f"Failed to load vector or whitening matrix:{e}. Please make sure you have successfully run 03_extract_vectors.py."
        )
        return

        # --- 3. Set up drawing ---
    if args.plot_all_layers:
        layers_to_plot = sorted(list(whitening_transforms.keys()))
        logging.info(
            f"--plot_all_layers detected. will draw all{len(layers_to_plot)}available tiers."
        )
    else:
        layers_to_plot = args.layers_to_plot

    n_layers = len(layers_to_plot)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False
    )
    axes = axes.flatten()

    sample_size = args.sample_size

    # --- 4. Loop through and draw each layer ---
    for i, layer in enumerate(
        tqdm(layers_to_plot, desc="Generating images for each layer")
    ):
        ax = axes[i]

        if layer not in whitening_transforms:
            logging.warning(
                f"No.{layer}Layer has no whitening matrix, skipping drawing."
            )
            ax.text(0.5, 0.5, f"Layer {layer}\nNo Data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        transform = whitening_transforms[layer]

        # Extract and whiten activation
        H_benign = benign_activations[layer]["content_window"]
        H_refusal = refusal_activations[layer]["content_window"]
        H_compliance = compliance_activations[layer]["content_window"]

        z_benign = apply_whitening(H_benign, transform)
        z_refusal = apply_whitening(H_refusal, transform)
        z_compliance = apply_whitening(H_compliance, transform)

        # --- Balance and sample data ---
        min_available_samples = min(len(z_benign), len(z_refusal), len(z_compliance))

        if sample_size > 0:
            # If the sampling size is set, the smaller of the user-specified value and the minimum number of samples available is taken
            final_sample_count = min(sample_size, min_available_samples)
        else:
            # If sample_size is 0 (meaning draw all), the smallest number of samples available is used to ensure balance
            final_sample_count = min_available_samples

        if i == 0:  # Only print the log once when processing the first layer
            logging.info(
                f"To ensure that each category has the same number of points, a number will be drawn for each category{final_sample_count}point."
            )

            # Sampling using a determined quantity
        z_benign = z_benign[torch.randperm(z_benign.size(0))[:final_sample_count]]
        z_refusal = z_refusal[torch.randperm(z_refusal.size(0))[:final_sample_count]]
        z_compliance = z_compliance[
            torch.randperm(z_compliance.size(0))[:final_sample_count]
        ]

        # Combine data and run PCA
        all_whitened = torch.cat([z_benign, z_refusal, z_compliance], dim=0)
        pca = PCA(n_components=2)
        pca.fit(all_whitened.numpy())

        # Transform data into two-dimensional space
        proj_benign = pca.transform(z_benign.numpy())
        proj_refusal = pca.transform(z_refusal.numpy())
        proj_compliance = pca.transform(z_compliance.numpy())

        # Create DataFrame for seaborn
        data_benign = pd.DataFrame(proj_benign, columns=["PC1", "PC2"])
        data_benign["Category"] = "Benign"
        data_refusal = pd.DataFrame(proj_refusal, columns=["PC1", "PC2"])
        data_refusal["Category"] = "Refusal"
        data_compliance = pd.DataFrame(proj_compliance, columns=["PC1", "PC2"])
        data_compliance["Category"] = "Compliance (Harmful)"
        # Reorder to control draw order: draw harmful and rejects first, benign last so that they are on top
        plot_df = pd.concat(
            [data_refusal, data_compliance, data_benign], ignore_index=True
        )

        # Use seaborn to draw scatter plots
        sns.scatterplot(
            data=plot_df,
            x="PC1",
            y="PC2",
            hue="Category",
            # Explicitly specify hue_order to ensure correct legend order
            hue_order=["Benign", "Refusal", "Compliance (Harmful)"],
            palette=colors,
            ax=ax,
            alpha=0.7,
            s=20,
            edgecolor="w",
            linewidth=0.5,
        )
        # Remove individual legends for each subplot
        if ax.get_legend() is not None:
            ax.get_legend().remove()

            # Transform and plot the c_l and v_l vectors
        v_l = intervention_vectors.get(layer)
        c_l = condition_vectors.get(layer)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        arrow_scale = np.mean([np.abs(xlim).sum(), np.abs(ylim).sum()]) * 0.2

        if c_l is not None and torch.norm(c_l) > 0:
            proj_c = pca.transform(c_l.numpy().reshape(1, -1))
            ax.quiver(
                0,
                0,
                proj_c[0, 0] * arrow_scale,
                proj_c[0, 1] * arrow_scale,
                color=vec_colors.get("c_vector"),
                scale=1,
                scale_units="xy",
                angles="xy",
                width=0.01,
                label=r"$c_l$ (Harmful Dir)",
            )

        if v_l is not None and torch.norm(v_l) > 0:
            proj_v = pca.transform(v_l.numpy().reshape(1, -1))
            ax.quiver(
                0,
                0,
                proj_v[0, 0] * arrow_scale,
                proj_v[0, 1] * arrow_scale,
                color=vec_colors.get("v_vector"),
                scale=1,
                scale_units="xy",
                angles="xy",
                width=0.01,
                label=r"$v_l$ (Refusal Dir)",
            )

        ax.set_title(f"Layer {layer}", fontsize=12)

        # --- 5. Clean and save image ---
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

        # Create a global legend
    handles, labels = [], []
    # Get legend items from scatter plot
    for cat, color in colors.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=cat,
                markerfacecolor=color,
                markersize=10,
            )
        )
        labels.append(cat)
        # Get legend item from vector arrow
    for cat, color in vec_colors.items():
        label_text = (
            r"$c_l$ (Harmful Dir)" if cat == "c_vector" else r"$v_l$ (Refusal Dir)"
        )
        handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label_text))
        labels.append(label_text)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.01),
            frameon=True,
            fontsize=12,
        )

    fig.suptitle(f"PCA of Hidden Activations for {llm_name}", fontsize=18, y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    output_path = output_dir / args.plot_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"The visualization was successfully saved to:{output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
