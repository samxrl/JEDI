# Modified in JEDI's CAST adapter:
# - suffix-only aggregation now uses the suffix aligned to each positive/negative
#   sample and derives its exact token span instead of reusing suffixes[0][0].
# - bf16 hidden states are converted to float32 before NumPy/scikit-learn use.
# - annotations are deferred for Python 3.9 compatibility.
# - NumPy PCA variance scalars are converted to native floats before JSON save.
# See ../../../../patches/README.md.
from __future__ import annotations

import dataclasses
import os, json
import typing
import warnings

import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import matplotlib.pyplot as plt

from activation_steering.malleable_model import MalleableModel, get_model_layer_list
from activation_steering.utils import ContrastivePair, custom_progress
from activation_steering.steering_dataset import SteeringDataset
from activation_steering.config import log, GlobalConfig




@dataclasses.dataclass
class SteeringVector:
    """
    A dataclass representing a steering vector used for guiding the language model.

    Attributes:
        model_type: The type of the model this vector is associated with.
        directions: A dictionary mapping layer IDs to numpy arrays of directions.
        explained_variances: A dictionary of explained variances.
    """
    model_type: str
    directions: dict[int, np.ndarray]
    explained_variances: dict

    @classmethod
    def train(cls, model: MalleableModel | PreTrainedModel, tokenizer: PreTrainedTokenizerBase, steering_dataset: SteeringDataset, **kwargs) -> "SteeringVector":
        """
        Train a SteeringVector for a given model and tokenizer using the provided dataset.

        Args:
            model: The model to train the steering vector for.
            tokenizer: The tokenizer associated with the model.
            steering_dataset: The dataset to use for training.
            **kwargs: Additional keyword arguments.

        Returns:
            A new SteeringVector instance.
        """
        log("Training steering vector", class_name="SteeringVector")
        # Set the pad_token_id of the tokenizer to 0
        tokenizer.pad_token_id = 0
        
        dirs, variances = read_representations(
            model,
            tokenizer,
            steering_dataset.formatted_dataset,
            suffixes=steering_dataset.suffixes,
            **kwargs,
        )
        
        return cls(model_type=model.config.model_type, 
                   directions=dirs, 
                   explained_variances=variances)
    
    def save(self, file_path: str):
        """
        Save the SteeringVector to a file.

        Args:
            file_path: The path to save the file to. If it doesn't end with '.svec', 
                       this extension will be added.
        """
        if not file_path.endswith('.svec'):
            file_path += '.svec'
        
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            log(f"Created directory: {directory}", class_name="SteeringVector")
        

        log(f"Saving SteeringVector to {file_path}", class_name="SteeringVector")
        data = {
            "model_type": self.model_type,
            "directions": {k: v.tolist() for k, v in self.directions.items()},
            "explained_variances": {
                k: float(v) for k, v in self.explained_variances.items()
            }
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        log(f"SteeringVector saved successfully", class_name="SteeringVector")

    @classmethod
    def load(cls, file_path: str) -> "SteeringVector":
        """
        Load a SteeringVector from a file.

        Args:
            file_path: The path to load the file from. If it doesn't end with '.svec', 
                       this extension will be added.

        Returns:
            A new SteeringVector instance loaded from the file.
        """
        if not file_path.endswith('.svec'):
            file_path += '.svec'
        
        log(f"Loading SteeringVector from {file_path}", class_name="SteeringVector")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        directions = {int(k): np.array(v) for k, v in data["directions"].items()}
        explained_variances = {int(k): v for k, v in data["explained_variances"].items()}
        
        log(f"Loaded directions for layers: {list(directions.keys())}", class_name="SteeringVector")
        log(f"Shape of first direction vector: {next(iter(directions.values())).shape}", class_name="SteeringVector")
        
        return cls(model_type=data["model_type"], 
               directions=directions, 
               explained_variances=explained_variances)


def read_representations(model: MalleableModel | PreTrainedModel, tokenizer: PreTrainedTokenizerBase, inputs: list[ContrastivePair], hidden_layer_ids: typing.Iterable[int] | None = None, batch_size: int = 32, method: typing.Literal["pca_diff", "pca_center", "pca_pairwise"] = "pca_pairwise", save_analysis: bool = False, output_dir: str = "activation_steering_figures", accumulate_last_x_tokens: typing.Union[int, str] = 1, suffixes: typing.List[typing.Tuple[str, str]] = None) -> dict[int, np.ndarray]:
    """
    Extract representations from the language model based on the contrast dataset.

    Args:
        model: The model to extract representations from.
        tokenizer: The tokenizer associated with the model.
        inputs: A list of ContrastivePair inputs.
        hidden_layer_ids: The IDs of hidden layers to extract representations from.
        batch_size: The batch size to use when processing inputs.
        method: The method to use for preparing training data ("pca_diff", "pca_center", or "pca_pairwise").
        save_analysis: Whether to save PCA analysis figures.
        output_dir: The directory to save analysis figures to.
        accumulate_last_x_tokens: How many tokens to accumulate for the hidden state.
        suffixes: List of suffixes to use when accumulating hidden states.

    Returns:
        A dictionary mapping layer IDs to numpy arrays of directions.
    """
    log(f"Reading representations for {len(inputs)} inputs", class_name="SteeringVector")
    # If hidden_layer_ids is not provided, use all layers of the model
    if hidden_layer_ids is None:
        hidden_layer_ids = range(model.config.num_hidden_layers)

    if accumulate_last_x_tokens == "all":
        # Accumulate hidden states of all tokens
        log("... accumulating all hidden states", class_name="SteeringVector")
    elif accumulate_last_x_tokens == "suffix-only":
        log(f"... accumulating suffix-only hidden states", class_name="SteeringVector")
    else:
        log(f"... accumulating last {accumulate_last_x_tokens} hidden states", class_name="SteeringVector")
        
    # Get the total number of layers in the model
    n_layers = len(get_model_layer_list(model))
    
    # Normalize the layer indexes if they are negative
    hidden_layer_ids = [i if i >= 0 else n_layers + i for i in hidden_layer_ids]
    
    # Prepare the input strings for the model by extracting the positive and negative examples from the contrastive pairs
    train_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]
    
    # Example:
    # inputs = [
    #     ContrastivePair(positive="I'm feeling happy today.", negative="I'm feeling sad today."),
    #     ContrastivePair(positive="The weather is great!", negative="The weather is terrible.")
    # ]
    # train_strs = [
    #     "I'm feeling happy today.", "I'm feeling sad today.",
    #     "The weather is great!", "The weather is terrible."
    # ]
    
    # Call the batched_get_hiddens function to get the hidden states for each specified layer
    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layer_ids, batch_size, accumulate_last_x_tokens, suffixes
    )
    if save_analysis:
        save_pca_figures(layer_hiddens, hidden_layer_ids, method, output_dir, inputs)

    # Initialize an empty dictionary to store the directions for each layer
    directions: dict[int, np.ndarray] = {}
    explained_variances: dict[int, float] = {}
    
    # Iterate over each specified layer
    for layer in custom_progress(hidden_layer_ids, description="Reading Hidden Representations ..."):
        # Retrieve the hidden states for the current layer
        h = layer_hiddens[layer]
        
        # Prepare the training data based on the specified method
        if method == "pca_diff":
            # Calculate the difference between positive and negative examples
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            # Calculate the mean center across all examples
            center = h.mean(axis=0)
            # Subtract the global mean from all examples
            train = h - center
        elif method == "pca_pairwise":
            # Calculate the center of positive and negative examples (pairwise centering)
            center = (h[::2] + h[1::2]) / 2
            train = h.copy()
            # Subtract the pairwise center from the examples
            train[::2] -= center
            train[1::2] -= center
        else:
            raise ValueError("unknown method " + method)
        
        # Perform PCA with 1 component on the training data to extract the direction vector
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)
        explained_variances[layer] = pca_model.explained_variance_ratio_[0]
        
        # Example:
        # Suppose the hidden states for the current layer are:
        # h = np.array([
        #     [0.1, 0.2, 0.3],  # Positive example 1
        #     [0.4, 0.5, 0.6],  # Negative example 1
        #     [0.2, 0.3, 0.4],  # Positive example 2
        #     [0.5, 0.6, 0.7]   # Negative example 2
        #     [0.2, 0.3, 0.4],  # Positive example 3
        #     [0.5, 0.6, 0.7]   # Negative example 3
        # ])
        #
        # Performing PCA on the training data (differences between positive and negative examples)
        # will extract the direction vector that captures the most significant variation.
        # The resulting direction vector might be something like [0.57735027, 0.57735027, 0.57735027].
        
        # Project the hidden states onto the direction vector
        projected_hiddens = project_onto_direction(h, directions[layer])
        
        # Calculate the mean of positive examples being smaller than negative examples
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        
        # Calculate the mean of positive examples being larger than negative examples
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        
        # If positive examples are smaller on average, flip the direction vector
        if positive_smaller_mean > positive_larger_mean:
            directions[layer] *= -1
    
    # Return the dictionary mapping layer IDs to their corresponding direction vectors
    return directions, explained_variances


def _expand_aligned_suffixes(
    inputs: list[str],
    suffixes: typing.Optional[
        typing.List[typing.Union[str, typing.Tuple[str, str]]]
    ],
) -> typing.Optional[list[str]]:
    """Expand suffix definitions in SteeringDataset's suffix-major order."""
    if not suffixes:
        return None
    denominator = 2 * len(suffixes)
    if len(inputs) % denominator != 0:
        raise ValueError(
            "suffix-only inputs cannot be aligned with the supplied suffix list"
        )
    pairs_per_suffix = len(inputs) // denominator
    aligned: list[str] = []
    for suffix in suffixes:
        if isinstance(suffix, tuple):
            positive_suffix, negative_suffix = suffix
        else:
            positive_suffix = negative_suffix = suffix
        for _ in range(pairs_per_suffix):
            aligned.extend((positive_suffix, negative_suffix))
    if len(aligned) != len(inputs):
        raise AssertionError("internal suffix alignment error")
    return aligned


def _suffix_token_indices(tokenizer, text: str, suffix: str) -> list[int]:
    """Return token positions overlapping the exact suffix."""
    if not suffix:
        warnings.warn("Empty suffix encountered; using the final token.")
        full_count = len(tokenizer.encode(text, add_special_tokens=True))
        return [full_count - 1]
    suffix_start = len(text) - len(suffix)
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and end > suffix_start
        ]
        if indices:
            return indices
    except (KeyError, NotImplementedError, TypeError, ValueError):
        pass

    # Slow-tokenizer fallback. Include one boundary token because BPE/SentencePiece
    # may merge the final prompt character with the first suffix character.
    full_count = len(tokenizer.encode(text, add_special_tokens=True))
    base_count = len(
        tokenizer.encode(text[:suffix_start], add_special_tokens=True)
    )
    count = max(1, full_count - base_count + 1)
    return list(range(max(0, full_count - count), full_count))


def _suffix_token_count(tokenizer, text: str, suffix: str) -> int:
    """Compatibility helper used by the local regression tests."""
    return len(_suffix_token_indices(tokenizer, text, suffix))


def _hidden_to_numpy(hidden_state: torch.Tensor) -> np.ndarray:
    """Convert fp16/bf16 model states to sklearn-compatible float32."""
    return hidden_state.detach().to(dtype=torch.float32).cpu().numpy()


def batched_get_hiddens(model, tokenizer, inputs: list[str], hidden_layer_ids: list[int],batch_size: int, accumulate_last_x_tokens: typing.Union[int, str] = 1, suffixes: typing.List[typing.Tuple[str, str]] = None) -> dict[int, np.ndarray]:
    """
    Retrieve the hidden states from the specified layers of the language model for the given input strings.

    Args:
        model: The model to get hidden states from.
        tokenizer: The tokenizer associated with the model.
        inputs: A list of input strings.
        hidden_layer_ids: The IDs of hidden layers to get states from.
        batch_size: The batch size to use when processing inputs.
        accumulate_last_x_tokens: How many tokens to accumulate for the hidden state.
        suffixes: List of suffixes to use when accumulating hidden states.

    Returns:
        A dictionary mapping layer IDs to numpy arrays of hidden states.
    """
    # Split the input strings into batches based on the specified batch size
    batch_starts = list(range(0, len(inputs), batch_size))
    batched_inputs = [inputs[p : p + batch_size] for p in batch_starts]
    aligned_suffixes = _expand_aligned_suffixes(inputs, suffixes)
    
    # Initialize an empty dictionary to store the hidden states for each specified layer
    hidden_states = {layer: [] for layer in hidden_layer_ids}
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate over each batch of input strings
        for batch_index, batch in enumerate(custom_progress(batched_inputs, description="Collecting Hidden Representations ...")):
            batch_start = batch_starts[batch_index]
            # Pass the batch through the language model and retrieve the hidden states
            out = model(
                **tokenizer(batch, padding=True, return_tensors="pt").to(model.device),
                output_hidden_states=True,
            )
            
            # Iterate over each specified layer ID
            for layer_id in hidden_layer_ids:
                # Adjust the layer index if it is negative
                hidden_idx = layer_id + 1 if layer_id >= 0 else layer_id
                
                # Iterate over each batch of hidden states
                for i, batch_hidden in enumerate(out.hidden_states[hidden_idx]):
                    if accumulate_last_x_tokens == "all":
                        accumulated_hidden_state = torch.mean(batch_hidden, dim=0)
                    elif accumulate_last_x_tokens == "suffix-only":
                        if aligned_suffixes:
                            suffix = aligned_suffixes[batch_start + i]
                            suffix_token_indices = _suffix_token_indices(
                                tokenizer,
                                batch[i],
                                suffix,
                            )
                            # Get the hidden states for this sample's exact suffix.
                            suffix_hidden = batch_hidden[suffix_token_indices, :]
                            accumulated_hidden_state = torch.mean(suffix_hidden, dim=0)
                        else:
                            warnings.warn("'suffix-only' option used but no suffixes provided. Using last token instead.")
                            accumulated_hidden_state = batch_hidden[-1, :]
                    else:
                        accumulated_hidden_state = torch.mean(batch_hidden[-accumulate_last_x_tokens:, :], dim=0)
                    
                    hidden_states[layer_id].append(
                        _hidden_to_numpy(accumulated_hidden_state.squeeze())
                    )
            
            
            # Delete the model output to free up memory
            del out
    
    # Stack the hidden states for each layer into a numpy array
    # Return the dictionary mapping layer IDs to their corresponding stacked hidden states
    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(H, direction):
    """
    Project a matrix H onto a direction vector.

    Args:
        H: The matrix to project.
        direction: The direction vector to project onto.

    Returns:
        The projected matrix.
    """
    # Calculate the magnitude (Euclidean norm) of the direction vector
    mag = np.linalg.norm(direction)
    
    # Assert that the magnitude is not infinite to ensure validity
    assert not np.isinf(mag)
    
    # Perform the projection by multiplying the matrix H with the direction vector
    # Divide the result by the magnitude of the direction vector to normalize the projection
    return (H @ direction) / mag


def save_pca_figures(layer_hiddens, hidden_layer_ids, method, output_dir, inputs):
    """
    Save PCA analysis figures for each hidden layer and create a macroscopic x-axis layer analysis plot.

    Args:
        layer_hiddens: A dictionary of hidden states for each layer.
        hidden_layer_ids: The IDs of hidden layers.
        method: The method used for preparing training data.
        output_dir: The directory to save the figures to.
        inputs: The input data used for the analysis.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store the variances and layer IDs for the macroscopic analysis
    variances = []
    layers = []

    for layer in custom_progress(hidden_layer_ids, description="Saving PCA Figures"):
        h = layer_hiddens[layer]

        if method == "pca_diff":
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            center = h.mean(axis=0)
            train = h - center
        elif method == "pca_pairwise":
            center = (h[::2] + h[1::2]) / 2
            train = h.copy()
            train[::2] -= center
            train[1::2] -= center
        else:
            raise ValueError("unknown method " + method)

        pca_model = PCA(n_components=2, whiten=False).fit(train)

        # Project the dataset points onto the first two principal components
        projected_data = pca_model.transform(h)

        # Separate the projected data into positive and negative examples
        positive_data = projected_data[::2]
        negative_data = projected_data[1::2]

        # Plot the projected points with separate colors for positive and negative examples
        plt.figure(figsize=(8, 6))
        plt.scatter(positive_data[:, 0], positive_data[:, 1], alpha=0.7, label="Positive Examples")
        plt.scatter(negative_data[:, 0], negative_data[:, 1], alpha=0.7, label="Negative Examples")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA Visualization - Layer {layer}")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f"pca_layer_{layer}.png"))
        plt.close()

        # Store the variance explained by PC1 and the corresponding layer ID for the macroscopic analysis
        variances.append(pca_model.explained_variance_ratio_[0])
        layers.append(layer)

    # Create the macroscopic x-axis layer analysis plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, variances, marker='o')
    plt.xlabel("Layer ID")
    plt.ylabel("Variance Explained by PC1")
    plt.title("Macroscopic X-Axis Layer Analysis")
    plt.grid(True)
    plt.xticks(layers)
    plt.tight_layout()

    # Save the macroscopic analysis figure
    plt.savefig(os.path.join(output_dir, "macroscopic_analysis.png"))
    plt.close()

