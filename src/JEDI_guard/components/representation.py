# -*- coding: utf-8 -*-
"""
This file implements core functionality related to processing model internal
representations (hidden states).

According to \"Method Flow.md\", before computing any risk scores, raw hidden states
must be normalized (centering and optional whitening). This removes scale
differences and correlations across dimensions, making subsequent projection
scores more stable and comparable.

This file mainly provides the `apply_transform` function, which performs this
normalization step.
"""

import torch
from typing import Tuple, Optional


def apply_transform(
        hidden_states: torch.Tensor,
        transform: Tuple[Optional[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
    Apply a precomputed transform (centering and optional whitening) to input
    hidden states.

    This function is a key step in the representation pipeline. It accepts a
    batch of hidden states and a tuple containing the whitening matrix 'W'
    (if enabled) and mean vector 'mu'.

    Args:
        hidden_states (torch.Tensor):
            Raw hidden states extracted from the model.
            Shape can be (N, D) for a single sequence aggregate, or
            (B, N, D) for batched per-token sequences, where B is batch size,
            N is sequence length, and D is hidden dimension.

        transform (Tuple[Optional[torch.Tensor], torch.Tensor]):
            A tuple `(W, mu)`, where:
            - `W` (torch.Tensor, optional): Whitening matrix, shape (D, D). If None,
              only centering is applied.
            - `mu` (torch.Tensor): Mean vector, shape (D,), used for centering.

    Returns:
        torch.Tensor:
            Transformed hidden states with the same shape as the input.

    Raises:
        ValueError: If `hidden_states` does not have 2 or 3 dimensions.
    """
    W, mu, _ = transform
    device = hidden_states.device

    # Ensure mu and W (if present) are on the same device as hidden_states
    mu_device = mu.to(device)

    # Step 1: Centering (subtract mean)
    centered_states = hidden_states - mu_device

    # Step 2: (Optional) apply whitening transform
    if W is not None:
        W_device = W.to(device)

        # Use einsum to handle 2D and 3D tensors elegantly
        if hidden_states.dim() == 2:  # Shape (N, D)
            # 'nd,cd->nc' -> (N, D) @ (D, D).T = (N, D)
            # Note: in RepEng, we typically use W @ (h-mu), hence 'cd'
            transformed_states = torch.einsum('nd,cd->nc', centered_states, W_device)
        elif hidden_states.dim() == 3:  # Shape (B, N, D)
            # 'bnd,cd->bnc' -> apply transform to each (N, D) matrix in the batch
            transformed_states = torch.einsum('bnd,cd->bnc', centered_states, W_device)
        else:
            raise ValueError(
                f"Unsupported hidden state dimensionality: {hidden_states.dim()}. "
                "Only 2D or 3D tensors are supported."
            )

        return transformed_states
    else:
        # If W is None, return the centered result only
        return centered_states


def invert_transform(
        transformed_states: torch.Tensor,
        transform: Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]
) -> torch.Tensor:
    """
    Invert the whitened/centered representation (z_t) back to the original
    hidden state (h_t).
    Compute: h_t = W_inv @ z_t + mu

    Args:
        transformed_states (torch.Tensor):
            Representation in whitened space (z_t).
            Shape can be (N, D) or (B, N, D).

        transform (Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]):
            A tuple `(W, mu, W_inv)`.

    Returns:
        torch.Tensor:
            Hidden states in original space (h_t), same shape as input.
    """
    _, mu, W_inv = transform  # Unpack triple, ignore W
    device = transformed_states.device

    mu_device = mu.to(device)

    # Step 1: (Optional) apply inverse whitening W_inv @ z_t
    if W_inv is not None:
        W_inv_device = W_inv.to(device)

        if transformed_states.dim() == 2:  # Shape (N, D)
            # 'nd,cd->nc' -> (N, D) @ (D, D).T = (N, D)
            # Note: W_inv is (D, D)
            de_whitened_states = torch.einsum('nd,cd->nc', transformed_states, W_inv_device)
        elif transformed_states.dim() == 3:  # Shape (B, N, D)
            # 'bnd,cd->bnc'
            de_whitened_states = torch.einsum('bnd,cd->bnc', transformed_states, W_inv_device)
        else:
            raise ValueError(
                f"Unsupported hidden state dimensionality: {transformed_states.dim()}. "
                "Only 2D or 3D tensors are supported."
            )
    else:
        # If W_inv is None (center-only), then z_t == h_t - mu
        de_whitened_states = transformed_states

    # Step 2: add mean back
    original_states = de_whitened_states + mu_device
    return original_states
