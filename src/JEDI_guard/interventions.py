# -*- coding: utf-8 -*-
"""
Interventions

[!] This file was modified to fix the "space mismatch" issue.
[!] Modified again:
- Removed the fixed strength parameter to support dynamic beta strength.
- `hook_func` now accepts a `dynamic_betas` tensor.

The core functionality is `create_intervention_hook_func`, which creates a
PyTorch hook function.

When a CUSUM alert triggers, this hook performs a full
"transform-intervene-invert" flow (h -> z -> z' -> h'):
1. (h_t -> z_t): Transform the original hidden state h_t into whitened space z_t.
2. (z_t -> z'_t): Apply ActAdd in whitened space: z'_t = z_t + (beta' * v_l).
   [!] `beta'` is provided dynamically at runtime.
3. (z'_t -> h'_t): Invert the intervened z'_t back to the original space h'_t.
"""

import torch
from torch.nn import Module
from typing import Callable, Tuple, Any, Optional
import logging

# [!] New import
from .components.representation import apply_transform, invert_transform

logger = logging.getLogger(__name__)


def create_intervention_hook_func(
        vector: torch.Tensor,
        # [!] No longer need fixed beta: float,
        transform: Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]],  # [!] Receive (W, mu, W_inv)
        device: str
) -> Callable:
    """
    Create a closure (hook function) to perform ActAdd intervention in *whitened space*.

    Args:
        vector (torch.Tensor):
            Intervention vector (v_l), i.e., the "refusal" direction. Shape (D,).
        transform (Tuple):
            The (W, mu, W_inv) transform tuple for the space of vector $v_l$
            (e.g., 'early_window').
        device (str):
            Device to run computations on.

    Returns:
        Callable:
            A PyTorch hook function.
    """

    # [!] Remove additive_vector precomputation
    # [!] Move the base vector to the device ahead of time
    vector_gpu = vector.to(device)

    # [!] New: move transform components to the device ahead of time
    W, mu, W_inv = transform
    transform_gpu = (
        W.to(device, non_blocking=True) if W is not None else None,
        mu.to(device, non_blocking=True),
        W_inv.to(device, non_blocking=True) if W_inv is not None else None
    )

    # *** [!] Updated function signature ***
    def hook_func(
            hidden_state: torch.Tensor,  # Receive hidden_state from pre-hook (args[0])
            indices: torch.Tensor,  # Receive indices from hook_manager (B,)
            dynamic_betas: torch.Tensor  # [!] Receive dynamic beta (B,)
    ) -> torch.Tensor:  # Return modified hidden_state
        """
        Actual PyTorch hook implementation (pre-hook compatible).
        Executes the (h -> z -> z' -> h') flow.

        Args:
            hidden_state (torch.Tensor):
                The module input hidden_state (B, SeqLen, D).
            indices (torch.Tensor):
                A boolean tensor (B,) indicating which batch indices need intervention.
            dynamic_betas (torch.Tensor):
                A float tensor (B,) containing the current intervention strength for
                *all* sequences.

        Returns:
            torch.Tensor: Modified hidden_state.
        """
        try:
            # 0. If no sequences need intervention, return immediately
            if not torch.any(indices):
                return hidden_state

            # 1. (h_t -> z_t) Forward transform (only the last token)
            # We only intervene on the last token in autoregressive generation
            # [B, SeqLen, D] -> [N_indices, 1, D]
            last_token_hidden_state = hidden_state[indices, -1:, :]

            z_t = apply_transform(last_token_hidden_state, transform_gpu)

            # 2. (z_t -> z'_t) [!] Apply *dynamic* intervention in whitened space

            # 2a. Get betas for active sequences
            # (B,)[indices] -> (N_indices,)
            betas_for_active = dynamic_betas[indices].to(hidden_state.dtype)

            # 2b. Prepare broadcasting
            # (N_indices,) -> (N_indices, 1, 1)
            betas_for_broadcast = betas_for_active.unsqueeze(-1).unsqueeze(-1)
            # (D,) -> (1, 1, D)
            vector_for_broadcast = vector_gpu.to(hidden_state.dtype).unsqueeze(0).unsqueeze(0)

            # 2c. Compute final additive vector (N_indices, 1, D)
            additive_vectors = betas_for_broadcast * vector_for_broadcast

            # 2d. Apply intervention
            z_prime_t = z_t + additive_vectors

            # 3. (z'_t -> h'_t) Invert back to original space
            h_prime_t = invert_transform(z_prime_t, transform_gpu)

            # 4. In-place modification
            # Write back h_prime_t with shape (N_indices, 1, D)
            hidden_state[indices, -1:, :] = h_prime_t.to(hidden_state.dtype)

            # 5. Return modified hidden_state
            return hidden_state

        except Exception as e:
            logger.error(f"JEDI intervention hook failed: {e}", exc_info=True)
            # On failure, return original hidden_state to avoid crashing the model
            return hidden_state

    # Return this inner function; it will be registered as a hook
    return hook_func
