# -*- coding: utf-8 -*-
"""
This file defines the `Scorer` class. Its core responsibility is to compute the
per-token "harmfulness" score for each generated token.

According to stage 4.1 of \"Method Flow.md\", the score is obtained by projecting
normalized hidden states onto a pre-extracted "condition vector" (c_l). A
bounded ReLU (subtracting threshold `theta` and taking the positive part) is
also applied to filter benign noise.

The `Scorer` class encapsulates this logic so the online defense system `Guard`
can easily evaluate the risk of each newly generated token.
"""

import torch
from typing import Tuple, Optional

from .representation import apply_transform


class Scorer:
    """
    Compute per-token raw risk scores (s_t) and thresholded risk scores (r_t).
    """

    def __init__(
            self,
            condition_vector: torch.Tensor,
            transform: Tuple[Optional[torch.Tensor], torch.Tensor],
            theta: float,
            device: str = 'cpu'
    ):
        """
        Initialize the Scorer component.

        Args:
            condition_vector (torch.Tensor):
                Condition vector `c_l` for detecting harmful semantics. Shape (D,).

            transform (Tuple[Optional[torch.Tensor], torch.Tensor]):
                A tuple `(W, mu)` containing the whitening matrix and mean vector
                for the content window. Loaded from the offline calibration stage.

            theta (float):
                Score threshold `theta`. This value is subtracted from the raw
                score `s_t` before computing the final risk score `r_t`.

            device (str):
                Device to run computations on (e.g., 'cuda:0' or 'cpu').
        """
        self.condition_vector = condition_vector.to(device, non_blocking=True)
        # Move transform matrices and vectors to the target device as well
        W, mu, W_inv = transform
        self.transform_gpu = (
            W.to(device, non_blocking=True) if W is not None else None,
            mu.to(device, non_blocking=True),
            W_inv.to(device, non_blocking=True) if W_inv is not None else None
        )
        self.theta = theta
        self.device = device

    @torch.no_grad()
    def calculate_scores(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute raw and risk scores for a given sequence of hidden states.

        Args:
            hidden_states (torch.Tensor):
                Hidden states for a batch of newly generated tokens.
                Expected shape (B, 1, D), where B is batch size and D is hidden dim.
                The middle '1' represents sequence length 1 (per-token processing).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            A tuple `(s_t, r_t)`, where:
            - `s_t` (torch.Tensor): Raw projection score, shape (B,).
            - `r_t` (torch.Tensor): Thresholded risk score, shape (B,).
        """
        if hidden_states.device.type != self.device:
            hidden_states = hidden_states.to(self.device, non_blocking=True)

        # 1. Normalize hidden states (whitening/centering)
        # hidden_states shape (B, 1, D) -> transformed_states shape (B, 1, D)
        # using the pre-moved self.transform_gpu
        transformed_states = apply_transform(hidden_states, self.transform_gpu)

        # 2. Project normalized representations onto the condition vector to get s_t
        # transformed_states shape (B, 1, D), condition_vector shape (D,)
        # -> s_t shape (B, 1)
        s_t_unsq = torch.einsum('bnd,d->bn', transformed_states, self.condition_vector)
        s_t = s_t_unsq.squeeze(1)  # Remove middle dimension -> (B,)

        # 3. Apply bounded ReLU to get the final risk score r_t
        # r_t = max(0, s_t - theta)
        r_t = torch.clamp(s_t - self.theta, min=0)

        return s_t.cpu(), r_t.cpu()
