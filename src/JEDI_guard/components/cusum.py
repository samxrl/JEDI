# -*- coding: utf-8 -*-
"""
This file implements the CUSUM (Cumulative Sum) control chart algorithm for
sequence drift detection.

Within the JEDI defense framework, CUSUM plays the role of \"memory\" and
\"accumulated evidence\". According to stage 5 of \"Method Flow.md\", relying only on
single-step risk scores `r_t` is vulnerable to noise or "gradual induction"
attacks. CUSUM accumulates "excess risk" to sensitively detect persistent—even
small—harmful tendencies.

The `CusumState` class is a state machine that maintains CUSUM statistics,
updates them when new risk scores arrive, and determines whether to trigger an
alert.

[!] Changes:
- `update` now returns cumulative score A_t instead of trigger indices.
- `update` no longer auto-resets state; reset logic moved to JEDILogitsProcessor.
"""

import torch


class CusumState:
    """
    A state machine for managing and updating CUSUM statistics.

    This implementation follows the one-sided upward detection form of the
    Page-Hinkley (PH) test:
    S_t = S_{t-1} + (r_t - mu_hat - kappa)
    M_t = min(M_{t-1}, S_t)
    A_t = S_t - M_t

    An alert triggers when A_t > alpha.
    """

    def __init__(self, mu_hat: float, kappa: float, alpha: float, batch_size: int, device: str = 'cpu'):
        """
        Initialize CUSUM state for a batch of samples.

        Args:
            mu_hat (float):
                Safe baseline. The mean of risk scores `r_t` observed on benign data.
                It represents the "normal" or "expected" risk level.

            kappa (float):
                Tolerance band to offset normal fluctuations in `r_t`, preventing
                false positives due to random noise. The cumulative sum increases
                only when `r_t` stays above `mu_hat + kappa`.

            alpha (float):
                Alert threshold. When CUSUM statistic `A_t` exceeds this value,
                it indicates significant harmful drift and should trigger
                intervention.

            batch_size (int):
                Number of independent sequences to process simultaneously.

            device (str):
                Device to run computations on.
        """
        self.mu_hat = mu_hat
        self.kappa = kappa
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device

        # S_t and M_t are core CUSUM state variables
        # S_t: cumulative sum
        # M_t: running minimum of S_t
        # A_t (trigger statistic) is computed dynamically in update
        self.S = torch.zeros(batch_size, device=device)
        self.M = torch.zeros(batch_size, device=device)

    def reset(self, indices: torch.Tensor = None):
        """
        Reset CUSUM state for specified indices or all sequences.

        Args:
            indices (torch.Tensor, optional):
                A boolean or integer tensor indicating which sequences to reset.
                If None, reset all sequences.
        """
        if indices is None:
            self.S.fill_(0)
            self.M.fill_(0)
        else:
            self.S[indices] = 0
            self.M[indices] = 0

    def update(self, r_t: torch.Tensor) -> torch.Tensor:
        """
        Update CUSUM state with a new batch of risk scores r_t and return A_t.

        [!] Change: this method no longer triggers a reset; it only returns A_t.

        Args:
            r_t (torch.Tensor):
                Latest risk scores, shape (B,), where B is batch size.

        Returns:
            torch.Tensor:
                A tensor of shape (B,) containing current CUSUM scores A_t.
        """
        if r_t.device.type != self.device:
            r_t = r_t.to(self.device, non_blocking=True)

        # Core CUSUM recursion
        # 1. Update cumulative sum S_t
        increment = r_t - self.mu_hat - self.kappa
        self.S += increment

        # 2. Update running minimum M_t
        # Note: in the original Page-Hinkley algorithm, M updates after S.
        # Using `torch.minimum` ensures this.
        self.M = torch.minimum(self.M, self.S)

        # 3. Compute current trigger statistic A_t
        A_t = self.S - self.M

        # 4. [!] Remove reset logic
        # triggered_indices = A_t > self.alpha
        # if torch.any(triggered_indices):
        #     self.reset(triggered_indices)

        # 5. [!] Return A_t
        return A_t.cpu()
