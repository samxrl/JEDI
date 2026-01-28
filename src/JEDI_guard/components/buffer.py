# -*- coding: utf-8 -*-
"""
This file implements the "commit buffer" mechanism.

According to stage 5 of \"Method Flow.md\", to allow graceful rollback when harmful
trends are detected (instead of exposing partially generated harmful content),
JEDI introduces a commit buffer.

The `CommitBuffer` class implements a FIFO queue that temporarily stores the
most recently generated tokens. Only when the CUSUM algorithm confirms the
current generation process is safe will tokens at the head of the buffer be
"committed" (i.e., safely shown to the user). Once an alert triggers, the
entire buffer can be cleared to enable seamless rollback.
"""

from collections import deque
from typing import List, Dict, Any
import torch


class CommitBuffer:
    """
    Manage a FIFO queue for temporarily storing tokens pending commit.

    This class is designed to operate in batches, maintaining an independent
    buffer for each parallel generation sequence.
    """

    def __init__(self, capacity: int, batch_size: int):
        """
        Initialize the commit buffer.

        Args:
            capacity (int):
                Maximum buffer capacity per sequence (number of tokens).
                This value also determines the maximum possible rollback length.

            batch_size (int):
                Number of parallel generation sequences to manage.
        """
        if capacity <= 0:
            raise ValueError("Buffer capacity must be a positive integer.")
        self.capacity = capacity
        self.batch_size = batch_size

        # Create a separate deque for each sequence in the batch
        # deque provides efficient append/pop from both ends
        self.buffers: List[deque] = [deque(maxlen=capacity) for _ in range(batch_size)]

    def add(self, tokens: List[Any]):
        """
        Add newly generated tokens to the end of each sequence buffer.

        Args:
            tokens (List[Any]):
                A list containing the newly generated token for each sequence
                in the batch. Length should equal `batch_size`.
        """
        if len(tokens) != self.batch_size:
            raise ValueError(
                f"Token count ({len(tokens)}) does not match batch size ({self.batch_size})."
            )

        for i in range(self.batch_size):
            self.buffers[i].append(tokens[i])

    def commit(self, num_tokens: int = 1) -> List[List[Any]]:
        """
        Commit (remove and return) a specified number of tokens from the head
        of each sequence buffer.

        This simulates sending safe tokens to the user.

        Args:
            num_tokens (int, optional):
                Number of tokens to commit per sequence. Defaults to 1.

        Returns:
            List[List[Any]]:
                A list where each sublist contains the committed tokens from the
                corresponding sequence buffer.
        """
        committed_batch = [[] for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            for _ in range(num_tokens):
                if self.buffers[i]:
                    committed_batch[i].append(self.buffers[i].popleft())
                else:
                    break  # Stop if the buffer is empty
        return committed_batch

    def rollback(self, indices: torch.Tensor):
        """
        Clear the buffers for specified sequence indices.

        This is called when CUSUM detects harmful trends.

        Args:
            indices (torch.Tensor):
                A boolean or integer tensor indicating which sequence buffers
                should be cleared.
        """
        # Convert PyTorch tensor to an iterable list of indices
        if indices.dtype == torch.bool:
            idx_list = indices.nonzero(as_tuple=True)[0]
        else:
            idx_list = indices

        for i in idx_list:
            self.buffers[i].clear()

    def flush_all(self) -> List[List[Any]]:
        """
        Clear and return any remaining content in all buffers.

        Called when a generation sequence completes normally (no alert), to
        ensure all buffered tokens are committed.

        Returns:
            List[List[Any]]:
                A list where each sublist contains all remaining tokens from the
                corresponding sequence buffer.
        """
        remaining_batch = []
        for i in range(self.batch_size):
            remaining_batch.append(list(self.buffers[i]))
            self.buffers[i].clear()
        return remaining_batch
