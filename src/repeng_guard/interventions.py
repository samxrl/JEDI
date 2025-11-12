# -*- coding: utf-8 -*-
"""
干预 (Interventions)

[!] 此文件已修改，以修复“空间不匹配”问题。

核心功能是 `create_intervention_hook_func`，它创建了一个
PyTorch 钩子函数。

当 CUSUM 警报触发时，此钩子将执行一个完整的“变换-干预-反转” (h -> z -> z' -> h') 流程:
1.  (h_t -> z_t): 将原始隐藏状态 h_t 变换到白化空间 z_t。
2.  (z_t -> z'_t): 在白化空间中应用 ActAdd: z'_t = z_t + (alpha * v_l)。
3.  (z'_t -> h'_t): 将干预后的 z'_t 反向变换回原始空间 h'_t。
"""

import torch
from torch.nn import Module
from typing import Callable, Tuple, Any, Optional
import logging

# [!] 新增导入
from .components.representation import apply_transform, invert_transform

logger = logging.getLogger(__name__)


def create_intervention_hook_func(
        vector: torch.Tensor,
        alpha: float,
        transform: Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]],  # [!] 接收 (W, mu, W_inv)
        device: str
) -> Callable:
    """
    创建一个闭包函数 (hook function)，用于在 *白化空间* 中执行 ActAdd 干预。

    Args:
        vector (torch.Tensor):
            干预向量 (v_l)，即“拒绝”方向。形状为 (D,)。
        alpha (float):
            干预强度。
        transform (Tuple):
            用于干预向量 $v_l$ 所在空间（例如 'early_window'）的
            (W, mu, W_inv) 变换元组。
        device (str):
            运行计算的设备。

    Returns:
        Callable:
            一个 PyTorch 钩子函数。
    """

    # 预先计算加法向量，并将其移动到目标设备
    additive_vector = (alpha * vector).to(device)

    # [!] 新增: 将变换组件预先移动到设备
    W, mu, W_inv = transform
    transform_gpu = (
        W.to(device, non_blocking=True) if W is not None else None,
        mu.to(device, non_blocking=True),
        W_inv.to(device, non_blocking=True) if W_inv is not None else None
    )

    # *** 这是修改后的函数签名 ***
    def hook_func(
            hidden_state: torch.Tensor,  # 接收来自 pre-hook (args[0]) 的 hidden_state
            indices: torch.Tensor  # 接收来自 hook_manager 的 indices
    ) -> torch.Tensor:  # 返回修改后的 hidden_state
        """
        实际的 PyTorch 钩子实现 (适配 pre-hook)。
        执行 (h -> z -> z' -> h') 流程。

        Args:
            hidden_state (torch.Tensor):
                模块的输入 hidden_state (B, SeqLen, D)。
            indices (torch.Tensor):
                一个布尔张量 (B,)，指示哪些批量索引需要被干预。

        Returns:
            torch.Tensor: 修改后的 hidden_state。
        """
        try:
            # 0. 如果没有序列需要干预，立即返回
            if not torch.any(indices):
                return hidden_state

            # 1. 确保加法向量与 hidden_state 的类型匹配
            add_vec_typed = additive_vector.to(hidden_state.dtype)

            # 2. (h_t -> z_t) 正向变换 (仅在最后一个 token)
            # 我们只干预自回归生成中的最后一个 token
            # [B, SeqLen, D] -> [N_indices, 1, D]
            last_token_hidden_state = hidden_state[indices, -1:, :]

            z_t = apply_transform(last_token_hidden_state, transform_gpu)

            # 3. (z_t -> z'_t) 在白化空间中应用干预
            # add_vec_typed (D,) -> (1, 1, D)
            z_prime_t = z_t + add_vec_typed.unsqueeze(0)

            # 4. (z'_t -> h'_t) 反向变换回原始空间
            h_prime_t = invert_transform(z_prime_t, transform_gpu)

            # 5. 就地修改 (in-place modification)
            # 将 (N_indices, 1, D) 形状的 h_prime_t 写回
            hidden_state[indices, -1:, :] = h_prime_t.to(hidden_state.dtype)

            # 6. 返回修改后的 hidden_state
            return hidden_state

        except Exception as e:
            logger.error(f"SARC 干预钩子执行失败: {e}", exc_info=True)
            # 如果失败，返回原始 hidden_state，避免使模型崩溃
            return hidden_state

    # 返回这个内部函数，它将被注册为钩子
    return hook_func