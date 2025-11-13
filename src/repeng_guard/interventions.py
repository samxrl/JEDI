# -*- coding: utf-8 -*-
"""
干预 (Interventions)

[!] 此文件已修改，以修复“空间不匹配”问题。
[!] 再次修改：
- 移除 `alpha` 参数，使其支持动态强度。
- `hook_func` 现在接受 `dynamic_alphas` 张量。

核心功能是 `create_intervention_hook_func`，它创建了一个
PyTorch 钩子函数。

当 CUSUM 警报触发时，此钩子将执行一个完整的“变换-干预-反转” (h -> z -> z' -> h') 流程:
1.  (h_t -> z_t): 将原始隐藏状态 h_t 变换到白化空间 z_t。
2.  (z_t -> z'_t): 在白化空间中应用 ActAdd: z'_t = z_t + (alpha' * v_l)。
   [!] `alpha'` 是在运行时动态传入的。
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
        # [!] 移除 alpha: float,
        transform: Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]],  # [!] 接收 (W, mu, W_inv)
        device: str
) -> Callable:
    """
    创建一个闭包函数 (hook function)，用于在 *白化空间* 中执行 ActAdd 干预。

    Args:
        vector (torch.Tensor):
            干预向量 (v_l)，即“拒绝”方向。形状为 (D,)。
        transform (Tuple):
            用于干预向量 $v_l$ 所在空间（例如 'early_window'）的
            (W, mu, W_inv) 变换元组。
        device (str):
            运行计算的设备。

    Returns:
        Callable:
            一个 PyTorch 钩子函数。
    """

    # [!] 移除 additive_vector 的预计算
    # [!] 预先将基础向量移动到设备
    vector_gpu = vector.to(device)

    # [!] 新增: 将变换组件预先移动到设备
    W, mu, W_inv = transform
    transform_gpu = (
        W.to(device, non_blocking=True) if W is not None else None,
        mu.to(device, non_blocking=True),
        W_inv.to(device, non_blocking=True) if W_inv is not None else None
    )

    # *** [!] 这是修改后的函数签名 ***
    def hook_func(
            hidden_state: torch.Tensor,  # 接收来自 pre-hook (args[0]) 的 hidden_state
            indices: torch.Tensor,  # 接收来自 hook_manager 的 indices (B,)
            dynamic_alphas: torch.Tensor  # [!] 接收动态 alpha (B,)
    ) -> torch.Tensor:  # 返回修改后的 hidden_state
        """
        实际的 PyTorch 钩子实现 (适配 pre-hook)。
        执行 (h -> z -> z' -> h') 流程。

        Args:
            hidden_state (torch.Tensor):
                模块的输入 hidden_state (B, SeqLen, D)。
            indices (torch.Tensor):
                一个布尔张量 (B,)，指示哪些批量索引需要被干预。
            dynamic_alphas (torch.Tensor):
                一个浮点张量 (B,)，包含 *所有* 序列的当前干预强度。

        Returns:
            torch.Tensor: 修改后的 hidden_state。
        """
        try:
            # 0. 如果没有序列需要干预，立即返回
            if not torch.any(indices):
                return hidden_state

            # 1. (h_t -> z_t) 正向变换 (仅在最后一个 token)
            # 我们只干预自回归生成中的最后一个 token
            # [B, SeqLen, D] -> [N_indices, 1, D]
            last_token_hidden_state = hidden_state[indices, -1:, :]

            z_t = apply_transform(last_token_hidden_state, transform_gpu)

            # 2. (z_t -> z'_t) [!] 在白化空间中应用 *动态* 干预

            # 2a. 获取需要干预的序列对应的 alphas
            # (B,)[indices] -> (N_indices,)
            alphas_for_active = dynamic_alphas[indices].to(hidden_state.dtype)

            # 2b. 准备广播
            # (N_indices,) -> (N_indices, 1, 1)
            alphas_for_broadcast = alphas_for_active.unsqueeze(-1).unsqueeze(-1)
            # (D,) -> (1, 1, D)
            vector_for_broadcast = vector_gpu.to(hidden_state.dtype).unsqueeze(0).unsqueeze(0)

            # 2c. 计算最终的加法向量 (N_indices, 1, D)
            additive_vectors = alphas_for_broadcast * vector_for_broadcast

            # 2d. 应用干预
            z_prime_t = z_t + additive_vectors

            # 3. (z'_t -> h'_t) 反向变换回原始空间
            h_prime_t = invert_transform(z_prime_t, transform_gpu)

            # 4. 就地修改 (in-place modification)
            # 将 (N_indices, 1, D) 形状的 h_prime_t 写回
            hidden_state[indices, -1:, :] = h_prime_t.to(hidden_state.dtype)

            # 5. 返回修改后的 hidden_state
            return hidden_state

        except Exception as e:
            logger.error(f"SARC 干预钩子执行失败: {e}", exc_info=True)
            # 如果失败，返回原始 hidden_state，避免使模型崩溃
            return hidden_state

    # 返回这个内部函数，它将被注册为钩子
    return hook_func

