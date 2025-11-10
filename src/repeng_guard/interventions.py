# -*- coding: utf-8 -*-
"""
干预 (Interventions)

该文件实现了 SARC 防御机制中的“干预”动作，
对应于“方法流程.md”文档中的阶段 6。

核心功能是 `create_intervention_hook_func`，它创建了一个
PyTorch 钩子函数，用于执行“激活添加” (ActAdd) 注入。
当 CUSUM 警报触发时，`Guard` 类会通过 `HookManager`
注册这个钩子，将模型在目标层的激活状态 (h_t) 转向
预先计算好的“拒绝”方向 (v_l)。
"""

import torch
from torch.nn import Module
from typing import Callable, Tuple, Any

import logging

logger = logging.getLogger(__name__)


def create_intervention_hook_func(
        vector: torch.Tensor,
        alpha: float,
        device: str
) -> Callable:
    """
    创建一个闭包函数 (hook function)，用于执行 ActAdd 干预。

    Args:
        vector (torch.Tensor):
            干预向量 (v_l)，即“拒绝”方向。形状为 (D,)。
        alpha (float):
            干预强度。h_t 将被修改为 h_t + (alpha * v_l)。
        device (str):
            运行计算的设备。

    Returns:
        Callable:
            一个 PyTorch 钩子函数。
            *** 修改 ***:
            此闭包现在匹配 pre-hook 的调用方式 (由 hook_manager._write_hook 调用)，
            直接接收 (hidden_state, indices) 并返回 (modified_hidden_state)。
    """

    # 预先计算加法向量，并将其移动到目标设备
    # `vector` 是从产物中加载的，可能在 CPU 上
    additive_vector = (alpha * vector).to(device)

    # *** 这是修改后的函数签名 ***
    def hook_func(
            hidden_state: torch.Tensor,  # 接收来自 pre-hook (args[0]) 的 hidden_state
            indices: torch.Tensor        # 接收来自 hook_manager 的 indices
    ) -> torch.Tensor:                   # 返回修改后的 hidden_state
        """
        实际的 PyTorch 钩子实现 (适配 pre-hook)。

        Args:
            hidden_state (torch.Tensor):
                模块的输入 hidden_state (B, SeqLen, D)。
            indices (torch.Tensor):
                一个布尔张量 (B,)，指示哪些批量索引需要被干预。

        Returns:
            torch.Tensor: 修改后的 hidden_state。
        """
        try:
            # 1. 确保加法向量与 hidden_state 的类型和设备匹配
            add_vec_typed = additive_vector.to(hidden_state.dtype, non_blocking=True)

            # 2. 只在最后一个 token 位置 (SeqLen-1) 和
            #    被 `indices` 标记的批量索引处添加向量

            # 在自回归生成时 (SeqLen=1)，-1 索引是正确的。
            # add_vec_typed (D,) -> add_vec_expanded (1, D)
            add_vec_expanded = add_vec_typed.unsqueeze(0)

            # 就地修改 (in-place modification)
            # `hidden_state` 是可变的，这会修改
            # hook_manager._write_hook 中的 `hidden_state` 变量
            hidden_state[indices, -1, :] = hidden_state[indices, -1, :] + add_vec_expanded

            # 3. 返回修改后的 hidden_state
            return hidden_state

        except Exception as e:
            logger.error(f"SARC 干预钩子执行失败: {e}", exc_info=True)
            # 如果失败，返回原始 hidden_state，避免使模型崩溃
            return hidden_state

    # 返回这个内部函数，它将被注册为钩子
    return hook_func
