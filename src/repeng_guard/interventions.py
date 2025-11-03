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
            一个 PyTorch 钩子函数。该函数接收 (module, inputs, outputs, indices)
            并返回修改后的 `outputs`。
    """

    # 预先计算加法向量，并将其移动到目标设备
    # `vector` 是从产物中加载的，可能在 CPU 上
    additive_vector = (alpha * vector).to(device)

    def hook_func(
            module: Module,
            inputs: Tuple[Any, ...],
            outputs: Tuple[Any, ...],
            indices: torch.Tensor
    ) -> Tuple[Any, ...]:
        """
        实际的 PyTorch 钩子实现。

        Args:
            module (Module): 钩子附加到的模块。
            inputs (Tuple[Any, ...]): 模块的输入。
            outputs (Tuple[Any, ...]): 模块的输出 (通常是 (hidden_state, ...))。
            indices (torch.Tensor):
                一个布尔张量 (B,)，指示哪些批量索引需要被干预。

        Returns:
            Tuple[Any, ...]: 修改后的 outputs 元组。
        """
        try:
            # 1. 复制一份 `outputs` 元组，因为它是不可变的
            # (注意: 我们只复制元组结构，底层的张量仍然是引用)
            new_outputs = list(outputs)

            # 2. `hidden_state` 是 `outputs` 的第一个元素
            hidden_state = new_outputs[0]  # 形状 (B, SeqLen, D)

            # 3. 确保加法向量与 hidden_state 的类型和设备匹配
            add_vec_typed = additive_vector.to(hidden_state.dtype, non_blocking=True)

            # 4. 只在最后一个 token 位置 (SeqLen-1) 和
            #    被 `indices` 标记的批量索引处添加向量

            # 形状 (B, SeqLen, D)
            # `indices` 形状 (B,)
            # 我们只想修改 `hidden_state[indices, -1, :]`

            # 为了正确广播，我们需要将 add_vec_typed (D,) 扩展为 (1, D)
            # 并将其添加到 (N_triggered, D) 的切片上
            add_vec_expanded = add_vec_typed.unsqueeze(0)

            hidden_state[indices, -1, :] = hidden_state[indices, -1, :] + add_vec_expanded

            # 5. 将修改后的 hidden_state 放回 new_outputs 列表
            new_outputs[0] = hidden_state

            # 6. 返回修改后的元组
            return tuple(new_outputs)

        except Exception as e:
            logger.error(f"SARC 干预钩子执行失败: {e}", exc_info=True)
            # 如果失败，返回原始输出，避免使模型崩溃
            return outputs

    # 返回这个内部函数，它将被注册为钩子
    return hook_func
