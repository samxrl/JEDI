# -*- coding: utf-8 -*-
"""
该文件定义了 `Scorer` 类，其核心职责是计算每个生成 token 的单步“有害度”分数。

根据“方法流程.md”文档的阶段 4.1，这个分数是通过将经过标准化的隐藏状态
投影到预先提取的“条件向量” (c_l) 上来得到的。此外，还应用了边界 ReLU
（即减去一个阈值 `theta` 并取正）来过滤掉良性噪声。

`Scorer` 类封装了这一逻辑，使得在线防御系统 `Guard` 可以方便地调用它来评估
每个新生成 token 的风险。
"""

import torch
from typing import Tuple, Optional

from .representation import apply_transform


class Scorer:
    """
    计算逐 token 的原始风险分数 (s_t) 和经过阈值处理后的风险分数 (r_t)。
    """

    def __init__(
            self,
            condition_vector: torch.Tensor,
            transform: Tuple[Optional[torch.Tensor], torch.Tensor],
            theta: float,
            device: str = 'cpu'
    ):
        """
        初始化 Scorer 组件。

        Args:
            condition_vector (torch.Tensor):
                条件向量 `c_l`，用于检测有害语义。形状为 (D,)。

            transform (Tuple[Optional[torch.Tensor], torch.Tensor]):
                一个元组 `(W, mu)`，包含用于内容窗口的白化矩阵和均值向量。
                这是从离线校准阶段加载的产物。

            theta (float):
                分数阈值 `theta`。在计算最终风险分数 `r_t` 之前，
                会从原始分数 `s_t` 中减去该值。

            device (str):
                指定运行计算的设备 (例如, 'cuda:0' 或 'cpu')。
        """
        self.condition_vector = condition_vector.to(device, non_blocking=True)
        # 将变换矩阵和向量也移动到指定设备
        W, mu = transform
        self.transform = (
            W.to(device, non_blocking=True) if W is not None else None,
            mu.to(device, non_blocking=True)
        )
        self.theta = theta
        self.device = device

    @torch.no_grad()
    def calculate_scores(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为给定的隐藏状态序列计算原始分数和风险分数。

        Args:
            hidden_states (torch.Tensor):
                一批新生成 token 的隐藏状态。
                期望形状为 (B, 1, D)，其中 B 是批量大小，D 是隐藏维度。
                中间的 '1' 代表序列长度为 1（因为我们是逐 token 处理）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            一个元组 `(s_t, r_t)`，其中：
            - `s_t` (torch.Tensor): 原始投影分数，形状为 (B,)。
            - `r_t` (torch.Tensor): 经过边界 ReLU 处理后的最终风险分数，形状为 (B,)。
        """
        if hidden_states.device.type != self.device:
            hidden_states = hidden_states.to(self.device, non_blocking=True)

        # 1. 对隐藏状态进行标准化（白化/中心化）
        # hidden_states 形状 (B, 1, D) -> transformed_states 形状 (B, 1, D)
        transformed_states = apply_transform(hidden_states, self.transform)

        # 2. 将标准化后的表征投影到条件向量上，得到原始分数 s_t
        # transformed_states 形状 (B, 1, D), condition_vector 形状 (D,)
        # -> s_t 形状 (B, 1)
        s_t_unsq = torch.einsum('bnd,d->bn', transformed_states, self.condition_vector)
        s_t = s_t_unsq.squeeze(1)  # 移除中间的维度 -> (B,)

        # 3. 应用边界 ReLU 得到最终的风险分数 r_t
        # r_t = max(0, s_t - theta)
        r_t = torch.clamp(s_t - self.theta, min=0)

        return s_t.cpu(), r_t.cpu()
