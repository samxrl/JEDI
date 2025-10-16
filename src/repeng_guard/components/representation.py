# -*- coding: utf-8 -*-
"""
该文件实现了与模型内部表征（隐藏状态）处理相关的核心功能。

根据“方法流程.md”文档，在计算任何风险分数之前，需要对原始的隐藏状态
进行标准化处理（中心化和可选的白化）。这样做可以消除不同维度间的尺度差异和相关性，
使得后续通过向量投影计算出的分数更加稳定和可比较。

本文件主要提供 `apply_transform` 函数，该函数负责执行此标准化步骤。
"""

import torch
from typing import Tuple, Optional


def apply_transform(
        hidden_states: torch.Tensor,
        transform: Tuple[Optional[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
    将预先计算好的变换（中心化和可选的白化）应用于输入的隐藏状态。

    此函数是表征工程流程中的关键一步。它接收一批隐藏状态以及一个包含
    白化矩阵 'W' (如果启用) 和均值向量 'mu' 的元组。

    Args:
        hidden_states (torch.Tensor):
            从模型中提取的原始隐藏状态张量。
            形状可以是 (N, D) 用于单个序列的聚合表示，
            或 (B, N, D) 用于批量处理的逐 token 序列，
            其中 B 是批量大小, N 是序列长度, D 是隐藏层维度。

        transform (Tuple[Optional[torch.Tensor], torch.Tensor]):
            一个元组 `(W, mu)`，其中:
            - `W` (torch.Tensor, optional): 白化矩阵，形状为 (D, D)。如果为 None，则只执行中心化。
            - `mu` (torch.Tensor): 均值向量，形状为 (D,)，用于中心化。

    Returns:
        torch.Tensor:
            经过变换（中心化和可选白化）后的隐藏状态，形状与输入 `hidden_states` 相同。

    Raises:
        ValueError: 如果输入 `hidden_states` 的维度不是 2 或 3，则会引发错误。
    """
    W, mu = transform
    device = hidden_states.device

    # 确保 mu 和 W (如果存在) 与隐藏状态在同一设备上
    mu_device = mu.to(device)

    # 步骤 1: 中心化 (减去均值)
    centered_states = hidden_states - mu_device

    # 步骤 2: (可选) 应用白化变换
    if W is not None:
        W_device = W.to(device)

        # 使用 einsum 以优雅地处理 2D 和 3D 张量
        if hidden_states.dim() == 2:  # 形状 (N, D)
            # 'nd,cd->nc' -> (N, D) @ (D, D).T = (N, D)
            # 注意：在 RepEng 中，通常使用 W @ (h-mu)，所以这里是 'cd'
            transformed_states = torch.einsum('nd,cd->nc', centered_states, W_device)
        elif hidden_states.dim() == 3:  # 形状 (B, N, D)
            # 'bnd,cd->bnc' -> 对批量中的每个 (N, D) 矩阵执行变换
            transformed_states = torch.einsum('bnd,cd->bnc', centered_states, W_device)
        else:
            raise ValueError(f"不支持的隐藏状态维度: {hidden_states.dim()}。只支持 2D 或 3D 张量。")

        return transformed_states
    else:
        # 如果 W 为 None，则只返回中心化后的结果
        return centered_states
