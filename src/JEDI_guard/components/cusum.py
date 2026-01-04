# -*- coding: utf-8 -*-
"""
该文件实现了 CUSUM (Cumulative Sum) 控制图算法，用于时间序列的漂移检测。

在 JEDI 防御框架中，CUSUM 扮演着“记忆”和“累积证据”的角色。
根据“方法流程.md”文档的阶段 5，仅仅依赖单步的风险分数 `r_t` 容易受到噪声干扰
或被“逐步诱导”式的攻击规避。CUSUM 通过累积“超出预期的风险”，能够灵敏地检测到
持续的、即使是微小的有害倾向。

本文件中的 `CusumState` 类是一个状态机，它维护着 CUSUM 统计量，并在每次
收到新的风险分数时进行更新，最终判断是否触发警报。

[!] 修改：
- `update` 方法现在返回累积分数 A_t，而不是触发索引。
- `update` 方法不再自动重置状态。重置逻辑已移至 SarcLogitsProcessor。
"""

import torch


class CusumState:
    """
    一个用于管理和更新 CUSUM 统计量的状态机。

    该实现遵循 Page-Hinkley (PH) 检验的单边上升检测形式：
    S_t = S_{t-1} + (r_t - mu_hat - kappa)
    M_t = min(M_{t-1}, S_t)
    A_t = S_t - M_t

    当 A_t > h 时，触发警报。
    """

    def __init__(self, mu_hat: float, kappa: float, h: float, batch_size: int, device: str = 'cpu'):
        """
        初始化一批样本的 CUSUM 状态。

        Args:
            mu_hat (float):
                安全基线。这是在良性数据上观测到的风险分数 `r_t` 的均值。
                它代表了“正常”或“预期”的风险水平。

            kappa (float):
                容忍带。用于抵消 `r_t` 的正常波动，防止因随机噪声导致的误报。
                只有当 `r_t` 持续高于 `mu_hat + kappa` 时，累积量才会显著增加。

            h (float):
                报警阈值。当 CUSUM 统计量 `A_t` 超过此值时，表明检测到了显著的
                有害倾向漂移，应触发干预。

            batch_size (int):
                要同时处理的独立序列的数量。

            device (str):
                运行计算的设备。
        """
        self.mu_hat = mu_hat
        self.kappa = kappa
        self.h = h
        self.batch_size = batch_size
        self.device = device

        # S_t 和 M_t 是 CUSUM 算法的核心状态变量
        # S_t: 累积和 (Cumulative Sum)
        # M_t: 至今为止 S_t 达到过的最小值 (Running Minimum)
        # A_t (触发统计量) 将在 update 方法中动态计算
        self.S = torch.zeros(batch_size, device=device)
        self.M = torch.zeros(batch_size, device=device)

    def reset(self, indices: torch.Tensor = None):
        """
        重置指定索引或所有序列的 CUSUM 状态。

        Args:
            indices (torch.Tensor, optional):
                一个布尔或长整型张量，指示哪些序列的状态需要被重置。
                如果为 None，则重置所有序列的状态。
        """
        if indices is None:
            self.S.fill_(0)
            self.M.fill_(0)
        else:
            self.S[indices] = 0
            self.M[indices] = 0

    def update(self, r_t: torch.Tensor) -> torch.Tensor:
        """
        使用新一批的风险分数 r_t 更新 CUSUM 状态，并返回当前的累积分数 A_t。

        [!] 修改：此方法不再触发重置，仅返回 A_t。

        Args:
            r_t (torch.Tensor):
                最新一步的风险分数，形状为 (B,)，其中 B 是批量大小。

        Returns:
            torch.Tensor:
                一个张量，形状为 (B,)，包含当前所有序列的 CUSUM 分数 A_t。
        """
        if r_t.device.type != self.device:
            r_t = r_t.to(self.device, non_blocking=True)

        # 核心 CUSUM 递推公式
        # 1. 更新累积和 S_t
        increment = r_t - self.mu_hat - self.kappa
        self.S += increment

        # 2. 更新运行最小值 M_t
        # 注意：在原始 Page-Hinkley 算法中，M 在 S 更新后更新。
        # 这里的实现 `torch.minimum` 保证了这一点。
        self.M = torch.minimum(self.M, self.S)

        # 3. 计算当前的触发统计量 A_t
        A_t = self.S - self.M

        # 4. [!] 移除重置逻辑
        # triggered_indices = A_t > self.h
        # if torch.any(triggered_indices):
        #     self.reset(triggered_indices)

        # 5. [!] 返回 A_t
        return A_t.cpu()