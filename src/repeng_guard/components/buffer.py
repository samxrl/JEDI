# -*- coding: utf-8 -*-
"""
该文件实现了“提交缓冲” (Commit Buffer) 机制。

根据“方法流程.md”文档的阶段 5，为了在检测到有害倾向时能够优雅地回滚，
而不是将已经部分生成的有害内容暴露给用户，SARC 引入了提交缓冲。

`CommitBuffer` 类实现了一个先进先出 (FIFO) 队列，它会暂存最新生成的
一小段 token 序列。只有当 CUSUM 算法确认当前生成过程安全时，缓冲队列
头部的 token 才会被“提交”（即可以安全地展示给用户）。一旦触发警报，
整个缓冲区可以被清空，从而实现无缝回滚。
"""

from collections import deque
from typing import List, Dict, Any


class CommitBuffer:
    """
    管理一个先进先出 (FIFO) 队列，用于暂存待提交的 token。

    这个类被设计为按批次 (batch) 工作，为每个并行的生成序列维护一个独立的缓冲区。
    """

    def __init__(self, capacity: int, batch_size: int):
        """
        初始化提交缓冲区。

        Args:
            capacity (int):
                每个序列的缓冲区最大容量（可以容纳的 token 数量）。
                这个值也决定了最大可能的回滚长度。

            batch_size (int):
                要同时管理的并行生成序列的数量。
        """
        if capacity <= 0:
            raise ValueError("缓冲区容量必须是正整数。")
        self.capacity = capacity
        self.batch_size = batch_size

        # 为批次中的每个序列创建一个独立的双端队列 (deque)
        # deque 提供了高效的从两端添加和弹出元素的操作
        self.buffers: List[deque] = [deque(maxlen=capacity) for _ in range(batch_size)]

    def add(self, tokens: List[Any]):
        """
        将新生成的 token 添加到对应序列的缓冲区末尾。

        Args:
            tokens (List[Any]):
                一个列表，包含批次中每个序列新生成的 token。
                列表的长度应等于 `batch_size`。
        """
        if len(tokens) != self.batch_size:
            raise ValueError(f"输入的 token 数量 ({len(tokens)}) 与批次大小 ({self.batch_size}) 不匹配。")

        for i in range(self.batch_size):
            self.buffers[i].append(tokens[i])

    def commit(self, num_tokens: int = 1) -> List[List[Any]]:
        """
        从每个序列的缓冲区头部“提交”（即移除并返回）指定数量的 token。

        这模拟了将安全的 token 发送给用户的过程。

        Args:
            num_tokens (int, optional):
                要为每个序列提交的 token 数量。默认为 1。

        Returns:
            List[List[Any]]:
                一个列表，其中每个子列表包含了从对应序列缓冲区提交的 token。
        """
        committed_batch = [[] for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            for _ in range(num_tokens):
                if self.buffers[i]:
                    committed_batch[i].append(self.buffers[i].popleft())
                else:
                    break  # 如果缓冲区为空，则停止提交
        return committed_batch

    def rollback(self, indices: torch.Tensor):
        """
        清空指定索引的序列的缓冲区。

        当 CUSUM 检测到有害倾向时调用此方法。

        Args:
            indices (torch.Tensor):
                一个布尔或长整型张量，指示哪些序列的缓冲区需要被清空。
        """
        # 将 PyTorch 张量转换为可迭代的索引列表
        if indices.dtype == torch.bool:
            idx_list = indices.nonzero(as_tuple=True)[0]
        else:
            idx_list = indices

        for i in idx_list:
            self.buffers[i].clear()

    def flush_all(self) -> List[List[Any]]:
        """
        清空并返回所有缓冲区中的剩余内容。

        当一个生成序列正常结束（未触发警报）时调用此方法，以确保
        所有暂存的 token 都被提交。

        Returns:
            List[List[Any]]:
                一个列表，其中每个子列表包含了从对应序列缓冲区取出的所有剩余 token。
        """
        remaining_batch = []
        for i in range(self.batch_size):
            remaining_batch.append(list(self.buffers[i]))
            self.buffers[i].clear()
        return remaining_batch
