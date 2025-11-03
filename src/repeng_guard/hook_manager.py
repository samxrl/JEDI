# -*- coding: utf-8 -*-
"""
HookManager (钩子管理器)

该文件实现了 `HookManager` 类，其核心职责是在 PyTorch 模型
(特别是 Hugging Face Transformers 模型) 的特定层上注册、管理和移除钩子。

根据“方法流程.md”，防御系统需要在两个时刻与模型交互：
1.  **读取 (Read)**: 在每个 token 生成步骤中，需要从目标层读取
    隐藏状态 (h_t)，以便 `Scorer` 计算风险分数 (阶段 5.3)。
    这通过注册一个 `register_forward_hook` 来实现。
2.  **写入 (Write/Intervention)**: 当 CUSUM 触发警报时，
    需要向目标层注入干预向量 (v_l) (阶段 6.1)。
    这也通过 `register_forward_hook` 实现，但该钩子会修改
    `output` 张量。

`HookManager` 封装了查找层、注册钩子、存储句柄 (handle) 以及
在防御结束时清理所有钩子的复杂逻辑。
"""

import torch
from torch.nn import Module
from typing import Callable, Optional, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def _find_target_layer(model: Module, layer_id: int) -> Optional[Module]:
    """
    一个辅助函数，用于在常见的 HF 模型结构中查找目标层模块。

    Args:
        model (Module): Hugging Face 模型。
        layer_id (int): 目标层的索引。

    Returns:
        Optional[Module]: 找到的 PyTorch 模块，如果未找到则返回 None。
    """
    # 尝试 LLaMA, Mistral, Gemma 等模型的常见结构
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        if 0 <= layer_id < len(model.model.layers):
            return model.model.layers[layer_id]

    # 尝试 GPT-2, OPT 等模型的常见结构
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        if 0 <= layer_id < len(model.transformer.h):
            return model.transformer.h[layer_id]

    # 尝试作为备选的 'layers' 属性
    if hasattr(model, 'layers'):
        if 0 <= layer_id < len(model.layers):
            return model.layers[layer_id]

    logger.warning(f"无法在模型 {type(model).__name__} 中自动定位第 {layer_id} 层。 "
                   f"请检查模型结构并可能需要调整 _find_target_layer 帮助函数。")
    return None


class HookManager:
    """
    管理模型钩子的注册、状态和移除。
    """

    def __init__(self, model: Module, layer_id: int, device: str = 'cpu'):
        """
        初始化钩子管理器。

        Args:
            model (Module): 要附加钩子的模型。
            layer_id (int): 目标层的索引。
            device (str): 目标设备。
        """
        self.model = model
        self.layer_id = layer_id
        self.device = device
        self.target_layer = _find_target_layer(model, layer_id)

        if self.target_layer is None:
            raise ValueError(f"无法在模型中找到第 {layer_id} 层。")

        # 存储 PyTorch 钩子句柄 (handle)，以便后续移除
        self._read_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._intervention_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # 存储从读钩子捕获的激活
        self._captured_activations: List[torch.Tensor] = []

        # 存储干预钩子的状态
        self._intervention_func: Optional[Callable] = None
        self._intervention_indices: Optional[torch.Tensor] = None

    def _read_hook(self, module: Module, inputs: Tuple[Any, ...], outputs: Tuple[Any, ...]):
        """
        *读钩子*的实现。
        此函数在 `target_layer` 的前向传播*之后*被调用。
        它捕获输出的隐藏状态并将其存储。
        """
        # `outputs` 通常是一个元组，第一个元素是隐藏状态
        hidden_state = outputs[0]

        # 我们只关心序列中的最后一个 token 的隐藏状态，
        # 因为这是用于预测 *下一个* token 的状态。
        # 形状: (B, SeqLen, D) -> (B, 1, D)
        last_token_hidden_state = hidden_state[:, -1:, :].detach().to(self.device, non_blocking=True)
        self._captured_activations.append(last_token_hidden_state)

    def _intervention_hook(self, module: Module, inputs: Tuple[Any, ...], outputs: Tuple[Any, ...]):
        """
        *干预钩子*的实现。
        此函数也在 `target_layer` 的前向传播*之后*被调用。
        如果干预被激活，它会*修改* `outputs` 元组。
        """
        if self._intervention_func is not None and self._intervention_indices is not None:
            # `self._intervention_func` 是 `create_intervention_hook_func`
            # 返回的函数，它会就地修改 `outputs`。
            # 我们传递 `outputs` 和要修改的批量索引。
            return self._intervention_func(module, inputs, outputs, self._intervention_indices)

        # 如果未激活干预，则不执行任何操作
        return outputs

    def attach_read_hook(self):
        """
        注册*读钩子*，用于捕获激活。
        """
        if self._read_hook_handle is not None:
            logger.warning("读钩子已注册。")
            return

        self._read_hook_handle = self.target_layer.register_forward_hook(self._read_hook)
        logger.debug(f"读钩子已附加到第 {self.layer_id} 层。")

    def set_intervention_state(self, func: Callable, indices: torch.Tensor):
        """
        设置下一次前向传播时要执行的干预。

        Args:
            func (Callable):
                从 `interventions.py` 创建的干预函数。
            indices (torch.Tensor):
                一个布尔张量 (B,)，指示哪些批量索引需要被干预。
        """
        self._intervention_func = func
        self._intervention_indices = indices

        # 确保干预钩子只在需要时被注册一次
        if self._intervention_hook_handle is None:
            self._intervention_hook_handle = self.target_layer.register_forward_hook(self._intervention_hook)
            logger.debug(f"干预钩子已动态附加到第 {self.layer_id} 层。")

    def clear_intervention_state(self):
        """
        清除干预状态。这不会移除钩子句柄，只是使其在下次调用时失效。
        """
        self._intervention_func = None
        self._intervention_indices = None

    def remove_all_hooks(self):
        """
        移除所有已注册的 PyTorch 钩子，以清理模型。
        """
        if self._read_hook_handle:
            self._read_hook_handle.remove()
            self._read_hook_handle = None
            logger.debug("读钩子已移除。")

        if self._intervention_hook_handle:
            self._intervention_hook_handle.remove()
            self._intervention_hook_handle = None
            logger.debug("干预钩子已移除。")

    def get_last_captured_activation(self) -> Optional[torch.Tensor]:
        """
        获取最近一次捕获的激活。
        """
        if not self._captured_activations:
            return None
        # 立即清除，确保每个 token 只被处理一次
        return self._captured_activations.pop()

    def clear_captured_activations(self):
        """
        清除激活缓冲区。
        """
        self._captured_activations.clear()
