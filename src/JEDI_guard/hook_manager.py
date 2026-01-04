# -*- coding: utf-8 -*-
"""
HookManager (钩子管理器)

该文件定义了 `HookManager` 类，是 `Guard` 的一个内部组件。

[!] 修改：
- `set_intervention_state` 现在额外接受 `dynamic_betas` 张量。
- `_write_hook` 将 `dynamic_betas` 传递给干预函数。

核心职责:
1.  充当模型 (`model`) 和 JEDI 处理器 (`SarcLogitsProcessor`) 之间的桥梁。
2.  提供 `attach_read_hook` 方法，在目标层注册一个 PyTorch
    `register_forward_hook`，用于“读取”隐藏状态。
3.  提供 `set_intervention_state` 方法，允许 JEDI 处理器
    动态地请求在下一
    个 forward 传递中“写入”（即干预）隐藏状态。
4.  管理读/写钩子的句柄 (`handle`)，并在 `detach` 时正确移除它们，
    防止内存泄漏并恢复模型原始行为。
"""

import torch
from torch.nn import Module
from typing import Callable, Optional, List, Dict, Any
import logging

# 设置一个日志记录器
logger = logging.getLogger(__name__)


class HookManager:
    """
    管理 PyTorch 钩子 (hooks) 以在模型的前向传播中读取和写入激活。
    """

    def __init__(self, model: Module, layer_id: int, device: str = 'cpu'):
        """
        初始化钩子管理器。

        Args:
            model (Module):
                要附加钩子的 Hugging Face 模型。
            layer_id (int):
                目标层的索引 (例如，Llama 模型的 `model.layers[layer_id]`)。
            device (str):
                运行计算的设备。
        """
        self.model = model
        self.layer_id = layer_id
        self.device = device

        # 尝试自动定位模型中的解码器层列表
        self.layer_module = self._find_target_layer(model, layer_id)
        if self.layer_module is None:
            msg = f"无法在模型中定位到第 {layer_id} 层。请检查模型结构和 layer_id。"
            logger.error(msg)
            raise ValueError(msg)

        # 句柄 (Handles) 用于在之后移除钩子
        self.read_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.write_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # 状态变量
        self.captured_activations: List[torch.Tensor] = []
        self.intervention_function: Optional[Callable] = None
        self.intervention_indices: Optional[torch.Tensor] = None
        self.dynamic_betas: Optional[torch.Tensor] = None  # [!] 新增：存储动态 beta

    def _find_target_layer(self, model: Module, layer_id: int) -> Optional[Module]:
        """
        尝试在模型中找到目标层模块。
        这适用于 Llama, Mistral, Gemma 等常见架构。
        """
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # 适用于 Llama, Mistral, Gemma, Phi-3 等
                return model.model.layers[layer_id]
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # 适用于 GPT-2, GPT-NeoX
                return model.transformer.h[layer_id]
            elif hasattr(model, 'layers'):
                # 备用，如果 model.layers 直接在顶层
                return model.layers[layer_id]
            else:
                logger.warning("未知的模型架构。无法自动定位 'layers' 属性。")
                return None
        except IndexError:
            logger.error(f"层索引 {layer_id} 超出范围。模型只有 "
                         f"{len(model.model.layers)} 层。")
            return None
        except Exception as e:
            logger.error(f"在定位目标层时出错: {e}", exc_info=True)
            return None

    def _read_hook(self, module: Module, args: tuple, output: Any):
        """
        “读”钩子 (Forward Hook)。
        在目标层的前向传播*之后*执行，用于捕获其输出的隐藏状态。
        """
        hidden_state = None
        if isinstance(output, tuple):
            # 大多数 HF 模型 (Llama, etc.) 的层输出是元组 (hidden_state, caches, ...)
            hidden_state = output[0]
        else:
            # 某些模型可能直接输出张量
            hidden_state = output

        if hidden_state is None:
            logger.warning(f"JEDI 读钩子在第 {self.layer_id} 层收到了空的输出。")
            return

        # --- 错误修复：---
        # 区分 3D (预填充) 和 2D (自回归) 的情况
        final_hidden_state_3d = None

        if hidden_state.dim() == 3:
            # 3D: (batch_size, seq_len, hidden_dim) - 这是预填充阶段
            # 我们只关心序列中的最后一个 token
            final_hidden_state_3d = hidden_state[:, -1:, :].detach().to(self.device, non_blocking=True)

        elif hidden_state.dim() == 2:
            # 2D: (batch_size, hidden_dim) - 这是自回归阶段 (seq_len=1)
            # 它已经是最后一个 token，我们只需添加 'seq_len' 维度
            final_hidden_state_3d = hidden_state.unsqueeze(1).detach().to(self.device, non_blocking=True)

        else:
            # 异常情况
            logger.warning(f"JEDI 读钩子: 收到意外的隐藏状态维度: "
                           f"{hidden_state.dim()}。跳过捕获。")
            return

        # 存储形状一致的 (batch_size, 1, hidden_dim) 张量
        self.captured_activations.append(final_hidden_state_3d)

        # 动态附加“写”钩子 (如果已被请求)
        self._dynamically_attach_write_hook(module)

    def _dynamically_attach_write_hook(self, module: Module):
        """
        如果干预已被请求，则附加“写”钩子 (Forward Hook)。
        “写”钩子在 `forward` 方法*之后*执行，用于修改其输出。
        """
        if self.intervention_function and self.write_hook_handle is None:
            # logger.debug(f"在第 {self.layer_id} 层动态附加干预钩子。")
            # [!] 修改：从 pre_hook 更改为 hook
            self.write_hook_handle = module.register_forward_hook(
                self._write_hook
            )

    # [!] 修改：更改了函数签名和内部逻辑
    def _write_hook(self, module: Module, args: tuple, output: Any) -> Any:
        """
        “写”钩子 (Forward Hook)。
        在目标层的前向传播*之后*执行，用于修改其输出 `output` (即 hidden_state)。
        """
        # [!] 检查所有必需的状态
        if (self.intervention_function is None or
                self.intervention_indices is None or
                self.dynamic_betas is None):
            return output  # [!] 如果未激活干预，必须返回原始 output

        # 1. 从 output 中提取 hidden_state
        original_hidden_state = None
        is_tuple_output = False

        if isinstance(output, tuple):
            original_hidden_state = output[0]
            is_tuple_output = True
        else:
            original_hidden_state = output

        if original_hidden_state is None:
            logger.warning(f"JEDI 写钩子在第 {self.layer_id} 层收到了空的输出。")
            return output

        # 2. 应用干预函数
        #    intervention_function 负责只修改 self.intervention_indices
        #    标记为 True 的那些序列。
        # [!] 传递 dynamic_betas
        modified_hidden_state = self.intervention_function(
            original_hidden_state,
            self.intervention_indices,
            self.dynamic_betas
        )

        # 3. 将修改后的 hidden_state 重新打包并返回
        if is_tuple_output:
            # [!] 返回修改后的元组
            return (modified_hidden_state,) + output[1:]
        else:
            # [!] 返回修改后的张量
            return modified_hidden_state

    def attach_read_hook(self):
        """
        在目标层上注册永久的“读”钩子。
        """
        if self.read_hook_handle:
            logger.warning("“读”钩子已被附加。将先移除旧钩子。")
            self.read_hook_handle.remove()

        self.read_hook_handle = self.layer_module.register_forward_hook(
            self._read_hook
        )
        # logger.debug(f"“读”钩子已附加到第 {self.layer_id} 层。")

    def set_intervention_state(self, func: Callable, indices: torch.Tensor, dynamic_betas: torch.Tensor):
        """
        由 SarcLogitsProcessor 调用，用于请求在下一个步骤激活干预。

        [!] 修改：新增 dynamic_betas 参数。
        """
        self.intervention_function = func
        self.intervention_indices = indices  # (B,) bool tensor
        self.dynamic_betas = dynamic_betas  # [!] (B,) float tensor

    def clear_intervention_state(self):
        """
        在 `generate` 调用开始时调用，重置干预状态。
        """
        self.intervention_function = None
        self.intervention_indices = None
        self.dynamic_betas = None  # [!] 清理 beta

        # “写”钩子是动态附加的，我们需要在每轮开始时将其移除
        if self.write_hook_handle:
            # logger.debug(f"在第 {self.layer_id} 层清理干预钩子。")
            self.write_hook_handle.remove()
            self.write_hook_handle = None

    def get_last_captured_activation(self) -> Optional[torch.Tensor]:
        """
        由 SarcLogitsProcessor 调用，用于获取最近一次“读”钩子捕获的激活。
        """
        if not self.captured_activations:
            return None

        # 在一次 forward 中，读钩子会按照“填充阶段 -> 自回归阶段”的顺序多次触发。
        # 我们只关心**最新**捕获的激活（对应当前 logits 的 token），
        # 因此应弹出列表末尾的元素，而不是队列头。否则会错用最早的前缀 token，
        # 造成检测延迟被整体平移一个 prompt 长度。
        return self.captured_activations.pop()

    def clear_captured_activations(self):
        """
        在 `generate` 调用开始时调用，清空上一轮的激活缓存。
        """
        self.captured_activations.clear()

    def remove_all_hooks(self):
        """
        在 `Guard.detach` 时调用，彻底清理所有钩子。
        """
        if self.read_hook_handle:
            self.read_hook_handle.remove()
            self.read_hook_handle = None

        self.clear_intervention_state()  # 这会移除 write_hook_handle
        self.clear_captured_activations()
        # logger.debug(f"第 {self.layer_id} 层的所有钩子已移除。")