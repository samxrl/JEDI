# -*- coding: utf-8 -*-
"""
【核心】Guard 类

该文件定义了 Guard 类，它是 SARC 防御系统的一站式入口，
封装了所有在线防御逻辑，实现了“方法流程.md”中的阶段 5 和 6。

核心职责:
1.  通过 `from_artifacts` 类方法加载所有离线准备好的防御产物。
2.  通过 `attach(model)` 方法，使用 'with' 上下文管理器将自身附加到
    Hugging Face 模型上。
3.  在 `attach` 期间，通过修补 (patch) 模型的 `generate` 方法来拦截生成流程。
4.  注入一个自定义的 `LogitsProcessor` (SarcLogitsProcessor)，
    该处理器在每个生成步骤执行以下操作：
    a. 从 `HookManager` 获取当前 token 的隐藏状态 (由读钩子捕获)。
    b. 调用 `Scorer` 计算单步风险分数 r_t (阶段 4.1, 5.3)。
    c. 将 r_t 送入 `CusumState` 进行时间序列累积 (阶段 5.4)。
    d. 如果 `CusumState` 触发警报，则激活干预 (阶段 5.5)。
5.  通过 `HookManager` 动态注册一个“写钩子”，
    该钩子执行 `interventions.py` 中定义的 ActAdd 注入 (阶段 6.1)。
6.  在 `with` 块结束时，自动 `detach`，清理所有钩子并恢复
    模型原始的 `generate` 方法。

注意：此实现侧重于检测和干预。`CommitBuffer` (提交缓冲) 逻辑
(阶段 5.2, 5.5) 在一个标准的、非流式的 `generate` 调用中难以
精确实现（因为它无法控制何时将 token 真正“显示”给用户）。
因此，此 `Guard` 实现专注于核心的“检测-触发-转向”流程，
而 `buffer.py` 模块可用于实现了自定义流式生成的高级框架中。
"""

import torch
import logging
from torch.nn import Module
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Callable, Optional, List, Dict, Any
from contextlib import contextmanager

from .components.scorer import Scorer
from .components.cusum import CusumState
from .components.representation import apply_transform
from .hook_manager import HookManager
from .interventions import create_intervention_hook_func
from .utils.config_loader import load_defense_artifacts

# 设置一个日志记录器
logger = logging.getLogger(__name__)


class SarcLogitsProcessor(LogitsProcessor):
    """
    SARC 防御的核心逻辑处理器。
    在 `generate` 循环的每个 token 生成步骤中被调用。
    """

    def __init__(self, guard_instance, batch_size: int):
        """
        初始化 SARC LogitsProcessor。

        Args:
            guard_instance (Guard):
                对主 Guard 实例的引用，用于访问 scorer, cusum, 和 hook_manager。
            batch_size (int):
                当前生成请求的批量大小。
        """
        self.guard = guard_instance
        self.batch_size = batch_size
        self.device = self.guard.device
        self.is_batch = batch_size > 1

        # 为这个特定的生成调用初始化一个 CUSUM 状态机
        self.cusum = CusumState(
            mu_hat=self.guard.mu_hat,
            kappa=self.guard.kappa,
            h=self.guard.h,
            batch_size=batch_size,
            device=self.device
        )
        # 跟踪哪些序列已经触发了干预
        self.intervention_active = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        在每个生成步骤中执行检测和干预逻辑。

        Args:
            input_ids (torch.LongTensor): 至今为止生成的 token 序列。
            scores (torch.FloatTensor): 当前步骤的原始 logits 分数。

        Returns:
            torch.FloatTensor: 可能被修改（也可能未被修改）的 logits 分数。
        """
        # 1. 从钩子获取刚刚被捕获的隐藏状态 (阶段 5.3)
        # 注意：钩子是在模型 forward 传播中、此 logits 处理器运行前触发的
        try:
            # 形状应为 (B, 1, D) 或 (B, D)
            hidden_state = self.guard.hook_manager.get_last_captured_activation()
            if hidden_state is None:
                logger.warning("SARC: 未能从 HookManager 获取隐藏状态。跳过本轮检测。")
                return scores

            # 确保形状为 (B, 1, D) 以便 scorer 处理
            if hidden_state.dim() == 2:
                hidden_state = hidden_state.unsqueeze(1)  # (B, D) -> (B, 1, D)

            # 确保隐藏状态与 logits 的批量大小一致
            if hidden_state.shape[0] != scores.shape[0]:
                logger.error(f"SARC: 隐藏状态批量大小 ({hidden_state.shape[0]}) 与 "
                             f"Logits 批量大小 ({scores.shape[0]}) 不匹配。")
                return scores

        except Exception as e:
            logger.error(f"SARC: 获取隐藏状态时出错: {e}", exc_info=True)
            return scores

        # 2. 计算风险分数 (阶段 5.3)
        s_t, r_t = self.guard.scorer.calculate_scores(hidden_state)

        # 3. 更新 CUSUM 状态机 (阶段 5.4)
        # `triggered_indices` 是一个布尔张量 (B,)，指示哪些序列 *在这一步* 触发了警报
        triggered_indices = self.cusum.update(r_t)

        # 4. 激活干预 (阶段 5.5, 6.1)
        if torch.any(triggered_indices):
            # 更新我们的状态，标记哪些序列 *从现在开始* 需要干预
            self.intervention_active |= triggered_indices.to(self.device)

            # 告诉 HookManager 在 *下一次* forward 传递时
            # 对已触发的序列应用干预钩子
            self.guard.hook_manager.set_intervention_state(
                self.guard.intervention_func,
                self.intervention_active
            )
            logger.info(f"SARC: CUSUM 触发警报。激活以下序列的干预: "
                        f"{triggered_indices.nonzero(as_tuple=True)[0].tolist()}")

        return scores


class Guard:
    """
    SARC 防御系统的主类。
    通过上下文管理器 (`with guard.attach(model): ...`) 来使用。
    """

    def __init__(
            self,
            layer_id: int,
            theta: float,
            mu_hat: float,
            kappa: float,
            h: float,
            scorer: Scorer,
            intervention_func: Callable,
            device: str = 'cpu'
    ):
        """
        初始化 Guard 实例。

        注意：推荐使用 `Guard.from_artifacts` 类方法来创建实例。
        """
        self.layer_id = layer_id
        self.device = device

        # CUSUM 参数
        self.mu_hat = mu_hat
        self.kappa = kappa
        self.h = h

        # 核心组件
        self.scorer = scorer
        self.intervention_func = intervention_func

        # 运行时状态（在 `attach` 时设置）
        self.model: Optional[Module] = None
        self.hook_manager: Optional[HookManager] = None
        self.original_generate: Optional[Callable] = None

        logger.info(f"Guard 实例已初始化。将在第 {layer_id} 层运行。")
        logger.info(f"防御参数: theta={theta:.4f}, mu_hat={mu_hat:.4f}, kappa={kappa:.4f}, h={h:.4f}")

    @classmethod
    def from_artifacts(cls, artifact_path: str, device: Optional[str] = None) -> "Guard":
        """
        【推荐】从离线产物目录加载并创建 Guard 实例。

        Args:
            artifact_path (str):
                包含所有必需产物文件的目录路径。
                (defense_params.yaml, transforms.pt,
                 condition_vectors.pt, intervention_vectors.pt)
            device (str, optional):
                要运行防御的设备 ('cuda', 'cpu')。
                如果为 None，将自动检测 CUDA。

        Returns:
            Guard: 一个配置好并准备就绪的 Guard 实例。
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. 使用工具函数加载所有文件
        try:
            artifacts = load_defense_artifacts(artifact_path, device)
            params = artifacts['defense_params']
        except FileNotFoundError as e:
            logger.error(f"加载防御产物失败: {e}。请确保路径 '{artifact_path}' 正确"
                         f"并且包含了所有必需的文件。")
            raise

        # 2. 提取防御参数 (阶段 4 校准结果)
        layer_id = params['best_layer']
        theta = params['theta']
        mu_hat = params['mu_hat']
        kappa = params['kappa']
        h = params['h']

        # 3. 准备 Scorer (阶段 4.1)
        transform = artifacts['transforms']['content_window'][layer_id]
        condition_vector = artifacts['condition_vectors'][layer_id]
        scorer = Scorer(
            condition_vector=condition_vector,
            transform=transform,
            theta=theta,
            device=device
        )

        # 4. 准备 Intervention (阶段 6.1)
        intervention_vector = artifacts['intervention_vectors'][layer_id]
        # 假设 alpha (干预强度) 也是一个可配置参数，这里使用一个合理的默认值
        alpha = params.get('alpha', 1.5)
        intervention_func = create_intervention_hook_func(
            vector=intervention_vector,
            alpha=alpha,
            device=device
        )

        # 5. 创建并返回 Guard 实例
        return cls(
            layer_id=layer_id,
            theta=theta,
            mu_hat=mu_hat,
            kappa=kappa,
            h=h,
            scorer=scorer,
            intervention_func=intervention_func,
            device=device
        )

    @contextmanager
    def attach(self, model: Module):
        """
        将防御系统附加到模型上，用作上下文管理器。

        用法:
            with guard.attach(model):
                model.generate(...)

        Args:
            model (Module): 要保护的 Hugging Face 模型。
        """
        if self.model is not None:
            logger.warning("Guard 已经附加到某个模型上。将先分离旧模型。")
            self.detach()

        try:
            self.model = model
            self.hook_manager = HookManager(
                model=model,
                layer_id=self.layer_id,
                device=self.device
            )

            # 1. 保存原始的 generate 方法
            self.original_generate = model.generate
            # 2. 修补 (Patch) generate 方法
            model.generate = self._guarded_generate

            # 3. 附加读钩子，用于捕获隐藏状态
            #    (干预钩子将在 SarcLogitsProcessor 触发时动态附加)
            self.hook_manager.attach_read_hook()

            logger.info(f"Guard 已附加到模型 {model.config.name_or_path} (层: {self.layer_id})。")
            yield self  # 进入 'with' 块

        finally:
            self.detach()  # 退出 'with' 块时自动分离

    def detach(self):
        """
        从模型分离防御系统，清理钩子并恢复原始方法。
        """
        if self.model and self.original_generate:
            # 恢复原始的 generate 方法
            self.model.generate = self.original_generate
            logger.info(f"已从模型恢复原始 'generate' 方法。")

        if self.hook_manager:
            # 移除所有已注册的 PyTorch 钩子
            self.hook_manager.remove_all_hooks()
            logger.info("已移除所有模型钩子。")

        # 清理状态
        self.model = None
        self.hook_manager = None
        self.original_generate = None
        logger.info("Guard 已成功分离。")

    def _guarded_generate(self, *args, **kwargs) -> Any:
        """
        这是修补后的 `generate` 方法，它会注入 SARC 防御逻辑。
        """
        if self.hook_manager is None or self.original_generate is None:
            raise RuntimeError("Guard 尚未附加到模型。请使用 `with guard.attach(model): ...`。")

        # 1. 确定批量大小
        # `input_ids` 通常是第一个位置参数或一个关键字参数
        input_ids = None
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
        elif len(args) > 0:
            input_ids = args[0]

        if input_ids is None:
            # 可能是 "inputs"
            if 'inputs' in kwargs:
                input_ids = kwargs['inputs']
            elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                input_ids = args[0]
            else:
                logger.warning("SARC: 无法在 generate 调用中确定 input_ids。假定批量为 1。")
                batch_size = 1
        else:
            batch_size = input_ids.shape[0]

        # 2. 清理上一轮的钩子状态
        self.hook_manager.clear_captured_activations()
        self.hook_manager.clear_intervention_state()

        # 3. 初始化 SARC LogitsProcessor
        sarc_processor = SarcLogitsProcessor(guard_instance=self, batch_size=batch_size)

        # 4. 将我们的处理器注入到 `generate` 调用中
        # 获取或创建 LogitsProcessorList
        processor_list = kwargs.get('logits_processor')
        if processor_list is None:
            processor_list = LogitsProcessorList()
        elif not isinstance(processor_list, LogitsProcessorList):
            processor_list = LogitsProcessorList(processor_list)

        processor_list.append(sarc_processor)
        kwargs['logits_processor'] = processor_list

        # 5. 调用原始的 `generate` 方法
        # 我们的 SarcLogitsProcessor 将在 `generate` 内部
        # 的每一步被自动调用
        return self.original_generate(*args, **kwargs)
