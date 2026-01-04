# -*- coding: utf-8 -*-
"""
【核心】Guard 类

该文件定义了 Guard 类，它是 JEDI 防御系统的一站式入口，
封装了所有在线防御逻辑，实现了“方法流程.md”中的阶段 5 和 6。

[!] 修改：
- `Guard` 和 `SarcLogitsProcessor` 现已更新，
  支持基于 CUSUM 分数 `A_t` 的动态干预强度 `beta'`。
- `SarcLogitsProcessor` 现在处理 `A_t` 的计算和状态跟踪。
- `CusumState` 不再自动重置。

核心职责:
1.  通过 `from_artifacts` 类方法加载所有离线准备好的防御产物。
2.  通过 `attach(model)` 方法，使用 'with' 上下文管理器将自身附加到
    Hugging Face 模型上。
3.  在 `attach` 期间，通过修补 (patch) 模型的 `generate` 方法来拦截生成流程。
4.  注入一个自定义的 `LogitsProcessor` (SarcLogitsProcessor)，
    该处理器在每个生成步骤执行以下操作：
    a. 从 `HookManager` 获取当前 token 的隐藏状态 (由读钩子捕获)。
    b. 调用 `Scorer` 计算单步风险分数 r_t (阶段 4.1, 5.3)。
    c. [!] 将 r_t 送入 `CusumState`，取回当前的累积分数 A_t (阶段 5.4)。
    d. [!] 如果 `A_t > h`，则激活干预，并计算动态 `beta'` (阶段 5.5)。
5.  通过 `HookManager` 动态注册一个“写钩子”，
    该钩子执行 `interventions.py` 中定义的 ActAdd 注入 (阶段 6.1)。
6.  在 `with` 块结束时，自动 `detach`，清理所有钩子并恢复
    模型原始的 `generate` 方法。
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
    JEDI 防御的核心逻辑处理器。
    在 `generate` 循环的每个 token 生成步骤中被调用。

    [!] 修改：实现动态 beta 计算和干预状态管理。
    """

    def __init__(self, guard_instance, batch_size: int, trigger_logs: List[int]):
        """
        初始化 JEDI LogitsProcessor。

        Args:
            guard_instance (Guard):
                对主 Guard 实例的引用，用于访问 scorer, cusum, 和 hook_manager。
            batch_size (int):
                当前生成请求的批量大小。
            trigger_logs (List[int]):
                一个长度为 batch_size 的列表 (由 Guard 实例持有)，
                用 -1 初始化。此处理器将在此列表中记录*首次*触发的 token 索引。
        """
        self.guard = guard_instance
        self.batch_size = batch_size
        self.device = self.guard.device
        self.is_batch = batch_size > 1

        # --- 日志记录 ---
        self.trigger_logs = trigger_logs  # 这是一个共享列表的引用
        self.current_step = 0  # 跟踪当前生成的 token 索引 (从 0 开始)

        # 为这个特定的生成调用初始化一个 CUSUM 状态机
        self.cusum = CusumState(
            mu_hat=self.guard.mu_hat,
            kappa=self.guard.kappa,
            h=self.guard.h,
            batch_size=batch_size,
            device=self.device
        )

        # --- [!] 新增：动态 Beta 和状态管理 ---
        # 跟踪哪些序列已经触发了干预
        self.intervention_active = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        # 从 Guard 获取 CUSUM 阈值
        self.h = self.guard.h
        # 从 Guard 获取基础干预强度
        self.base_beta = self.guard.base_beta
        # 初始化一个张量来存储每个序列的 *当前* 干预强度
        self.dynamic_betas = torch.full(
            (batch_size,), self.base_beta, device=self.device, dtype=torch.float32
        )
        # --- 结束新增 ---

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
        try:
            hidden_state = self.guard.hook_manager.get_last_captured_activation()
            if hidden_state is None:
                logger.warning("JEDI: 未能从 HookManager 获取隐藏状态。跳过本轮检测。")
                self.current_step += 1  # [!] 确保步骤计数器增加
                return scores

            if hidden_state.dim() == 2:
                hidden_state = hidden_state.unsqueeze(1)  # (B, D) -> (B, 1, D)

            if hidden_state.shape[0] != scores.shape[0]:
                logger.error(f"JEDI: 隐藏状态批量大小 ({hidden_state.shape[0]}) 与 "
                             f"Logits 批量大小 ({scores.shape[0]}) 不匹配。")
                self.current_step += 1  # [!] 确保步骤计数器增加
                return scores

        except Exception as e:
            logger.error(f"JEDI: 获取隐藏状态时出错: {e}", exc_info=True)
            self.current_step += 1  # [!] 确保步骤计数器增加
            return scores

        # 2. 计算风险分数 (阶段 5.3)
        s_t, r_t = self.guard.scorer.calculate_scores(hidden_state)

        # 3. [!] 更新 CUSUM 状态机并获取 A_t (阶段 5.4)
        # A_t 是 (B,) 张量，在 CPU 上
        A_t = self.cusum.update(r_t)
        A_t_device = A_t.to(self.device)

        # 4. [!] 检查触发器并更新干预状态

        # 4a. 确定哪些序列 *当前* 应该被触发
        currently_triggered = A_t_device > self.h

        # 4b. 确定哪些是 *新* 触发的
        newly_triggered = currently_triggered & (~self.intervention_active)

        if torch.any(newly_triggered):
            # 4c. 将新触发的序列标记为 "永久" 激活
            self.intervention_active |= newly_triggered

            # 4d. 记录 *首次* 触发的步骤
            for i in newly_triggered.nonzero(as_tuple=True)[0]:
                idx = i.item()
                if self.trigger_logs[idx] == -1:  # 再次检查，确保只记录一次
                    self.trigger_logs[idx] = self.current_step

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"JEDI: CUSUM 在第 {self.current_step} 步 *首次* 触发。激活以下序列: "
                             f"{newly_triggered.nonzero(as_tuple=True)[0].tolist()}")

        # 5. [!] 如果 *任何* 序列（新的或旧的）处于激活状态，则计算 beta' 并设置钩子
        if torch.any(self.intervention_active):
            active_indices = self.intervention_active

            # 5a. 计算动态 Betas
            # (A_t / h)  clamped at 1.0 然后取 gamma 次方
            ratios = (A_t_device[active_indices] / self.h).clamp(min=1.0)
            gamma = 1.2  # 或 1.5
            ratios = ratios.pow(gamma)
            self.dynamic_betas[active_indices] = self.base_beta * ratios

            if logger.isEnabledFor(logging.DEBUG):
                if torch.any(newly_triggered):  # 只在首次触发时记录
                    active_idxs_list = active_indices.nonzero(as_tuple=True)[0].tolist()
                    betas_list = self.dynamic_betas[active_indices].tolist()
                    logger.debug(f"  > Betas: { {idx: beta for idx, beta in zip(active_idxs_list, betas_list)} }")

            # 5b. 告诉 HookManager 在 *下一次* forward 传递时
            #     应用干预钩子
            self.guard.hook_manager.set_intervention_state(
                self.guard.intervention_func,  # 干预函数
                self.intervention_active,  # (B,) bool, 哪些序列要干预
                self.dynamic_betas  # (B,) float, 所有序列的 beta 值
            )

        # 6. 递增生成步骤计数器
        self.current_step += 1

        return scores


class Guard:
    """
    JEDI 防御系统的主类。
    通过上下文管理器 (`with guard.attach(model): ...`) 来使用。
    """

    def __init__(
            self,
            layer_id: int,
            theta: float,
            mu_hat: float,
            kappa: float,
            h: float,
            base_beta: float,  # [!] 新增：基础 beta
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

        # [!] 干预参数
        self.base_beta = base_beta

        # 核心组件
        self.scorer = scorer
        self.intervention_func = intervention_func

        # 运行时状态（在 `attach` 时设置）
        self.model: Optional[Module] = None
        self.hook_manager: Optional[HookManager] = None
        self.original_generate: Optional[Callable] = None

        # --- 新增：用于在 Guard 和 SarcLogitsProcessor 之间传递日志列表 ---
        self.current_batch_trigger_logs: Optional[List[int]] = None
        # --- 结束新增 ---

        logger.info(f"Guard 实例已初始化。将在第 {layer_id} 层运行。")
        logger.info(f"防御参数: theta={theta:.4f}, mu_hat={mu_hat:.4f}, kappa={kappa:.4f}, h={h:.4f}, base_beta={base_beta:.2f}")

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
        transform_cont = artifacts['transforms']['content_window'][layer_id]
        condition_vector = artifacts['condition_vectors'][layer_id]
        scorer = Scorer(
            condition_vector=condition_vector,
            transform=transform_cont,  # [!] 传入 content_window 变换
            theta=theta,
            device=device
        )

        # 4. 准备 Intervention (阶段 6.1)
        transform_early = artifacts['transforms']['early_window'][layer_id]
        intervention_vector = artifacts['intervention_vectors'][layer_id]

        # [!] 从 defense_params.yaml 加载 *基础* beta（兼容旧键 alpha）
        base_beta = params.get('beta', params.get('alpha', 2.0))  # 尝试键 'beta'，否则回退
        if 'intervention_beta' in params:  # 备用键
            base_beta = params.get('intervention_beta', base_beta)

        logger.info(f"使用基础干预强度 (base_beta): {base_beta}")

        # [!] 创建干预函数时不再传入 beta
        intervention_func = create_intervention_hook_func(
            vector=intervention_vector,
            # beta=base_beta, <-- [!] 移除
            transform=transform_early,  # 传入 early_window 变换
            device=device
        )

        # 5. 创建并返回 Guard 实例
        return cls(
            layer_id=layer_id,
            theta=theta,
            mu_hat=mu_hat,
            kappa=kappa,
            h=h,
            base_beta=base_beta,  # [!] 传入基础 beta
            scorer=scorer,
            intervention_func=intervention_func,
            device=device
        )

    # --- 新增：用于设置和清理日志目标的方法 ---
    def set_batch_log_target(self, log_list: List[int]):
        """
        在 `_guarded_generate` 之前，从外部 (run_evaluation.py) 设置一个列表
        用于 SarcLogitsProcessor 记录触发步骤。
        """
        self.current_batch_trigger_logs = log_list

    def clear_batch_log_target(self):
        """
        在 `_guarded_generate` 之后，清理日志目标引用。
        """
        self.current_batch_trigger_logs = None

    # --- 结束新增 ---

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
        self.clear_batch_log_target()  # 确保也清理了日志目标
        logger.info("Guard 已成功分离。")

    def _guarded_generate(self, *args, **kwargs) -> Any:
        """
        这是修补后的 `generate` 方法，它会注入 JEDI 防御逻辑。
        """
        if self.hook_manager is None or self.original_generate is None:
            raise RuntimeError("Guard 尚未附加到模型。请使用 `with guard.attach(model): ...`。")

        # 1. 确定批量大小
        input_ids = None
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
        elif len(args) > 0:
            input_ids = args[0]

        if input_ids is None:
            if 'inputs' in kwargs:
                input_ids = kwargs['inputs']
            elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                input_ids = args[0]
            else:
                logger.warning("JEDI: 无法在 generate 调用中确定 input_ids。假定批量为 1。")
                batch_size = 1
        else:
            batch_size = input_ids.shape[0]

        # 2. 清理上一轮的钩子状态
        self.hook_manager.clear_captured_activations()
        self.hook_manager.clear_intervention_state()

        # 3. 初始化 JEDI LogitsProcessor，并传入日志列表
        # --- 修改：从 Guard 实例获取日志列表 ---
        trigger_logs = self.current_batch_trigger_logs
        if trigger_logs is None:
            logger.warning("JEDI: Guarded generate 被调用，但没有设置批量日志目标 "
                           "(set_batch_log_target)。触发步骤将不会被记录。")
            # 创建一个临时的 dummy 列表以防止崩溃
            trigger_logs = [-1] * batch_size
        elif len(trigger_logs) != batch_size:
            logger.error(f"JEDI: 提供的日志列表长度 ({len(trigger_logs)}) 与 "
                         f"批量大小 ({batch_size}) 不匹配。")
            # 同样使用 dummy 列表
            trigger_logs = [-1] * batch_size

        sarc_processor = SarcLogitsProcessor(
            guard_instance=self,
            batch_size=batch_size,
            trigger_logs=trigger_logs  # 传入共享列表
        )
        # --- 结束修改 ---

        # 4. 将我们的处理器注入到 `generate` 调用中
        processor_list = kwargs.get('logits_processor')
        if processor_list is None:
            processor_list = LogitsProcessorList()
        elif not isinstance(processor_list, LogitsProcessorList):
            processor_list = LogitsProcessorList([processor_list])  # 确保是列表

        processor_list.append(sarc_processor)
        kwargs['logits_processor'] = processor_list

        # 5. 调用原始的 `generate` 方法
        return self.original_generate(*args, **kwargs)