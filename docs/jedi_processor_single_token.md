# JEDI 处理器导致只生成一个 token 的可能机制

## 关键流程回顾
- `JEDILogitsProcessor` 在每个生成步骤读取隐藏态，打分并更新 CUSUM 统计量。如果 `A_t > alpha`，会把 `intervention_active` 置为真，并计算 `dynamic_betas`。
- 处理器随后通过 `HookManager.set_intervention_state` 将干预函数、需要干预的序列布尔掩码、以及按 `ratio^gamma * base_beta` 放大的动态强度传给钩子管理器。该状态会在下一次 forward 被读取。
- `HookManager._write_hook` 一旦被激活，就会把目标层输出的最后一个 token 隐藏态替换为 `ActAdd` 结果；替换操作直接修改张量然后返回给模型后续层。干预掩码在 `intervention_active` 为真时不会自动清空，因此触发后会一直应用。

## 为什么会“一步即停”
1. **初步触发后干预永久打开**：CUSUM 触发条件是 `A_t > alpha`。一旦 `intervention_active` 变为真，就不会在 `JEDILogitsProcessor` 内被重置；后续步骤的所有序列都会继续使用同一掩码。随着步数增加，`dynamic_betas` 还会按 `(A_t/alpha)^gamma` 继续放大。【F:src/JEDI_guard/guard.py†L137-L190】
2. **写钩子直接改写当前 token 的表示**：当干预开启时，`_write_hook` 会把目标层输出中对应批次的最后一个 token 隐藏态用 `invert_transform(transform(z)+beta*v)` 的结果覆盖。这一操作会影响当前步 logits 的主导方向，可能将最高分推向 `<eos>` 或安全 token，导致贪心解码立刻终止，从而只生成 1 个 token。【F:src/JEDI_guard/hook_manager.py†L95-L192】【F:src/JEDI_guard/interventions.py†L66-L133】
3. **动态 beta 的指数放大**：触发后的强度计算为 `base_beta * (A_t/alpha)^gamma`，默认 `gamma=5`。如果首步风险分数就高于阈值，`dynamic_betas` 会迅速增大，干预向量在第一轮就可能产生巨大偏移，使模型更倾向输出结束符或模板化拒绝，从而出现“任何输入都只出一个 token”现象。【F:src/JEDI_guard/guard.py†L164-L187】

## 调试建议
- 在触发前后打印 `A_t`、`dynamic_betas` 与当前 logits 的 top-k，确认是否首步就被放大到极高值。
- 检查 `mu_hat`/`kappa`/`alpha` 是否过低，或 `base_beta` 是否过高；必要时降低 `gamma` 或在触发后适度衰减 `intervention_active`。这样可以避免一次触发就让模型迅速输出 `<eos>`。
