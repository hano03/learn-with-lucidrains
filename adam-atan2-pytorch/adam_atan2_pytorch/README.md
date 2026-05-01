# adam_atan2_pytorch 代码对比分析

本文基于目录 `adam_atan2_pytorch/` 下全部源码（不含 `__pycache__`）做对比，重点回答：

1. 基线 `AdamAtan2`（`adam_atan2.py`）做了什么。
2. 其他“变体”相对基线改了什么、适用场景是什么。

## 1. 基线：`adam_atan2.py` (`AdamAtan2`)

基线实现可理解为“Adam 的分母除法改成 `atan2` 角度形式”：

- 一阶/二阶动量：
  - `exp_avg <- beta1 * exp_avg + (1 - beta1) * grad`
  - `exp_avg_sq <- beta2 * exp_avg_sq + (1 - beta2) * grad^2`
- 偏置修正后更新方向：
  - `update = atan2(exp_avg / bias_correction1, b * sqrt(exp_avg_sq / bias_correction2))`
- 参数更新：
  - `p <- p - lr * a * update`

和标准 Adam 的核心差异：

- 不再使用 `grad / (sqrt(v) + eps)`，而是 `atan2(num, den)`，避免手工 `eps`。
- 更新值被限制在角度范围（`atan2` 有界），在极值/尺度变化时更稳定。

基线还集成了 4 个附加机制：

- `regen_reg_rate`：再生正则（把参数拉回初始化点）。
- `decoupled_wd`：解耦权重衰减（在 `__init__` 中按初始学习率重标定）。
- `cautious_factor`：与梯度方向不一致的更新降权（Cautious Optimizer 思路）。
- `cautious_wd`：权重衰减也可按“方向一致性”掩码执行。

## 2. 各变体与基线的差异总览

| 文件 | 核心更新形式 | 关键状态 | 与基线最重要差异 |
|---|---|---|---|
| `adopt.py` (`Adopt`) | `grad / max(sqrt(v), eps)` + 裁剪 | `m`, `v`, `steps` | 不是 atan2；首步不更新；有 `eps` 与 `t^0.25` 裁剪 |
| `adopt_atan2.py` (`AdoptAtan2`) | `atan2(grad, b*sqrt(v))` 再进 `m` | `m`, `v`, `steps` | ADOPT 主体 + atan2；首步不更新；支持 cautious 和 regen |
| `adam_atan2_with_orthog_grad.py` | 同基线 atan2 | `exp_avg`, `exp_avg_sq`, `grad_emas` | 可对当前梯度做“对历史梯度正交投影” |
| `adam_atan2_with_wasserstein_reg.py` | 同基线 atan2 | `exp_avg`, `exp_avg_sq`, `param_init(sorted)` | 再生正则目标换成“初始化参数的排序统计匹配” |
| `foreach.py` | 同基线 atan2（foreach 批量算子） | `exp_avg`, `exp_avg_sq` | 用 `_foreach_*` 做批量张量更新，偏向吞吐性能 |
| `muon_adam_atan2.py` (`MuonAdamAtan2`) | 普通参数用 atan2；指定参数用 Muon/Newton-Schulz | 普通组: `exp_avg/exp_avg_sq`; Muon组: `exp_avg` | 一个优化器里混合两套更新规则+两组学习率 |
| `polar_adam_atan2.py` (`PolarAdamAtan2`) | 普通参数用 atan2；指定参数用 Polar Express | 普通组: `exp_avg/exp_avg_sq`; Polar组: `exp_avg` | 与 Muon 类似，但矩阵符号迭代替换为 Polar Express |

## 3. 逐个变体详细说明

### 3.1 `adopt.py`（纯 ADOPT）

这不是 `adam_atan2` 变体，而是并列基线（ADOPT 原版思路）：

- 分母仍是 `sqrt(v)`，需要 `eps`（`clamp(min=eps)`）。
- 更新先做 `clip_value = steps**0.25` 的裁剪，再进入一阶动量 `m`。
- `steps == 0` 的第一步只累计统计量，不更新参数。
- `decoupled_wd=True` 默认开启（与 `adam_atan2.py` 默认值不同）。

结论：`adopt.py` 更像对照组，用来对比“是否引入 atan2”。

### 3.2 `adopt_atan2.py`（ADOPT + atan2）

相对 `adopt.py` 的变化：

- 把 `grad / sqrt(v)` 改为 `atan2(grad, b * sqrt(v))`。
- 保留 ADOPT 的“首步不更新”机制。
- 增加 `regen_reg_rate`、`cautious_factor`、`a/b` 两个缩放参数。

相对 `adam_atan2.py` 的变化：

- 一阶动量 `m` 不是直接积累原始梯度，而是积累 `atan2` 后的更新量。
- 没有偏置修正项 `bias_correction1/2` 的显式参与更新公式（路径不同）。

### 3.3 `adam_atan2_with_orthog_grad.py`（正交梯度版）

这是最“任务场景化”的扩展，面向时序/多损失梯度冲突：

- 在进入 `exp_avg/exp_avg_sq` 前，先把当前梯度投影到某个历史梯度 EMA 的正交子空间：
  - `new_grad = orthog_proj(grad, grad_emas[key])`
- 通过 `store_grad_key` 和 `orthog_against_key` 控制“存谁”“对谁正交”。
- 新增 `reset_(key=None)`，可重置全部或指定 key 的梯度 EMA。
- 提供 `orthog_proj_double_precision`，可用双精度做投影避免数值误差。

其余 Adam-atan2 主干、cautious、cautious_wd 与基线一致。

### 3.4 `adam_atan2_with_wasserstein_reg.py`（Wasserstein 再生正则）

核心差异只在再生正则目标：

- 普通再生正则：参数拉回“逐元素的初始值”。
- 此版本：参数拉回“按排序统计量对齐的初始分布”。
  - 初始化时保存排序后的 `param_init`。
  - 每步按当前参数的排序索引重排 `param_init` 作为目标，再 `lerp` 拉回。

直观上它更关心权重分布形状，而不是每个坐标一一对应的回拉。

### 3.5 `foreach.py`（批量内核版）

目标是减少 Python 循环开销，尽量用 `torch._foreach_*`：

- 把同组参数的 `mul/lerp/sqrt/add/atan2` 变成批量原地更新。
- 数学形式与基线基本一致（仍是 atan2 版本）。
- 提供 `foreach_atan2_fn` 注入；若环境没有 `_foreach_atan2_`，回退到逐张量循环。

工程注意点：

- 该实现按“参数组”统一使用最后一次循环得到的 bias correction 标量，默认假设同组 `steps` 同步；若同组参数步数不一致，语义会偏离逐参数实现。

### 3.6 `muon_adam_atan2.py`（混合 Muon 版）

一个优化器中包含两类参数组：

- 普通参数组：用 Adam-atan2（基线逻辑）。
- `muon_params` 参数组：只保留一阶动量 `exp_avg`，再走 `newtonschulz5` 近似矩阵符号更新（Muon 思路），并做 RMS 缩放。

关键点：

- 可单独设置 `muon_lr`、`muon_beta1`、`muon_steps`、`muon_rms_factor`。
- `remove_muon_params_from_params=True` 默认会把 Muon 参数从普通组移除，避免重复更新。
- 对 Muon 组会将 `a` 强制置为 `1.`，普通组仍用传入的 `a`。

### 3.7 `polar_adam_atan2.py`（混合 Polar Express 版）

结构与 `muon_adam_atan2.py` 高度类似，区别在“矩阵符号近似器”：

- Muon 版使用 `newtonschulz5`。
- Polar 版使用 `polar_express` 多项式迭代（可选 `bfloat16` 计算）。

同样是两组参数混合优化，同样保留普通参数组的 Adam-atan2 更新。

## 4. 额外代码层差异（易忽略但重要）

1. 导出接口：`__init__.py` 里 `Adopt = AdoptAtan2`，默认对外暴露的是 atan2 版 ADOPT，不是 `adopt.py` 的纯 ADOPT。
2. 同名类：`adam_atan2.py`、`adam_atan2_with_orthog_grad.py`、`adam_atan2_with_wasserstein_reg.py` 都定义了类名 `AdamAtan2`，必须按模块路径区分导入。
3. 默认超参差异较大：例如 `decoupled_wd` 在不同变体默认值不一致，迁移时不能直接复用旧配置。

## 5. 怎么选（实践建议）

- 想最小改动替代 Adam：先用 `adam_atan2.py`。
- 你本来就在用 ADOPT：优先比较 `adopt.py` vs `adopt_atan2.py`。
- 有多任务/时序梯度冲突：用 `adam_atan2_with_orthog_grad.py`。
- 更关注长期塑性与分布保持：试 `adam_atan2_with_wasserstein_reg.py`（配合 `regen_reg_rate`）。
- 大模型且想把 2D/3D 权重做专门更新：用 `muon_adam_atan2.py` 或 `polar_adam_atan2.py`，并明确参数分组策略。
- 训练吞吐瓶颈在优化器开销：试 `foreach.py`（先确认语义假设满足你的参数步数同步情况）。

