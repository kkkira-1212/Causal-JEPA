# Causal-JEPA 问题追踪与架构说明

更新时间：2026-04-29

---

## 🏗️ 当前架构（I-JEPA 风格，2026-04-29 教授会议确认）

### 核心变化总览

| 模块 | 旧设计 | 新设计 |
|---|---|---|
| TTM backbone | r3（56 patches, d_model=128, 10层, 100/162权重）| **r2（9 patches, d_model=192, 2层, 103/103权重）** |
| 输出表示 | pool → (B, D, 64) 全局向量 | **patch-level (B, D, 9, 128)，保留时序结构** |
| proj 层 | Linear(128→64) | **Linear(192→128)，hidden_dim=128** |
| mask 策略 | context 随机 mask 30% patch（置零） | **context/target 独立随机各选 M=3 个 patch** |
| predictor | 固定 MLP，输入 pooled z | **cross-attention + MLP，输入 patch 序列 + position query** |
| L_JEPA 来源 | w1·ẑ_self + w2·ẑ_pa vs z_target | **只用 P_self 输出（实验版本，见注）** |

### 前向流程

```
x_context (B, 512, D)
    ↓ 随机选 M_c=3 个 patch 位置 → 置零
x_ctx_masked → TTM r2 backbone → last_hidden_state: (B, D, 9, 192)
    ↓ Linear(192→128)，作用于最后一维
s_ctx: (B, D, 9, 128)

x_target (B, 512, D)  ← 不 mask
    → target encoder (EMA 副本，no grad) → Linear(192→128)
s_target: (B, D, 9, 128)

── CausalLayer ──────────────────────────────────────────────
s_ctx.mean(dim=2) → (B, D, 128)   ← pool 仅用于图学习
A = σ((Θ + g·noise_scale) / τ): (D, D)

── 独立随机选 M_t=3 个 target patch 位置 {j1, j2, j3} ────────
（与 context mask 独立采样，更接近 I-JEPA 原版）

── 对每个 target patch j ────────────────────────────────────
query_j = mask_token + pos_embed[j]   # learnable (128,)

Parent 聚合（patch-level）：
  s_pa[b,d,:,:]  = Σ_i A[d,i] * s_ctx[b,i,:,:] / (Σ A[d,:] + ε)
Non-parent 聚合：
  s_non[b,d,:,:] = Σ_i (1-A[d,i]) * s_ctx[b,i,:,:] / (Σ(1-A[d,:]) + ε)

P_self:  cross-attn(Q=query_j, KV=s_ctx) + MLP → ŝ_self_j  (B, D, 128)
P_pa:    cross-attn(Q=query_j, KV=s_pa)  + MLP → ŝ_pa_j    (B, D, 128)
P_non:   cross-attn(Q=query_j, KV=s_non) + MLP → ŝ_non_j   (B, D, 128)

── Loss ─────────────────────────────────────────────────────
s_target_j = s_target[:, :, j, :]    # (B, D, 128)

L_JEPA = mean over {j} MSE(ŝ_self_j, s_target_j)
L_ADV  = mean over {j} log(MSE(ŝ_pa_j, s_target_j) / MSE(ŝ_non_j, s_target_j))
L_DAG  = Tr(exp(A⊙A)) − D
```

> ⚠️ **L_JEPA 设计说明（非最终版本）**
>
> 教授原始建议：combined prediction `ŝ = w1·ŝ_self + w2·ŝ_pa` 进 L_JEPA，self 和 parent predictor 同时被 JEPA loss 训练。
> 当前版本（L_JEPA 只用 P_self）是**实验性职责分离**，观察 Θ 从 L_JEPA 解耦后 L_ADV 信号是否更干净。
> 若效果不如预期，回退到 combined prediction 方案。

### 为什么 context 和 target patch 独立采样

I-JEPA 原版关键设计：predictor 看到的 context 区域和要预测的 target 区域不同，必须真正"推断"而非从对应位置复制。若 context mask = target predict，predictor 可能只学会"忽略零位置"，任务退化为去噪。

### 参数更新汇总表（新架构）

```
              φ(encoder)   Θ(A矩阵)   P_self   P_pa    P_non   mask_token  pos_embed
L_JEPA         ✅ 主要        ✗        ✅ 主要    ✗        ✗       ✅           ✅
L_ADV          ✅ 间接      ✅ 主要      ✗      ✅ 直接  ✅（唯一）  ✗           ✗
L_DAG            ✗          ✅ 主要      ✗        ✗        ✗        ✗           ✗
```

---

## 🏗️ 旧架构（已废弃，保留供参考）

```
x_context → TTM r3 骨干（freeze） → last_hidden_state(B, D, 56, 128)
    → mean/last pool → (B, D, 128) → Linear(128→64) → z_ctx (B, D, 64)
x_target  → target encoder (EMA)                   → z_target (B, D, 64)

z_ctx → CausalLayer → A (D×D)
z_ctx → P_self(z_ctx) → ẑ_self
z_ctx → P_pa(z_pa)   → ẑ_pa      z_pa = Σ A_ij * z_i（变量维度聚合）
z_ctx → P_non(z_non) → ẑ_non

ẑ = 0.5·ẑ_self + 0.5·ẑ_pa  →  L_JEPA = MSE(ẑ, z_target)
L_ADV = log(loss_pa / loss_non)
```

**废弃原因：**
- r3 只加载 100/162 权重，patch_mixer 随机初始化，时间 mixing 能力打折
- mean pool 把 56 个 patch 压成无时序信息的全局摘要，因果层拿到的表示没有时序方向
- frozen backbone 是性能天花板（probe MSE 锁死在 0.965~0.967，改 pooling/masking 无用）
- JEPA 任务过于简单：相邻 512 步窗口全局统计特性高度相似，l_jepa→0.0004

---

## 🔬 TTM r2 Backbone 内部架构

### 变量维度 D 与 embedding 维度 d_model 的区别

```
D=5       = 有几条时间序列（变量数），CausalLayer 在此维度上学因果
d_model=192 = 每个 64步 patch 被压缩成多少维向量，与变量数无关
```

TTM 把 D 个变量拆开，每个变量独立走同一套 backbone 权重（共享参数，不共享信息）：

```
输入 (B, 512, D=5)
    ↓ 拆变量，拼进 batch 维
backbone 看到: (B×5, 8 patches, 64步/patch)  ← r2 patch_length=64，512/64=8（最后一段）
    ↓ 处理
输出: (B, D=5, 9, 192)  ← r2 实际 num_patches=9（含补位）
```

### r2 Backbone 结构（2 层，103/103 权重完整加载）

```
TinyTimeMixerStdScaler           每样本除以自身 std
    ↓
Patchify                         512步 → 9 patches（64步/patch）
    ↓
Linear(64 → 192)                 每个 patch 嵌入为 192 维
    ↓
2 × TinyTimeMixerLayer
    ├── PatchMixerBlock（跨时间）：9 patches 间混合
    └── FeatureMixerBlock（跨特征）：192 维内部混合
    ↓
last_hidden_state: (B, D, 9, 192)
```

注：TTM 没有位置编码（`use_positional_encoding: False`）。r2 需要 `freq_token` 参数，通过 `full_model.config.use_fft_embedding` 检测（必须在 `del full_model` 之前读取，否则此 attribute 不在 `backbone.config` 上）。

---

## ⚙️ Loss 函数梯度分析

### L_JEPA（新：只用 P_self）

```
L_JEPA → ŝ_self_j → P_self 权重 ✅ → mask_token / pos_embed ✅ → s_ctx → φ ✅
       ✗ s_target 无梯度（EMA detach）
       ✗ Θ 与 L_JEPA 完全解耦（新架构职责分离）
```

### L_ADV = mean log(MSE(ŝ_pa) / MSE(ŝ_non))

```
L_ADV → ŝ_pa  → P_pa 权重 ✅ → s_pa → A_ij ✅（因果学习主信号）→ φ ✅
      → ŝ_non → P_non 权重 ✅（P_non 唯一信号）→ s_non → A_ij ✅

∂L_ADV/∂A_ij 方向：若 i 对 j 有真实预测力
  → 增大 A_ij → s_pa[j] 更多来自 i → MSE(ŝ_pa)↓，MSE(ŝ_non)↑ → L_ADV 更负 ✅
```

### L_DAG = Tr(exp(A⊙A)) − D

仅流向 Θ，不影响 φ 或 predictor。有环路的位置（i→j 且 j→i）梯度最大。

### L_ADV 设计演进历史

**尝试 1：sgn + threshold（v1，失败）**
- `sgn` 不可微；P_pa 从 warmup 天然占优，threshold 全程满足，L_ADV=0

**尝试 2：ReLU hinge（失败）**
- `ReLU(log(loss_pa) - log(loss_non) + log(1+m))`
- warmup 后 log_ratio 已为负数，ReLU 全程输出 0

**尝试 3：log-ratio（当前，有梯度但 margin 是摆设）**
- `log(loss_pa + ε) - log(loss_non + ε) + math.log(1 + margin)`
- `math.log(1+margin)` 是纯常数，梯度为 0，训练行为与无 margin 完全相同
- **待教授确认**：margin 的正确语义（绝对倍数 vs 动态相对提升）

---

## 📊 实验记录

### v1：初始版本（ReLU threshold，uniform 初始化）
- L_ADV 全程为 0，A 由 L_sparse 随机压稀疏，无因果学习信号

### v2：Normal(0,1) 初始化 + margin=0.3
- Debug 验证：ratio=0.91，L_ADV=0.388，A std=0.28
- 完整训练仍全程 L_ADV=0，根因：P_pa 从初始化就赢 30%+，threshold 从未触碰

### v3：log-ratio + Gumbel-Sigmoid + λ_adv=5.0
- L_ADV 从 epoch 6 起持续非零（-8.6 → -13.4）✅
- λ_adv=5.0 → ADV 贡献 -65，统治 total loss；L_JEPA 从 0.0007 上升到 0.017 ❌
- L_DAG 较好控制（2.0~3.7）✅

### v4：λ_adv=0.1
- L_ADV 贡献 -1.3，与 L_DAG 量级相当 ✅
- L_JEPA 不再退步，基本稳定 ✅

### v7：I-JEPA 重构 + Patch-level Routing（2026-04-29）

**Pure JEPA baseline（step 11）：** l_jepa 0.49→0.019，probe val MSE=0.8518（↑12% vs 旧架构 0.9646）✅

**Full Causal-JEPA（step 12）：**
- probe val MSE=0.8520（与 baseline 相当）
- **图恢复完全崩溃：F1=0 全程**（vs Granger 0.541 / 旧架构 0.571）❌
- 根因：A→0 退化不动点（详见下方崩溃分析）

**崩溃分析（step 12）**

| Epoch | A mean | A sparsity | L_ADV | TPR | F1 |
|---|---|---|---|---|---|
| 5 (warmup end) | — | — | 0 | — | — |
| 6 (first full, λ_adv=0) | 0.085 | 0.68 | +0.201 | 0.250 | 0.400 |
| 7 (λ_adv=0.01 kicks in) | 0.026 | 0.96 | -8.640 | 0.000 | 0.000 |
| 8-20 | ~0.02-0.11 | 0.84-1.00 | -13~-19 | 0.000 | 0.000 |

**根因：** `_route` 里当 A→0，`s_pa = einsum(s_ctx,A) / (sum(A)+eps)` 退化为 0/eps ≈ 0（KV 全零向量）。P_pa 的 cross-attention attn_out=0，输出退化成 `queries_expanded + mlp(queries_expanded)`——纯位置先验预测器，与 A 完全解耦。P_pa 在退化模式下 MSE 很低 → L_ADV 极负 → ∂MSE_pa/∂A ≈ 0 → A 无梯度推力 → 卡在退化不动点。

**修复（step 13）：** s_pa 加单位自环（identity self-loop）：
```python
A_pa = A + torch.eye(D)          # 每个变量 j 始终从自身接收，权重=1
s_pa = einsum(s_ctx, A_pa) / A_pa.sum(dim=0)   # 分母 ≥ 1.0，KV 永远非零
```
当 A=0：s_pa[j]=s_ctx[j]（自身）；当 A_ij↑（真实父节点）：s_pa[j] 获得更多 i 的信息 → MSE_pa↓ → 梯度信号恢复 ✅

---

### v6：Pooling / Masking 消融实验（2026-04-23，Pure JEPA baseline 模式）

**背景**：发现 mean pool 将 56 个 patch 压成无时序信息的全局摘要，JEPA 任务 trivially easy（l_jepa≈0.0004）。

| 实验 | pool_mode | mask_ratio | l_jepa ep1 | l_jepa ep20 | probe val MSE |
|---|---|---|---|---|---|
| 原始 baseline | mean | 0.3 | 0.0291 | 0.0004 | 0.9646 |
| Causal-JEPA（对照）| mean | 0.3 | — | — | 0.9650 |
| Exp1 | last | 0.0 | 0.2147 | 0.0135 | 0.9671 |
| Exp2 | last | 0.3 | 0.2240 | 0.0066 | 0.9671 |

**Causal-JEPA 图恢复结果（mean pool + mask 0.3）：**
- Epoch 20：TPR=0.500，FPR=0.048，F1=0.571（Granger baseline F1=0.541）✅

**关键结论：**
1. last patch 让任务变难 34×（l_jepa 0.0004→0.0135），但 masking 在 last patch 上无额外收益
2. probe MSE 锁死在 0.965~0.967：瓶颈在 frozen backbone，而非 pooling/masking 策略
3. 参照系：naive baseline（猜均值）MSE≈1.0；三个实验均只比猜均值好 3%

---

## 📦 数据集：CAUKER 风格 SCM 合成数据

### 设计决策

**为什么用多条独立序列而非单条长序列：**
原设计单条 T=10000 序列，相邻 window 重叠 511/512 步，train/val split 无意义。
新设计固定 DAG，用不同 gp_seed 生成 N 条独立序列，按序列划分 train/val/test（标准 Monte Carlo evaluation）。

### 与 CAUKER 的对齐与偏差

| 项目 | CAUKER 原版 | 我们 | 说明 |
|---|---|---|---|
| DAG | 每 sample 不同 | **固定**（dag_seed） | 模型学单一 Theta，必须固定 |
| 因果时序 | 同时刻 X_i(t)→X_j(t) | **lag-1** X_i(t-1)→X_j(t) | 与 Granger 因果对齐，有意偏差 |
| Kernel bank | 33 个，T-scaling | 直接复用 CAUKER 代码 | ✓ 对齐 |
| 归一化 | 无 | per-series per-variable z-score | TTM 输入需要数值稳定 |

### 当前数据文件

```
data/scm_d5_n30_t1500_dag42.npz   # 正式数据集（30 条序列）
data/scm_d5_n6_t1500_dag42.npz    # 小批量验证集（6 条序列）

参数：D=5, N=30, T=1500, max_parents=2, noise_std=0.1, dag_seed=42
DAG：4 edges / 20 possible (density=20%)
Split：train=24 / val=3 / test=3（按序列划分）
Windows per series：477
```

### 数据质量验证（N=30 正式数据集）

```
无 NaN / Inf：✓    per-series per-var std=1.0：✓    Durbin-Watson 均值：0.782
Granger F1 mean=0.541，std=0.132（30 条序列）
极端值：|x|>5 占 0.01%（anomaly_mean spike 孤立点，不影响训练）
```

**当前 baseline：Granger mean F1 = 0.541**（模型须超过此值才算优于线性方法）

---

## 🔧 Granger 初始化

`A = sigmoid(Θ/τ)` → 反推 `Θ = τ × logit(A_init)`

**形式 1（软初始化，当前使用）：**
```python
# high_conf=0.65 → A≈0.66，low_conf=0.35 → A≈0.41（软边界，避免 L_DAG 爆炸）
granger_init_theta(data[:2000], num_vars, maxlag=1, alpha=0.05, tau, high_conf=0.65, low_conf=0.35)
```

**形式 2（p 值连续映射）：**
```python
# p 越小 → logit(1-p) 越大 → Θ 越高，保留强度信息
theta_init[i,j] = tau * logit(clip(1-p, 0.05, 0.95))
```

**与 warmup 的关系：** Granger 初始化给 Θ 有意义的起点（训练前一次），warmup 让 P_self/P_pa 热身使训练量对齐，两者不互替。

---

## 🔵 设计决策记录

### L_JEPA 是核心 Loss

其他三个 loss（L_ADV、L_DAG、L_sparse）本质上都是对 A 矩阵的约束，是辅助手段。
教授建议（2026-03-27）：考虑大权重 `lambda_jepa >> 1.0`，当前设为 1000.0（config 已更新）。

### 静态 Θ vs Graph_Gen(z_ctx)

- **静态 Θ（当前）**：全局一张图，适合平稳系统（SCM、ETTh1）
- **Graph_Gen(z_ctx)**：样本条件化，每 batch 一张图，适合非平稳/多系统场景（future work）

### 双层优化（Bilevel Optimization，future work）

Θ 被 L_JEPA / L_ADV / L_DAG 四个方向同时推，无主次感。

```
外层（主导）：L_JEPA 先给 Θ 定方向 → 知道哪些边对预测有用
内层（约束）：固定 φ/predictors，枚举 λ 组合 → G1, G2, G3（结构合法候选图）
反馈：选让 L_JEPA 最小的 G 更新 Θ
```

待教授确认：G1/G2/G3 是否来自不同 λ 组合；反馈是 argmin 选择还是梯度加权。

---

## 📋 当前行动计划（2026-04-29）

**数据集：** `data/scm_d5_n30_t1500_dag42.npz`
**backbone：** TTM r2（512-96-ft-r2.1，9 patches，d_model=192，103/103权重完整加载）
**目标：** F1 > 0.541（超越 Granger baseline）

### 架构改造 To-Do（按顺序执行）

| 步骤 | 文件 | 任务 | 预计时长 | 状态 |
|---|---|---|---|---|
| 1 | `encoder.py` | freq_token 检测修复（在 del full_model 前读 use_fft_embedding） | — | ✅ 完成 |
| 2 | `encoder.py` | 去掉 pool_mode，输出 patch-level (B,D,9,192)；proj 改 Linear(192→128) | 20 min | ✅ 完成 |
| 3 | `causal_jepa.py` | context mask 返回 mask 位置；forward 独立采样 M_t=3 个 target patch | 30 min | ✅ 完成 |
| 4 | `causal_jepa.py` | 新增 `mask_token: nn.Parameter(128,)` + `pos_embed: nn.Embedding(9, 128)` | 10 min | ✅ 完成 |
| 5 | `models/predictors.py` | 重写 PatchPredictor：cross-attn(Q=query, KV=patches) + MLP；P_self/P_pa/P_non 共用结构 | 45 min | ✅ 完成 |
| 6 | `causal_jepa.py` | Parent/Non-parent 聚合改 patch-level：`einsum('bjnp,dj->bdnp', s_ctx, A)`；CausalLayer 输入用 `s_ctx.mean(dim=2)` | 20 min | ✅ 完成 |
| 7 | `losses/losses.py` | L_JEPA = mean MSE over target patches（只用 P_self）；L_ADV 改 patch-level log-ratio | 25 min | ✅ 完成 |
| 8 | `models/downstream.py` + `train.py` | DownstreamHead 输入 dim 改 128；probe 输入改为 `s_ctx.mean(patch)` | 15 min | ✅ 完成 |
| 9 | `configs/default.yaml` | hidden_dim: 128，删除 pool_mode，更新 mask_ratio 注释 | 5 min | ✅ 完成 |
| 10 | — | `python -c "from models.causal_jepa import CausalJEPA"` 确认 import | 5 min | ✅ 完成 |
| 11 | — | 跑 `--baseline` pure JEPA，确认 shape 正确、l_jepa 量级合理 | 20 min | ✅ 完成 |
| 12 | — | 跑完整 Causal-JEPA，对比 F1 vs 0.541 | ~20 min 运行 | ✅ 完成（F1=0，A 坍塌，已定位根因）|
| 13 | `causal_jepa.py` | **修复 A→0 退化不动点**：`_route` 里 s_pa 加单位自环，破坏 P_pa position-only 退化模式 | 10 min | ❌ **未实现（交接后第一优先任务，见 README Known Issues #1）** |

**预计总实现时长：~3 小时**

### 完成历史

| 任务 | 时间 |
|---|---|
| TTM r2 切换 + freq_token 检测修复 | 2026-04-29 |
| Pooling/Masking 消融实验（frozen backbone 是瓶颈）| 2026-04-23 |
| 多序列 SCM 数据集 + train/val/test 按序列划分 | 2026-04-23 |
| CAUKER 代码对齐（33-kernel bank，eigh 采样，mean functions）| 2026-04-23 |
| per-series per-variable z-score 归一化 | 2026-04-23 |
| D=5 N=30 T=1500 数据集生成并验证（Granger F1=0.541）| 2026-04-23 |
| Muon optimizer（Theta 用 Muon，encoder/predictor 用 AdamW）| 2026-04-22 |
| Granger 软初始化（high_conf=0.65，low_conf=0.35）| 2026-04-22 |
| log-ratio L_ADV + λ_adv=0.1 调通 | 2026-04-14 |
| Gumbel-Sigmoid + noise_scale 线性退火 | 2026-04-14 |
