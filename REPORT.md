# 分块精确 Attention 报告

## 数学定义

输入 `Q∈R^{B×H×Lq×Dq}`、`K∈R^{B×H×Lk×Dq}`、`V∈R^{B×H×Lk×Dv}`，缩放 logits 为

`S_ij = scale · Q_i · K_j`。

可见性 `M_ij` 同时包含 causal 条件 `j <= i`（使用绝对 query/key 位置）和 key padding。对每个 query 行：

`P_ij = M_ij · exp(S_ij - max_k:M_ik=1 S_ik) / Z_i`

`Z_i = Σ_k M_ik · exp(S_ik - m_i)`，`O_i = Σ_j P_ij V_j`。

如果某行没有任何可见 key，约定 `O_i=0`、`dQ_i=0`、所有 key 对该行的 `dK/dV` 贡献为 0。

## 前向在线合并

处理 key 块 `t` 时只构造大小 `Br×Bc` 的 score tile。令：

- `b_i^t = max`：当前块内可见 logits 的最大值；空块记为 `-inf`。
- `w_ij^t = M_ij · exp(S_ij - b_i^t)`。
- `z_i^t = Σ_j w_ij^t`。

前一块保存全局最大值 `m_i`、全局非归一化和 `l_i=Σ旧 exp(S-m_i)` 以及非归一化分子 `a_i=Σ旧 exp(S-m_i)V`。合并到新全局最大值

`m_i' = max(m_i, b_i^t)`

后，旧块和当前块分别重标定：

`c_old = exp(m_i - m_i')`

`c_block = exp(b_i^t - m_i')`

`l_i' = c_old·l_i + c_block·z_i^t`

`a_i' = c_old·a_i + c_block·Σ_j w_ij^t V_j`

最终 `O_i=a_i/l_i`。实现中空块的重标定系数直接置零，不计算 `exp(-inf)`；无可见行用安全分母 1 计算后显式输出零，避免 `0/0`。

## 自定义反向

对可见 logits，softmax Jacobian 为

`dS_ij = scale·P_ij·(dO_i·V_j - Σ_k P_ik(dO_i·V_k))`。

第二项是完整 query 行的全局点积。由于 `O_i=Σ_k P_ik V_k`，实现保存输出后直接计算

`r_i = dO_i·O_i`

，然后在每个 key tile 内重算 `P_ij`，得到：

`dV_j += Σ_i P_ij dO_i`

`dS_ij = scale·P_ij·(dO_i·V_j - r_i)`

`dQ_i += Σ_j dS_ij K_j`

`dK_j += Σ_i dS_ij Q_i`

反向保存 `Q、K、V、O、m、l`，逐块重算 score/probability；不会物化完整 `Lq×Lk` 分数或概率矩阵。不可见位置的指数输入先填成 `-inf` 再 `exp`，避免隐藏位置存在极大 logits 时出现 `inf·0=NaN`。

## 复杂度与存储

- 算术量仍是 `O(Lq·Lk·D)`；分块降低内存，不把二次复杂度变成线性。
- 保存张量：无 padding mask 时为 6 个稠密张量（`Q、K、V、O、m、l`）；传入 mask 时第 7 个保存对象是外部 mask（可能是广播视图）。
- 保存容量：
  - `Q/K/V/O`：`4BHLD` 个元素（`Dq=Dv=D` 时）。
  - `m/l`：`2BHL` 个元素。
  - 总计 `BHL(4D+2)` 个元素，随序列长度线性增长。
- 主路径额外 tile 临时量上界约为 `O(BH(BrBc + (Br+Bc)D))`，随块大小增长，不随完整 `Lq·Lk` 增长。
- 块大小可配置，自动处理最后不完整块；Q/K/V 不要求连续，反向梯度保持输入逻辑形状和步幅。

## 测试

运行：

`python -m unittest -v`

覆盖内容：

- 双精度稠密前向对照，块大小 `1/3/16`，覆盖末尾不完整块。
- 双精度稠密 `dQ/dK/dV` 对照。
- `torch.autograd.gradcheck`：矩形 causal、非连续 Q/K/V 和 padding 视图、全行被 mask。
- 非方阵 causal 使用绝对全局位置，而不是套用方阵 `tril`。
- 不同 Q/K 长度、head 级 padding、零 key 长度。
- `scale=2500` 的极大正负 logits、`scale=10000` 且极大 logits 落在 causal/padding 不可见位置。
- 无可见 key 行输出为零且相关梯度为零、无 NaN/Inf。

“精确”指与同一数学 softmax attention 等价，差异只来自浮点求和顺序和舍入；不承诺逐位相同。

## 实测数据

环境：PyTorch 2.14.0 CPU，无 CUDA；`B=1,H=4,D=64,float32,causal=True`，`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，预热 5 次、重复 7 次，表中为“中位数/最快值”。Python 分块实现包含大量小算子和 Python 循环，不预期比优化 fused kernel 更快。

| Lq | Lk | Br | Bc | forward median/min (ms) | fwd+bwd median/min (ms) | 保存 (MiB) | tile 上界 (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 32 | 32 | 4.02/3.12 | 12.67/11.65 | 0.25 | 0.23 |
| 64 | 64 | 64 | 64 | 2.28/1.73 | 6.89/5.44 | 0.25 | 0.56 |
| 128 | 128 | 32 | 32 | 20.53/6.83 | 71.07/37.65 | 0.50 | 0.23 |
| 128 | 128 | 64 | 64 | 6.35/5.29 | 18.72/16.31 | 0.50 | 0.56 |
| 256 | 256 | 32 | 32 | 26.86/25.58 | 239.86/145.52 | 1.01 | 0.23 |
| 256 | 256 | 64 | 64 | 23.41/21.33 | 121.20/59.66 | 1.01 | 0.56 |
| 512 | 512 | 32 | 32 | 189.34/102.79 | 732.74/602.75 | 2.02 | 0.23 |
| 512 | 512 | 64 | 64 | 151.18/71.57 | 230.46/220.39 | 2.02 | 0.56 |

保存内存随 `L` 近似线性增长（0.25 → 0.50 → 1.01 → 2.02 MiB），而算法 tile 临时量在固定块大小下保持约 0.23 或 0.56 MiB。时间随长度显示二次缩放：例如块大小 64 的 forward+backward 中位数从 6.89 ms 增至 18.72 ms、121.20 ms、230.46 ms；最大长度处继续保持约 4× 的总体趋势，但 Python 循环、小 GEMM 启动和 CPU 缓存效应会让比例不严格等于 4。可用 `benchmark_tiled_attention.py` 重新测量。
