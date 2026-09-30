# 分块精确 Attention 推导与实现说明

## 记号

对单个 batch/head 的一行 query，令

\[
s_{ij}=\operatorname{scale}\,q_i^\top k_j,\qquad
p_{ij}=\frac{\exp(s_{ij})[j\in V_i]}{\sum_{t\in V_i}\exp(s_{it})},\qquad
o_i=\sum_j p_{ij}v_j .
\]

其中 \(V_i\) 是该行 query 可见的 key。可见性由显式绝对位置决定：

\[
j\in V_i \iff (\neg M^{\mathrm{key}}_b(j))\land
(\neg\text{causal}\ \lor\ q_{\mathrm{start}}+i\ge j).
\]

因此非方阵时使用的是 \(q_{\mathrm{start}}+i\) 与 key 的绝对下标 \(j\)，不假设 \(L_q=L_k\)，也不套用方阵下三角。

若 \(V_i=\varnothing\)，定义 \(o_i=0\)，且该行对 `Q/K/V` 的梯度全部为 0。

## 在线前向合并

把 key 维切成块。处理块 \(C\) 时，块内逐行最大值与块内质量为

\[
m_i^C=\max_{j\in C\cap V_i} s_{ij},\qquad
b_i^C=\sum_{j\in C\cap V_i}\exp(s_{ij}-m_i^C).
\]

设此前已经处理的键集合为 \(A\)，保存

\[
m_i^A=\max_{t\in A\cap V_i}s_{it},\quad
\ell_i^A=\sum_{t\in A\cap V_i}\exp(s_{it}-m_i^A),\quad
a_i^A=\sum_{t\in A\cap V_i}\exp(s_{it}-m_i^A)v_t.
\]

合并后令 \(m_i'=\max(m_i^A,m_i^C)\)，则

\[
\alpha_i^A=\exp(m_i^A-m_i'),\qquad
\alpha_i^C=\exp(m_i^C-m_i'),
\]

\[
\ell_i'=\alpha_i^A\ell_i^A+\alpha_i^Cb_i^C,
\]

\[
a_i'=\alpha_i^A a_i^A+
\alpha_i^C\sum_{j\in C\cap V_i}\exp(s_{ij}-m_i^C)v_j.
\]

最终输出为 \(o_i=a_i/\ell_i\)；\(\ell_i=0\) 时输出零。全部更新只使用当前 \(B_q\times B_k\) 的分数块，不物化完整 \(L_q\times L_k\) 分数或概率矩阵。

## 极大正负 logits

所有指数都先减块最大值，再合并时减全局行最大值，避免 `exp(+large)` 溢出。不可见位置的分数不参与最大值；一个块完全不可见时保留旧状态，不把旧最大值改成 \(-\infty\)。

数学上的 \(+\infty\) 作为边界情形单独处理：若一行有 \(c\) 个可见的 \(+\infty\)，它们形成均匀 tie，输出这些 key 对应 value 的平均，有限分数权重为零。实现保存正无穷 tie 计数，并在后续有限块中冻结该 tie 状态；遇到新的正无穷时合并计数和 value 和。可见性为 false 的 \(+\infty\) 不参与计数，因而不会产生 NaN。若所有可见值都是 \(-\infty\) 且不存在正无穷，该行按无可见 key 处理；有限但极大的负 logits 会自然下溢为 0。

## 自定义反向

反向不保存概率，也不重新物化完整概率矩阵。保存：

- 原始 `Q/K/V`；
- 输出 `O`；
- 每行最终最大值 `M`；
- 每行未归一化质量 `L = \sum_j exp(s_ij-m_i)`；
- 每行正无穷 tie 计数（没有 `key_padding_mask` 时共 7 个张量，有 mask 时 8 个）。

对每个 query/key 块重新计算：

\[
S_{ij}=Q_iK_j^\top \operatorname{scale},\qquad
P_{ij}=\frac{\exp(S_{ij}-m_i)[j\in V_i]}{\ell_i}.
\]

正无穷行使用 \(1/c_i\) 在可见正无穷 key 间均匀分配。然后

\[
dV_j=\sum_i P_{ij}\,dO_i,
\]

\[
dP_{ij}=dO_i^\top V_j,\qquad
dS_{ij}=P_{ij}(dP_{ij}-\delta_i),\qquad
\delta_i=\sum_d dO_{id}O_{id},
\]

\[
dQ_i=\operatorname{scale}\sum_j dS_{ij}K_j,\qquad
dK_j=\operatorname{scale}\sum_i dS_{ij}Q_i.
\]

空行的 `delta` 不使用，块概率和 `dS` 都置零。除以归一化质量前先用 1 替换 0，只用于避免 0/0；最终再用可见性掩码清零。

## 复杂度与临时存储

设形状为 \([B,H,L_q,D]\) 和 \([B,H,L_k,D]\)：

- 算术量仍是 \(O(BHL_qL_kD)\)，即仍是二次 attention；分块不把二次算术变成线性。
- 固定块大小下，分数/概率临时张量为 \(O(BHB_qB_k)\)，不随 \(L_qL_k\) 增长。
- 持久保存量为 \(O(BH(L_qD+L_kD+L_q)))\)；本实现保存三份输入、一份输出、三份逐行统计，带 mask 时外加 mask。
- 末尾不完整块自然由 `min(L,B)` 处理。
- 输入不调用 `.contiguous()`；reshape 只改变视图，矩阵乘和切片读取都接受非连续输入。

## 使用

```python
from tiled_attention import tiled_attention

out = tiled_attention(
    q, k, v,
    scale=q.size(-1) ** -0.5,
    causal=True,
    key_padding_mask=padding_bool,  # True 表示该 key 是 padding
    q_start_pos=q_abs_start,
    q_block=32,
    k_block=32,
)
```

主路径只调用基础矩阵乘、归约和指数，不调用 `scaled_dot_product_attention`、FlashAttention 或其他完整 attention API。`tiled_attention.reference.dense_attention_reference` 是独立稠密参照，仅供测试核验。

## 实测数据

环境：PyTorch 2.14.0 CPU-only，`B=1, H=2, D=32, q_block=k_block=32, float32`，计时取 3 次重复中的最小值。

| N | attention pairs | forward (ms) | forward+backward (ms) | 持久保存估算 (MiB) | 分数块 (MiB) | 块工作区估算上界 (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4,096 | 3.425 | 6.909 | 0.064 | 0.008 | 0.062 |
| 128 | 16,384 | 12.493 | 23.739 | 0.128 | 0.008 | 0.062 |
| 256 | 65,536 | 49.777 | 92.651 | 0.256 | 0.008 | 0.062 |
| 512 | 262,144 | 198.433 | 367.307 | 0.512 | 0.008 | 0.062 |

保存张量数量为 7；带 `key_padding_mask` 时为 8。固定块大小时，最大块分数/概率张量不随 N 增长；持久保存量随 \(O(N)\) 增长，算术工作量随 \(O(N^2)\) 增长。Python 分块实现主要用于说明精确算法和内存边界，不预期超过优化库内核。
