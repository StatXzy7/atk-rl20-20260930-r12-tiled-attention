# 分块精确 Attention

PyTorch 自定义 autograd 算子：前向用在线 softmax 合并，反向用保存的行统计和分块重算求 `dQ/dK/dV`，不生成完整 N×N 分数或概率矩阵。

## 运行

```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python scripts/benchmark_tiled_attention.py --lengths 64,128,256,512
```

关键实现位于 `tiled_attention/attention.py:35`，独立稠密参照位于 `tiled_attention/reference.py:10`，推导与复杂度见 `docs/TILED_ATTENTION.md:1`。
