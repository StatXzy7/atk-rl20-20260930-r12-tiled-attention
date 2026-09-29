# Tiled Exact Attention

纯 PyTorch 分块精确 softmax attention，主路径不使用
`scaled_dot_product_attention`、FlashAttention 或完整 attention API。

```python
from tiled_attention import tiled_attention

output = tiled_attention(
    q,
    k,
    v,
    scale=None,              # 默认 1 / sqrt(D)
    causal=False,
    key_padding_mask=mask,   # bool，True 表示 padded/invalid key
    block_size=64,
)
```

`Q/K/V` 形状为 `(B, H, L, D)`，支持不同 Q/K 长度、非连续输入、末尾不完整块和零长度 key。causal 使用绝对 query/key 位置 `j <= i`。

- 推导、保存张量和实测数据：`REPORT.md`
- 小规模稠密参照：`dense_reference.py`
- 双精度 gradcheck 和边界测试：`python -m unittest -v`
- 基准：`python benchmark_tiled_attention.py --lengths 64 128 256 512`
