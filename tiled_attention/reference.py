from __future__ import annotations

from typing import Optional

import torch


def dense_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    causal: bool = False,
    *,
    key_padding_mask: Optional[torch.Tensor] = None,
    q_start_pos: int = 0,
) -> torch.Tensor:
    """Small, independent dense implementation used only for testing."""
    q3 = q.ndim == 3
    if q3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    lq, lk = scores.shape[-2:]
    qpos = torch.arange(q_start_pos, q_start_pos + lq, device=q.device)
    kpos = torch.arange(lk, device=q.device)
    visible = torch.ones_like(scores, dtype=torch.bool)
    if causal:
        visible = qpos.view(1, 1, lq, 1) >= kpos.view(1, 1, 1, lk)
    if key_padding_mask is not None:
        visible = visible & (~key_padding_mask.bool()).view(q.shape[0], 1, 1, lk)

    scores = scores.masked_fill(~visible, -torch.inf)
    max_scores = torch.where(
        visible.any(dim=-1, keepdim=True),
        scores.amax(dim=-1, keepdim=True),
        torch.zeros_like(scores[..., :1]),
    )
    weights = torch.exp(scores - max_scores)
    weights = torch.where(visible, weights, torch.zeros_like(weights))
    normalizer = weights.sum(dim=-1, keepdim=True)
    safe_normalizer = torch.where(normalizer > 0, normalizer, torch.ones_like(normalizer))
    out = torch.matmul(weights / safe_normalizer, v)
    out = torch.where(normalizer > 0, out, torch.zeros_like(out))
    return out.squeeze(1) if q3 else out
