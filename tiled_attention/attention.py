from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DenseTiledAttentionStats:
    q_block: int
    k_block: int
    saved_tensors: int


_LAST_STATS: Optional[DenseTiledAttentionStats] = None


def _canonicalize(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 3:
        return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape [B, L, D] or [B, H, L, D]")
    return x


def _visibility(
    q_start_pos: int,
    j0: int,
    bq: int,
    bk: int,
    k_padding_mask: Optional[torch.Tensor],
    causal: bool,
    batch: int,
    heads: int,
    device: torch.device,
) -> torch.Tensor:
    qpos = torch.arange(q_start_pos, q_start_pos + bq, device=device)
    kpos = torch.arange(j0, j0 + bk, device=device)
    visible = torch.ones((1, 1, bq, bk), dtype=torch.bool, device=device)
    if causal:
        visible = qpos[None, None, :, None] >= kpos[None, None, None, :]
    if k_padding_mask is not None:
        key_ok = ~k_padding_mask[:, j0 : j0 + bk].to(torch.bool)
        visible = visible & key_ok.reshape(key_ok.shape[0], 1, 1, bk)
    return visible.expand(batch, heads, bq, bk)


class _TiledAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        causal: bool,
        q_start_pos: int,
        k_padding_mask: Optional[torch.Tensor],
        q_block: int,
        k_block: int,
    ) -> torch.Tensor:
        q3 = q.ndim == 3
        qq = _canonicalize(q, "q")
        kk = _canonicalize(k, "k")
        vv = _canonicalize(v, "v")
        if qq.shape[:2] != kk.shape[:2] or qq.shape[:2] != vv.shape[:2]:
            raise ValueError("q, k, and v must have matching batch and head dimensions")
        if kk.shape[1] != vv.shape[1] or kk.shape[2] != vv.shape[2]:
            raise ValueError("k and v must have the same number of keys")
        if qq.shape[3] != kk.shape[3]:
            raise ValueError("q and k must have matching head dimensions")
        if k_padding_mask is not None:
            if k_padding_mask.shape != (kk.shape[0], kk.shape[2]):
                raise ValueError("k_padding_mask must have shape [batch, key_length]")
        if q_block <= 0 or k_block <= 0:
            raise ValueError("block sizes must be positive")
        if not torch.isfinite(torch.tensor(scale)):
            raise ValueError("scale must be a finite number")
        if q_start_pos < 0:
            raise ValueError("q_start_pos must be non-negative")

        work_dtype = torch.float32 if qq.dtype in (torch.float16, torch.bfloat16) else qq.dtype
        qw = qq.to(work_dtype)
        kw = kk.to(work_dtype)
        vw = vv.to(work_dtype)
        batch, heads, lq, dhead = qw.shape
        lk = kw.shape[2]
        device = qw.device
        neg_inf = torch.full((), -torch.inf, dtype=work_dtype, device=device)

        m = torch.full((batch, heads, lq, 1), -torch.inf, dtype=work_dtype, device=device)
        ell = torch.zeros((batch, heads, lq, 1), dtype=work_dtype, device=device)
        acc = torch.zeros((batch, heads, lq, dhead), dtype=work_dtype, device=device)
        inf_count = torch.zeros((batch, heads, lq, 1), dtype=work_dtype, device=device)

        for i0 in range(0, lq, q_block):
            i1 = min(i0 + q_block, lq)
            bq = i1 - i0
            for j0 in range(0, lk, k_block):
                j1 = min(j0 + k_block, lk)
                bk = j1 - j0
                visible = _visibility(
                    q_start_pos + i0, j0, bq, bk, k_padding_mask, causal, batch, heads, device
                )
                scores = torch.matmul(qw[:, :, i0:i1], kw[:, :, j0:j1].transpose(-1, -2)) * scale
                raw_inf = torch.isposinf(scores)
                scores = scores.masked_fill(~visible, -torch.inf)
                has_pos_inf = (raw_inf & visible).any(dim=-1, keepdim=True)
                finite_scores = scores.masked_fill(raw_inf | ~visible, -torch.inf)
                bmax = finite_scores.amax(dim=-1, keepdim=True)
                any_valid = visible.any(dim=-1, keepdim=True)
                bmax = torch.where(any_valid, bmax, neg_inf)

                old_m = m[:, :, i0:i1]
                old_l = ell[:, :, i0:i1]
                old_a = acc[:, :, i0:i1]
                old_c = inf_count[:, :, i0:i1]
                old_inf = old_c > 0

                new_m = torch.where(any_valid, torch.maximum(old_m, bmax), old_m)
                alpha_old = torch.where(
                    old_inf,
                    torch.zeros_like(old_l),
                    torch.nan_to_num(
                        torch.exp(torch.where(old_m.isfinite(), old_m, torch.zeros_like(old_m)) - new_m),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ),
                )
                alpha_block = torch.nan_to_num(
                    torch.exp(torch.where(any_valid, bmax, torch.zeros_like(bmax)) - new_m),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                block_w = torch.where(
                    bmax.isfinite() & any_valid,
                    torch.exp(finite_scores - bmax) * visible,
                    torch.zeros_like(scores),
                )
                bsum = block_w.sum(dim=-1, keepdim=True)
                new_inf_count = old_c + has_pos_inf.to(work_dtype)

                scaled_a = torch.where(old_inf & ~has_pos_inf, old_a, old_a * alpha_old)
                bv_inf = torch.where(
                    has_pos_inf,
                    torch.matmul((raw_inf & visible).to(work_dtype), vw[:, :, j0:j1]),
                    torch.zeros_like(old_a),
                )
                bv_fin = torch.matmul(block_w, vw[:, :, j0:j1]) * alpha_block
                bv = torch.where(
                    has_pos_inf,
                    bv_inf,
                    torch.where(old_inf, torch.zeros_like(bv_fin), bv_fin),
                )

                finite_new_l = old_l * alpha_old + bsum * alpha_block
                new_l = torch.where(
                    has_pos_inf,
                    new_inf_count,
                    torch.where(old_inf, old_c, finite_new_l),
                )
                new_acc = scaled_a + bv

                m[:, :, i0:i1] = torch.where(has_pos_inf, torch.full_like(new_m, torch.inf), new_m)
                ell[:, :, i0:i1] = new_l
                acc[:, :, i0:i1] = new_acc
                inf_count[:, :, i0:i1] = new_inf_count

        safe_l = torch.where(ell > 0, ell, torch.ones_like(ell))
        out = acc / safe_l
        out = torch.where(ell > 0, out, torch.zeros_like(out)).to(q.dtype)

        ctx.q3 = q3
        ctx.causal = causal
        ctx.q_start_pos = q_start_pos
        ctx.q_block = q_block
        ctx.k_block = k_block
        ctx.work_dtype = work_dtype
        ctx.scale = float(scale)
        ctx.save_for_backward(q, k, v, out, m, ell, inf_count, k_padding_mask)
        return out if not q3 else out.reshape_as(q)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, out, m, ell, inf_count, pad = ctx.saved_tensors
        gout = _canonicalize(grad_out, "grad_out")
        qw = _canonicalize(q, "q").to(ctx.work_dtype)
        kw = _canonicalize(k, "k").to(ctx.work_dtype)
        vw = _canonicalize(v, "v").to(ctx.work_dtype)
        ow = _canonicalize(out, "out").to(ctx.work_dtype)
        gow = gout.to(ctx.work_dtype)
        batch, heads, lq, dhead = qw.shape
        lk = kw.shape[2]
        device = qw.device

        dq = torch.zeros_like(qw)
        dk = torch.zeros_like(kw)
        dv = torch.zeros_like(vw)
        valid = ell > 0
        safe_l = torch.where(valid, ell, torch.ones_like(ell))
        delta = torch.sum(gow * ow, dim=-1, keepdim=True)

        for i0 in range(0, lq, ctx.q_block):
            i1 = min(i0 + ctx.q_block, lq)
            bq = i1 - i0
            for j0 in range(0, lk, ctx.k_block):
                j1 = min(j0 + ctx.k_block, lk)
                bk = j1 - j0
                visible = _visibility(
                    ctx.q_start_pos + i0,
                    j0,
                    bq,
                    bk,
                    pad,
                    ctx.causal,
                    batch,
                    heads,
                    device,
                )
                scores = (
                    torch.matmul(qw[:, :, i0:i1], kw[:, :, j0:j1].transpose(-1, -2))
                    * ctx.scale
                )
                raw_inf = torch.isposinf(scores)
                row_inf = inf_count[:, :, i0:i1] > 0
                block_inf = raw_inf & visible
                shifted = torch.where(
                    row_inf,
                    torch.zeros_like(scores),
                    scores - m[:, :, i0:i1],
                )
                probs = torch.where(visible, torch.exp(shifted), torch.zeros_like(scores))
                probs = probs / safe_l[:, :, i0:i1]
                probs = torch.where(
                    row_inf,
                    block_inf.to(ctx.work_dtype) / inf_count[:, :, i0:i1],
                    probs,
                )
                probs = torch.where(valid[:, :, i0:i1], probs, torch.zeros_like(probs))

                dv[:, :, j0:j1] += torch.matmul(probs.transpose(-1, -2), gow[:, :, i0:i1])
                dp = torch.matmul(gow[:, :, i0:i1], vw[:, :, j0:j1].transpose(-1, -2))
                ds = probs * (dp - delta[:, :, i0:i1])
                ds = torch.where(visible, ds, torch.zeros_like(ds))
                ds = torch.where(valid[:, :, i0:i1], ds, torch.zeros_like(ds))
                dq_update = torch.matmul(ds, kw[:, :, j0:j1]) * ctx.scale
                dq[:, :, i0:i1] += torch.where(row_inf, torch.zeros_like(dq_update), dq_update)
                safe_q = torch.where(
                    valid[:, :, i0:i1], qw[:, :, i0:i1], torch.zeros_like(qw[:, :, i0:i1])
                )
                safe_q = torch.where(row_inf, torch.zeros_like(safe_q), safe_q)
                dk[:, :, j0:j1] += torch.matmul(ds.transpose(-1, -2), safe_q) * ctx.scale

        return (
            (dq if not ctx.q3 else dq.reshape_as(q)).to(q.dtype),
            (dk if not ctx.q3 else dk.reshape_as(k)).to(k.dtype),
            (dv if not ctx.q3 else dv.reshape_as(v)).to(v.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    causal: bool = False,
    *,
    key_padding_mask: Optional[torch.Tensor] = None,
    q_start_pos: int = 0,
    q_block: int = 32,
    k_block: int = 32,
) -> torch.Tensor:
    """Exact tiled dense attention with a custom block-recomputing backward pass."""
    global _LAST_STATS
    result = _TiledAttention.apply(
        q,
        k,
        v,
        float(scale),
        bool(causal),
        int(q_start_pos),
        key_padding_mask,
        int(q_block),
        int(k_block),
    )
    saved_tensors = 8 if key_padding_mask is not None else 7
    _LAST_STATS = DenseTiledAttentionStats(q_block, k_block, saved_tensors)
    return result


def last_tiled_attention_stats() -> Optional[DenseTiledAttentionStats]:
    return _LAST_STATS
