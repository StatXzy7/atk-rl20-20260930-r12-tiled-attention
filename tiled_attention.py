import math

import torch


def _canonicalize_key_padding_mask(mask, batch, heads, key_length, device):
    if mask is None:
        return None
    if mask.device != device:
        raise ValueError("key_padding_mask must be on the same device as Q/K/V")
    if mask.dtype is not torch.bool:
        raise TypeError("key_padding_mask must be a boolean tensor")
    if mask.dim() == 2:
        if mask.shape != (batch, key_length):
            raise ValueError(
                f"expected key_padding_mask shape {(batch, key_length)}, got {tuple(mask.shape)}"
            )
        return mask.unsqueeze(1).expand(batch, heads, key_length)
    if mask.dim() == 3:
        if mask.shape != (batch, heads, key_length):
            raise ValueError(
                f"expected key_padding_mask shape {(batch, heads, key_length)}, got {tuple(mask.shape)}"
            )
        return mask
    raise ValueError("key_padding_mask must have shape (B, Lk) or (B, H, Lk)")


def _block_visibility(key_padding_mask, causal, query_start, query_end, key_start, key_end, device):
    if causal:
        query_positions = torch.arange(query_start, query_end, device=device).view(-1, 1)
        key_positions = torch.arange(key_start, key_end, device=device).view(1, -1)
        visible = key_positions <= query_positions
        if key_padding_mask is not None:
            valid_keys = ~key_padding_mask[:, :, key_start:key_end].unsqueeze(2)
            visible = visible.unsqueeze(0).unsqueeze(0) & valid_keys
        return visible
    if key_padding_mask is not None:
        return (~key_padding_mask[:, :, key_start:key_end]).unsqueeze(2)
    return None


class _TiledAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, key_padding_mask, causal, block_size):
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("Q, K and V must have shape (B, H, L, D)")
        if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
            raise ValueError("Q, K and V must have the same batch and head dimensions")
        if k.shape[3] != q.shape[3]:
            raise ValueError("Q and K must have the same head dimension")
        if v.shape[2] != k.shape[2]:
            raise ValueError("K and V must have the same key length")
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not torch.isfinite(torch.as_tensor(scale, device=q.device, dtype=q.dtype)):
            raise ValueError("scale must be finite")

        batch, heads, query_length, head_dim = q.shape
        key_length = k.shape[2]
        mask = _canonicalize_key_padding_mask(
            key_padding_mask, batch, heads, key_length, q.device
        )

        min_float = torch.finfo(q.dtype).min
        running_max = q.new_full((batch, heads, query_length, 1), min_float)
        running_sum = q.new_zeros(batch, heads, query_length, 1)
        output_accumulator = q.new_zeros(batch, heads, query_length, v.shape[-1])
        neg_inf = torch.as_tensor(float("-inf"), dtype=q.dtype, device=q.device)
        min_float_tensor = q.new_tensor(min_float)
        zero = q.new_zeros(())

        for key_start in range(0, key_length, block_size):
            key_end = min(key_start + block_size, key_length)
            key_block = k[:, :, key_start:key_end, :]
            value_block = v[:, :, key_start:key_end, :]
            scores = torch.matmul(q, key_block.transpose(-2, -1)) * scale
            visible = _block_visibility(
                mask, causal, 0, query_length, key_start, key_end, q.device
            )

            if visible is None:
                block_max = scores.amax(dim=-1, keepdim=True)
                block_weights = torch.exp(scores - block_max)
            else:
                masked_scores = scores.masked_fill(~visible, min_float_tensor)
                block_max = masked_scores.amax(dim=-1, keepdim=True)
                block_exponents = (scores - block_max).masked_fill(
                    ~visible, neg_inf
                )
                block_weights = torch.exp(block_exponents)

            block_sum_local = block_weights.sum(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)
            old_correction = torch.where(
                running_sum > 0,
                torch.exp(running_max - new_max),
                zero,
            )
            block_correction = torch.where(
                block_sum_local > 0,
                torch.exp(block_max - new_max),
                zero,
            )
            running_sum = (
                running_sum * old_correction + block_sum_local * block_correction
            )
            output_accumulator = output_accumulator * old_correction + (
                torch.matmul(block_weights, value_block) * block_correction
            )
            running_max = new_max

        safe_sum = torch.where(
            running_sum > 0, running_sum, torch.ones_like(running_sum)
        )
        output = output_accumulator / safe_sum
        output = torch.where(
            (running_sum > 0).expand_as(output), output, torch.zeros_like(output)
        )

        ctx.save_for_backward(q, k, v, output, running_max, running_sum, mask)
        ctx.scale = float(scale)
        ctx.block_size = block_size
        ctx.causal = bool(causal)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, running_max, running_sum, mask = ctx.saved_tensors
        batch, heads, query_length, head_dim = q.shape
        key_length = k.shape[2]
        block_size = ctx.block_size
        scale = ctx.scale

        grad_q_work = torch.zeros_like(q, memory_format=torch.contiguous_format)
        grad_k_work = torch.zeros_like(k, memory_format=torch.contiguous_format)
        grad_v_work = torch.zeros_like(v, memory_format=torch.contiguous_format)
        neg_inf = torch.as_tensor(float("-inf"), dtype=q.dtype, device=q.device)

        positive_sum = running_sum > 0
        safe_sum = torch.where(positive_sum, running_sum, torch.ones_like(running_sum))
        safe_max = torch.where(
            positive_sum, running_max, torch.zeros_like(running_max)
        )

        for query_start in range(0, query_length, block_size):
            query_end = min(query_start + block_size, query_length)
            query_block = q[:, :, query_start:query_end, :]
            grad_query_tile = torch.zeros_like(
                query_block, memory_format=torch.contiguous_format
            )
            row_safe_sum = safe_sum[:, :, query_start:query_end, :]
            row_safe_max = safe_max[:, :, query_start:query_end, :]
            row_logit_dot = torch.sum(
                grad_output[:, :, query_start:query_end, :] * output[:, :, query_start:query_end, :],
                dim=-1,
                keepdim=True,
            )

            for key_start in range(0, key_length, block_size):
                key_end = min(key_start + block_size, key_length)
                key_block = k[:, :, key_start:key_end, :]
                value_block = v[:, :, key_start:key_end, :]
                grad_output_block = grad_output[:, :, query_start:query_end, :]

                scores = torch.matmul(query_block, key_block.transpose(-2, -1)) * scale
                visible = _block_visibility(
                    mask,
                    ctx.causal,
                    query_start,
                    query_end,
                    key_start,
                    key_end,
                    q.device,
                )
                if visible is None:
                    probabilities = torch.exp(scores - row_safe_max) / row_safe_sum
                else:
                    block_exponents = (scores - row_safe_max).masked_fill(
                        ~visible, neg_inf
                    )
                    probabilities = (
                        torch.exp(block_exponents) / row_safe_sum
                    )

                grad_v_work[:, :, key_start:key_end, :] += torch.matmul(
                    probabilities.transpose(-2, -1), grad_output_block
                )
                grad_logits_value = torch.matmul(
                    grad_output_block, value_block.transpose(-2, -1)
                )
                grad_scores = (
                    scale
                    * probabilities
                    * (grad_logits_value - row_logit_dot)
                )
                grad_query_tile += torch.matmul(grad_scores, key_block)
                grad_k_work[:, :, key_start:key_end, :] += torch.matmul(
                    grad_scores.transpose(-2, -1), query_block
                )

            grad_q_work[:, :, query_start:query_end, :] = grad_query_tile

        grad_q = torch.empty_strided(
            q.shape, q.stride(), dtype=q.dtype, device=q.device
        )
        grad_k = torch.empty_strided(
            k.shape, k.stride(), dtype=k.dtype, device=k.device
        )
        grad_v = torch.empty_strided(
            v.shape, v.stride(), dtype=v.dtype, device=v.device
        )
        for query_start in range(0, query_length, block_size):
            query_end = min(query_start + block_size, query_length)
            grad_q[:, :, query_start:query_end, :].copy_(
                grad_q_work[:, :, query_start:query_end, :]
            )
        for key_start in range(0, key_length, block_size):
            key_end = min(key_start + block_size, key_length)
            grad_k[:, :, key_start:key_end, :].copy_(
                grad_k_work[:, :, key_start:key_end, :]
            )
            grad_v[:, :, key_start:key_end, :].copy_(
                grad_v_work[:, :, key_start:key_end, :]
            )

        return grad_q, grad_k, grad_v, None, None, None, None


def tiled_attention(
    q,
    k,
    v,
    scale=None,
    causal=False,
    key_padding_mask=None,
    block_size=64,
):
    """Exact tiled softmax attention.

    Q/K/V have shape ``(B, H, L, D)``. ``key_padding_mask`` is boolean, where
    ``True`` marks a padded/invalid key. Causal positions use absolute global
    positions ``j <= i`` and therefore work for unequal Q and K lengths.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return _TiledAttention.apply(
        q, k, v, float(scale), key_padding_mask, causal, int(block_size)
    )
