import torch


def dense_attention_reference(
    q,
    k,
    v,
    scale,
    causal=False,
    key_padding_mask=None,
):
    """Small-only dense reference used by tests, not the operator path."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        query_positions = torch.arange(q.shape[-2], device=q.device)
        key_positions = torch.arange(k.shape[-2], device=k.device)
        scores = scores.masked_fill(
            key_positions.view(1, -1) > query_positions.view(-1, 1),
            float("-inf"),
        )

    if key_padding_mask is not None:
        if key_padding_mask.dim() == 2:
            invalid = key_padding_mask.unsqueeze(1).unsqueeze(-2)
        else:
            invalid = key_padding_mask.unsqueeze(-2)
        scores = scores.masked_fill(invalid, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0)
    return torch.matmul(probabilities, v)
