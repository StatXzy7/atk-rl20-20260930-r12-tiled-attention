from __future__ import annotations

import pytest
import torch

from tiled_attention import dense_attention_reference, last_tiled_attention_stats, tiled_attention


def make_case(batch=1, heads=2, lq=5, lk=7, dhead=3, *, dtype=torch.float64, mask=None):
    torch.manual_seed(1234)
    q = torch.randn(batch, heads, lq, dhead, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, heads, lk, dhead, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, heads, lk, dhead, dtype=dtype, requires_grad=True)
    return q, k, v


def compare(q, k, v, scale=0.7, causal=True, q_start_pos=0, mask=None, q_block=2, k_block=3, tol=None):
    rtol, atol = tol or (1e-10, 1e-10)
    q1 = q.detach().clone().requires_grad_()
    k1 = k.detach().clone().requires_grad_()
    v1 = v.detach().clone().requires_grad_()
    q2 = q.detach().clone().requires_grad_()
    k2 = k.detach().clone().requires_grad_()
    v2 = v.detach().clone().requires_grad_()

    out_tiled = tiled_attention(
        q1, k1, v1, scale, causal, key_padding_mask=mask, q_start_pos=q_start_pos,
        q_block=q_block, k_block=k_block,
    )
    out_ref = dense_attention_reference(
        q2, k2, v2, scale, causal, key_padding_mask=mask, q_start_pos=q_start_pos
    )
    torch.testing.assert_close(out_tiled, out_ref, rtol=rtol, atol=atol)

    grad = torch.randn_like(out_tiled)
    out_tiled.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q1.grad, q2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(k1.grad, k2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(v1.grad, v2.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("lq,lk,q_start_pos,causal", [
    (5, 7, 0, True),
    (7, 5, 2, True),
    (6, 9, 3, False),
    (1, 1, 4, True),
])
@pytest.mark.parametrize("q_block,k_block", [(1, 1), (3, 2), (8, 8)])
def test_forward_and_gradient_match_dense(lq, lk, q_start_pos, causal, q_block, k_block):
    q, k, v = make_case(batch=2, heads=2, lq=lq, lk=lk, dhead=4)
    mask = torch.zeros(2, lk, dtype=torch.bool)
    mask[0, -1] = True
    mask[1, lk // 2:] = True
    compare(q, k, v, causal=causal, q_start_pos=q_start_pos, mask=mask,
            q_block=q_block, k_block=k_block)


def test_gradcheck_double_precision():
    q, k, v = make_case(batch=1, heads=2, lq=4, lk=6, dhead=3)
    mask = torch.tensor([[False, False, False, True, True, False]])
    q = q.detach().requires_grad_()
    k = k.detach().requires_grad_()
    v = v.detach().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b, c: tiled_attention(
            a, b, c, 0.55, True, key_padding_mask=mask, q_start_pos=1,
            q_block=2, k_block=3,
        ),
        (q, k, v),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_noncontiguous_3d_inputs_and_ragged_tail():
    q, k, v = make_case(batch=3, heads=3, lq=6, lk=8, dhead=4)
    q = q[:, 1].detach().requires_grad_()
    k = k[:, 0].detach().requires_grad_()
    v = v[:, 2].detach().requires_grad_()
    mask = torch.tensor([
        [False, False, False, False, True, True, True, True],
        [False, False, True, True, False, False, True, True],
        [False, False, False, False, False, False, False, False],
    ])
    compare(q, k, v, causal=False, mask=mask, q_block=4, k_block=3, tol=(1e-9, 1e-9))


def test_noncontiguous_4d_inputs():
    q, k, v = make_case(batch=3, heads=4, lq=5, lk=7, dhead=4)
    q = q.permute(1, 0, 2, 3).detach().requires_grad_()
    k = k.permute(1, 0, 2, 3).detach().requires_grad_()
    v = v.permute(1, 0, 2, 3).detach().requires_grad_()
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()
    compare(q, k, v, causal=False, q_block=2, k_block=5, tol=(1e-9, 1e-9))


def test_rows_without_visible_keys_are_zero_without_nan():
    torch.manual_seed(7)
    q = torch.randn(2, 1, 4, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 1, 5, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 1, 5, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([
        [False, False, False, True, True],
        [True, True, True, True, True],
    ])
    out = tiled_attention(q, k, v, 0.4, True, key_padding_mask=mask, q_block=2, k_block=3)
    assert torch.isfinite(out).all()
    assert out[1].abs().eq(0).all()
    out.sum().backward()
    for grad in (q.grad, k.grad, v.grad):
        assert torch.isfinite(grad).all()
    assert q.grad[1].abs().eq(0).all()


def test_positive_infinity_ties_and_masked_infinity_without_nan():
    q = torch.zeros(3, 1, 1, 2, dtype=torch.float64)
    k = torch.zeros(3, 1, 4, 2, dtype=torch.float64)
    v = torch.arange(8, dtype=torch.float64).reshape(1, 1, 4, 2).repeat(3, 1, 1, 1)
    q[0, 0, 0, 0] = torch.inf
    q[1, 0, 0, 1] = torch.inf
    q[2, 0, 0, 0] = torch.inf
    k[:, 0, 0, 0] = 1.0
    k[:, 0, 1, 1] = 1.0
    k[:, 0, 2, 0] = 1.0
    k[:, 0, 3, 1] = 1.0
    mask = torch.tensor([
        [False, True, False, True],
        [True, False, False, True],
        [True, True, True, True],
    ])
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    out = tiled_attention(q, k, v, 1.0, False, key_padding_mask=mask, q_block=2, k_block=2)
    expected = torch.stack([
        torch.tensor([2.0, 2.5], dtype=torch.float64),
        v[0, 0, 1],
        torch.zeros(2, dtype=torch.float64),
    ]).reshape(3, 1, 1, 2)
    torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-10)
    assert torch.isfinite(out).all()
    out.backward(torch.ones_like(out))
    assert all(torch.isfinite(grad).all() for grad in (q.grad, k.grad, v.grad))


def test_extreme_but_finite_logits():
    torch.manual_seed(9)
    q = torch.randn(1, 1, 3, 2, dtype=torch.float64) * 150
    k = torch.randn(1, 1, 4, 2, dtype=torch.float64) * 150
    v = torch.randn(1, 1, 4, 2, dtype=torch.float64)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    out = tiled_attention(q, k, v, 1.0, False, q_block=2, k_block=3)
    ref = dense_attention_reference(q.detach().requires_grad_(False), k, v, 1.0, False)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    out.sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in (q, k, v))


def test_causal_uses_absolute_positions_for_non_square_lengths():
    q = torch.zeros(1, 1, 3, 2, dtype=torch.float64)
    k = torch.zeros(1, 1, 5, 2, dtype=torch.float64)
    v = torch.arange(10, dtype=torch.float64).reshape(1, 1, 5, 2)
    q[0, 0, 0] = torch.tensor([10.0, 0.0])
    q[0, 0, 1] = torch.tensor([0.0, 10.0])
    q[0, 0, 2] = torch.tensor([0.0, 10.0])
    k[0, 0, 0] = torch.tensor([0.0, 0.0])
    k[0, 0, 1] = torch.tensor([0.0, 0.0])
    k[0, 0, 2] = torch.tensor([0.0, 0.0])
    k[0, 0, 3] = torch.tensor([10.0, -10.0])
    k[0, 0, 4] = torch.tensor([0.0, 1.0])
    mask = torch.tensor([[True, True, True, False, False]])
    out = tiled_attention(
        q, k, v, 1.0, True, key_padding_mask=mask, q_start_pos=3, q_block=2, k_block=2
    )
    expected = torch.stack([v[0, 0, 3], v[0, 0, 4], v[0, 0, 4]]).unsqueeze(0).unsqueeze(0)
    torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-10)


def test_saved_tensor_count_reported():
    q, k, v = make_case(lq=2, lk=3, dhead=2)
    tiled_attention(q, k, v, 0.5, True, q_block=2, k_block=2)
    assert last_tiled_attention_stats().saved_tensors == 7
    mask = torch.zeros(1, 3, dtype=torch.bool)
    tiled_attention(q, k, v, 0.5, True, key_padding_mask=mask, q_block=2, k_block=2)
    assert last_tiled_attention_stats().saved_tensors == 8
