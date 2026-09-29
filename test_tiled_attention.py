import unittest

import torch

from dense_reference import dense_attention_reference
from tiled_attention import tiled_attention


class TiledAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def assertAgainstDense(
        self,
        q,
        k,
        v,
        scale,
        causal=False,
        key_padding_mask=None,
        block_size=3,
        atol=1e-10,
        rtol=1e-8,
    ):
        actual = tiled_attention(
            q,
            k,
            v,
            scale,
            causal=causal,
            key_padding_mask=key_padding_mask,
            block_size=block_size,
        )
        expected = dense_attention_reference(
            q,
            k,
            v,
            scale,
            causal=causal,
            key_padding_mask=key_padding_mask,
        )
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        self.assertTrue(torch.isfinite(actual).all())

    def test_masks_rectangular_lengths_and_tail_blocks(self):
        q = torch.randn(2, 3, 7, 5, dtype=torch.float64)
        k = torch.randn(2, 3, 11, 5, dtype=torch.float64)
        v = torch.randn(2, 3, 11, 4, dtype=torch.float64)
        scale = 0.21
        padding = torch.tensor(
            [
                [False, False, False, False, True, True] + [True] * 5,
                [False, False, False, True, True] + [True] * 6,
            ]
        )

        for causal in (False, True):
            for mask in (None, padding):
                for block_size in (1, 3, 16):
                    with self.subTest(
                        causal=causal, mask=mask is not None, block_size=block_size
                    ):
                        self.assertAgainstDense(
                            q, k, v, scale, causal, mask, block_size
                        )

    def test_head_specific_padding_and_noncontiguous_inputs(self):
        q = torch.randn(2, 4, 6, 8, dtype=torch.float64).permute(0, 1, 3, 2)
        k = torch.randn(2, 4, 6, 13, dtype=torch.float64).permute(0, 1, 3, 2)
        v = torch.randn(2, 4, 5, 13, dtype=torch.float64).permute(0, 1, 3, 2)

        mask = (torch.rand(4, 2, 13) < 0.22).permute(1, 0, 2)
        mask[:, :, 0] = False
        self.assertFalse(q.is_contiguous())
        self.assertFalse(k.is_contiguous())
        self.assertFalse(v.is_contiguous())
        self.assertAgainstDense(q, k, v, 0.17, True, mask, block_size=5)

    def test_causal_uses_global_positions_not_square_tril(self):
        q = torch.randn(1, 1, 5, 3, dtype=torch.float64)
        k = torch.randn(1, 1, 8, 3, dtype=torch.float64)
        v = torch.randn(1, 1, 8, 3, dtype=torch.float64)
        out = tiled_attention(q, k, v, scale=0.3, causal=True, block_size=2)

        expected_first_rows = []
        for query_index in range(5):
            visible_scores = (q[0, 0, query_index] @ k[0, 0].T) * 0.3
            visible_scores[query_index + 1 :] = float("-inf")
            expected_first_rows.append(
                torch.softmax(visible_scores, dim=-1) @ v[0, 0]
            )
        torch.testing.assert_close(
            out[0, 0], torch.stack(expected_first_rows), atol=1e-11, rtol=1e-10
        )

    def test_rows_without_visible_keys_are_exact_zero_and_have_zero_grads(self):
        q = torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True)
        k0 = torch.randn(1, 2, 6, 3, dtype=torch.float64)
        v0 = torch.randn(1, 2, 6, 3, dtype=torch.float64)
        k = k0.detach().clone().requires_grad_(True)
        v = v0.detach().clone().requires_grad_(True)
        mask = torch.zeros(1, 2, 6, dtype=torch.bool)
        mask[0, 1, :] = True

        output = tiled_attention(
            q, k, v, scale=0.4, causal=True, key_padding_mask=mask, block_size=3
        )
        self.assertTrue(torch.equal(output[0, 1], torch.zeros(4, 3)))
        output.sum().backward()

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertTrue(torch.equal(q.grad[0, 1], torch.zeros_like(q.grad[0, 1])))
        self.assertTrue(torch.equal(k.grad[0, 1], torch.zeros_like(k.grad[0, 1])))
        self.assertTrue(torch.equal(v.grad[0, 1], torch.zeros_like(v.grad[0, 1])))

    def test_zero_key_length(self):
        q = torch.randn(2, 2, 3, 4, dtype=torch.float64)
        k = torch.randn(2, 2, 0, 4, dtype=torch.float64)
        v = torch.randn(2, 2, 0, 4, dtype=torch.float64)
        out = tiled_attention(q, k, v, scale=0.5, block_size=4)
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_empty_first_key_block_then_visible_extreme_negative_logit(self):
        q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
        k = torch.tensor(
            [[[[-1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float64
        )
        v = torch.tensor([[[[9.0], [4.0]]]], dtype=torch.float64)
        mask = torch.tensor([[True, False]])

        output = tiled_attention(
            q, k, v, scale=10000.0, key_padding_mask=mask, block_size=1
        )
        torch.testing.assert_close(
            output, torch.full_like(output, 4.0), atol=0.0, rtol=0.0
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_extreme_logits(self):
        base_q = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
        base_k = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0],
            ],
            dtype=torch.float64,
        )
        q = base_q.view(1, 1, 1, 4).clone().requires_grad_(True)
        k = base_k.view(1, 1, 3, 4).clone().requires_grad_(True)
        v = torch.tensor(
            [[[[1.0], [2.0], [3.0]]]], dtype=torch.float64
        ).requires_grad_(True)

        out = tiled_attention(q, k, v, scale=2500.0, block_size=2)
        torch.testing.assert_close(out, torch.ones_like(out))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        for gradient in (q.grad, k.grad, v.grad):
            self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(q.grad, torch.zeros_like(q.grad), atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(k.grad, torch.zeros_like(k.grad), atol=1e-9, rtol=1e-9)
        expected_v_grad = torch.zeros_like(v.grad)
        expected_v_grad[0, 0, 0, 0] = 1.0
        torch.testing.assert_close(v.grad, expected_v_grad, atol=1e-9, rtol=1e-9)

    def test_extreme_logits_at_masked_and_causal_hidden_positions(self):
        q = torch.tensor(
            [[[[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float64
        ).requires_grad_(True)
        k = torch.tensor(
            [[[[1.0, -1.0], [-1.0, 1.0]]]], dtype=torch.float64
        ).requires_grad_(True)
        v = torch.tensor(
            [[[[7.0], [2.0]]]], dtype=torch.float64
        ).requires_grad_(True)
        padding = torch.tensor([[False, True]])

        output = tiled_attention(
            q,
            k,
            v,
            scale=10000.0,
            causal=True,
            key_padding_mask=padding,
            block_size=1,
        )
        expected = torch.tensor(
            [[[[7.0], [7.0], [7.0]]]], dtype=torch.float64
        )
        torch.testing.assert_close(output, expected, atol=1e-12, rtol=1e-12)
        output.sum().backward()
        for gradient in (q.grad, k.grad, v.grad):
            self.assertTrue(torch.isfinite(gradient).all())

    def test_dense_gradient_comparison(self):
        q0 = torch.randn(2, 2, 6, 4, dtype=torch.float64)
        k0 = torch.randn(2, 2, 9, 4, dtype=torch.float64)
        v0 = torch.randn(2, 2, 9, 5, dtype=torch.float64)
        mask = torch.tensor(
            [
                [False, False, False, True, True] + [True] * 4,
                [False, False, True, True, True] + [True] * 4,
            ]
        )
        grad_out = torch.randn(2, 2, 6, 5, dtype=torch.float64)

        def run(op, q, k, v):
            return op(q, k, v, 0.31, causal=True, key_padding_mask=mask)

        q_a = q0.clone().requires_grad_(True)
        k_a = k0.clone().requires_grad_(True)
        v_a = v0.clone().requires_grad_(True)
        output_a = run(tiled_attention, q_a, k_a, v_a)
        output_a.backward(grad_out)

        q_b = q0.clone().requires_grad_(True)
        k_b = k0.clone().requires_grad_(True)
        v_b = v0.clone().requires_grad_(True)
        output_b = run(dense_attention_reference, q_b, k_b, v_b)
        output_b.backward(grad_out)

        torch.testing.assert_close(output_a, output_b, atol=1e-10, rtol=1e-9)
        for actual, expected in (
            (q_a.grad, q_b.grad),
            (k_a.grad, k_b.grad),
            (v_a.grad, v_b.grad),
        ):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)

    def test_double_gradcheck_rectangular_causal_noncontiguous(self):
        q = torch.randn(1, 2, 3, 4, dtype=torch.float64).permute(0, 1, 3, 2)
        q = q.detach().requires_grad_(True)
        k = torch.randn(1, 2, 3, 7, dtype=torch.float64).permute(0, 1, 3, 2)
        k = k.detach().requires_grad_(True)
        v = torch.randn(1, 2, 2, 7, dtype=torch.float64).permute(0, 1, 3, 2)
        v = v.detach().requires_grad_(True)
        mask = torch.tensor(
            [[False, False, False, False, True, True, True]]
        ).expand(1, 7)
        mask = mask.transpose(0, 1).transpose(0, 1)

        self.assertTrue(torch.autograd.gradcheck(
            lambda q, k, v: tiled_attention(
                q, k, v, 0.5, causal=True, key_padding_mask=mask, block_size=3
            ),
            (q, k, v),
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
        ))

    def test_double_gradcheck_with_all_masked_row(self):
        q = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, True]])

        def function(q, k, v):
            return tiled_attention(
                q, k, v, 0.7, causal=False, key_padding_mask=mask, block_size=2
            )

        self.assertTrue(torch.autograd.gradcheck(function, (q, k, v), eps=1e-6))


if __name__ == "__main__":
    unittest.main()
