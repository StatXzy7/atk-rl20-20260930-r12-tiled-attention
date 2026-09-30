from __future__ import annotations

import argparse
import time

import torch

from tiled_attention import last_tiled_attention_stats, tiled_attention


def measure(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        seconds.append(time.perf_counter() - start)
    return min(seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="64,128,256,512", help="comma-separated query/key lengths")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--q-block", type=int, default=32)
    parser.add_argument("--k-block", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float32
    element_bytes = torch.finfo(torch.float32).bits // 8
    block_score_elements = args.batch * args.heads * args.q_block * args.k_block
    block_score_bytes = block_score_elements * element_bytes

    print(
        "n  pairs      fwd_ms   fwd+bwd_ms  saved_mb  "
        "block_score_mb  est_block_work_mb  q_tiles k_tiles"
    )
    for length in [int(x) for x in args.lengths.split(",")]:
        q = torch.randn(args.batch, args.heads, length, args.dim, dtype=dtype, requires_grad=True)
        k = torch.randn(args.batch, args.heads, length, args.dim, dtype=dtype, requires_grad=True)
        v = torch.randn(args.batch, args.heads, length, args.dim, dtype=dtype, requires_grad=True)

        def run_forward():
            return tiled_attention(
                q, k, v, 1.0 / (args.dim ** 0.5), False,
                q_block=args.q_block, k_block=args.k_block,
            )

        out = run_forward()

        def run_forward_backward():
            q.grad = None
            k.grad = None
            v.grad = None
            output = tiled_attention(
                q, k, v, 1.0 / (args.dim ** 0.5), False,
                q_block=args.q_block, k_block=args.k_block,
            )
            output.backward(torch.ones_like(output), retain_graph=False)

        fwd = measure(run_forward, args.warmup, args.repeats)
        both = measure(run_forward_backward, args.warmup, args.repeats)
        stats = last_tiled_attention_stats()
        saved_tensors = stats.saved_tensors if stats is not None else 0
        leading = args.batch * args.heads
        saved_bytes = (
            3 * leading * length * args.dim
            + leading * length * args.dim
            + 3 * leading * length
        ) * element_bytes
        block_work_bytes = block_score_bytes * 4 + leading * args.q_block * args.dim * element_bytes * 4
        print(
            f"{length:3d} {length * length:9d} "
            f"{1000 * fwd:9.3f} {1000 * both:12.3f}  "
            f"{saved_bytes / 2**20:9.3f} {block_score_bytes / 2**20:14.3f}  "
            f"{block_work_bytes / 2**20:18.3f}  "
            f"{(length + args.q_block - 1) // args.q_block:7d} "
            f"{(length + args.k_block - 1) // args.k_block:7d} saved_tensors={saved_tensors}"
        )


if __name__ == "__main__":
    main()
