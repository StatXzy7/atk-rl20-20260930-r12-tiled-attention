import argparse
import statistics
import time

import torch

from tiled_attention import tiled_attention


def timed_run(q, k, v, block_size, repeats, backward):
    for _ in range(5):
        with torch.no_grad():
            output = tiled_attention(q, k, v, causal=True, block_size=block_size)
        if backward:
            q.grad = k.grad = v.grad = None
            output = tiled_attention(q, k, v, causal=True, block_size=block_size)
            output.sum().backward()

    times = []
    for _ in range(repeats):
        if backward:
            q.grad = k.grad = v.grad = None
            start = time.perf_counter()
            output = tiled_attention(q, k, v, causal=True, block_size=block_size)
            output.sum().backward()
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            with torch.no_grad():
                output = tiled_attention(q, k, v, causal=True, block_size=block_size)
            end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return statistics.median(times), min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    bytes_per_float = torch.finfo(torch.float32).bits // 8

    print(
        "| Lq | Lk | Br | Bc | forward median/min ms | fwd+bwd median/min ms | "
        "saved MiB | algorithmic tile peak MiB |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for length in args.lengths:
        q = torch.randn(
            args.batch,
            args.heads,
            length,
            args.dim,
            dtype=torch.float32,
            requires_grad=True,
        )
        k = torch.randn(
            args.batch,
            args.heads,
            length,
            args.dim,
            dtype=torch.float32,
            requires_grad=True,
        )
        v = torch.randn(
            args.batch,
            args.heads,
            length,
            args.dim,
            dtype=torch.float32,
            requires_grad=True,
        )
        rows = args.batch * args.heads * length
        saved_elements = (
            rows * args.dim * 3
            + rows * args.dim
            + rows * 2
        )
        saved_mib = saved_elements * bytes_per_float / 2**20

        for block_size in args.block_sizes:
            forward_ms, forward_min = timed_run(
                q, k, v, block_size, args.repeats, False
            )
            backward_ms, backward_min = timed_run(
                q, k, v, block_size, args.repeats, True
            )
            br = min(block_size, length)
            bc = min(block_size, length)
            tile_elements = args.batch * args.heads * (
                br * bc * 3
                + br * args.dim * 3
                + bc * args.dim * 3
            )
            tile_mib = tile_elements * bytes_per_float / 2**20
            print(
                f"| {length} | {length} | {br} | {bc} | "
                f"{forward_ms:.2f}/{forward_min:.2f} | "
                f"{backward_ms:.2f}/{backward_min:.2f} | "
                f"{saved_mib:.2f} | {tile_mib:.2f} |"
            )


if __name__ == "__main__":
    main()
