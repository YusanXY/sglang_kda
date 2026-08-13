#!/usr/bin/env python3
"""Sweep V2 one-shot all-reduce CUDA-Graph latency for GLM-5.2 decode."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from sglang.jit_kernel.all_reduce import AllReduceAlgo
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    get_world_group,
    init_distributed_environment,
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
        local_rank=rank,
        backend="nccl",
    )
    group = get_world_group().cpu_group
    device = torch.device("cuda", rank)
    comm = CustomAllReduceV2(
        group=group,
        device=device,
        max_pull_size=0,
        max_push_size=1024 * 1024,
        max_pull_blocks=0,
        max_push_blocks=args.blocks,
    )
    if comm.disabled:
        raise RuntimeError("CustomAllReduceV2 is disabled")
    comm.override_algo = AllReduceAlgo.ONE_SHOT_PUSH

    # Every DP rank holds exactly eight requests in the Req64/DP8 workload.
    # Use rank-distinct values so a stale/non-reducing kernel cannot pass.
    x = torch.full(
        (args.rows, args.hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    with comm.capture():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = comm.custom_all_reduce(x)
    graph.replay()
    torch.cuda.synchronize()
    expected = sum(range(1, dist.get_world_size() + 1))
    torch.testing.assert_close(
        y.float(), torch.full_like(y.float(), expected), rtol=0, atol=0
    )

    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
    for i in range(args.repeats):
        begin[i].record()
        graph.replay()
        end[i].record()
    torch.cuda.synchronize()
    us = [a.elapsed_time(b) * 1000 for a, b in zip(begin, end)]
    local = {
        "rank": rank,
        "blocks": args.blocks,
        "bytes": x.numel() * x.element_size(),
        "median_us": statistics.median(us),
        "p90_us": percentile(us, 0.90),
        "p99_us": percentile(us, 0.99),
        "mean_us": statistics.fmean(us),
    }
    rows = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local, rows, dst=0)
    if rank == 0:
        assert rows is not None
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "blocks": args.blocks,
                    "bytes_per_rank": local["bytes"],
                    "rank_median_us": [row["median_us"] for row in rows],
                    "rank_median_us_max": max(row["median_us"] for row in rows),
                    "rank_p99_us_max": max(row["p99_us"] for row in rows),
                    "ranks": rows,
                },
                sort_keys=True,
            )
        )
    comm.close()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
