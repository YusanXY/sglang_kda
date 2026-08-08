import os
import sys

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as symm_mem

from sglang.jit_kernel.dsv4 import (
    fused_q_norm_rope,
    fused_q_norm_rope_tp4_bulk_route,
)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    heads, head_dim = 16, 512
    torch.manual_seed(20260809 + rank)
    q_input = torch.randn(
        (tokens, heads, head_dim), dtype=torch.bfloat16, device=device
    )
    reference_local = torch.empty_like(q_input)
    reference_recv = torch.empty_like(q_input)
    bulk_local = torch.empty_like(q_input)
    routed_recv = symm_mem.empty(
        (131072, heads, head_dim), dtype=torch.bfloat16, device=device
    )
    handle = symm_mem.rendezvous(routed_recv, dist.group.WORLD)
    peers = tuple(
        handle.get_buffer(peer, routed_recv.shape, routed_recv.dtype)
        for peer in range(4)
    )
    freqs = torch.ones((1, 32), dtype=torch.complex64, device=device)
    positions = torch.zeros((tokens,), dtype=torch.int64, device=device)

    fused_q_norm_rope(q_input, reference_local, 1.0e-6, freqs, positions)
    dist.all_to_all_single(reference_recv, reference_local)
    fused_q_norm_rope_tp4_bulk_route(
        q_input, bulk_local, peers, rank, 1.0e-6, freqs, positions
    )
    handle.barrier(channel=0)
    torch.cuda.synchronize()

    routed = routed_recv[:tokens]
    equal = torch.equal(routed, reference_recv)
    max_diff = (routed.float() - reference_recv.float()).abs().max()
    equal_tensor = torch.tensor(int(equal), device=device)
    dist.all_reduce(equal_tensor, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            f"tp4_q_route_equal={bool(equal_tensor.item())} "
            f"rank0_max_diff={max_diff.item()}"
        )
    if not bool(equal_tensor.item()):
        raise RuntimeError("fused TP4 Q route differs from NCCL reference")


if __name__ == "__main__":
    main()
