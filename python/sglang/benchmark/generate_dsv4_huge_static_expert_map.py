#!/usr/bin/env python3
"""Generate validated static expert placement presets for DSV4 Huge.

The presets in this file are workload-specific.  They must not be selected
automatically: the caller explicitly provides the generated file through
SGLANG_DSV4_STATIC_EXPERT_MAP.
"""

import argparse
from pathlib import Path

import torch


NUM_LAYERS = 43
NUM_EXPERTS = 256

# Validated on four B300 GPUs with cached=16384 and new=4096 per request.
# Cross-rank placement can change FP4 reduction order, so this deliberately
# balances only the final MoE layer and preserves every other layer exactly.
REQ128_V70J_SWAPS = (
    (42, 119, 238),
    (42, 18, 98),
    (42, 14, 74),
    (42, 7, 195),
    (42, 25, 176),
)


def build_mapping(swaps: tuple[tuple[int, int, int], ...]) -> torch.Tensor:
    mapping = torch.arange(NUM_EXPERTS, dtype=torch.int64).repeat(NUM_LAYERS, 1)
    for layer, left, right in swaps:
        mapping[layer, left], mapping[layer, right] = (
            mapping[layer, right].clone(),
            mapping[layer, left].clone(),
        )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("req128_v70j", "identity"),
        default="req128_v70j",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    swaps = REQ128_V70J_SWAPS if args.preset == "req128_v70j" else ()
    mapping = build_mapping(swaps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "physical_to_logical_map": mapping,
            "preset": args.preset,
            "validated_workloads": {
                "cuda_graph": False,
                "tp_size": 4,
                "ep_size": 4,
                "cached_tokens_per_request": 16384,
                "new_tokens_per_request": 4096,
                "correctness_request_counts": (16, 128),
                "recommended_request_counts": (128,),
            },
        },
        args.output,
    )
    print(f"preset={args.preset}")
    print(f"swaps={len(swaps)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
