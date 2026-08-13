#!/usr/bin/env python3
"""Byte-level check for fused Top-K physical remap and FlashInfer packing."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_gpu,
    biased_topk_jit_kernel_impl,
)


@triton.jit
def _unpack_kernel(packed, ids, weight_bits, n: tl.constexpr):
    offsets = tl.arange(0, n)
    value = tl.load(packed + offsets)
    tl.store(ids + offsets, value >> 16)
    tl.store(weight_bits + offsets, value & 0xFFFF)


def main() -> None:
    torch.manual_seed(4199)
    for rows in (8, 2048):
        scores = torch.randn(rows, 256, device="cuda", dtype=torch.float32)
        bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.01
        hidden = torch.empty(rows, 6144, device="cuda", dtype=torch.bfloat16)
        # Exercise a non-identity physical placement, not only the trivial map.
        physical_map = torch.randperm(256, device="cuda", dtype=torch.int64)
        info = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=physical_map,
            partial_logical_to_all_physical_map=physical_map[:, None],
            partial_logical_to_all_physical_map_num_valid=torch.ones(
                256, device="cuda", dtype=torch.int64
            ),
            num_physical_experts=256,
        )
        for apply_scale in (False, True):
            ref_w, logical_ids = biased_topk_jit_kernel_impl(
                hidden,
                scores,
                bias,
                topk=8,
                renormalize=True,
                scoring_func="sigmoid",
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=apply_scale,
            )
            ref_ids = physical_map[logical_ids].to(torch.int32)
            reference = PackTopkIds.execute(ref_ids.contiguous(), ref_w.contiguous())

            fused = torch.empty_like(reference)
            got_w, got_ids = biased_topk_jit_kernel_impl(
                hidden,
                scores,
                bias,
                topk=8,
                renormalize=True,
                scoring_func="sigmoid",
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=apply_scale,
                expert_location_dispatch_info=info,
                packed_out=fused,
            )
            torch.cuda.synchronize()
            assert torch.equal(got_ids, ref_ids), (rows, apply_scale, "ids")
            assert torch.equal(got_w, ref_w), (rows, apply_scale, "weights")
            assert torch.equal(fused, reference), (rows, apply_scale, "packed")
            unpacked_ids = torch.empty_like(got_ids)
            unpacked_weight_bits = torch.empty_like(got_ids)
            _unpack_kernel[(1,)](
                fused,
                unpacked_ids,
                unpacked_weight_bits,
                n=fused.numel(),
            )
            expected_weight_bits = (
                got_w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
            )
            assert torch.equal(unpacked_ids, got_ids), (rows, apply_scale, "unpack ids")
            assert torch.equal(unpacked_weight_bits, expected_weight_bits), (
                rows,
                apply_scale,
                "unpack weights",
            )
            print(f"PASS rows={rows} apply_scale={apply_scale}")

        # GLM-5.2 uses the biased_grouped_topk API with n_group=topk_group=1.
        # Also retain a genuinely grouped case to cover the generic helper.
        for num_groups, topk_groups in ((1, 1), (8, 4)):
            grouped_ref_w, grouped_ref_ids = biased_grouped_topk_gpu(
                hidden,
                scores,
                bias,
                topk=8,
                renormalize=True,
                num_expert_group=num_groups,
                topk_group=topk_groups,
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=True,
            )
            grouped_packed = torch.empty_like(grouped_ref_ids)
            grouped_w, grouped_ids = biased_grouped_topk_gpu(
                hidden,
                scores,
                bias,
                topk=8,
                renormalize=True,
                num_expert_group=num_groups,
                topk_group=topk_groups,
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=True,
                packed_out=grouped_packed,
            )
            grouped_reference = PackTopkIds.execute(
                grouped_ref_ids.contiguous(), grouped_ref_w.contiguous()
            )
            grouped_self_reference = PackTopkIds.execute(
                grouped_ids.contiguous(), grouped_w.contiguous()
            )
            torch.cuda.synchronize()
            abs_error = (grouped_w - grouped_ref_w).abs()
            id_diff = (grouped_ids != grouped_ref_ids).sum().item()
            packed_diff = (grouped_packed != grouped_reference).sum().item()
            print(
                "grouped diagnostics "
                f"rows={rows} groups={num_groups}/{topk_groups} "
                f"max_abs={abs_error.max().item():.9g} "
                f"mean_abs={abs_error.mean().item():.9g} "
                f"id_diff={id_diff}/{grouped_ids.numel()} "
                f"packed_diff={packed_diff}/{grouped_packed.numel()}"
            )
            # FlashInfer and Triton use slightly different sigmoid/reduction rounding.
            # Random scores can therefore flip a near-boundary expert, but the fused
            # carrier must always be byte-identical to separately packing *its own*
            # IDs and weights. End-to-end token/logprob validation is the acceptance
            # gate for the router substitution.
            assert id_diff / grouped_ids.numel() <= 1e-3, (
                rows,
                num_groups,
                "grouped id drift",
            )
            assert abs_error.mean().item() <= 2e-6, (
                rows,
                num_groups,
                "grouped mean weight drift",
            )
            assert torch.equal(grouped_packed, grouped_self_reference), (
                rows,
                num_groups,
                "grouped packed carrier",
            )
            print(f"PASS grouped rows={rows} groups={num_groups}/{topk_groups}")


if __name__ == "__main__":
    main()
