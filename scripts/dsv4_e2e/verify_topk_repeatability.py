"""Verify deterministic DSV4 C4 top-k at aggregate ForwardBatch M=4096.

Use (requests, query-per-request, raw-prefix) of (1, 4096, 65536),
(16, 256, 16384), or (128, 32, 16384) for the supported benchmark cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--requests", type=int, choices=(1, 16, 128), required=True)
    parser.add_argument("--query-per-request", type=int, required=True)
    parser.add_argument("--raw-prefix", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo / "python"))
    from sglang.jit_kernel.dsv4 import plan_topk_v2, topk_transform_512_v2

    torch.cuda.set_device(0)
    torch.manual_seed(20260804)
    reqs = args.requests
    query_per_request = args.query_per_request
    total_q = reqs * query_per_request
    if total_q != 4096:
        raise ValueError(f"aggregate M must be 4096, got {total_q}")

    positions = torch.arange(
        args.raw_prefix,
        args.raw_prefix + query_per_request,
        dtype=torch.int32,
        device="cuda",
    ).repeat(reqs)
    query_start_loc = torch.arange(
        0, total_q + 1, query_per_request, dtype=torch.int32, device="cuda"
    )
    full_seq_lens = torch.full(
        (reqs,),
        args.raw_prefix + query_per_request,
        dtype=torch.int32,
        device="cuda",
    )
    compressed_seq_lens = ((positions + 1) // 4).to(torch.int32)
    max_compressed_len = (args.raw_prefix + query_per_request + 3) // 4
    score_stride = (max_compressed_len + 3) // 4 * 4
    scores = torch.randn(
        total_q, score_stride, dtype=torch.float32, device="cuda"
    )

    page_size = 256
    pages_per_req = (max_compressed_len + page_size - 1) // page_size
    request_pages = torch.arange(
        reqs * pages_per_req, dtype=torch.int32, device="cuda"
    ).view(reqs, pages_per_req)
    page_table = request_pages.repeat_interleave(query_per_request, dim=0).contiguous()
    page_indices = torch.empty((total_q, 512), dtype=torch.int32, device="cuda")
    raw_indices = torch.empty_like(page_indices)
    combined_indices = torch.empty((total_q, 640), dtype=torch.int32, device="cuda")
    combined_lens = torch.empty(total_q, dtype=torch.int32, device="cuda")
    gather_lens = torch.full((reqs,), 128, dtype=torch.int32, device="cuda")
    compressed_base = (
        torch.arange(reqs, dtype=torch.int32, device="cuda")
        * max_compressed_len
    )
    swa_base = reqs * max_compressed_len + torch.arange(
        reqs, dtype=torch.int32, device="cuda"
    ) * 128
    metadata = plan_topk_v2(compressed_seq_lens)

    def launch() -> None:
        topk_transform_512_v2(
            scores,
            compressed_seq_lens,
            page_table,
            page_indices,
            page_size,
            metadata,
            raw_indices,
            positions=positions,
            query_start_loc=query_start_loc,
            full_seq_lens=full_seq_lens,
            swa_gather_lens=gather_lens,
            compressed_base=compressed_base,
            swa_base=swa_base,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
        )

    launch()
    torch.cuda.synchronize()
    baseline_raw = raw_indices.clone()
    baseline_page = page_indices.clone()
    baseline_combined = combined_indices.clone()
    raw_mismatches = []
    page_mismatches = []
    combined_mismatches = []
    for _ in range(args.iterations):
        launch()
        torch.cuda.synchronize()
        raw_mismatches.append(int((raw_indices != baseline_raw).sum().item()))
        page_mismatches.append(int((page_indices != baseline_page).sum().item()))
        combined_mismatches.append(
            int((combined_indices != baseline_combined).sum().item())
        )

    sorted_set_equal = torch.equal(
        torch.sort(raw_indices, dim=1).values,
        torch.sort(baseline_raw, dim=1).values,
    )
    valid_lens = torch.clamp(compressed_seq_lens, min=0, max=512)
    adjacent_valid = (
        torch.arange(511, device="cuda")[None, :]
        < (valid_lens - 1).clamp_min(0)[:, None]
    )
    output_is_sorted = not bool(
        torch.any(
            (raw_indices[:, 1:] < raw_indices[:, :-1]) & adjacent_valid
        ).item()
    )
    reference_rows = []
    for row in sorted({0, total_q // 2, total_q - 1}):
        seq_len = int(compressed_seq_lens[row].item())
        if seq_len <= 512:
            continue
        expected = torch.topk(
            scores[row, :seq_len], 512, sorted=False
        ).indices.sort().values.to(torch.int32)
        reference_rows.append(
            {
                "row": row,
                "seq_len": seq_len,
                "set_equal": torch.equal(raw_indices[row], expected),
            }
        )
    passed = (
        max(raw_mismatches) == 0
        and max(page_mismatches) == 0
        and max(combined_mismatches) == 0
        and sorted_set_equal
        and output_is_sorted
        and bool(reference_rows)
        and all(row["set_equal"] for row in reference_rows)
    )
    print(
        json.dumps(
            {
                "status": "PASS" if passed else "FAILED",
                "requests": reqs,
                "query_per_request": query_per_request,
                "raw_prefix": args.raw_prefix,
                "aggregate_m": total_q,
                "max_compressed_len": max_compressed_len,
                "score_stride": score_stride,
                "iterations": args.iterations,
                "raw_exact_every_run": max(raw_mismatches) == 0,
                "raw_mismatch_min": min(raw_mismatches),
                "raw_mismatch_max": max(raw_mismatches),
                "page_mismatch_max": max(page_mismatches),
                "combined_mismatch_max": max(combined_mismatches),
                "selected_set_equal_after_sort": sorted_set_equal,
                "output_is_sorted": output_is_sorted,
                "reference_rows": reference_rows,
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
