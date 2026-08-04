import pytest
import torch

from sglang.jit_kernel.dsv4 import (
    plan_topk_v2,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    combine_topk_swa_indices,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="DSV4 fused top-k sparse epilogue requires Blackwell CUDA",
)
@pytest.mark.parametrize("version", ("v1", "v2"))
@pytest.mark.parametrize("num_reqs", (2, 128))
def test_topk_sparse_epilogue_matches_standalone_combiner(version, num_reqs):
    device = torch.device("cuda")
    torch.manual_seed(7)

    if num_reqs == 2:
        query_lens = (3, 5)
        query_start_loc = torch.tensor(
            (0, 3, 8), dtype=torch.int32, device=device
        )
        positions = torch.tensor(
            (2048, 2049, 2050, 4094, 4095, 4096, 4097, 4098),
            dtype=torch.int32,
            device=device,
        )
    else:
        query_lens = (1,) * num_reqs
        query_start_loc = torch.arange(
            num_reqs + 1, dtype=torch.int32, device=device
        )
        positions = torch.arange(
            2048, 2048 + num_reqs, dtype=torch.int32, device=device
        )
    compressed_seq_lens = ((positions + 1) // 4).to(torch.int32)
    max_compressed_len = int(compressed_seq_lens.max().item())
    scores = torch.randn(
        len(positions), max_compressed_len, dtype=torch.float32, device=device
    )

    page_size = 256
    num_pages = (max_compressed_len + page_size - 1) // page_size
    per_req_pages = (
        torch.arange(
            num_reqs * num_pages, dtype=torch.int32, device=device
        ).view(num_reqs, num_pages)
        * 17
        + 13
    )
    page_table = per_req_pages.repeat_interleave(
        torch.tensor(query_lens, device=device), dim=0
    ).contiguous()

    page_ref = torch.empty((len(positions), 512), dtype=torch.int32, device=device)
    raw_ref = torch.empty_like(page_ref)
    metadata = plan_topk_v2(compressed_seq_lens) if version == "v2" else None

    def transform(page_indices, raw_indices, **kwargs):
        if version == "v2":
            topk_transform_512_v2(
                scores,
                compressed_seq_lens,
                page_table,
                page_indices,
                page_size,
                metadata,
                raw_indices,
                **kwargs,
            )
        else:
            topk_transform_512(
                scores,
                compressed_seq_lens,
                page_table,
                page_indices,
                page_size,
                raw_indices,
                **kwargs,
            )

    transform(page_ref, raw_ref)

    last_query = query_start_loc[1:].to(torch.int64) - 1
    full_seq_lens = positions.index_select(0, last_query) + 1
    gather_lens = torch.minimum(full_seq_lens, torch.full_like(full_seq_lens, 132))
    c4_max = max_compressed_len
    compressed_base = (
        torch.arange(num_reqs, dtype=torch.int32, device=device) * c4_max
    )
    swa_offsets = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    swa_offsets[1:] = torch.cumsum(gather_lens[:-1], dim=0)
    swa_base = num_reqs * c4_max + swa_offsets
    page_out = torch.empty_like(page_ref)
    raw_out = torch.empty_like(raw_ref)
    combined_out = torch.full(
        (len(positions), 640), -1, dtype=torch.int32, device=device
    )
    lens_out = torch.empty(len(positions), dtype=torch.int32, device=device)
    transform(
        page_out,
        raw_out,
        positions=positions,
        query_start_loc=query_start_loc,
        full_seq_lens=full_seq_lens,
        swa_gather_lens=gather_lens,
        compressed_base=compressed_base,
        swa_base=swa_base,
        combined_indices=combined_out,
        combined_lens=lens_out,
    )
    torch.cuda.synchronize()

    # Radix selection uses atomic output slots, so independent launches do not
    # promise an ordering for the selected set.  Attention is order-invariant.
    torch.testing.assert_close(
        page_out.sort(dim=1).values, page_ref.sort(dim=1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        raw_out.sort(dim=1).values, raw_ref.sort(dim=1).values, rtol=0, atol=0
    )

    # Compare the epilogue against the standalone kernel using this same
    # launch's raw ordering; every combined-index position must match exactly.
    combined_ref, lens_ref = combine_topk_swa_indices(
        topk_indices=raw_out,
        query_start_loc=query_start_loc,
        seq_lens=full_seq_lens,
        gather_lens=gather_lens,
        compressed_base=compressed_base,
        swa_base=swa_base,
        window_size=128,
        compress_ratio=4,
        topk=512,
    )
    torch.testing.assert_close(lens_out, lens_ref, rtol=0, atol=0)
    torch.testing.assert_close(combined_out, combined_ref, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="DSV4 Huge bitset top-k requires Blackwell CUDA",
)
def test_topk_v2_huge_bitset_range_is_exact_sorted_and_deterministic():
    """Cover the K=512, Register5-range deterministic bitset compactor."""

    device = torch.device("cuda")
    torch.manual_seed(29)
    batch = 8
    width = 17408
    scores = torch.randn((batch, width), dtype=torch.float32, device=device)
    seq_lens = torch.tensor(
        (16385, 16512, 16640, 16896, 17024, 17152, 17280, 17408),
        dtype=torch.int32,
        device=device,
    )
    page_size = 64
    num_pages = (width + page_size - 1) // page_size
    page_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).expand(batch, -1).contiguous()
    page_indices = torch.full(
        (batch, 512), -1, dtype=torch.int32, device=device
    )
    raw_indices = torch.full_like(page_indices, -1)
    metadata = plan_topk_v2(seq_lens)

    topk_transform_512_v2(
        scores,
        seq_lens,
        page_table,
        page_indices,
        page_size,
        metadata,
        raw_indices,
    )
    torch.cuda.synchronize()

    reference = torch.stack(
        [
            torch.topk(scores[row, : int(seq_lens[row])], 512, sorted=False)
            .indices.sort()
            .values.to(torch.int32)
            for row in range(batch)
        ]
    )
    torch.testing.assert_close(raw_indices, reference, rtol=0, atol=0)
    # With an identity page table the transformed slots equal the raw slots.
    torch.testing.assert_close(page_indices, raw_indices, rtol=0, atol=0)

    first_raw = raw_indices.clone()
    first_pages = page_indices.clone()
    for _ in range(8):
        topk_transform_512_v2(
            scores,
            seq_lens,
            page_table,
            page_indices,
            page_size,
            metadata,
            raw_indices,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(raw_indices, first_raw, rtol=0, atol=0)
        torch.testing.assert_close(page_indices, first_pages, rtol=0, atol=0)

    # Tie-heavy inputs exercise the uniqueness precondition of the bitset
    # compactor. The selected set may differ from torch.topk's arbitrary tie
    # choice and from another launch, but every launch must contain exactly 512
    # unique, in-range sorted slots.
    scores.zero_()
    topk_transform_512_v2(
        scores,
        seq_lens,
        page_table,
        page_indices,
        page_size,
        metadata,
        raw_indices,
    )
    torch.cuda.synchronize()
    assert torch.all(raw_indices[:, 1:] > raw_indices[:, :-1])
    assert torch.all(raw_indices >= 0)
    assert torch.all(raw_indices < seq_lens[:, None])
    torch.testing.assert_close(page_indices, raw_indices, rtol=0, atol=0)

    for _ in range(8):
        topk_transform_512_v2(
            scores,
            seq_lens,
            page_table,
            page_indices,
            page_size,
            metadata,
            raw_indices,
        )
        torch.cuda.synchronize()
        assert torch.all(raw_indices[:, 1:] > raw_indices[:, :-1])
        assert torch.all(raw_indices >= 0)
        assert torch.all(raw_indices < seq_lens[:, None])
        torch.testing.assert_close(page_indices, raw_indices, rtol=0, atol=0)
