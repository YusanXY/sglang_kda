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
def test_topk_sparse_epilogue_matches_standalone_combiner(version):
    device = torch.device("cuda")
    torch.manual_seed(7)

    query_lens = (3, 5)
    query_start_loc = torch.tensor((0, 3, 8), dtype=torch.int32, device=device)
    positions = torch.tensor(
        (2048, 2049, 2050, 4094, 4095, 4096, 4097, 4098),
        dtype=torch.int32,
        device=device,
    )
    compressed_seq_lens = ((positions + 1) // 4).to(torch.int32)
    max_compressed_len = int(compressed_seq_lens.max().item())
    scores = torch.randn(
        len(positions), max_compressed_len, dtype=torch.float32, device=device
    )

    page_size = 256
    num_pages = (max_compressed_len + page_size - 1) // page_size
    per_req_pages = torch.tensor(
        ([13, 29, 47, 61, 79], [101, 127, 149, 167, 191]),
        dtype=torch.int32,
        device=device,
    )[:, :num_pages]
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

    full_seq_lens = torch.tensor((2051, 4099), dtype=torch.int32, device=device)
    gather_lens = torch.tensor((130, 132), dtype=torch.int32, device=device)
    c4_max = max_compressed_len
    compressed_base = torch.tensor((0, c4_max), dtype=torch.int32, device=device)
    swa_base = torch.tensor(
        (2 * c4_max, 2 * c4_max + 130), dtype=torch.int32, device=device
    )
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
