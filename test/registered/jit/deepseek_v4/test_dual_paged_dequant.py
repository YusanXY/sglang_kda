import pytest
import torch

from sglang.jit_kernel.dsv4 import dual_paged_dequantize_k_cache_bf16
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    dequantize_k_cache_paged,
)


def _cache(page_size: int, num_pages: int) -> torch.Tensor:
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    bytes_per_page *= NOPE_ROPE_BYTES
    return torch.randint(
        0,
        256,
        (num_pages, bytes_per_page),
        dtype=torch.uint8,
        device="cuda",
    )


@pytest.mark.parametrize("compressed_page_size", [64, 2])
def test_dual_paged_dequant_matches_two_triton_launches(
    compressed_page_size: int,
) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA only")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("B200/SM100 or B300/SM103 specialization")

    torch.manual_seed(20260803 + compressed_page_size)
    swa_page_size = 128
    compressed_cache = _cache(compressed_page_size, 19)
    swa_cache = _cache(swa_page_size, 11)
    compressed_ids = torch.randint(
        0,
        compressed_page_size * compressed_cache.shape[0],
        (257,),
        dtype=torch.int32,
        device="cuda",
    )
    swa_ids = torch.randint(
        0,
        swa_page_size * swa_cache.shape[0],
        (131,),
        dtype=torch.int32,
        device="cuda",
    )

    expected_a = dequantize_k_cache_paged(
        compressed_cache, compressed_ids, compressed_page_size
    )
    expected_b = dequantize_k_cache_paged(swa_cache, swa_ids, swa_page_size)
    expected = torch.cat((expected_a, expected_b))
    output = torch.empty_like(expected)
    actual = dual_paged_dequantize_k_cache_bf16(
        compressed_cache,
        compressed_ids,
        compressed_page_size,
        swa_cache,
        swa_ids,
        swa_page_size,
        output,
    )
    output_ptr = actual.data_ptr()
    actual = dual_paged_dequantize_k_cache_bf16(
        compressed_cache,
        compressed_ids,
        compressed_page_size,
        swa_cache,
        swa_ids,
        swa_page_size,
        output,
    )
    torch.cuda.synchronize()

    assert actual.data_ptr() == output_ptr == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
