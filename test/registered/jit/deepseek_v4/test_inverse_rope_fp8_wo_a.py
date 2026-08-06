import pytest
import torch

from sglang.jit_kernel.dsv4 import (
    fused_rope_inplace,
    inverse_rope_fp8_wo_a_ue8m0,
    sglang_per_token_group_quant_fp8_dsv4_wo_a,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=90,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
)


def _freqs(max_position: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(20260803)
    angles = torch.randn(max_position, 32, device=device, dtype=torch.float32)
    return torch.polar(torch.ones_like(angles), angles)


@pytest.mark.parametrize(
    ("num_tokens", "padded_tp_heads"),
    [(1, False), (17, False), (4096, False), (17, True)],
)
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_inverse_rope_fp8_wo_a_matches_two_kernel_boundary(
    num_tokens: int,
    padded_tp_heads: bool,
    position_dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA only")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("B200/SM100 or B300/SM103 specialization")

    device = torch.device("cuda")
    torch.manual_seed(17 + num_tokens)
    if padded_tp_heads:
        # Non-unified TP4 attention can return a [T, 64, 512] output and slice
        # the first 16 local heads.  The resulting [T, 2, 4096] view keeps a
        # padded token stride and is the actual huge-path input contract.
        source_storage = torch.randn(
            num_tokens,
            64,
            512,
            device=device,
            dtype=torch.bfloat16,
        )
        source = source_storage[:, :16, :].view(num_tokens, 2, 4096)
        assert not source.is_contiguous()
    else:
        source = torch.randn(
            num_tokens,
            2,
            4096,
            device=device,
            dtype=torch.bfloat16,
        )
    source_before = source.clone()
    positions = torch.randint(
        0, 73728, (num_tokens,), device=device, dtype=position_dtype
    )
    freqs_cis = _freqs(73728, device)

    reference = source.clone()
    reference_heads = reference.view(num_tokens, 16, 512)
    fused_rope_inplace(
        reference_heads[..., -64:],
        None,
        freqs_cis,
        positions,
        inverse=True,
    )
    expected_q, expected_s = sglang_per_token_group_quant_fp8_dsv4_wo_a(
        reference
    )
    import deep_gemm

    expected_s = torch.stack(
        [
            deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                expected_s[:, group]
            ).transpose(0, 1)
            for group in range(2)
        ]
    ).permute(2, 0, 1)[:num_tokens]

    output_q = torch.empty(
        source.shape,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    aligned_tokens = (num_tokens + 3) // 4 * 4
    output_s_storage = torch.empty(
        (2, 8, aligned_tokens),
        device=device,
        dtype=torch.int32,
    )
    actual_q, actual_s = inverse_rope_fp8_wo_a_ue8m0(
        source,
        freqs_cis,
        positions,
        output_q,
        output_s_storage,
    )
    q_ptr = actual_q.data_ptr()
    s_ptr = actual_s.data_ptr()
    # A second layer launch reuses the exact same ForwardBatch workspace.
    actual_q, actual_s = inverse_rope_fp8_wo_a_ue8m0(
        source,
        freqs_cis,
        positions,
        output_q,
        output_s_storage,
    )
    torch.cuda.synchronize()

    # The fused API must not materialize inverse-RoPE back into its input.
    assert torch.equal(source, source_before)
    assert actual_q.data_ptr() == q_ptr == output_q.data_ptr()
    assert actual_s.data_ptr() == s_ptr == output_s_storage.data_ptr()
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_s, expected_s)


def test_inverse_rope_fp8_wo_a_rejects_wrong_shape() -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA only")
    x = torch.empty(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    freqs = _freqs(2, x.device)
    positions = torch.zeros(1, device=x.device, dtype=torch.int32)
    output_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    output_s_storage = torch.empty(1, 8, 4, device=x.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match=r"requires \[T, 2, 4096\]"):
        inverse_rope_fp8_wo_a_ue8m0(
            x,
            freqs,
            positions,
            output_q,
            output_s_storage,
        )
