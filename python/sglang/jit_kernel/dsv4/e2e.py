"""JIT CUDA primitives owned by the DSV4 whole-layer executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HEAD_DIM = 512
_ROPE_DIM = 64
_QUANT_GROUP_SIZE = 128


@cache_once
def _jit_inverse_rope_fp8_wo_a_module(
    input_dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
    use_pdl: bool,
) -> Module:
    args = make_cpp_args(input_dtype, head_dim, rope_dim, use_pdl)
    return load_jit(
        make_name("inverse_rope_fp8_wo_a_ue8m0"),
        *args,
        cuda_files=["deepseek_v4/inverse_rope_fp8_wo_a.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"InverseRopeFP8WoAQuantUE8M0Kernel<{args}>::run",
            )
        ],
        # Match the existing dedicated WO_A quantizer.  Do not add another
        # arithmetic mode here without a numerical equivalence test.
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="dsv4_inverse_rope_fp8_wo_a_ue8m0",
    mutates_args=["output_q", "output_s"],
)
def _inverse_rope_fp8_wo_a_custom_op(
    input: torch.Tensor,
    freqs_real: torch.Tensor,
    positions: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    module = _jit_inverse_rope_fp8_wo_a_module(
        input.dtype,
        _HEAD_DIM,
        _ROPE_DIM,
        is_arch_support_pdl(),
    )
    module.run(input, freqs_real, positions, output_q, output_s)


@debug_kernel_api
def inverse_rope_fp8_wo_a_ue8m0(
    input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    output_q: torch.Tensor,
    output_s_storage: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse inverse RoPE and B200 group-major WO_A activation quantization.

    Args:
        input: Attention output viewed as ``[T, G, D]`` BF16.  The token
            stride may include padded TP heads; the group and hidden axes must
            remain dense.
            For the TP4 Flash model, ``G=2`` and ``D=4096`` (8 heads/group,
            512 values/head).
        freqs_cis: The model complex64 RoPE table.
        positions: One int32/int64 position per token.
        output_q: Preallocated contiguous FP8 ``[T, 2, 4096]`` workspace.
        output_s_storage: Preallocated contiguous FP32 group-major
            ``[2, T, 32]`` workspace.

    Returns:
        FP8 codes in contiguous ``[T, G, D]`` layout and a logical
        ``[T, G, D/128]`` FP32 UE8M0 scale view backed by group-major
        ``[G, T, D/128]`` storage, exactly as DeepGEMM's WO_A einsum expects.

    This API is strict.  It never runs the separate RoPE and quant kernels.
    """

    if input.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("dsv4 inverse-RoPE/WO_A fusion requires NVIDIA CUDA")
    if input.dtype != torch.bfloat16:
        raise RuntimeError(
            f"dsv4 inverse-RoPE/WO_A fusion requires bf16, got {input.dtype}"
        )
    if input.ndim != 3:
        raise RuntimeError(
            "dsv4 inverse-RoPE/WO_A fusion requires a [T, G, D] input"
        )
    num_tokens, num_groups, hidden = input.shape
    if num_groups != 2 or hidden != 4096:
        raise RuntimeError(
            "dsv4 TP4 inverse-RoPE/WO_A specialization requires [T, 2, 4096], "
            f"got {tuple(input.shape)}"
        )
    if input.stride(2) != 1 or input.stride(1) != hidden:
        raise RuntimeError(
            "dsv4 inverse-RoPE/WO_A fusion requires dense group/hidden axes; "
            f"got strides={input.stride()}"
        )
    if positions.ndim != 1 or positions.shape[0] != num_tokens:
        raise RuntimeError(
            "positions must be [T] and match the attention output token count"
        )
    if positions.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"positions must be int32/int64, got {positions.dtype}")
    if positions.device != input.device or freqs_cis.device != input.device:
        raise RuntimeError("input, positions and freqs_cis must share one device")
    if freqs_cis.dtype != torch.complex64 or freqs_cis.shape[-1] * 2 != _ROPE_DIM:
        raise RuntimeError(
            "freqs_cis must be complex64 [max_position, 32] for rope_dim=64"
        )
    if hidden % _QUANT_GROUP_SIZE:
        raise AssertionError("WO_A quantized hidden size must be divisible by 128")
    if (
        output_q.shape != input.shape
        or output_q.dtype != torch.float8_e4m3fn
        or output_q.device != input.device
        or not output_q.is_contiguous()
    ):
        raise RuntimeError(
            "output_q must be preallocated contiguous CUDA FP8 "
            f"{tuple(input.shape)} on {input.device}"
        )
    expected_scale_shape = (
        num_groups,
        num_tokens,
        hidden // _QUANT_GROUP_SIZE,
    )
    if (
        output_s_storage.shape != expected_scale_shape
        or output_s_storage.dtype != torch.float32
        or output_s_storage.device != input.device
        or not output_s_storage.is_contiguous()
    ):
        raise RuntimeError(
            "output_s_storage must be preallocated contiguous CUDA FP32 "
            f"group-major {expected_scale_shape} on {input.device}"
        )
    if input.numel() != 0:
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
        _inverse_rope_fp8_wo_a_custom_op(
            input,
            freqs_real,
            positions,
            output_q,
            output_s_storage,
        )
    return output_q, output_s_storage.transpose(0, 1)


__all__ = ["inverse_rope_fp8_wo_a_ue8m0"]
