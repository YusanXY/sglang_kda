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
def _jit_mhc_post_vec8_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        make_name("mhc_post_vec8_hc4_h4096_v3_occupancy"),
        *args,
        cuda_files=["deepseek_v4/mhc_post_vec8.cuh"],
        cuda_wrappers=[("run", f"MhcPostVec8Kernel<{args}>::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def load_mhc_post_vec8_extension() -> None:
    """Compile/load the strict HC=4/H=4096 post kernel before serving."""

    _jit_mhc_post_vec8_module(False)


@register_custom_op(
    op_name="dsv4_mhc_post_vec8",
    mutates_args=["output"],
)
def _mhc_post_vec8_custom_op(
    hidden_in: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    output: torch.Tensor,
) -> None:
    module = _jit_mhc_post_vec8_module(False)
    module.run(hidden_in, residual, post_mix, comb_mix, output)


@debug_kernel_api
def mhc_post_vec8(
    hidden_in: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Apply DSV4 Flash mHC post into a caller-owned GPU output buffer."""

    m = hidden_in.shape[0]
    expected = (
        (hidden_in, (m, 4096), torch.bfloat16),
        (residual, (m, 4, 4096), torch.bfloat16),
        (post_mix, (m, 4), torch.float32),
        (comb_mix, (m, 4, 4), torch.float32),
        (output, (m, 4, 4096), torch.bfloat16),
    )
    device = hidden_in.device
    for tensor, shape, dtype in expected:
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"invalid DSV4 mHC post tensor: expected contiguous {shape} "
                f"{dtype}, got {tuple(tensor.shape)} {tensor.dtype}"
            )
        if device.type != "cuda" or tensor.device != device:
            raise RuntimeError(
                "all DSV4 mHC post tensors must share one CUDA device"
            )
    _mhc_post_vec8_custom_op(hidden_in, residual, post_mix, comb_mix, output)
    return output


@cache_once
def _jit_mhc_pre_norm_mxfp8_quant_module(threads: int, use_pdl: bool) -> Module:
    args = make_cpp_args(threads, use_pdl)
    return load_jit(
        make_name("mhc_pre_norm_mxfp8_quant_double_buffer_final_v1"),
        *args,
        cuda_files=["deepseek_v4/mhc_pre_norm_mxfp8_quant.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"MhcPreNormMxfp8QuantKernel<{args}>::run",
            )
        ],
    )


def load_mhc_pre_norm_mxfp8_quant_extension(threads: int = 160) -> None:
    """Compile/load the strict DSV4 mHC producer before the first request."""

    if threads not in (64, 96, 128, 160, 256, 512, 1024):
        raise RuntimeError(
            "mHC fused quant threads must be 64/96/128/160/256/512/1024"
        )
    _jit_mhc_pre_norm_mxfp8_quant_module(threads, is_arch_support_pdl())


@register_custom_op(
    op_name="dsv4_mhc_pre_norm_mxfp8_quant",
    mutates_args=[
        "post_mix",
        "comb_mix",
        "layer_input",
        "output_fp8",
        "output_scale",
        "routed_output_fp8",
        "routed_output_scale",
    ],
)
def _mhc_pre_norm_mxfp8_quant_custom_op(
    gemm_mul: torch.Tensor,
    gemm_sq: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    output_fp8: torch.Tensor,
    output_scale: torch.Tensor,
    routed_output_fp8: torch.Tensor,
    routed_output_scale: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    emit_routed_quant: bool,
    threads: int,
) -> None:
    module = _jit_mhc_pre_norm_mxfp8_quant_module(
        threads, is_arch_support_pdl()
    )
    module.run(
        gemm_mul,
        gemm_sq,
        hc_scale,
        hc_base,
        residual,
        norm_weight,
        post_mix,
        comb_mix,
        layer_input,
        output_fp8,
        output_scale,
        routed_output_fp8,
        routed_output_scale,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult,
        sinkhorn_repeat,
        norm_eps,
        emit_routed_quant,
    )


@debug_kernel_api
def mhc_pre_norm_mxfp8_quant(
    gemm_mul: torch.Tensor,
    gemm_sq: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
    output_fp8: torch.Tensor,
    output_scale_storage: torch.Tensor,
    routed_output_fp8: torch.Tensor,
    routed_output_scale: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    emit_routed_quant: bool,
    threads: int = 160,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fuse mHC finalization, following RMSNorm, and block-FP8 quantization.

    This is the strict DSV4-Flash M=4096-oriented producer used immediately
    before attention/shared-expert block-FP8 linears.  ``output_scale_storage``
    is DeepGEMM's physical ``[8, align(M,4)]`` packed-UE8M0 layout.
    """

    m = residual.shape[0]
    if residual.shape != (m, 4, 4096) or residual.dtype != torch.bfloat16:
        raise RuntimeError("mHC fused quant requires BF16 residual [M,4,4096]")
    if gemm_mul.ndim != 3 or gemm_mul.shape[1:] != (m, 24):
        raise RuntimeError("gemm_mul must be [splits,M,24]")
    if gemm_sq.shape != gemm_mul.shape[:2]:
        raise RuntimeError("gemm_sq must be [splits,M]")
    if gemm_mul.dtype != torch.float32 or gemm_sq.dtype != torch.float32:
        raise RuntimeError("mHC GEMM partials must be FP32")
    if hc_scale.shape != (3,) or hc_base.shape != (24,):
        raise RuntimeError("invalid DSV4 mHC scale/base shape")
    if norm_weight.shape != (4096,) or norm_weight.dtype != torch.bfloat16:
        raise RuntimeError("mHC fused quant requires BF16 norm weight [4096]")
    if threads not in (64, 96, 128, 160, 256, 512, 1024):
        raise RuntimeError(
            "mHC fused quant threads must be 64/96/128/160/256/512/1024"
        )
    expected = (
        (post_mix, (m, 4), torch.float32),
        (comb_mix, (m, 4, 4), torch.float32),
        (layer_input, (m, 4096), torch.bfloat16),
        (output_fp8, (m, 4096), torch.float8_e4m3fn),
        (routed_output_fp8, (m, 4096), torch.float8_e4m3fn),
        (routed_output_scale, (m, 128), torch.uint8),
    )
    for tensor, shape, dtype in expected:
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise RuntimeError(
                f"invalid mHC fused output: expected contiguous {shape} {dtype}, "
                f"got {tuple(tensor.shape)} {tensor.dtype}"
            )
    aligned_m = (m + 3) // 4 * 4
    if (
        output_scale_storage.shape != (8, aligned_m)
        or output_scale_storage.dtype != torch.int32
        or not output_scale_storage.is_contiguous()
    ):
        raise RuntimeError(
            f"mHC packed scale storage must be contiguous int32 {(8, aligned_m)}"
        )
    device = residual.device
    tensors = (
        gemm_mul,
        gemm_sq,
        hc_scale,
        hc_base,
        norm_weight,
        post_mix,
        comb_mix,
        layer_input,
        output_fp8,
        output_scale_storage,
        routed_output_fp8,
        routed_output_scale,
    )
    if device.type != "cuda" or any(t.device != device for t in tensors):
        raise RuntimeError("all mHC fused quant tensors must share one CUDA device")
    _mhc_pre_norm_mxfp8_quant_custom_op(
        gemm_mul,
        gemm_sq,
        hc_scale,
        hc_base,
        residual,
        norm_weight,
        post_mix,
        comb_mix,
        layer_input,
        output_fp8,
        output_scale_storage,
        routed_output_fp8,
        routed_output_scale,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult,
        sinkhorn_repeat,
        norm_eps,
        emit_routed_quant,
        threads,
    )
    logical_scale = output_scale_storage.transpose(0, 1)[:m]
    return (
        post_mix,
        comb_mix,
        layer_input,
        output_fp8,
        logical_scale,
        routed_output_fp8,
        routed_output_scale,
    )


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
        output_s_storage: Preallocated contiguous int32 packed-scale storage
            ``[2, 8, align(T, 4)]``.

    Returns:
        FP8 codes in contiguous ``[T, G, D]`` layout and a logical
        ``[T, G, D/512]`` int32 view.  Each int32 packs four UE8M0 exponent
        bytes and is backed by DeepGEMM's physical
        ``[G, D/512, align(T, 4)]`` TMA layout.

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
    aligned_tokens = (num_tokens + 3) // 4 * 4
    expected_scale_shape = (
        num_groups,
        hidden // (_QUANT_GROUP_SIZE * 4),
        aligned_tokens,
    )
    if (
        output_s_storage.shape != expected_scale_shape
        or output_s_storage.dtype != torch.int32
        or output_s_storage.device != input.device
        or not output_s_storage.is_contiguous()
    ):
        raise RuntimeError(
            "output_s_storage must be preallocated contiguous CUDA int32 "
            f"packed-scale {expected_scale_shape} on {input.device}"
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
    return output_q, output_s_storage.permute(2, 0, 1)[:num_tokens]


_TP4_PACKED_OUTPUT_ROW_BYTES = 2 * 4096 + 2 * (4096 // _QUANT_GROUP_SIZE)


@cache_once
def _jit_tp4_packed_output_module(
    input_dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
    use_pdl: bool,
) -> Module:
    args = make_cpp_args(input_dtype, head_dim, rope_dim, use_pdl)
    return load_jit(
        make_name("tp4_packed_attention_output_ue8m0"),
        *args,
        cuda_files=["deepseek_v4/inverse_rope_fp8_wo_a.cuh"],
        cuda_wrappers=[
            ("pack", f"TP4PackedOutputUE8M0Kernel<{args}>::pack"),
            ("unpack", f"TP4PackedOutputUE8M0Kernel<{args}>::unpack"),
        ],
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="dsv4_tp4_pack_attention_output_ue8m0",
    mutates_args=["packed_output"],
)
def _tp4_pack_attention_output_custom_op(
    input: torch.Tensor,
    freqs_real: torch.Tensor,
    shard_positions: torch.Tensor,
    packed_output: torch.Tensor,
) -> None:
    module = _jit_tp4_packed_output_module(
        input.dtype,
        _HEAD_DIM,
        _ROPE_DIM,
        is_arch_support_pdl(),
    )
    module.pack(input, freqs_real, shard_positions, packed_output)


@register_custom_op(
    op_name="dsv4_tp4_unpack_attention_output_ue8m0",
    mutates_args=["output_q", "output_s"],
)
def _tp4_unpack_attention_output_custom_op(
    packed_input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    module = _jit_tp4_packed_output_module(
        torch.bfloat16,
        _HEAD_DIM,
        _ROPE_DIM,
        is_arch_support_pdl(),
    )
    module.unpack(packed_input, output_q, output_s)


@debug_kernel_api
def tp4_pack_attention_output_ue8m0(
    input: torch.Tensor,
    freqs_cis: torch.Tensor,
    shard_positions: torch.Tensor,
    packed_output: torch.Tensor,
) -> torch.Tensor:
    """Inverse-RoPE, quantize, and pack one token-sharded C4 output.

    ``input`` is destination-major ``[4 * shard_T, 2, 4096]`` BF16.  Each
    packed row contains 8192 FP8 values followed by 64 UE8M0 bytes so a single
    uint8 all-to-all replaces the old BF16 output collective.
    """

    if input.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("TP4 packed attention output requires NVIDIA CUDA")
    if input.dtype != torch.bfloat16 or input.shape[1:] != (2, 4096):
        raise RuntimeError(
            "TP4 packed attention input must be BF16 [4*shard_T, 2, 4096]"
        )
    if not input.is_contiguous() or input.shape[0] % 4:
        raise RuntimeError("TP4 packed attention input must be contiguous and T%4=0")
    shard_tokens = input.shape[0] // 4
    if (
        shard_positions.ndim != 1
        or shard_positions.shape[0] != shard_tokens
        or shard_positions.dtype not in (torch.int32, torch.int64)
        or shard_positions.device != input.device
    ):
        raise RuntimeError("TP4 shard positions must be CUDA int32/int64 [T/4]")
    if freqs_cis.device != input.device or freqs_cis.dtype != torch.complex64:
        raise RuntimeError("TP4 packed output requires a CUDA complex64 RoPE table")
    if (
        packed_output.shape != (input.shape[0], _TP4_PACKED_OUTPUT_ROW_BYTES)
        or packed_output.dtype != torch.uint8
        or packed_output.device != input.device
        or not packed_output.is_contiguous()
    ):
        raise RuntimeError(
            "packed_output must be contiguous CUDA uint8 "
            f"[{input.shape[0]}, {_TP4_PACKED_OUTPUT_ROW_BYTES}]"
        )
    if input.numel():
        _tp4_pack_attention_output_custom_op(
            input,
            torch.view_as_real(freqs_cis).flatten(-2),
            shard_positions,
            packed_output,
        )
    return packed_output


@debug_kernel_api
def tp4_unpack_attention_output_ue8m0(
    packed_input: torch.Tensor,
    output_q: torch.Tensor,
    output_s_storage: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack a received TP4 FP8 row into DeepGEMM WO_A operands."""

    if (
        packed_input.ndim != 2
        or packed_input.shape[1] != _TP4_PACKED_OUTPUT_ROW_BYTES
        or packed_input.dtype != torch.uint8
        or packed_input.device.type != "cuda"
        or not packed_input.is_contiguous()
    ):
        raise RuntimeError(
            "packed_input must be contiguous CUDA uint8 [T, 8256]"
        )
    num_tokens = packed_input.shape[0]
    if (
        output_q.shape != (num_tokens, 2, 4096)
        or output_q.dtype != torch.float8_e4m3fn
        or output_q.device != packed_input.device
        or not output_q.is_contiguous()
    ):
        raise RuntimeError("TP4 unpack output_q must be contiguous FP8 [T,2,4096]")
    aligned_tokens = (num_tokens + 3) // 4 * 4
    if (
        output_s_storage.shape != (2, 8, aligned_tokens)
        or output_s_storage.dtype != torch.int32
        or output_s_storage.device != packed_input.device
        or not output_s_storage.is_contiguous()
    ):
        raise RuntimeError(
            "TP4 unpack scale storage must be contiguous int32 [2,8,align(T,4)]"
        )
    if packed_input.numel():
        _tp4_unpack_attention_output_custom_op(
            packed_input,
            output_q,
            output_s_storage,
        )
    return output_q, output_s_storage.permute(2, 0, 1)[:num_tokens]


__all__ = [
    "inverse_rope_fp8_wo_a_ue8m0",
    "load_mhc_post_vec8_extension",
    "load_mhc_pre_norm_mxfp8_quant_extension",
    "mhc_post_vec8",
    "mhc_pre_norm_mxfp8_quant",
    "tp4_pack_attention_output_ue8m0",
    "tp4_unpack_attention_output_ue8m0",
]
