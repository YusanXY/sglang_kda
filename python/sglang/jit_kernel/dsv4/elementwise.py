from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_hip, is_xpu

from .utils import make_name

_is_hip = is_hip()
_is_xpu = is_xpu()


@cache_once
def _jit_fused_rope_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("fused_rope"),
        *args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


@cache_once
def _jit_main_q_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
):
    """Main MLA path Q kernel: rmsnorm-self + RoPE, warp per (token, head)."""
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name("main_q_norm_rope"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQNormRopeKernel<{args}>::forward"),
            ("tp4_route", f"FusedQNormRopeKernel<{args}>::tp4_route"),
            (
                "tp4_bulk_route_forward",
                f"FusedQNormRopeKernel<{args}>::tp4_bulk_route_forward",
            ),
        ],
    )


@cache_once
def _jit_main_k_norm_rope_flashmla_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
    page_size: int,
):
    """Main MLA path K kernel: rmsnorm + RoPE + write to FlashMLA paged cache."""
    args = make_cpp_args(dtype, head_dim, rope_dim, page_size, is_arch_support_pdl())
    return load_jit(
        make_name("main_k_norm_rope_flashmla"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedKNormRopeFlashMLAKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_main_q_indexer_rope_hadamard_quant_module(dtype: torch.dtype):
    """C4 indexer Q kernel: RoPE + 128-pt Hadamard + fp8 act-quant"""
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("main_q_indexer_rope_hadamard_quant"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward"),
        ],
    )


# V3.2 lays q out as [rope | nope] (V4 is [nope | rope]) -> kRopeFirst=true, and
# drops the Hadamard rotation (kHadamard=false).
@cache_once
def _jit_main_q_indexer_rope_first_quant_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl(), True, False)
    return load_jit(
        make_name("main_q_indexer_rope_first_quant"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_main_q_indexer_rope_hadamard_fp4_quant_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("main_q_indexer_rope_hadamard_fp4_quant"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardFp4QuantKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_rmsnorm_mxfp8_quant_module(
    dtype: torch.dtype,
    hidden_size: int,
    group_size: int,
):
    """DSV4 q_lora RMSNorm producing BF16 and DeepGEMM MXFP8 together."""
    args = make_cpp_args(
        dtype,
        hidden_size,
        group_size,
        is_arch_support_pdl(),
    )
    return load_jit(
        make_name("rmsnorm_mxfp8_quant"),
        *args,
        cuda_files=["deepseek_v4/rmsnorm_mxfp8_quant.cuh"],
        cuda_wrappers=[
            ("forward", f"RmsnormMxfp8QuantKernel<{args}>::run"),
        ],
    )


def fused_rope_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """Apply rotary embeddings to both Q and K in a single fused CUDA kernel.

    Args:
        q: [batch_size, num_q_heads, rope_dim] bfloat16
        k: [batch_size, num_k_heads, rope_dim] bfloat16 or None
        freqs_cis: [max_seq_len, rope_dim // 2] complex64 (full table)
        positions: [batch_size] int32 or int64, indices into freqs_cis
        inverse: if True, apply inverse rotation (conjugate freqs)
    """
    if _is_hip or _is_xpu:
        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            apply_rotary_emb_triton,
        )

        apply_rotary_emb_triton(q, freqs_cis, positions=positions, inverse=inverse)
        if k is not None:
            apply_rotary_emb_triton(k, freqs_cis, positions=positions, inverse=inverse)
        return

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    module = _jit_fused_rope_module()
    module.forward(q, k, freqs_real, positions, inverse)


def fused_q_norm_rope(
    q_input: torch.Tensor,
    q_output: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    head_dim = q_input.shape[-1]
    rope_dim = freqs_real.shape[-1]
    module = _jit_main_q_norm_rope_module(q_input.dtype, head_dim, rope_dim)
    module.forward(q_input, q_output, freqs_real, positions, eps)


def fused_q_norm_rope_tp4_route(
    q_input: torch.Tensor,
    q_output_peers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    source_rank: int,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Fuse Q norm/RoPE with the C4 TP4 token-shard exchange.

    Each source rank writes its four contiguous token quarters directly into
    the corresponding peer-visible receive buffer. The destination layout is
    source-major, so FlashMLA can consume it without a relayout or NCCL A2A.
    """
    if len(q_output_peers) != 4:
        raise RuntimeError("TP4 routed Q producer requires exactly four peers")
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    head_dim = q_input.shape[-1]
    rope_dim = freqs_real.shape[-1]
    module = _jit_main_q_norm_rope_module(q_input.dtype, head_dim, rope_dim)
    module.tp4_route(
        q_input,
        q_output_peers[0],
        q_output_peers[1],
        q_output_peers[2],
        q_output_peers[3],
        freqs_real,
        positions,
        source_rank,
        eps,
    )


def fused_q_norm_rope_tp4_bulk_route(
    q_input: torch.Tensor,
    q_local: torch.Tensor,
    q_output_peers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    source_rank: int,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Fuse the Host boundary around local Q production and bulk TP4 routing."""
    if len(q_output_peers) != 4:
        raise RuntimeError("TP4 bulk Q route requires exactly four peers")
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    head_dim = q_input.shape[-1]
    rope_dim = freqs_real.shape[-1]
    module = _jit_main_q_norm_rope_module(q_input.dtype, head_dim, rope_dim)
    module.tp4_bulk_route_forward(
        q_input,
        q_local,
        q_output_peers[source_rank],
        q_output_peers[0],
        q_output_peers[1],
        q_output_peers[2],
        q_output_peers[3],
        freqs_real,
        positions,
        source_rank,
        eps,
    )


def rmsnorm_mxfp8_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_bf16: torch.Tensor,
    output_fp8: torch.Tensor,
    output_scale: torch.Tensor,
    eps: float,
    group_size: int = 128,
) -> None:
    """Fuse q_lora RMSNorm with DeepGEMM block-FP8 quantization.

    ``output_scale`` is DeepGEMM's token-contiguous, TMA-aligned packed UE8M0
    layout. All outputs are caller-owned so Huge can reuse one model-scoped
    GPU workspace across all decoder layers.
    """
    if _is_hip or _is_xpu:
        raise RuntimeError("DSV4 Huge RMSNorm+MXFP8 fusion requires NVIDIA CUDA")
    module = _jit_rmsnorm_mxfp8_quant_module(
        input.dtype,
        input.shape[-1],
        group_size,
    )
    module.forward(input, weight, output_bf16, output_fp8, output_scale, eps)


def fused_q_indexer_rope_hadamard_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    if _is_hip:
        torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant(
            q_input,
            q_fp8,
            weight,
            weights_out,
            float(weight_scale),
            freqs_real,
            positions,
        )
    elif _is_xpu:
        from sgl_kernel import fused_q_indexer_rope_hadamard_quant

        fused_q_indexer_rope_hadamard_quant(
            q_input,
            q_fp8,
            weight,
            weights_out,
            float(weight_scale),
            freqs_real,
            positions,
        )
    else:
        module = _jit_main_q_indexer_rope_hadamard_quant_module(q_input.dtype)
        module.forward(
            q_input,
            q_fp8,
            weight,
            weights_out,
            float(weight_scale),
            freqs_real,
            positions,
        )
    return q_fp8, weights_out


def fused_q_indexer_rope_first_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V3.2 only. Indexer Q: RoPE on the leading dims + fp8 act-quant. CUDA only."""
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        cos_sin_cache,
        positions,
    )
    return q_fp8, weights_out


def fused_q_indexer_rope_hadamard_fp4_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if _is_hip:
        raise RuntimeError("DeepSeek V4 FP4 indexer requires the CUDA fused Q path.")
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    q_fp4 = torch.empty(
        (*q_input.shape[:-1], q_input.shape[-1] // 2),
        dtype=torch.int8,
        device=q_input.device,
    )
    q_sf = torch.empty(q_input.shape[:-1], dtype=torch.int32, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_hadamard_fp4_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp4,
        q_sf,
        weight,
        weights_out,
        float(weight_scale),
        freqs_real,
        positions,
    )
    return (q_fp4, q_sf), weights_out


def fused_k_norm_rope_flashmla(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
) -> None:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    head_dim = kv.shape[-1]
    rope_dim = freqs_real.shape[-1]
    module = _jit_main_k_norm_rope_flashmla_module(
        kv.dtype, head_dim, rope_dim, page_size
    )
    module.forward(kv, kv_weight, freqs_real, positions, out_loc, kvcache, eps)
