"""CUDA fused paged-cache dequantization for DSV4 sparse prefill."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

_OUTPUT_DIM = 512


@cache_once
def _jit_dual_paged_dequant_module(use_pdl: bool) -> Module:
    args = make_cpp_args(torch.bfloat16, use_pdl)
    return load_jit(
        make_name("dual_paged_dequant_bf16"),
        *args,
        cuda_files=["deepseek_v4/dual_paged_dequant.cuh"],
        cuda_wrappers=[("run", f"DualPagedDequantKernel<{args}>::run")],
    )


@register_custom_op(
    op_name="dsv4_dual_paged_dequant_bf16",
    mutates_args=["output"],
)
def _dual_paged_dequant_custom_op(
    cache_a_u8: torch.Tensor,
    token_ids_a: torch.Tensor,
    page_size_a: int,
    cache_b_u8: torch.Tensor,
    token_ids_b: torch.Tensor,
    page_size_b: int,
    output: torch.Tensor,
) -> None:
    module = _jit_dual_paged_dequant_module(is_arch_support_pdl())
    module.run(
        cache_a_u8,
        token_ids_a,
        page_size_a,
        cache_b_u8,
        token_ids_b,
        page_size_b,
        output,
    )


@debug_kernel_api
def dual_paged_dequantize_k_cache_bf16(
    cache_a: torch.Tensor,
    token_ids_a: torch.Tensor,
    page_size_a: int,
    cache_b: torch.Tensor,
    token_ids_b: torch.Tensor,
    page_size_b: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Dequantize compressed and SWA page lists into one flat workspace.

    The two lists are concatenated logically as ``[A, B]``.  No allocation,
    CPU readback, intermediate tensor, or second kernel launch is permitted.
    """

    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("DSV4 dual paged dequant requires NVIDIA CUDA")
    if cache_a.device.type != "cuda" or cache_b.device != cache_a.device:
        raise RuntimeError("both paged caches must share one CUDA device")
    if not cache_a.is_contiguous() or not cache_b.is_contiguous():
        raise RuntimeError("both paged caches must be contiguous")
    for name, ids in (("token_ids_a", token_ids_a), ("token_ids_b", token_ids_b)):
        if (
            ids.device != cache_a.device
            or ids.dtype != torch.int32
            or ids.ndim != 1
            or not ids.is_contiguous()
        ):
            raise RuntimeError(f"{name} must be contiguous CUDA int32 [N]")
    if page_size_a <= 0 or page_size_b <= 0:
        raise RuntimeError("page sizes must be positive")
    total_tokens = token_ids_a.shape[0] + token_ids_b.shape[0]
    if (
        output.device != cache_a.device
        or output.dtype != torch.bfloat16
        or output.shape != (total_tokens, 1, _OUTPUT_DIM)
        or not output.is_contiguous()
    ):
        raise RuntimeError(
            "output must be contiguous CUDA BF16 "
            f"[{total_tokens}, 1, {_OUTPUT_DIM}]"
        )

    cache_a_u8 = cache_a.view(torch.uint8)
    cache_b_u8 = cache_b.view(torch.uint8)
    if cache_a_u8.ndim != 2 or cache_b_u8.ndim != 2:
        raise RuntimeError("paged cache byte views must be two-dimensional")
    _dual_paged_dequant_custom_op(
        cache_a_u8,
        token_ids_a,
        page_size_a,
        cache_b_u8,
        token_ids_b,
        page_size_b,
        output,
    )
    return output


__all__ = ["dual_paged_dequantize_k_cache_bf16"]
