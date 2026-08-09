"""SM100 clustered FP8 paged-MQA logits for the strict DSV4 huge runtime.

The CUDA kernel consumes sixteen adjacent causal queries per two-CTA cluster.
Grouping metadata is built once per ForwardBatch and reused by all 21 C4
layers; the layer hot path only submits the already-bound CUDA extension.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sglang.jit_kernel.utils import cache_once

HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
QUERIES_PER_CLUSTER = 16
# One true req=16 high-load forward contributes 16*4096 query rows.  This is
# a capacity bound only; req=1 continues to take an exact 4096-row view.
MAX_TOTAL_Q = 131072
MAX_C4_CONTEXT = 18432

_HERE = Path(__file__).resolve().parent
_SOURCE_DIR = _HERE.parent / "csrc" / "dsv4" / "clustered_mqa_logits"
_SOURCES = (_SOURCE_DIR / "v2_binding.cpp", _SOURCE_DIR / "v2_kernel.cu")
_TOPK_V2 = _HERE.parent / "csrc" / "deepseek_v4" / "topk_v2.cuh"
_TOPK_IMPL = _HERE.parent / "include" / "sgl_kernel" / "deepseek_v4" / "topk_impl.cuh"
_DEPENDENCIES = (
    *_SOURCES,
    _SOURCE_DIR / "v2_mqa_logits_layout.cuh",
    _SOURCE_DIR / "v2_sm100_mqa_logits.cuh",
    _SOURCE_DIR / "v2_sm100_paged_mqa_logits.cuh",
    _TOPK_V2,
    _TOPK_IMPL,
)


@dataclass(frozen=True)
class ClusteredMqaMetadata:
    """Device metadata shared by every C4 layer in one forward."""

    seq_lens: torch.Tensor
    page_table: torch.Tensor
    schedule: torch.Tensor
    logits_workspace: torch.Tensor
    max_context: int
    total_q: int
    global_total_q: int
    query_begin: int
    query_end: int
    request_begin: int
    request_end: int
    extension: Any


def _plan_tp_query_shard(
    lengths: tuple[int, ...],
    total_q: int,
    *,
    tp_rank: int | None,
    tp_size: int | None,
) -> tuple[int, int, int, int]:
    """Return request-aligned query/request bounds for the TP-local owner.

    C4 sparse attention already assigns one contiguous token quarter to each
    TP rank.  The indexer must use the identical partition so its local top-k
    outputs can be consumed directly without an all-gather.  Request alignment
    is mandatory because the sparse epilogue uses per-request cache geometry.
    """

    if (tp_rank is None) != (tp_size is None):
        raise RuntimeError("clustered MQA TP rank and size must be provided together")
    if tp_rank is None:
        return 0, total_q, 0, len(lengths)
    if tp_size != 4 or not 0 <= tp_rank < tp_size:
        raise RuntimeError(
            "DSV4 Huge TP-local C4 indexer requires TP4 and a valid rank; "
            f"got rank={tp_rank}, size={tp_size}"
        )
    if total_q % tp_size:
        raise RuntimeError(
            "DSV4 Huge TP-local C4 indexer requires M divisible by TP4; "
            f"got M={total_q}"
        )

    query_begin = total_q * tp_rank // tp_size
    query_end = total_q * (tp_rank + 1) // tp_size
    request_offsets = [0]
    for length in lengths:
        request_offsets.append(request_offsets[-1] + length)
    try:
        request_begin = request_offsets.index(query_begin)
        request_end = request_offsets.index(query_end)
    except ValueError as exc:
        raise RuntimeError(
            "DSV4 Huge TP-local C4 indexer requires request-aligned token "
            f"quarters; query range [{query_begin},{query_end}) does not align "
            f"with request offsets {request_offsets}"
        ) from exc
    return query_begin, query_end, request_begin, request_end


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in _DEPENDENCIES:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@contextmanager
def _torch_arch(arch: str):
    key = "TORCH_CUDA_ARCH_LIST"
    previous = os.environ.get(key)
    os.environ[key] = arch
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@cache_once
def load_clustered_mqa_extension():
    """Compile/load the SM100 kernel once during Huge runtime binding."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            "clustered DSV4 MQA requires SM100/SM103, got "
            f"sm{capability[0]}{capability[1]}"
        )
    import deep_gemm
    import tvm_ffi
    from torch.utils.cpp_extension import load

    deep_gemm_include = Path(deep_gemm.__file__).resolve().parent / "include"
    tvm_ffi_include = Path(tvm_ffi.__file__).resolve().parent / "include"
    arch = "10.0a" if capability == (10, 0) else "10.3a"
    module_name = f"sglang_dsv4_clustered_mqa_{_source_digest()}"
    with _torch_arch(arch):
        return load(
            name=module_name,
            sources=[str(path) for path in _SOURCES],
            extra_include_paths=[
                str(_SOURCE_DIR),
                str(deep_gemm_include),
                str(_HERE.parent / "include"),
                str(tvm_ffi_include),
            ],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-lineinfo",
                f"-DSGL_CUDA_ARCH={capability[0] * 100 + capability[1] * 10}",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                # torch.utils.cpp_extension disables CUDA half/bfloat
                # conversions globally.  The exact v2 radix kernel uses a
                # deliberate FP32 -> FP16 coarse-bin conversion, matching its
                # standalone TVM-FFI build, so restore the CUDA operators for
                # this translation unit.
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-DSGLANG_DSV4_TOPK_WARP_BITSET_SORT=1",
            ],
            extra_ldflags=["-lcuda"],
            with_cuda=True,
            verbose=os.environ.get("SGLANG_DSV4_CLUSTERED_MQA_VERBOSE", "0") == "1",
        )


def prepare_clustered_mqa_metadata(
    *,
    indexer_metadata: Any,
    extend_lens_cpu: Sequence[int],
    logits_workspace: torch.Tensor | None = None,
    tp_rank: int | None = None,
    tp_size: int | None = None,
) -> ClusteredMqaMetadata | None:
    """Build one grouped device schedule when requests have complete Q16 groups.

    Non-Q16 tails remain on the explicit DeepGEMM CUDA specialization.  This
    is not a native-layer fallback: both paths stay inside the Huge C4 executor
    and kernel failures propagate unchanged.
    """

    lengths = tuple(int(length) for length in extend_lens_cpu)
    if any(length % QUERIES_PER_CLUSTER for length in lengths):
        return None

    global_c4_seq_lens = indexer_metadata.c4_seq_lens.reshape(-1)
    global_total_q = int(global_c4_seq_lens.numel())
    if global_total_q == 0 or global_total_q % QUERIES_PER_CLUSTER:
        raise RuntimeError(
            "clustered DSV4 MQA requires a nonempty aggregate M divisible by 16"
        )
    if sum(lengths) != global_total_q:
        raise RuntimeError(
            "clustered DSV4 MQA host/device shape mismatch: "
            f"extend lengths sum to {sum(lengths)}, c4 rows={global_total_q}"
        )
    global_page_table = indexer_metadata.page_table
    if global_page_table.ndim != 2 or global_page_table.shape[0] != global_total_q:
        raise RuntimeError(
            "clustered DSV4 MQA requires one page-table row per query token"
        )

    query_begin, query_end, request_begin, request_end = _plan_tp_query_shard(
        lengths,
        global_total_q,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    c4_seq_lens = global_c4_seq_lens[query_begin:query_end]
    page_table = global_page_table[query_begin:query_end]
    total_q = query_end - query_begin
    if total_q == 0 or total_q % QUERIES_PER_CLUSTER:
        raise RuntimeError(
            "clustered DSV4 MQA TP-local M must be nonempty and divisible by 16; "
            f"got {total_q}"
        )

    grouped_lens = c4_seq_lens.view(-1, QUERIES_PER_CLUSTER)
    # Request lengths are Q16 aligned, so every strided row starts a group and
    # no group crosses requests. The kernel accepts an arbitrary outer stride.
    grouped_page_table = page_table[::QUERIES_PER_CLUSTER]

    import deep_gemm

    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        grouped_lens,
        PAGE_SIZE,
        deep_gemm.get_num_sms() // 2,
    )
    # Breakable capture pads the page table beyond the model's legal context
    # capacity. The extra columns are address-stability sentinels, not keys the
    # model may attend to. Keep the clustered logits stride and CUDA v2 bound
    # at the strict DSV4-Flash C4 limit.
    max_context = min(int(indexer_metadata.max_c4_seq_len), MAX_C4_CONTEXT)
    padded_context = (max_context + 255) // 256 * 256
    if logits_workspace is None:
        # Standalone/operator callers retain the old allocation contract.
        logits_workspace = torch.empty(
            (total_q, padded_context),
            dtype=torch.float32,
            device=c4_seq_lens.device,
        )
    else:
        if (
            logits_workspace.device != c4_seq_lens.device
            or logits_workspace.dtype != torch.float32
            or logits_workspace.ndim != 2
            or logits_workspace.shape[0] < total_q
            or logits_workspace.shape[1] < padded_context
            or logits_workspace.shape[1] % 256 != 0
            or not logits_workspace.is_contiguous()
        ):
            raise RuntimeError(
                "clustered DSV4 MQA requires a contiguous FP32 prebound "
                f"workspace covering [{total_q},{padded_context}], got "
                f"shape={tuple(logits_workspace.shape)}, "
                f"dtype={logits_workspace.dtype}, device={logits_workspace.device}"
            )
        # Slice rows only and preserve the fixed full-context row stride.  This
        # remains contiguous and avoids a new allocator request when M changes.
        logits_workspace = logits_workspace[:total_q]
    return ClusteredMqaMetadata(
        seq_lens=grouped_lens,
        page_table=grouped_page_table,
        schedule=schedule,
        logits_workspace=logits_workspace,
        max_context=max_context,
        total_q=total_q,
        global_total_q=global_total_q,
        query_begin=query_begin,
        query_end=query_end,
        request_begin=request_begin,
        request_end=request_end,
        extension=load_clustered_mqa_extension(),
    )


def refresh_clustered_mqa_metadata_for_graph_replay_(
    captured: ClusteredMqaMetadata,
    *,
    indexer_metadata: Any,
) -> None:
    """Refresh a captured Q16 schedule without rebinding graph pointers."""

    live = prepare_clustered_mqa_metadata(
        indexer_metadata=indexer_metadata,
        extend_lens_cpu=(captured.total_q,),
        logits_workspace=captured.logits_workspace,
    )
    if live is None:
        raise RuntimeError("captured clustered MQA replay unexpectedly has a Q16 tail")
    scalar_fields = (
        "max_context",
        "total_q",
        "global_total_q",
        "query_begin",
        "query_end",
        "request_begin",
        "request_end",
    )
    for name in scalar_fields:
        current = getattr(captured, name)
        incoming = getattr(live, name)
        if current != incoming:
            raise RuntimeError(
                "clustered MQA Graph replay shape changed: "
                f"{name}={current!r} vs {incoming!r}"
            )
    if captured.extension is not live.extension:
        raise RuntimeError("clustered MQA extension changed during Graph replay")
    for name in ("seq_lens", "page_table", "schedule"):
        dst = getattr(captured, name)
        src = getattr(live, name)
        if dst.shape != src.shape or dst.dtype != src.dtype:
            raise RuntimeError(
                "clustered MQA Graph replay tensor changed: "
                f"{name}: captured={tuple(dst.shape)}/{dst.dtype}, "
                f"live={tuple(src.shape)}/{src.dtype}"
            )
        # seq_lens/page_table are views into the already-refreshed captured
        # indexer metadata, so this is redundant for values but intentionally
        # enforces the stable-view contract alongside the independent schedule.
        dst.copy_(src)


def clustered_fp8_paged_mqa_logits(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    metadata: ClusteredMqaMetadata,
) -> torch.Tensor:
    """Submit the preplanned CUDA kernel without per-layer metadata work."""

    if q.shape != (metadata.total_q, HEADS, HEAD_DIM):
        raise RuntimeError(
            f"clustered DSV4 MQA Q must be [M,64,128], got {tuple(q.shape)}"
        )
    return metadata.extension.forward_out(
        q,
        kv_cache,
        weights,
        metadata.seq_lens,
        metadata.page_table,
        metadata.schedule,
        metadata.logits_workspace,
        metadata.max_context,
    )


def clustered_fp8_paged_mqa_topk(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    metadata: ClusteredMqaMetadata,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    full_seq_lens: torch.Tensor,
    swa_gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
) -> None:
    """Submit clustered logits and exact v2 top-k through one C++ entry."""

    if q.shape != (metadata.total_q, HEADS, HEAD_DIM):
        raise RuntimeError(
            f"clustered DSV4 MQA Q must be [M,64,128], got {tuple(q.shape)}"
        )
    metadata.extension.forward_topk(
        q,
        kv_cache,
        weights,
        metadata.seq_lens,
        metadata.page_table,
        metadata.schedule,
        metadata.logits_workspace,
        page_table,
        page_indices,
        raw_indices,
        positions,
        query_start_loc,
        full_seq_lens,
        swa_gather_lens,
        compressed_base,
        swa_base,
        combined_indices,
        combined_lens,
        metadata.max_context,
    )


__all__ = [
    "ClusteredMqaMetadata",
    "MAX_C4_CONTEXT",
    "MAX_TOTAL_Q",
    "clustered_fp8_paged_mqa_logits",
    "clustered_fp8_paged_mqa_topk",
    "load_clustered_mqa_extension",
    "prepare_clustered_mqa_metadata",
    "refresh_clustered_mqa_metadata_for_graph_replay_",
]
