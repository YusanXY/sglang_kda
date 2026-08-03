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

_HERE = Path(__file__).resolve().parent
_SOURCE_DIR = _HERE.parent / "csrc" / "dsv4" / "clustered_mqa_logits"
_SOURCES = (_SOURCE_DIR / "v2_binding.cpp", _SOURCE_DIR / "v2_kernel.cu")
_DEPENDENCIES = (
    *_SOURCES,
    _SOURCE_DIR / "v2_mqa_logits_layout.cuh",
    _SOURCE_DIR / "v2_sm100_mqa_logits.cuh",
    _SOURCE_DIR / "v2_sm100_paged_mqa_logits.cuh",
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
    extension: Any


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
    from torch.utils.cpp_extension import load

    deep_gemm_include = Path(deep_gemm.__file__).resolve().parent / "include"
    arch = "10.0a" if capability == (10, 0) else "10.3a"
    module_name = f"sglang_dsv4_clustered_mqa_{_source_digest()}"
    with _torch_arch(arch):
        return load(
            name=module_name,
            sources=[str(path) for path in _SOURCES],
            extra_include_paths=[str(_SOURCE_DIR), str(deep_gemm_include)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-lineinfo",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
            extra_ldflags=["-lcuda"],
            with_cuda=True,
            verbose=os.environ.get("SGLANG_DSV4_CLUSTERED_MQA_VERBOSE", "0") == "1",
        )


def prepare_clustered_mqa_metadata(
    *, indexer_metadata: Any, extend_lens_cpu: Sequence[int]
) -> ClusteredMqaMetadata | None:
    """Build one grouped device schedule when requests have complete Q16 groups.

    Non-Q16 tails remain on the explicit DeepGEMM CUDA specialization.  This
    is not a native-layer fallback: both paths stay inside the Huge C4 executor
    and kernel failures propagate unchanged.
    """

    lengths = tuple(int(length) for length in extend_lens_cpu)
    if any(length % QUERIES_PER_CLUSTER for length in lengths):
        return None

    c4_seq_lens = indexer_metadata.c4_seq_lens.reshape(-1)
    total_q = int(c4_seq_lens.numel())
    if total_q == 0 or total_q % QUERIES_PER_CLUSTER:
        raise RuntimeError(
            "clustered DSV4 MQA requires a nonempty aggregate M divisible by 16"
        )
    if sum(lengths) != total_q:
        raise RuntimeError(
            "clustered DSV4 MQA host/device shape mismatch: "
            f"extend lengths sum to {sum(lengths)}, c4 rows={total_q}"
        )
    page_table = indexer_metadata.page_table
    if page_table.ndim != 2 or page_table.shape[0] != total_q:
        raise RuntimeError(
            "clustered DSV4 MQA requires one page-table row per query token"
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
    padded_context = (indexer_metadata.max_c4_seq_len + 255) // 256 * 256
    # Decoder layers execute serially on the current stream.  Keep one device
    # buffer alive for the whole ForwardBatch instead of entering the PyTorch
    # allocator once per C4 layer (21 times for DSV4 Flash).
    logits_workspace = torch.empty(
        (total_q, padded_context),
        dtype=torch.float32,
        device=c4_seq_lens.device,
    )
    return ClusteredMqaMetadata(
        seq_lens=grouped_lens,
        page_table=grouped_page_table,
        schedule=schedule,
        logits_workspace=logits_workspace,
        max_context=int(indexer_metadata.max_c4_seq_len),
        total_q=total_q,
        extension=load_clustered_mqa_extension(),
    )


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


__all__ = [
    "ClusteredMqaMetadata",
    "clustered_fp8_paged_mqa_logits",
    "load_clustered_mqa_extension",
    "prepare_clustered_mqa_metadata",
]
