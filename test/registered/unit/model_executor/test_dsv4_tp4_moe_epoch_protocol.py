from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.jit_kernel.dsv4 import e2e


REPO_ROOT = Path(__file__).resolve().parents[4]
CUDA_SOURCE = (
    REPO_ROOT
    / "python/sglang/jit_kernel/csrc/deepseek_v4/tp4_moe_mhc_post.cuh"
)


def _flags(*, device: str = "cuda", dtype: torch.dtype = torch.int32):
    mode = FakeTensorMode()
    with mode:
        return tuple(
            torch.empty((4,), dtype=dtype, device=device) for _ in range(4)
        )


def _epoch_inputs(
    *,
    global_m: int = 57344,
    local_m: int = 12288,
    device: str = "cuda",
):
    mode = FakeTensorMode()
    with mode:
        partials = tuple(
            torch.empty(
                (global_m, 4096), dtype=torch.bfloat16, device=device
            )
            for _ in range(4)
        )
        shared_hidden = torch.empty(
            (local_m, 4096), dtype=torch.bfloat16, device=device
        )
        residual = torch.empty(
            (local_m, 4, 4096), dtype=torch.bfloat16, device=device
        )
        post_mix = torch.empty(
            (local_m, 4), dtype=torch.float32, device=device
        )
        comb_mix = torch.empty(
            (local_m, 4, 4), dtype=torch.float32, device=device
        )
        output = torch.empty(
            (local_m, 4, 4096), dtype=torch.bfloat16, device=device
        )
        flags = tuple(
            torch.empty((4,), dtype=torch.int32, device=device)
            for _ in range(4)
        )
    return (
        partials,
        shared_hidden,
        residual,
        post_mix,
        comb_mix,
        output,
        flags,
    )


def test_wait_slot_first_use_still_reaches_immediate_cuda_kernel(
    monkeypatch,
) -> None:
    flags = _flags()
    calls = []
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_wait_slot_reusable_custom_op",
        lambda *args: calls.append(args),
    )

    assert e2e.tp4_moe_wait_slot_reusable(flags, 0, 0) is None
    assert calls == [(*flags, 0, 0)]


@pytest.mark.parametrize(
    ("slot", "epoch"),
    [(-1, 0), (2, 0), (True, 0), (0.0, 0), (0, -1), (0, True)],
)
def test_wait_slot_rejects_invalid_metadata(
    monkeypatch, slot, epoch
) -> None:
    flags = _flags()
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_wait_slot_reusable_custom_op",
        lambda *_: pytest.fail("invalid metadata reached CUDA"),
    )
    with pytest.raises(RuntimeError, match="slot|epoch"):
        e2e.tp4_moe_wait_slot_reusable(flags, slot, epoch)


def test_wait_slot_requires_four_contiguous_cuda_int32_flags(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_wait_slot_reusable_custom_op",
        lambda *_: pytest.fail("invalid flags reached CUDA"),
    )
    with pytest.raises(RuntimeError, match="four peer flags"):
        e2e.tp4_moe_wait_slot_reusable(_flags()[:3], 0, 0)
    with pytest.raises(RuntimeError, match="int32"):
        e2e.tp4_moe_wait_slot_reusable(_flags(dtype=torch.int64), 0, 0)
    with pytest.raises(RuntimeError, match="CUDA"):
        e2e.tp4_moe_wait_slot_reusable(_flags(device="cpu"), 0, 0)


def test_epoch_wrapper_accepts_variable_bucket_and_forwards_protocol(
    monkeypatch,
) -> None:
    args = _epoch_inputs()
    calls = []
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_epoch_custom_op",
        lambda *op_args: calls.append(op_args),
    )

    result = e2e.tp4_moe_local_slice_shared_mhc_post_epoch(
        args[0],
        32768,
        *args[1:6],
        args[6],
        3,
        1,
        29,
    )

    assert result is args[5]
    assert len(calls) == 1
    assert calls[0][4] == 32768
    assert calls[0][9] is args[5]
    assert all(
        forwarded is expected
        for forwarded, expected in zip(calls[0][10:14], args[6])
    )
    assert calls[0][14:] == (3, 1, 29)


def test_split_epoch_wrapper_uses_distinct_custom_op_and_same_protocol(
    monkeypatch,
) -> None:
    args = _epoch_inputs()
    calls = []
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_epoch_split_custom_op",
        lambda *op_args: calls.append(op_args),
    )

    result = e2e.tp4_moe_local_slice_shared_mhc_post_epoch_split(
        args[0],
        32768,
        *args[1:6],
        args[6],
        3,
        1,
        29,
    )

    assert result is args[5]
    assert len(calls) == 1
    assert calls[0][4] == 32768
    assert calls[0][9] is args[5]
    assert all(
        forwarded is expected
        for forwarded, expected in zip(calls[0][10:14], args[6])
    )
    assert calls[0][14:] == (3, 1, 29)


def test_counter_epoch_wrapper_forwards_local_completion_state(monkeypatch) -> None:
    args = _epoch_inputs()
    mode = FakeTensorMode()
    with mode:
        completion = torch.zeros((2,), dtype=torch.int64, device="cuda")
    calls = []
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_epoch_counter_custom_op",
        lambda *op_args: calls.append(op_args),
    )

    result = e2e.tp4_moe_local_slice_shared_mhc_post_epoch_counter(
        args[0],
        32768,
        *args[1:6],
        args[6],
        completion,
        3,
        1,
        29,
    )

    assert result is args[5]
    assert len(calls) == 1
    assert calls[0][14] is completion
    assert calls[0][15:] == (3, 1, 29)


def test_nvls_counter_wrapper_forwards_anchor_pointer_and_protocol(
    monkeypatch,
) -> None:
    args = _epoch_inputs()
    mode = FakeTensorMode()
    with mode:
        completion = torch.zeros((2,), dtype=torch.int64, device="cuda")
    calls = []
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_"
        "epoch_counter_multimem_custom_op",
        lambda *op_args: calls.append(op_args),
    )
    multicast_local_ptr = 0x100000000

    result = e2e.tp4_moe_local_slice_shared_mhc_post_epoch_counter_multimem(
        multicast_local_ptr,
        args[0][3],
        32768,
        *args[1:6],
        args[6],
        completion,
        3,
        1,
        29,
    )

    assert result is args[5]
    assert len(calls) == 1
    assert calls[0][0] == multicast_local_ptr
    assert calls[0][1] is args[0][3]
    assert calls[0][2] == 32768
    assert calls[0][7] is args[5]
    assert all(
        forwarded is expected
        for forwarded, expected in zip(calls[0][8:12], args[6])
    )
    assert calls[0][12] is completion
    assert calls[0][13:] == (3, 1, 29)


@pytest.mark.parametrize(
    "multicast_local_ptr",
    [0, -16, 17, True, 1 << 63],
)
def test_nvls_counter_wrapper_rejects_invalid_multicast_pointer(
    monkeypatch, multicast_local_ptr
) -> None:
    args = _epoch_inputs()
    mode = FakeTensorMode()
    with mode:
        completion = torch.zeros((2,), dtype=torch.int64, device="cuda")
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_"
        "epoch_counter_multimem_custom_op",
        lambda *_: pytest.fail("invalid multicast pointer reached CUDA"),
    )

    with pytest.raises(RuntimeError, match="multicast VA"):
        e2e.tp4_moe_local_slice_shared_mhc_post_epoch_counter_multimem(
            multicast_local_ptr,
            args[0][0],
            32768,
            *args[1:6],
            args[6],
            completion,
            0,
            0,
            1,
        )


@pytest.mark.parametrize(
    ("rank", "slot", "epoch"),
    [(-1, 0, 1), (4, 0, 1), (True, 0, 1), (0, 2, 1), (0, 0, 0),
     (0, 0, 0x1_0000_0000)],
)
def test_epoch_wrapper_rejects_invalid_protocol_metadata(
    monkeypatch, rank, slot, epoch
) -> None:
    args = _epoch_inputs()
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_epoch_custom_op",
        lambda *_: pytest.fail("invalid metadata reached CUDA"),
    )
    with pytest.raises(RuntimeError, match="rank|slot|epoch"):
        e2e.tp4_moe_local_slice_shared_mhc_post_epoch(
            args[0],
            32768,
            *args[1:6],
            args[6],
            rank,
            slot,
            epoch,
        )


def test_epoch_wrapper_requires_flags_on_data_device(monkeypatch) -> None:
    args = _epoch_inputs()
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_epoch_custom_op",
        lambda *_: pytest.fail("invalid flags reached CUDA"),
    )
    with pytest.raises(RuntimeError, match="device"):
        e2e.tp4_moe_local_slice_shared_mhc_post_epoch(
            args[0],
            32768,
            *args[1:6],
            _flags(device="cpu"),
            0,
            0,
            1,
        )


def test_epoch_cuda_protocol_is_cooperative_resident_and_system_scoped() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    wait_begin = source.index(
        "__global__ void tp4_moe_wait_slot_reusable_kernel"
    )
    epoch_begin = source.index(
        "void tp4_moe_local_slice_shared_mhc_post_epoch_kernel"
    )
    ready_begin = source.index(
        "__global__ void tp4_moe_publish_ready_wait_kernel",
        wait_begin,
    )
    owner_begin = source.index("// Two-stage owner protocol", epoch_begin)
    wait_kernel = source[wait_begin:ready_begin]
    epoch_kernel = source[epoch_begin:owner_begin]

    assert "expected_done_epoch == 0" in wait_kernel
    assert "kTp4MoeEpochDoneBase + params.slot" in wait_kernel
    assert wait_kernel.count("tp4_moe_mhc_load_acquire_sys") == 4
    assert "__launch_bounds__(kTp4MoeMhcThreads, 4)" in source[
        epoch_begin - 100:epoch_begin
    ]
    assert "__threadfence_system()" in epoch_kernel
    assert epoch_kernel.count("tp4_moe_mhc_store_release_sys") == 2
    assert epoch_kernel.count("grid.sync()") == 2
    assert "local_token += gridDim.x" in epoch_kernel
    assert "tp4_moe_local_slice_shared_mhc_post_token" in epoch_kernel
    assert "PDL" not in epoch_kernel

    host_run = source[source.index("static void run_local_slice_shared_epoch"):]
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in host_run
    assert "cudaDevAttrMultiProcessorCount" in host_run
    assert "cudaDevAttrCooperativeLaunch" in host_run
    assert "kTp4OwnerPersistentBlocks" in host_run
    assert "cudaLaunchCooperativeKernel" in host_run


def test_split_epoch_cuda_protocol_keeps_one_host_boundary_without_grid_sync() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    ready_begin = source.index(
        "__global__ void tp4_moe_publish_ready_wait_kernel"
    )
    done_begin = source.index("__global__ void tp4_moe_publish_done_kernel")
    cooperative_begin = source.index(
        "void tp4_moe_local_slice_shared_mhc_post_epoch_kernel"
    )
    ready_kernel = source[ready_begin:done_begin]
    done_kernel = source[done_begin:cooperative_begin]
    assert "tp4_moe_mhc_store_release_sys" in ready_kernel
    assert ready_kernel.count("tp4_moe_mhc_load_acquire_sys") == 4
    assert "tp4_moe_mhc_store_release_sys" in done_kernel

    split_begin = source.index(
        "static void run_local_slice_shared_epoch_split"
    )
    split_end = source.index("static void run_multimem", split_begin)
    split_host = source[split_begin:split_end]
    assert "tp4_moe_publish_ready_wait_kernel" in split_host
    assert "tp4_moe_local_slice_shared_mhc_post_kernel<false>" in split_host
    assert "tp4_moe_publish_done_kernel" in split_host
    assert "cudaLaunchCooperativeKernel" not in split_host
    assert "cudaDeviceGetAttribute" not in split_host
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" not in split_host


def test_counter_epoch_cuda_protocol_is_noncooperative_and_self_reusing() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    kernel_begin = source.index(
        "void tp4_moe_local_slice_shared_mhc_post_epoch_counter_kernel"
    )
    cooperative_begin = source.index(
        "void tp4_moe_local_slice_shared_mhc_post_epoch_kernel",
        kernel_begin,
    )
    kernel = source[kernel_begin:cooperative_begin]
    assert "atomicCAS" in kernel
    assert "atomicAdd" in kernel
    assert "tp4_moe_mhc_atomic_add_acq_rel_gpu" in kernel
    assert "tp4_moe_mhc_load_acquire_gpu_u64" in kernel
    assert "kTp4MoeEpochLocalReadyBit" in kernel
    # Four peer-ready loads are executed only by the CAS winner; the four
    # remaining system loads are in the last-CTA alternate-slot reuse check.
    assert kernel.count("tp4_moe_mhc_load_acquire_sys") == 8
    assert "kTp4MoeEpochReadyBase" in kernel
    assert "kTp4MoeEpochDoneBase" in kernel
    assert "next_slot = params.slot ^ 1" in kernel
    assert "this_grid" not in kernel
    assert "grid.sync" not in kernel

    host_begin = source.index(
        "static void run_local_slice_shared_epoch_counter"
    )
    host_end = source.index(
        "static void run_local_slice_shared_epoch_counter_multimem",
        host_begin,
    )
    host = source[host_begin:host_end]
    assert "TensorMatcher({2})" in host
    assert "with_dtype<int64_t>()" in host
    assert "kTp4OwnerPersistentBlocks" in host
    assert "cudaLaunchCooperativeKernel" not in host
    assert "cudaDeviceGetAttribute" not in host
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" not in host


def test_nvls_counter_cuda_path_replaces_peer_staging_and_fences_aliases() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    token_begin = source.index(
        "SGL_DEVICE void "
        "tp4_moe_local_slice_multimem_shared_mhc_post_token"
    )
    token_end = source.index(
        "template <bool kUsePDL>\n"
        "__global__ void tp4_moe_local_slice_shared_mhc_post_kernel",
        token_begin,
    )
    token = source[token_begin:token_end]
    assert "tp4_moe_mhc_multimem_reduce_bf16x8" in token
    assert "peer_stages" not in token
    assert "rounded_sum = __float22bfloat162_rn" in token

    counter_begin = source.index(
        "void tp4_moe_local_slice_shared_mhc_post_epoch_counter_kernel"
    )
    counter_end = source.index(
        "void tp4_moe_local_slice_shared_mhc_post_epoch_kernel",
        counter_begin,
    )
    counter = source[counter_begin:counter_end]
    assert "if constexpr (kUseMultimem)" in counter
    assert counter.count("tp4_moe_mhc_fence_proxy_alias") == 2
    assert "tp4_moe_local_slice_multimem_shared_mhc_post_token" in counter
    assert "peer_stages" in counter  # only the discarded <false> branch

    host_begin = source.index(
        "static void run_local_slice_shared_epoch_counter_multimem"
    )
    host_end = source.index("static void run_multimem", host_begin)
    host = source[host_begin:host_end]
    assert "local_partial_anchor" in host
    assert "multicast_local_ptr" in host
    assert (
        "tp4_moe_local_slice_shared_mhc_post_epoch_counter_kernel<true>"
        in host
    )

    jit_source = Path(e2e.__file__).read_text(encoding="utf-8")
    assert "owner_v65_nvls_local_epoch" in jit_source
    assert "run_local_slice_shared_epoch_counter_multimem" in jit_source


def test_epoch_addition_keeps_bringup_abi_separate() -> None:
    bringup = inspect.signature(e2e.tp4_moe_local_slice_shared_mhc_post)
    epoch = inspect.signature(
        e2e.tp4_moe_local_slice_shared_mhc_post_epoch
    )
    assert "flags" not in bringup.parameters
    assert tuple(epoch.parameters)[-4:] == ("flags", "rank", "slot", "epoch")
