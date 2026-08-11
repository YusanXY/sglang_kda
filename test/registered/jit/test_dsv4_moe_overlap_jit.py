from __future__ import annotations

import pytest

from sglang.jit_kernel.dsv4_moe_overlap.jit import (
    _DSV4_JIT_MODULE_NAME,
    _EXPORT_ANCHOR,
    _FP4_MULTI_TILE_END,
    _FP4_MULTI_TILE_START,
    _INCLUDE_ANCHOR,
    _MOE_RUNNER_CONSTRUCTION,
    _MOE_RUNNER_MEMBER,
    _MOE_TACTIC_AND_WORKSPACE,
    _NAMESPACE_ANCHOR,
    _POST_ROUTING_PREPARE,
    _RUN_PROLOGUE,
    _patch_launcher,
)


def _source() -> str:
    # The production launcher currently has 48 temporary allocations.  Two
    # live in the tactic/workspace anchor below; synthesize the remaining 46
    # so this hermetic fixture exercises the same strict source-layout guard.
    allocations = "\n".join(
        f"auto allocation_{index} = alloc_tensor(shape, dtype, device);"
        for index in range(46)
    )
    multi_tile = """  std::unordered_map<int32_t, int> launchers_map;
  for (int32_t curr_tile_N : mSupportedTileN) {
    std::make_unique<FP4BlockScaleLauncher>();
  }
  resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k,
                          local_num_experts);
  launchers_map.find(0);
"""
    return (
        f"{_INCLUDE_ANCHOR}{_NAMESPACE_ANCHOR}"
        f"prefix\n{_MOE_RUNNER_MEMBER}\n{_MOE_RUNNER_CONSTRUCTION}"
        f"{_MOE_TACTIC_AND_WORKSPACE}\n{allocations}\n"
        f"{_RUN_PROLOGUE}routing body\n{_POST_ROUTING_PREPARE}"
        f"middle\n{_FP4_MULTI_TILE_START}{multi_tile}{_FP4_MULTI_TILE_END}"
        f"suffix\n{_EXPORT_ANCHOR}"
    )


def test_patch_moves_prepare_before_routing() -> None:
    patched = _patch_launcher(_source())

    prepare = patched.index("prepare_moe(moe_tactic)")
    routing = patched.index("// Execute routing")
    moe_stream = patched.index("cudaStream_t moe_stream")
    assert prepare < routing < moe_stream
    assert patched.count("prepare_moe(moe_tactic)") == 1
    assert "for (int32_t curr_tile_N : mSupportedTileN)" not in patched
    assert "launchers_map" not in patched
    assert patched.count("std::make_unique<FP4BlockScaleLauncher>") == 1
    assert patched.count("dsv4MoeFinalizeSharedKernel") >= 2
    assert patched.count("dsv4MoeFinalizeDpRoutedTopK6Kernel") >= 2
    assert patched.count("dsv4_set_shared_finalize") >= 3
    assert patched.count("dsv4_set_dp_routed_finalize") >= 3
    assert patched.count("dsv4_set_dp_deferred_raw") >= 3
    assert (
        "do_finalize && !dsv4_fuse_shared && !dsv4_fuse_dp_routed"
        in patched
    )
    assert "result[0], topk_ids, result[2]" in patched
    assert "RoutingInputMode::PackedPrecomputed" in patched
    assert "static_cast<uint32_t>(packed) & 0xffffu" in patched
    assert "__ushort_as_bfloat16(bits)" in patched
    assert (
        patched.count("dsv4_packed_weight_to_float(packed_topk[expanded])")
        == 2
    )
    assert "result[0], result[1], result[2]" not in patched
    assert "result[0], topk_weights, result[2]" not in patched
    assert (
        _DSV4_JIT_MODULE_NAME
        == "sgl_dsv4_fused_moe_trtllm_sm100_overlap_v76_stage1"
    )


def test_dp_routed_finalize_packs_two_bf16_rounds_without_shared_add() -> None:
    patched = _patch_launcher(_source())
    kernel_start = patched.index(
        "__global__ void dsv4MoeFinalizeDpRoutedTopK6Kernel"
    )
    kernel_end = patched.index("void dsv4_launch_dp_routed_finalize", kernel_start)
    kernel = patched[kernel_start:kernel_end]

    first_round = kernel.index(
        "__nv_bfloat162 const finalized = __floats2bfloat162_rn("
    )
    scale = kernel.index("__hmul2_rn(finalized, scale2)")
    output_store = kernel.index("output_vec[token * kHiddenVecs + vec] = result")
    assert first_round < scale < output_store
    assert "for (int pair = 0; pair < 4; ++pair)" in kernel
    assert "__float2bfloat16_rn(accum[element])" not in kernel
    assert "shared_output" not in kernel
    assert "shared_vec" not in kernel


def test_dp_routed_topk6_compaction_preserves_route_order() -> None:
    patched = _patch_launcher(_source())
    kernel_start = patched.index(
        "__global__ void dsv4MoeFinalizeDpRoutedTopK6Kernel"
    )
    kernel_end = patched.index("void dsv4_launch_dp_routed_finalize", kernel_start)
    kernel = patched[kernel_start:kernel_end]

    assert "constexpr int kTopK = 6" in kernel
    assert "__shared__ int compact_sources[kTopK]" in kernel
    assert "__shared__ float compact_weights[kTopK]" in kernel
    ballot = kernel.index("__ballot_sync(0xffffffffu")
    stable_slot = kernel.index(
        "__popc(valid_mask & ((1u << lane) - 1u))"
    )
    barrier = kernel.index("__syncthreads()")
    serial_loop = kernel.index(
        "for (int slot = 0; slot < valid_count; ++slot)"
    )
    first_read = kernel.index("gemm_vec[source * padded_vecs + vec]")
    assert ballot < stable_slot < barrier < serial_loop < first_read
    assert "#pragma unroll 1" in kernel
    assert "if (source < 0) continue" not in kernel


def test_dp_routed_topk6_strict_contract_and_thread_dispatch() -> None:
    patched = _patch_launcher(_source())
    launch_start = patched.index("void dsv4_launch_dp_routed_finalize")
    launch = patched[launch_start:]

    assert "TVM_FFI_ICHECK_GT(state.num_tokens, 0)" in launch
    assert "TVM_FFI_ICHECK_LE(state.num_tokens, 131072)" in launch
    assert "TVM_FFI_ICHECK_EQ(top_k, 6)" in launch
    assert "TVM_FFI_ICHECK_EQ(state.hidden_dim, 4096)" in launch
    assert "TVM_FFI_ICHECK_EQ(gemm2_output.size(1), 4096)" in launch
    assert "TVM_FFI_ICHECK_EQ(state.routed_scale, 1.5f)" in launch
    dispatch = launch.index("state.num_tokens <= 4096 ? 256u : 128u")
    kernel_launch = launch.index(
        "dsv4MoeFinalizeDpRoutedTopK6Kernel<<<state.num_tokens, threads"
    )
    assert dispatch < kernel_launch


def test_dp_routed_finalize_rejects_non_dsv4_or_non_bf16_scale() -> None:
    patched = _patch_launcher(_source())
    setter_start = patched.index("void dsv4_set_dp_routed_finalize")
    setter_end = patched.index("struct alignas(16) Dsv4Bf16x8", setter_start)
    setter = patched[setter_start:setter_end]

    round_to_bf16 = setter.index("__float2bfloat16(routed_scale_float)")
    exact_check = setter.index(
        "__bfloat162float(routed_scale_bf16), routed_scale_float"
    )
    dsv4_check = setter.index("TVM_FFI_ICHECK_EQ(routed_scale, 1.5)")
    state_write = setter.index(
        "dsv4_dp_routed_finalize_state.routed_scale = routed_scale_float"
    )
    assert round_to_bf16 < exact_check < dsv4_check < state_write
    assert "must be exactly BF16-representable" in setter
    assert "requires routed_scale=1.5" in setter


def test_packed_epilogue_does_not_modify_shared_finalize() -> None:
    patched = _patch_launcher(_source())
    kernel_start = patched.index("__global__ void dsv4MoeFinalizeSharedKernel")
    kernel_end = patched.index("void dsv4_launch_shared_finalize", kernel_start)
    kernel = patched[kernel_start:kernel_end]

    assert "__floats2bfloat162_rn" not in kernel
    assert "__hmul2_rn" not in kernel
    assert "__float2bfloat16_rn(accum[element])" in kernel
    assert "__bfloat162float(shared.value[element])" in kernel
    assert "compact_sources" not in kernel
    assert "valid_mask" not in kernel
    assert "kTopK = 6" not in kernel


def test_dp_routed_finalize_is_one_shot_and_mutually_exclusive() -> None:
    patched = _patch_launcher(_source())

    assert "DSV4 finalize modes are mutually exclusive" in patched
    assert "Dsv4FinalizeStateResetGuard" in patched
    assert "dsv4_cancel_finalize" in patched
    assert "DSV4 DP routed-finalize descriptor was not consumed" in patched
    assert "DSV4 DP deferred-raw descriptor was not consumed" in patched
    assert "reset_dp_deferred_raw" in patched


def test_dp_deferred_raw_returns_owning_tuple_without_launcher_finalize() -> None:
    patched = _patch_launcher(_source())
    launch = patched.index("Array<Tensor> result = launcher->run(config, enable_pdl)")
    raw = patched.index("if (dsv4_defer_dp_raw)", launch)
    routed = patched.index("if (dsv4_fuse_dp_routed)", raw)
    block = patched[raw:routed]

    assert "TVM_FFI_ICHECK_EQ(result.size(), 3)" in block
    assert "return result" in block
    assert "dsv4_launch_dp_routed_finalize" not in block


@pytest.mark.parametrize(
    "source",
    [
        "unrelated source",
        _source() + _RUN_PROLOGUE,
        _source() + _POST_ROUTING_PREPARE,
        _source() + _FP4_MULTI_TILE_START,
    ],
)
def test_patch_rejects_unexpected_source_layout(source: str) -> None:
    with pytest.raises(RuntimeError, match="expected"):
        _patch_launcher(source)
