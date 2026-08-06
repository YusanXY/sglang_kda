from __future__ import annotations

import pytest

from sglang.jit_kernel.dsv4_moe_overlap.jit import (
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
    assert patched.count("dsv4_set_shared_finalize") >= 3
    assert "args->do_finalize = do_finalize && !dsv4_fuse_shared" in patched
    assert "result[0], topk_weights, result[2]" in patched


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
