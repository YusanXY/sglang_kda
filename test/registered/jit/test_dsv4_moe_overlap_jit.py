from __future__ import annotations

import pytest

from sglang.jit_kernel.dsv4_moe_overlap.jit import (
    _POST_ROUTING_PREPARE,
    _RUN_PROLOGUE,
    _patch_launcher,
)


def _source() -> str:
    return f"prefix\n{_RUN_PROLOGUE}routing body\n{_POST_ROUTING_PREPARE}suffix\n"


def test_patch_moves_prepare_before_routing() -> None:
    patched = _patch_launcher(_source())

    prepare = patched.index("prepare_moe(moe_tactic)")
    routing = patched.index("// Execute routing")
    moe_stream = patched.index("cudaStream_t moe_stream")
    assert prepare < routing < moe_stream
    assert patched.count("prepare_moe(moe_tactic)") == 1


@pytest.mark.parametrize(
    "source",
    [
        "unrelated source",
        _source() + _RUN_PROLOGUE,
        _source() + _POST_ROUTING_PREPARE,
    ],
)
def test_patch_rejects_unexpected_source_layout(source: str) -> None:
    with pytest.raises(RuntimeError, match="expected exactly one"):
        _patch_launcher(source)
