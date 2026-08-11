from __future__ import annotations

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


def _inputs(
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
    return (
        partials,
        shared_hidden,
        residual,
        post_mix,
        comb_mix,
        output,
    )


def test_local_slice_wrapper_accepts_variable_dp_bucket(monkeypatch) -> None:
    args = _inputs()
    calls = []
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_custom_op",
        lambda *op_args: calls.append(op_args),
    )

    output = e2e.tp4_moe_local_slice_shared_mhc_post(
        args[0], 32768, *args[1:]
    )

    assert output is args[-1]
    assert len(calls) == 1
    assert calls[0][4] == 32768
    assert calls[0][-1] is args[-1]


@pytest.mark.parametrize("offset", [-4096, 2048, 49152, True, 0.0])
def test_local_slice_wrapper_rejects_invalid_offset(
    monkeypatch, offset
) -> None:
    args = _inputs()
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_custom_op",
        lambda *_: pytest.fail("invalid metadata reached the CUDA op"),
    )

    with pytest.raises(RuntimeError, match="offset|interval"):
        e2e.tp4_moe_local_slice_shared_mhc_post(
            args[0], offset, *args[1:]
        )


def test_local_slice_wrapper_rejects_noncontiguous_partial(monkeypatch) -> None:
    args = list(_inputs())
    mode = FakeTensorMode()
    with mode:
        bad = torch.empty(
            (4096, 57344), dtype=torch.bfloat16, device="cuda"
        ).t()
    args[0] = (bad, *args[0][1:])
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_custom_op",
        lambda *_: pytest.fail("noncontiguous metadata reached the CUDA op"),
    )

    with pytest.raises(RuntimeError, match="contiguous"):
        e2e.tp4_moe_local_slice_shared_mhc_post(
            args[0], 32768, *args[1:]
        )


def test_local_slice_wrapper_rejects_wrong_dtype(monkeypatch) -> None:
    args = list(_inputs())
    mode = FakeTensorMode()
    with mode:
        args[1] = torch.empty(
            (12288, 4096), dtype=torch.float32, device="cuda"
        )
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_custom_op",
        lambda *_: pytest.fail("wrong dtype reached the CUDA op"),
    )

    with pytest.raises(RuntimeError, match="expected contiguous"):
        e2e.tp4_moe_local_slice_shared_mhc_post(
            args[0], 32768, *args[1:]
        )


def test_local_slice_wrapper_rejects_wrong_shape(monkeypatch) -> None:
    args = list(_inputs())
    mode = FakeTensorMode()
    with mode:
        args[3] = torch.empty(
            (12288, 5), dtype=torch.float32, device="cuda"
        )
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_custom_op",
        lambda *_: pytest.fail("wrong shape reached the CUDA op"),
    )

    with pytest.raises(RuntimeError, match="expected contiguous"):
        e2e.tp4_moe_local_slice_shared_mhc_post(
            args[0], 32768, *args[1:]
        )


def test_local_slice_wrapper_rejects_non_cuda_device(monkeypatch) -> None:
    args = _inputs(device="cpu")
    monkeypatch.setattr(
        e2e,
        "_tp4_moe_local_slice_shared_mhc_post_custom_op",
        lambda *_: pytest.fail("CPU metadata reached the CUDA op"),
    )

    with pytest.raises(RuntimeError, match="CUDA device"):
        e2e.tp4_moe_local_slice_shared_mhc_post(
            args[0], 32768, *args[1:]
        )


def test_cuda_source_preserves_nccl_order_and_bf16_shared_boundary() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    begin = source.index(
        "SGL_DEVICE void tp4_moe_local_slice_shared_mhc_post_token"
    )
    end = source.index(
        "__global__ void tp4_moe_local_slice_shared_mhc_post_kernel", begin
    )
    token_body = source[begin:end]

    assert "global_hidden_base" in token_body
    assert "first_element % kNCCLPeriodElements" in token_body
    assert "tp4_moe_mhc_add4_ordered" in token_body
    assert "rounded_sum = __float22bfloat162_rn" in token_body
    assert "params.shared_hidden" in token_body
