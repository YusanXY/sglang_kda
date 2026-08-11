import re
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.topk import (
    PackedOnlyTopKOutput,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    TopKOutputChecker,
    TopKOutputFormat,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    _resolve_mxfp4_packed_topk,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


M = 8
TOP_K = 6


def _standard_output(*, packed=None):
    weights = torch.rand((M, TOP_K), dtype=torch.float32)
    ids = torch.zeros((M, TOP_K), dtype=torch.int32)
    logits = torch.empty((M, 0), dtype=torch.float32)
    if packed is None:
        return StandardTopKOutput(weights, ids, logits)
    return StandardTopKOutputPacked(weights, ids, logits, packed)


def test_packed_only_carrier_has_distinct_format():
    output = PackedOnlyTopKOutput(torch.empty((M, TOP_K), dtype=torch.int32))

    assert output.format == TopKOutputFormat.PACKED_ONLY
    assert TopKOutputChecker.format_is_packed_only(output)
    assert not TopKOutputChecker.format_is_standard(output)


def test_native_standard_output_keeps_existing_pack_path():
    output = _standard_output()
    expected = torch.full((M, TOP_K), 7, dtype=torch.int32)

    with patch(
        "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe."
        "PackTopkIds.execute",
        return_value=expected,
    ) as pack:
        packed, top_k = _resolve_mxfp4_packed_topk(
            output,
            torch.empty((M, 128), dtype=torch.bfloat16),
            dsv4_worker_backend="native",
            layer_id=3,
        )

    pack.assert_called_once_with(output.topk_ids, output.topk_weights)
    assert packed is expected
    assert top_k == TOP_K


def test_packed_only_is_huge_only_and_never_falls_back_to_pack():
    output = PackedOnlyTopKOutput(torch.empty((M, TOP_K), dtype=torch.int32))

    with patch(
        "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe."
        "PackTopkIds.execute"
    ) as pack, pytest.raises(RuntimeError, match="restricted to DSV4 Huge"):
        _resolve_mxfp4_packed_topk(
            output,
            torch.empty((M, 128), dtype=torch.bfloat16),
            dsv4_worker_backend="native",
            layer_id=4,
        )

    pack.assert_not_called()


def test_huge_standard_output_requires_fused_packed_routing():
    with patch(
        "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe."
        "PackTopkIds.execute"
    ) as pack, pytest.raises(RuntimeError, match="requires fused packed routing"):
        _resolve_mxfp4_packed_topk(
            _standard_output(),
            torch.empty((M, 128), dtype=torch.bfloat16),
            dsv4_worker_backend="huge_kernel",
            layer_id=5,
        )

    pack.assert_not_called()


@pytest.mark.parametrize(
    ("packed", "error_fragment"),
    [
        (torch.empty((M, TOP_K - 1), dtype=torch.int32), "shape=(8, 6)"),
        (torch.empty((M, TOP_K), dtype=torch.int64), "dtype=torch.int32"),
        (
            torch.empty((M, TOP_K * 2), dtype=torch.int32)[:, ::2],
            "contiguous=True",
        ),
        (torch.empty((M, TOP_K), dtype=torch.int32), "(CUDA)"),
    ],
)
def test_huge_packed_only_rejects_invalid_workspace(packed, error_fragment):
    with pytest.raises(RuntimeError, match=re.escape(error_fragment)):
        _resolve_mxfp4_packed_topk(
            PackedOnlyTopKOutput(packed),
            torch.empty((M, 128), dtype=torch.bfloat16),
            dsv4_worker_backend="huge_kernel",
            layer_id=6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_huge_packed_only_accepts_exact_cuda_workspace():
    hidden_states = torch.empty((M, 128), dtype=torch.bfloat16, device="cuda")
    expected = torch.empty((M, TOP_K), dtype=torch.int32, device="cuda")

    packed, top_k = _resolve_mxfp4_packed_topk(
        PackedOnlyTopKOutput(expected),
        hidden_states,
        dsv4_worker_backend="huge_kernel",
        layer_id=7,
    )

    assert packed is expected
    assert top_k == TOP_K
