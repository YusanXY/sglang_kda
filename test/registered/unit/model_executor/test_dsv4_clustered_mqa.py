"""Hermetic contracts for the Huge-only clustered paged-MQA path."""

from types import SimpleNamespace
from unittest import mock

import torch
from sglang.jit_kernel.dsv4.clustered_mqa_logits import (
    prepare_clustered_mqa_metadata,
)


def test_grouped_metadata_is_built_once_without_copying_page_rows():
    c4_seq_lens = torch.arange(1, 33, dtype=torch.int32)
    page_table = torch.arange(32 * 5, dtype=torch.int32).view(32, 5)
    indexer_metadata = SimpleNamespace(
        c4_seq_lens=c4_seq_lens,
        page_table=page_table,
        max_c4_seq_len=320,
    )
    schedule = torch.tensor([[7, 11]], dtype=torch.int32)
    extension = object()
    deep_gemm = SimpleNamespace(
        get_num_sms=lambda: 148,
        get_paged_mqa_logits_metadata=mock.Mock(return_value=schedule),
    )

    with (
        mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
        mock.patch(
            "sglang.jit_kernel.dsv4.clustered_mqa_logits.load_clustered_mqa_extension",
            return_value=extension,
        ),
    ):
        metadata = prepare_clustered_mqa_metadata(
            indexer_metadata=indexer_metadata,
            extend_lens_cpu=[16, 16],
        )

    assert metadata is not None
    assert metadata.seq_lens.shape == (2, 16)
    assert metadata.page_table.shape == (2, 5)
    assert metadata.page_table.stride(0) == page_table.stride(0) * 16
    assert torch.equal(metadata.page_table[0], page_table[0])
    assert torch.equal(metadata.page_table[1], page_table[16])
    assert metadata.schedule is schedule
    assert metadata.extension is extension
    deep_gemm.get_paged_mqa_logits_metadata.assert_called_once_with(
        metadata.seq_lens, 64, 74
    )


def test_non_q16_request_uses_explicit_deepgemm_cuda_specialization():
    indexer_metadata = SimpleNamespace(
        c4_seq_lens=torch.ones(17, dtype=torch.int32),
        page_table=torch.zeros((17, 2), dtype=torch.int32),
        max_c4_seq_len=128,
    )
    assert (
        prepare_clustered_mqa_metadata(
            indexer_metadata=indexer_metadata,
            extend_lens_cpu=[17],
        )
        is None
    )
