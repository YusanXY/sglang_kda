"""Hermetic contracts for the Huge-only clustered paged-MQA path."""

from types import SimpleNamespace
from unittest import mock

import torch
from sglang.jit_kernel.dsv4.clustered_mqa_logits import (
    ClusteredMqaMetadata,
    clustered_fp8_paged_mqa_topk,
    prepare_clustered_mqa_metadata,
)
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata


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
    assert metadata.logits_workspace.shape == (32, 512)
    assert metadata.logits_workspace.dtype == torch.float32
    deep_gemm.get_paged_mqa_logits_metadata.assert_called_once_with(
        metadata.seq_lens, 64, 74
    )


def test_grouped_metadata_reuses_prebound_logits_workspace():
    c4_seq_lens = torch.arange(1, 33, dtype=torch.int32)
    indexer_metadata = SimpleNamespace(
        c4_seq_lens=c4_seq_lens,
        page_table=torch.arange(32 * 5, dtype=torch.int32).view(32, 5),
        max_c4_seq_len=320,
    )
    workspace = torch.empty((64, 768), dtype=torch.float32)
    deep_gemm = SimpleNamespace(
        get_num_sms=lambda: 148,
        get_paged_mqa_logits_metadata=lambda *_: torch.empty(
            (1, 2), dtype=torch.int32
        ),
    )

    with (
        mock.patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
        mock.patch(
            "sglang.jit_kernel.dsv4.clustered_mqa_logits.load_clustered_mqa_extension",
            return_value=object(),
        ),
    ):
        metadata = prepare_clustered_mqa_metadata(
            indexer_metadata=indexer_metadata,
            extend_lens_cpu=[16, 16],
            logits_workspace=workspace,
        )

    assert metadata is not None
    assert metadata.logits_workspace.shape == (32, 768)
    assert metadata.logits_workspace.is_contiguous()
    assert (
        metadata.logits_workspace.untyped_storage().data_ptr()
        == workspace.untyped_storage().data_ptr()
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


def test_huge_q16_skips_stock_schedule_until_cuda_tail_needs_it():
    metadata = object.__new__(PagedIndexerMetadata)
    metadata.prefer_clustered_mqa = True
    metadata.deep_gemm_metadata = None
    schedule = torch.tensor([[3, 5]], dtype=torch.int32)

    with mock.patch.object(
        PagedIndexerMetadata,
        "_build_deep_gemm_metadata",
        return_value=schedule,
    ) as build:
        assert metadata.deep_gemm_metadata is None
        assert metadata.ensure_deep_gemm_metadata() is schedule
        assert metadata.ensure_deep_gemm_metadata() is schedule

    build.assert_called_once_with()


def test_huge_clustered_metadata_skips_unused_generic_topk_v2_plan():
    with (
        mock.patch(
            "sglang.srt.layers.attention.dsv4.metadata.envs.SGLANG_OPT_USE_TOPK_V2.get",
            return_value=True,
        ),
        mock.patch("sglang.jit_kernel.dsv4.plan_topk_v2") as plan,
    ):
        metadata = PagedIndexerMetadata(
            page_size=256,
            page_table=torch.zeros((16, 2), dtype=torch.int32),
            c4_seq_lens=torch.ones(16, dtype=torch.int32),
            prefer_clustered_mqa=True,
        )

    plan.assert_not_called()
    assert metadata.topk_metadata.numel() == 0
    assert metadata.topk_metadata.device == metadata.c4_seq_lens.device


def test_clustered_topk_v2_submits_logits_and_epilogue_through_one_cpp_entry():
    extension = mock.Mock()
    q = torch.empty((2, 64, 128))
    metadata = ClusteredMqaMetadata(
        seq_lens=mock.sentinel.seq_lens,
        page_table=mock.sentinel.grouped_page_table,
        schedule=mock.sentinel.schedule,
        logits_workspace=mock.sentinel.workspace,
        max_context=320,
        total_q=2,
        extension=extension,
    )
    args = {
        "kv_cache": mock.sentinel.kv_cache,
        "weights": mock.sentinel.weights,
        "page_table": mock.sentinel.query_page_table,
        "page_indices": mock.sentinel.page_indices,
        "raw_indices": mock.sentinel.raw_indices,
        "positions": mock.sentinel.positions,
        "query_start_loc": mock.sentinel.query_start_loc,
        "full_seq_lens": mock.sentinel.full_seq_lens,
        "swa_gather_lens": mock.sentinel.swa_gather_lens,
        "compressed_base": mock.sentinel.compressed_base,
        "swa_base": mock.sentinel.swa_base,
        "combined_indices": mock.sentinel.combined_indices,
        "combined_lens": mock.sentinel.combined_lens,
    }

    assert clustered_fp8_paged_mqa_topk(q=q, metadata=metadata, **args) is None

    extension.forward_topk.assert_called_once_with(
        q,
        args["kv_cache"],
        args["weights"],
        metadata.seq_lens,
        metadata.page_table,
        metadata.schedule,
        metadata.logits_workspace,
        args["page_table"],
        args["page_indices"],
        args["raw_indices"],
        args["positions"],
        args["query_start_loc"],
        args["full_seq_lens"],
        args["swa_gather_lens"],
        args["compressed_base"],
        args["swa_base"],
        args["combined_indices"],
        args["combined_lens"],
        metadata.max_context,
    )
    extension.forward_out.assert_not_called()
