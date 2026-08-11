"""Integration contracts for DSV4 Huge Attention-DP local routing.

These tests intentionally exercise the Python orchestration boundary with CPU
tensors and mocked collectives.  CUDA kernel correctness is covered elsewhere;
the contracts here pin workspace lifetime, local routed-MXFP8 ownership, and
fail-closed control flow without requiring a distributed process group.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.layers.moe.topk import (
    PackedOnlyTopKOutput,
    StandardTopKOutputPacked,
)
from sglang.srt.models import deepseek_v2, deepseek_v4, dsv4_whole_layer_runtime


class _ExtendMode:
    @staticmethod
    def is_idle() -> bool:
        return False

    @staticmethod
    def is_extend_without_speculative() -> bool:
        return True


def _make_forward_batch(
    extend_lens: tuple[int, ...],
    global_rows: int,
    *,
    global_sizes: tuple[int, ...] | None = None,
):
    local_rows = sum(extend_lens)
    requests = len(extend_lens)
    if global_sizes is None:
        global_sizes = (local_rows,) * 4
    return SimpleNamespace(
        forward_mode=_ExtendMode(),
        req_pool_indices=torch.arange(requests, dtype=torch.int32),
        seq_lens=torch.full((requests,), local_rows, dtype=torch.int32),
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=extend_lens,
        out_cache_loc=torch.arange(local_rows, dtype=torch.int32),
        global_dp_buffer_len=global_rows,
        global_num_tokens_cpu=list(global_sizes),
        dp_padding_mode=None,
    )


def _make_runtime_with_bound_route_workspace(
    *,
    symmetric_moe_post: bool = False,
    symmetric_rank: int = 0,
    moe_epoch: bool = False,
    moe_epoch_split: bool = False,
    moe_epoch_counter: bool = False,
    moe_nvls: bool = False,
):
    runtime = object.__new__(dsv4_whole_layer_runtime.DSV4WholeLayerRuntime)
    route_storage = torch.empty((131072, 6), dtype=torch.int32)
    # Keep the unit fixture small while retaining the production tensor shapes.
    # bind_after_weight_load is separately inspected below to prove that the
    # real fixed-address workspaces are contiguous full-capacity allocations.
    routed_q_storage = torch.empty((1, 4096), dtype=torch.float8_e4m3fn).expand(
        131072, 4096
    )
    routed_scale_storage = torch.empty((1, 128), dtype=torch.uint8).expand(
        131072, 128
    )
    scratch = torch.empty(0)

    runtime._active = None
    runtime._handles = (object(),)
    runtime._generation = 1
    runtime._attention_dp4 = True
    runtime._max_local_tokens = 32768
    runtime._use_tp4_token_shard_attention = False
    runtime._use_clustered_mqa = False
    runtime._use_dp_local_routing = True
    runtime._use_dp_symmetric_moe_post = symmetric_moe_post
    runtime._use_dp_moe_epoch = moe_epoch
    runtime._use_dp_moe_epoch_split = moe_epoch_split
    runtime._use_dp_moe_epoch_counter = moe_epoch_counter
    runtime._use_dp_moe_nvls = moe_nvls
    runtime._tp4_symmetric_moe_epoch_workspace = None
    runtime._dp_moe_epoch_layer_count = 2
    runtime._dp_moe_last_planned_epoch = 0
    runtime._dp_moe_epoch_poisoned = False
    runtime._use_tp4_local_wob = False
    runtime._dp_packed_route_workspace = (route_storage.device, route_storage)
    runtime._dp_routed_quant_workspace = (
        routed_q_storage.device,
        routed_q_storage,
        routed_scale_storage,
    )
    if symmetric_moe_post:
        # One physical row expanded to max capacity keeps this CPU orchestration
        # fixture small. Production binding is separately inspected to require
        # a real contiguous symmetric [131072,4096] allocation per rank.
        partials = tuple(
            torch.empty((1, 4096), dtype=torch.bfloat16).expand(131072, 4096)
            for _ in range(4)
        )
        if moe_epoch_counter:
            double_partials = tuple(
                torch.empty((1, 1, 4096), dtype=torch.bfloat16).expand(
                    2, 131072, 4096
                )
                for _ in range(4)
            )
            runtime._tp4_symmetric_moe_workspace = None
            runtime._tp4_symmetric_moe_epoch_workspace = (
                double_partials[0].device,
                SimpleNamespace(barrier=mock.Mock()),
                double_partials[symmetric_rank],
                double_partials,
                symmetric_rank,
                torch.zeros((2,), dtype=torch.int64),
                0x100000000 if moe_nvls else 0,
            )
        else:
            runtime._tp4_symmetric_moe_workspace = (
                partials[0].device,
                SimpleNamespace(barrier=mock.Mock()),
                partials[symmetric_rank],
                partials,
                symmetric_rank,
            )
        if moe_epoch:
            epoch_flags = tuple(
                torch.zeros((4,), dtype=torch.int32) for _ in range(4)
            )
            runtime._tp4_moe_epoch_flags_workspace = (
                epoch_flags[0].device,
                object(),
                epoch_flags[symmetric_rank],
                epoch_flags,
                symmetric_rank,
            )
        else:
            runtime._tp4_moe_epoch_flags_workspace = None
    else:
        runtime._tp4_symmetric_moe_workspace = None
        runtime._tp4_moe_epoch_flags_workspace = None
    runtime._get_wo_a_workspace = mock.Mock(
        return_value=(scratch, scratch, scratch, scratch)
    )
    runtime._get_q_lora_workspace = mock.Mock(
        return_value=(scratch, scratch, scratch)
    )
    runtime._get_mhc_pre_workspace = mock.Mock(return_value=(scratch,) * 12)
    runtime._get_shared_down_workspace = mock.Mock(
        return_value=(scratch, scratch)
    )
    return runtime, route_storage


def _begin_forward(
    runtime,
    extend_lens: tuple[int, ...],
    global_rows: int,
    *,
    global_sizes: tuple[int, ...] | None = None,
    attn_dp_rank: int = 0,
    is_graph_capture: bool = False,
):
    local_rows = sum(extend_lens)
    forward_batch = _make_forward_batch(
        extend_lens, global_rows, global_sizes=global_sizes
    )
    metadata = SimpleNamespace(
        core_attn_metadata=object(),
        indexer_metadata=object(),
        clustered_mqa_metadata=None,
    )
    attn_backend = SimpleNamespace(forward_metadata=metadata)

    with mock.patch.object(
        dsv4_whole_layer_runtime,
        "get_is_capture_mode",
        return_value=is_graph_capture,
    ), mock.patch.object(
        dsv4_whole_layer_runtime, "get_attn_backend", return_value=attn_backend
    ), mock.patch.object(
        dsv4_whole_layer_runtime,
        "get_parallel",
        return_value=SimpleNamespace(attn_dp_rank=attn_dp_rank),
    ):
        return runtime.begin_forward(
            forward_batch=forward_batch,
            positions=torch.arange(local_rows, dtype=torch.int64),
            input_ids=torch.arange(local_rows, dtype=torch.int64),
            input_ids_global=torch.arange(global_rows, dtype=torch.int64),
        )


def test_runtime_binds_route_storage_once_outside_the_forward_hot_path():
    bind_source = textwrap.dedent(
        inspect.getsource(
            dsv4_whole_layer_runtime.DSV4WholeLayerRuntime.bind_after_weight_load
        )
    )
    bind_tree = ast.parse(bind_source)
    route_assignments = [
        node
        for node in ast.walk(bind_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_dp_packed_route_workspace"
            for target in node.targets
        )
    ]
    assert len(route_assignments) == 1
    route_binding = ast.get_source_segment(
        bind_source, route_assignments[0].value
    )
    assert route_binding is not None
    assert "_MAX_FORWARD_TOKENS" in route_binding
    assert "torch.int32" in route_binding
    assert "6" in route_binding

    routed_quant_assignments = [
        node
        for node in ast.walk(bind_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_dp_routed_quant_workspace"
            for target in node.targets
        )
    ]
    assert len(routed_quant_assignments) == 1
    routed_quant_binding = ast.get_source_segment(
        bind_source, routed_quant_assignments[0].value
    )
    assert routed_quant_binding is not None
    assert routed_quant_binding.count("_MAX_FORWARD_TOKENS") == 2
    assert "4096" in routed_quant_binding
    assert "128" in routed_quant_binding
    assert "torch.float8_e4m3fn" in routed_quant_binding
    assert "torch.uint8" in routed_quant_binding

    begin_source = textwrap.dedent(
        inspect.getsource(
            dsv4_whole_layer_runtime.DSV4WholeLayerRuntime.begin_forward
        )
    )
    begin_tree = ast.parse(begin_source)
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_dp_packed_route_workspace"
            for target in node.targets
        )
        for node in ast.walk(begin_tree)
    )
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_dp_routed_quant_workspace"
            for target in node.targets
        )
        for node in ast.walk(begin_tree)
    )
    assert "route_workspace[1][:moe_num_tokens]" in begin_source
    assert "routed_quant_workspace[1][:moe_num_tokens]" in begin_source
    assert "routed_quant_workspace[2][:moe_num_tokens]" in begin_source


def test_local_route_and_symmetric_post_selection_use_only_rank_shared_metadata():
    begin_source = textwrap.dedent(
        inspect.getsource(
            dsv4_whole_layer_runtime.DSV4WholeLayerRuntime.begin_forward
        )
    )
    begin_tree = ast.parse(begin_source)
    shared_bucket_selection = next(
        node.value
        for node in ast.walk(begin_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "shared_dp_bucket"
            for target in node.targets
        )
    )
    selection_names = {
        node.id
        for node in ast.walk(shared_bucket_selection)
        if isinstance(node, ast.Name)
    }

    assert "global_dp_sizes" in selection_names
    assert "moe_num_tokens" in selection_names
    assert "num_tokens" not in selection_names
    assert "attn_dp_rank" not in selection_names
    assert "get_parallel" not in selection_names

    for selected_name in (
        "use_dp_local_routing",
        "use_dp_symmetric_moe_post",
    ):
        selection = next(
            node.value
            for node in ast.walk(begin_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == selected_name
                for target in node.targets
            )
        )
        dynamic_names = {
            node.id for node in ast.walk(selection) if isinstance(node, ast.Name)
        }
        assert "shared_dp_bucket" in dynamic_names
        assert "num_tokens" not in dynamic_names
        assert "attn_dp_rank" not in dynamic_names
        assert "get_parallel" not in dynamic_names


def test_runtime_descriptors_reuse_fixed_packed_route_storage_across_buckets():
    runtime, route_storage = _make_runtime_with_bound_route_workspace()

    req16 = _begin_forward(runtime, (4096,) * 4, 65536)
    assert req16.moe_packed_route.shape == (65536, 6)
    assert req16.moe_routed_q_global.shape == (65536, 4096)
    assert req16.moe_routed_scale_global.shape == (65536, 128)
    assert req16.moe_packed_route.untyped_storage().data_ptr() == (
        route_storage.untyped_storage().data_ptr()
    )
    runtime.end_forward(req16)

    req32 = _begin_forward(runtime, (4096,) * 8, 131072)
    assert req32.moe_packed_route.shape == (131072, 6)
    assert req32.moe_routed_q_global.shape == (131072, 4096)
    assert req32.moe_routed_scale_global.shape == (131072, 128)
    assert req32.moe_packed_route.untyped_storage().data_ptr() == (
        route_storage.untyped_storage().data_ptr()
    )
    assert runtime._dp_packed_route_workspace[1] is route_storage
    assert req32.moe_routed_q_global.untyped_storage().data_ptr() == (
        runtime._dp_routed_quant_workspace[1].untyped_storage().data_ptr()
    )
    assert req32.moe_routed_scale_global.untyped_storage().data_ptr() == (
        runtime._dp_routed_quant_workspace[2].untyped_storage().data_ptr()
    )
    runtime.end_forward(req32)


def test_nonuniform_host_known_tp4_sizes_enable_local_route_on_every_rank():
    runtime, _ = _make_runtime_with_bound_route_workspace()
    global_sizes = (16384, 16384, 12288, 12288)
    global_rows = sum(global_sizes)

    for attn_dp_rank, local_rows in enumerate(global_sizes):
        descriptor = _begin_forward(
            runtime,
            (4096,) * (local_rows // 4096),
            global_rows,
            global_sizes=global_sizes,
            attn_dp_rank=attn_dp_rank,
        )

        assert descriptor.moe_packed_route.shape == (global_rows, 6)
        assert descriptor.moe_routed_q_global.shape == (global_rows, 4096)
        assert descriptor.moe_routed_scale_global.shape == (global_rows, 128)
        runtime.end_forward(descriptor)


@pytest.mark.parametrize(
    ("extend_lens", "global_rows", "global_sizes", "attn_dp_rank"),
    [
        ((4096,), 12288, (4096, 4096, 4096, 0), 0),
        ((4096,), 14336, (4096, 4096, 4096, 2048), 0),
        ((4096,), 12288, (4096, 4096, 4096), 0),
        ((4096,), 20480, (4096, 4096, 4096, 4096), 0),
        ((4096,), 49152, (4096, 36864, 4096, 4096), 0),
    ],
)
def test_unsupported_tp4_size_vectors_keep_the_legacy_huge_path(
    extend_lens, global_rows, global_sizes, attn_dp_rank
):
    runtime, _ = _make_runtime_with_bound_route_workspace()

    descriptor = _begin_forward(
        runtime,
        extend_lens,
        global_rows,
        global_sizes=global_sizes,
        attn_dp_rank=attn_dp_rank,
    )

    assert descriptor.moe_packed_route is None
    assert descriptor.moe_routed_q_global is None
    assert descriptor.moe_routed_scale_global is None
    runtime.end_forward(descriptor)


def test_selected_tp4_vector_rejects_rank_local_size_mismatch():
    runtime, _ = _make_runtime_with_bound_route_workspace()

    with pytest.raises(RuntimeError, match="local_M to match"):
        _begin_forward(
            runtime,
            (4096,),
            20480,
            global_sizes=(4096, 8192, 4096, 4096),
            attn_dp_rank=1,
        )


def test_symmetric_moe_workspace_is_bound_once_at_max_global_capacity():
    bind_source = textwrap.dedent(
        inspect.getsource(
            dsv4_whole_layer_runtime.DSV4WholeLayerRuntime.bind_after_weight_load
        )
    )
    assert "self._use_dp_symmetric_moe_post" in bind_source
    assert "self._bind_tp4_symmetric_moe_workspace(workspace_device)" in bind_source

    bind_workspace = getattr(
        dsv4_whole_layer_runtime.DSV4WholeLayerRuntime,
        "_bind_tp4_symmetric_moe_workspace",
    )
    workspace_source = textwrap.dedent(
        inspect.getsource(bind_workspace)
    )
    assert workspace_source.count("symm_mem.empty(") == 1
    assert "(_MAX_FORWARD_TOKENS, 4096)" in workspace_source
    assert "dtype=torch.bfloat16" in workspace_source
    assert "symm_mem.rendezvous" in workspace_source


def test_symmetric_moe_descriptor_uses_rank_shared_vector_and_local_offset():
    global_sizes = (16384, 16384, 12288, 12288)
    global_rows = sum(global_sizes)
    expected_offsets = (0, 16384, 32768, 45056)

    for rank, (local_rows, expected_offset) in enumerate(
        zip(global_sizes, expected_offsets, strict=True)
    ):
        runtime, _ = _make_runtime_with_bound_route_workspace(
            symmetric_moe_post=True, symmetric_rank=rank
        )
        descriptor = _begin_forward(
            runtime,
            (4096,) * (local_rows // 4096),
            global_rows,
            global_sizes=global_sizes,
            attn_dp_rank=rank,
        )

        assert descriptor.dp_symmetric_moe_post_selected
        assert descriptor.dp_moe_global_num_tokens == global_rows
        assert descriptor.dp_moe_attention_rank == rank
        assert descriptor.dp_moe_local_row_offset == expected_offset
        assert descriptor.moe_partial_local.shape == (global_rows, 4096)
        assert descriptor.moe_partial_local is descriptor.__getattribute__(
            f"moe_partial_peer{rank}"
        )
        runtime.end_forward(descriptor)


@pytest.mark.parametrize(
    ("global_rows", "global_sizes"),
    [
        (12288, (4096, 4096, 4096, 0)),
        (14336, (4096, 4096, 4096, 2048)),
        (12288, (4096, 4096, 4096)),
        (20480, (4096, 4096, 4096, 4096)),
        (49152, (4096, 36864, 4096, 4096)),
    ],
)
def test_invalid_shared_vector_keeps_symmetric_post_on_existing_huge_path(
    global_rows, global_sizes
):
    runtime, _ = _make_runtime_with_bound_route_workspace(
        symmetric_moe_post=True
    )
    descriptor = _begin_forward(
        runtime,
        (4096,),
        global_rows,
        global_sizes=global_sizes,
    )

    assert not descriptor.dp_symmetric_moe_post_selected
    assert descriptor.dp_moe_attention_rank == -1
    assert descriptor.dp_moe_local_row_offset == 0
    assert descriptor.moe_partial_local is None
    runtime.end_forward(descriptor)


def test_explicit_nvls_leaves_unbalanced_prefix_build_on_existing_huge_path():
    runtime, _ = _make_runtime_with_bound_route_workspace(
        symmetric_moe_post=True,
        moe_epoch=True,
        moe_epoch_counter=True,
        moe_nvls=True,
    )
    descriptor = _begin_forward(
        runtime,
        (4096,),
        12288,
        global_sizes=(4096, 4096, 4096, 0),
    )

    assert not descriptor.dp_symmetric_moe_post_selected
    assert not descriptor.dp_moe_epoch_counter_selected
    assert not descriptor.dp_moe_nvls_selected
    assert descriptor.moe_partial_local is None
    runtime.end_forward(descriptor)


def test_selected_symmetric_post_rejects_local_size_and_rank_mismatch():
    runtime, _ = _make_runtime_with_bound_route_workspace(
        symmetric_moe_post=True, symmetric_rank=0
    )
    with pytest.raises(RuntimeError, match="local_M to match"):
        _begin_forward(
            runtime,
            (4096,),
            20480,
            global_sizes=(4096, 8192, 4096, 4096),
            attn_dp_rank=1,
        )

    runtime, _ = _make_runtime_with_bound_route_workspace(
        symmetric_moe_post=True, symmetric_rank=0
    )
    with pytest.raises(RuntimeError, match="TP rank must match"):
        _begin_forward(
            runtime,
            (4096,),
            16384,
            global_sizes=(4096, 4096, 4096, 4096),
            attn_dp_rank=1,
        )


def test_symmetric_post_gate_is_one_time_and_rejects_tp_only(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_HUGE_DP_SYMM_MOE_POST", "1")
    with mock.patch.object(
        dsv4_whole_layer_runtime.DSV4WholeLayerRuntime,
        "_validate_static_config",
    ), pytest.raises(RuntimeError, match="attention-DP4 Eager"):
        dsv4_whole_layer_runtime.DSV4WholeLayerRuntime(
            config=object(),
            server_args=SimpleNamespace(enable_dp_attention=False),
        )

    with mock.patch.object(
        dsv4_whole_layer_runtime.DSV4WholeLayerRuntime,
        "_validate_static_config",
    ), mock.patch(
        "sglang.srt.layers.dp_attention.enable_dp_gatherv_for_dsv4_huge"
    ):
        runtime = dsv4_whole_layer_runtime.DSV4WholeLayerRuntime(
            config=object(),
            server_args=SimpleNamespace(enable_dp_attention=True),
        )
    monkeypatch.setenv("SGLANG_DSV4_HUGE_DP_SYMM_MOE_POST", "0")
    assert runtime._use_dp_symmetric_moe_post is True


def test_nvls_gate_requires_the_full_epoch_counter_stack(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_HUGE_DP_SYMM_MOE_POST", "1")
    monkeypatch.setenv("SGLANG_DSV4_HUGE_DP_MOE_NVLS", "1")
    monkeypatch.delenv("SGLANG_DSV4_HUGE_DP_MOE_EPOCH", raising=False)
    monkeypatch.delenv(
        "SGLANG_DSV4_HUGE_DP_MOE_EPOCH_COUNTER", raising=False
    )
    with mock.patch.object(
        dsv4_whole_layer_runtime.DSV4WholeLayerRuntime,
        "_validate_static_config",
    ), mock.patch(
        "sglang.srt.layers.dp_attention.enable_dp_gatherv_for_dsv4_huge"
    ), pytest.raises(RuntimeError, match="epoch-counter"):
        dsv4_whole_layer_runtime.DSV4WholeLayerRuntime(
            config=object(),
            server_args=SimpleNamespace(enable_dp_attention=True),
        )


def test_symmetric_post_gate_rejects_graph_capture_before_hot_path():
    runtime, _ = _make_runtime_with_bound_route_workspace(
        symmetric_moe_post=True
    )
    with pytest.raises(RuntimeError, match="symmetric MoE post is Eager-only"):
        _begin_forward(
            runtime,
            (4096,),
            16384,
            global_sizes=(4096, 4096, 4096, 4096),
            is_graph_capture=True,
        )


def test_symmetric_post_dispatch_requires_a_shared_expert_layer():
    selected = SimpleNamespace(dp_symmetric_moe_post_selected=True)
    unselected = SimpleNamespace(dp_symmetric_moe_post_selected=False)
    shared_layer = SimpleNamespace(mlp=SimpleNamespace(shared_experts=object()))
    dense_layer = SimpleNamespace(mlp=SimpleNamespace())

    assert dsv4_whole_layer_runtime._should_run_dp_symmetric_moe_post(
        shared_layer, selected
    )
    assert not dsv4_whole_layer_runtime._should_run_dp_symmetric_moe_post(
        dense_layer, selected
    )
    assert not dsv4_whole_layer_runtime._should_run_dp_symmetric_moe_post(
        shared_layer, unselected
    )


def test_symmetric_moe_post_uses_external_buffer_fused_cuda_and_two_barriers():
    global_m = 8
    local_m = 2
    partials = tuple(
        torch.empty((global_m, 4096), dtype=torch.bfloat16) for _ in range(4)
    )
    barrier = mock.Mock()
    shared_hidden = torch.empty((local_m, 4096), dtype=torch.bfloat16)
    output = torch.empty((local_m, 4, 4096), dtype=torch.bfloat16)
    descriptor = SimpleNamespace(
        moe_partial_local=partials[1],
        moe_partial_peer0=partials[0],
        moe_partial_peer1=partials[1],
        moe_partial_peer2=partials[2],
        moe_partial_peer3=partials[3],
        moe_partial_handle=SimpleNamespace(barrier=barrier),
        dp_moe_global_num_tokens=global_m,
        dp_moe_attention_rank=1,
        dp_moe_local_row_offset=2,
        dp_moe_epoch_selected=False,
        moe_epoch_flag0=None,
        moe_epoch_flag1=None,
        moe_epoch_flag2=None,
        moe_epoch_flag3=None,
        num_tokens=local_m,
        forward_batch=object(),
        input_ids=torch.arange(local_m),
        input_ids_global=torch.arange(global_m),
        moe_packed_route=None,
        moe_routed_q_global=None,
        moe_routed_scale_global=None,
        mhc_residual_out=output,
    )
    run_moe = mock.Mock(return_value=(partials[1], shared_hidden))
    layer = SimpleNamespace(_run_moe_ffn_dp_sync=run_moe)
    hidden_states = torch.empty((local_m, 4096), dtype=torch.bfloat16)
    residual = torch.empty((local_m, 4, 4096), dtype=torch.bfloat16)
    post = torch.empty((local_m, 4), dtype=torch.float32)
    comb = torch.empty((local_m, 4, 4), dtype=torch.float32)

    from sglang.jit_kernel.dsv4 import e2e
    from sglang.srt.layers.moe.moe_runner import base

    fused_post = mock.Mock(return_value=output)
    output_ctx = mock.Mock(side_effect=lambda *_args, **_kwargs: nullcontext())
    with mock.patch.object(
        base, "moe_output_buffer_ctx", output_ctx
    ), mock.patch.object(
        e2e,
        "tp4_moe_local_slice_shared_mhc_post",
        fused_post,
    ):
        result = dsv4_whole_layer_runtime._run_dp_symmetric_moe_mhc_post(
            layer=layer,
            handle=SimpleNamespace(layer_id=2, dp_moe_sequence_index=-1),
            descriptor=descriptor,
            hidden_states=hidden_states,
            shared_x_quant=(torch.empty(0), torch.empty(0)),
            routed_x_quant=None,
            use_dp_routed_prequant=False,
            residual=residual,
            post=post,
            comb=comb,
        )

    assert result is output
    output_ctx.assert_called_once_with(partials[1], external_symmetric=True)
    assert run_moe.call_args.kwargs["defer_shared_expert_add"] is True
    assert run_moe.call_args.kwargs["defer_dp_output_combine"] is True
    assert run_moe.call_args.kwargs["dp_routed_quant_output"] is None
    fused_post.assert_called_once_with(
        partials,
        2,
        shared_hidden,
        residual,
        post,
        comb,
        output,
    )
    assert barrier.call_args_list == [mock.call(channel=0), mock.call(channel=1)]


def test_gpu_epoch_moe_post_waits_before_producer_and_uses_no_barrier():
    global_m = 8
    local_m = 2
    partials = tuple(
        torch.empty((global_m, 4096), dtype=torch.bfloat16) for _ in range(4)
    )
    flags = tuple(torch.zeros((4,), dtype=torch.int32) for _ in range(4))
    barrier = mock.Mock()
    shared_hidden = torch.empty((local_m, 4096), dtype=torch.bfloat16)
    output = torch.empty((local_m, 4, 4096), dtype=torch.bfloat16)
    descriptor = SimpleNamespace(
        moe_partial_local=partials[1],
        moe_partial_peer0=partials[0],
        moe_partial_peer1=partials[1],
        moe_partial_peer2=partials[2],
        moe_partial_peer3=partials[3],
        moe_partial_handle=SimpleNamespace(barrier=barrier),
        dp_moe_global_num_tokens=global_m,
        dp_moe_attention_rank=1,
        dp_moe_local_row_offset=2,
        dp_moe_epoch_selected=True,
        dp_moe_epoch_split_selected=True,
        dp_moe_epoch_counter_selected=False,
        dp_moe_epoch_base=41,
        dp_moe_expected_done_epoch=40,
        moe_epoch_flag0=flags[0],
        moe_epoch_flag1=flags[1],
        moe_epoch_flag2=flags[2],
        moe_epoch_flag3=flags[3],
        num_tokens=local_m,
        forward_batch=object(),
        input_ids=torch.arange(local_m),
        input_ids_global=torch.arange(global_m),
        moe_packed_route=None,
        moe_routed_q_global=None,
        moe_routed_scale_global=None,
        mhc_residual_out=output,
    )
    order = []
    run_moe = mock.Mock(
        side_effect=lambda *_args, **_kwargs: (
            order.append("producer") or (partials[1], shared_hidden)
        )
    )
    layer = SimpleNamespace(_run_moe_ffn_dp_sync=run_moe)

    from sglang.jit_kernel.dsv4 import e2e
    from sglang.srt.layers.moe.moe_runner import base

    wait = mock.Mock(side_effect=lambda *_args: order.append("wait"))
    fused_epoch_split = mock.Mock(
        side_effect=lambda *_args: order.append("consumer") or output
    )
    with mock.patch.object(
        base, "moe_output_buffer_ctx", return_value=nullcontext()
    ), mock.patch.object(
        e2e, "tp4_moe_wait_slot_reusable", wait
    ), mock.patch.object(
        e2e,
        "tp4_moe_local_slice_shared_mhc_post_epoch_split",
        fused_epoch_split,
    ):
        result = dsv4_whole_layer_runtime._run_dp_symmetric_moe_mhc_post(
            layer=layer,
            handle=SimpleNamespace(layer_id=7, dp_moe_sequence_index=1),
            descriptor=descriptor,
            hidden_states=torch.empty((local_m, 4096), dtype=torch.bfloat16),
            shared_x_quant=(torch.empty(0), torch.empty(0)),
            routed_x_quant=None,
            use_dp_routed_prequant=False,
            residual=torch.empty((local_m, 4, 4096), dtype=torch.bfloat16),
            post=torch.empty((local_m, 4), dtype=torch.float32),
            comb=torch.empty((local_m, 4, 4), dtype=torch.float32),
        )

    assert result is output
    assert order == ["wait", "producer", "consumer"]
    wait.assert_called_once_with(flags, 0, 41)
    assert fused_epoch_split.call_args.args[-4:] == (flags, 1, 0, 42)
    barrier.assert_not_called()


def test_counter_epoch_uses_alternate_data_slot_and_removes_wait_launch():
    global_m = 8
    local_m = 2
    slot0 = tuple(
        torch.empty((global_m, 4096), dtype=torch.bfloat16) for _ in range(4)
    )
    slot1 = tuple(
        torch.empty((global_m, 4096), dtype=torch.bfloat16) for _ in range(4)
    )
    flags = tuple(torch.zeros((4,), dtype=torch.int32) for _ in range(4))
    completion = torch.zeros((2,), dtype=torch.int64)
    shared_hidden = torch.empty((local_m, 4096), dtype=torch.bfloat16)
    output = torch.empty((local_m, 4, 4096), dtype=torch.bfloat16)
    descriptor = SimpleNamespace(
        moe_partial_local=slot0[1],
        moe_partial_peer0=slot0[0],
        moe_partial_peer1=slot0[1],
        moe_partial_peer2=slot0[2],
        moe_partial_peer3=slot0[3],
        moe_partial_slot1_local=slot1[1],
        moe_partial_slot1_peer0=slot1[0],
        moe_partial_slot1_peer1=slot1[1],
        moe_partial_slot1_peer2=slot1[2],
        moe_partial_slot1_peer3=slot1[3],
        moe_partial_handle=SimpleNamespace(barrier=mock.Mock()),
        dp_moe_global_num_tokens=global_m,
        dp_moe_attention_rank=1,
        dp_moe_local_row_offset=2,
        dp_moe_epoch_selected=True,
        dp_moe_epoch_split_selected=False,
        dp_moe_epoch_counter_selected=True,
        dp_moe_nvls_selected=False,
        dp_moe_epoch_base=1,
        dp_moe_expected_done_epoch=0,
        moe_epoch_flag0=flags[0],
        moe_epoch_flag1=flags[1],
        moe_epoch_flag2=flags[2],
        moe_epoch_flag3=flags[3],
        moe_epoch_completion_state=completion,
        moe_multicast_local_slot0_ptr=0,
        moe_multicast_local_slot1_ptr=0,
        num_tokens=local_m,
        forward_batch=object(),
        input_ids=torch.arange(local_m),
        input_ids_global=torch.arange(global_m),
        moe_packed_route=None,
        moe_routed_q_global=None,
        moe_routed_scale_global=None,
        mhc_residual_out=output,
    )
    run_moe = mock.Mock(return_value=(slot1[1], shared_hidden))
    layer = SimpleNamespace(_run_moe_ffn_dp_sync=run_moe)

    from sglang.jit_kernel.dsv4 import e2e
    from sglang.srt.layers.moe.moe_runner import base

    output_ctx = mock.Mock(return_value=nullcontext())
    counter_post = mock.Mock(return_value=output)
    with mock.patch.object(
        base, "moe_output_buffer_ctx", output_ctx
    ), mock.patch.object(
        e2e,
        "tp4_moe_wait_slot_reusable",
        side_effect=AssertionError("counter path must not launch slot wait"),
    ), mock.patch.object(
        e2e,
        "tp4_moe_local_slice_shared_mhc_post_epoch_counter",
        counter_post,
    ):
        result = dsv4_whole_layer_runtime._run_dp_symmetric_moe_mhc_post(
            layer=layer,
            handle=SimpleNamespace(layer_id=7, dp_moe_sequence_index=1),
            descriptor=descriptor,
            hidden_states=torch.empty((local_m, 4096), dtype=torch.bfloat16),
            shared_x_quant=(torch.empty(0), torch.empty(0)),
            routed_x_quant=None,
            use_dp_routed_prequant=False,
            residual=torch.empty((local_m, 4, 4096), dtype=torch.bfloat16),
            post=torch.empty((local_m, 4), dtype=torch.float32),
            comb=torch.empty((local_m, 4, 4), dtype=torch.float32),
        )

    assert result is output
    output_ctx.assert_called_once_with(slot1[1], external_symmetric=True)
    assert all(
        actual is expected
        for actual, expected in zip(
            counter_post.call_args.args[0], slot1, strict=True
        )
    )
    assert counter_post.call_args.args[-4:] == (completion, 1, 1, 2)


def test_nvls_counter_epoch_selects_multicast_slot_without_peer_fallback():
    global_m = 8
    local_m = 2
    slot0 = tuple(
        torch.empty((global_m, 4096), dtype=torch.bfloat16) for _ in range(4)
    )
    slot1 = tuple(
        torch.empty((global_m, 4096), dtype=torch.bfloat16) for _ in range(4)
    )
    flags = tuple(torch.zeros((4,), dtype=torch.int32) for _ in range(4))
    completion = torch.zeros((2,), dtype=torch.int64)
    shared_hidden = torch.empty((local_m, 4096), dtype=torch.bfloat16)
    output = torch.empty((local_m, 4, 4096), dtype=torch.bfloat16)
    multicast_slot0 = 0x100000000
    multicast_slot1 = multicast_slot0 + (1 << 30)
    descriptor = SimpleNamespace(
        moe_partial_local=slot0[1],
        moe_partial_peer0=slot0[0],
        moe_partial_peer1=slot0[1],
        moe_partial_peer2=slot0[2],
        moe_partial_peer3=slot0[3],
        moe_partial_slot1_local=slot1[1],
        moe_partial_slot1_peer0=slot1[0],
        moe_partial_slot1_peer1=slot1[1],
        moe_partial_slot1_peer2=slot1[2],
        moe_partial_slot1_peer3=slot1[3],
        moe_partial_handle=SimpleNamespace(barrier=mock.Mock()),
        dp_moe_global_num_tokens=global_m,
        dp_moe_attention_rank=1,
        dp_moe_local_row_offset=2,
        dp_moe_epoch_selected=True,
        dp_moe_epoch_split_selected=False,
        dp_moe_epoch_counter_selected=True,
        dp_moe_nvls_selected=True,
        dp_moe_epoch_base=1,
        dp_moe_expected_done_epoch=0,
        moe_epoch_flag0=flags[0],
        moe_epoch_flag1=flags[1],
        moe_epoch_flag2=flags[2],
        moe_epoch_flag3=flags[3],
        moe_epoch_completion_state=completion,
        moe_multicast_local_slot0_ptr=multicast_slot0,
        moe_multicast_local_slot1_ptr=multicast_slot1,
        num_tokens=local_m,
        forward_batch=object(),
        input_ids=torch.arange(local_m),
        input_ids_global=torch.arange(global_m),
        moe_packed_route=None,
        moe_routed_q_global=None,
        moe_routed_scale_global=None,
        mhc_residual_out=output,
    )
    run_moe = mock.Mock(return_value=(slot1[1], shared_hidden))
    layer = SimpleNamespace(_run_moe_ffn_dp_sync=run_moe)

    from sglang.jit_kernel.dsv4 import e2e
    from sglang.srt.layers.moe.moe_runner import base

    output_ctx = mock.Mock(return_value=nullcontext())
    nvls_post = mock.Mock(return_value=output)
    with mock.patch.object(
        base, "moe_output_buffer_ctx", output_ctx
    ), mock.patch.object(
        e2e,
        "tp4_moe_local_slice_shared_mhc_post_epoch_counter_multimem",
        nvls_post,
    ), mock.patch.object(
        e2e,
        "tp4_moe_local_slice_shared_mhc_post_epoch_counter",
        side_effect=AssertionError("NVLS must not use the peer-load consumer"),
    ):
        result = dsv4_whole_layer_runtime._run_dp_symmetric_moe_mhc_post(
            layer=layer,
            handle=SimpleNamespace(layer_id=7, dp_moe_sequence_index=1),
            descriptor=descriptor,
            hidden_states=torch.empty((local_m, 4096), dtype=torch.bfloat16),
            shared_x_quant=(torch.empty(0), torch.empty(0)),
            routed_x_quant=None,
            use_dp_routed_prequant=False,
            residual=torch.empty((local_m, 4, 4096), dtype=torch.bfloat16),
            post=torch.empty((local_m, 4), dtype=torch.float32),
            comb=torch.empty((local_m, 4, 4), dtype=torch.float32),
        )

    assert result is output
    output_ctx.assert_called_once_with(slot1[1], external_symmetric=True)
    assert nvls_post.call_args.args[0] == multicast_slot1
    assert nvls_post.call_args.args[1] is slot1[1]
    assert nvls_post.call_args.args[2] == 2
    assert nvls_post.call_args.args[-4:] == (completion, 1, 1, 2)


def test_gpu_epoch_descriptor_advances_only_for_selected_batches_and_poison_fails_closed():
    runtime, _ = _make_runtime_with_bound_route_workspace(
        symmetric_moe_post=True,
        symmetric_rank=0,
        moe_epoch=True,
        moe_epoch_split=True,
    )
    first = _begin_forward(
        runtime,
        (4096,),
        16384,
        global_sizes=(4096, 4096, 4096, 4096),
    )
    assert first.dp_moe_epoch_selected
    assert first.dp_moe_epoch_split_selected
    assert first.dp_moe_epoch_base == 1
    assert first.dp_moe_expected_done_epoch == 0
    runtime.end_forward(first)

    # This invalid shared vector stays on the existing Huge path and must not
    # consume an epoch that a peer will never publish.
    unselected = _begin_forward(
        runtime,
        (4096,),
        12288,
        global_sizes=(4096, 4096, 4096, 0),
    )
    assert not unselected.dp_moe_epoch_selected
    runtime.end_forward(unselected)

    second = _begin_forward(
        runtime,
        (4096,),
        16384,
        global_sizes=(4096, 4096, 4096, 4096),
    )
    assert second.dp_moe_epoch_base == 3
    assert second.dp_moe_expected_done_epoch == 2
    runtime.abort_forward(second)
    with pytest.raises(RuntimeError, match="poisoned"):
        _begin_forward(
            runtime,
            (4096,),
            12288,
            global_sizes=(4096, 4096, 4096, 0),
        )


def test_counter_epoch_descriptor_binds_two_real_slots_and_local_counter():
    runtime, _ = _make_runtime_with_bound_route_workspace(
        symmetric_moe_post=True,
        symmetric_rank=2,
        moe_epoch=True,
        moe_epoch_counter=True,
    )
    descriptor = _begin_forward(
        runtime,
        (4096,),
        16384,
        global_sizes=(4096, 4096, 4096, 4096),
        attn_dp_rank=2,
    )

    assert descriptor.dp_moe_epoch_counter_selected
    assert not descriptor.dp_moe_epoch_split_selected
    assert descriptor.moe_partial_local.shape == (16384, 4096)
    assert descriptor.moe_partial_slot1_local.shape == (16384, 4096)
    assert (
        descriptor.moe_partial_slot1_local
        is descriptor.moe_partial_slot1_peer2
    )
    assert descriptor.moe_epoch_completion_state.shape == (2,)
    assert descriptor.moe_epoch_completion_state.dtype == torch.int64
    runtime.end_forward(descriptor)


def test_nvls_descriptor_precomputes_fixed_capacity_local_slot_addresses():
    runtime, _ = _make_runtime_with_bound_route_workspace(
        symmetric_moe_post=True,
        symmetric_rank=2,
        moe_epoch=True,
        moe_epoch_counter=True,
        moe_nvls=True,
    )
    descriptor = _begin_forward(
        runtime,
        (4096,),
        16384,
        global_sizes=(4096, 4096, 4096, 4096),
        attn_dp_rank=2,
    )

    multicast_base = 0x100000000
    row_bytes = (4096 + 4096) * 4096 * 2
    assert descriptor.dp_moe_nvls_selected
    assert descriptor.moe_multicast_local_slot0_ptr == (
        multicast_base + row_bytes
    )
    assert descriptor.moe_multicast_local_slot1_ptr == (
        multicast_base
        + dsv4_whole_layer_runtime._TP4_MOE_SLOT_BYTES
        + row_bytes
    )
    # The physical slot stride is fixed at the 131072-token allocation
    # capacity, independent of this forward's active global_M=16384.
    assert (
        descriptor.moe_multicast_local_slot1_ptr
        - descriptor.moe_multicast_local_slot0_ptr
        == 1 << 30
    )
    runtime.end_forward(descriptor)


def test_symmetric_moe_post_fails_closed_when_flashinfer_ignores_buffer():
    partials = tuple(
        torch.empty((8, 4096), dtype=torch.bfloat16) for _ in range(4)
    )
    descriptor = SimpleNamespace(
        moe_partial_local=partials[0],
        moe_partial_peer0=partials[0],
        moe_partial_peer1=partials[1],
        moe_partial_peer2=partials[2],
        moe_partial_peer3=partials[3],
        moe_partial_handle=SimpleNamespace(barrier=mock.Mock()),
        dp_moe_global_num_tokens=8,
        dp_moe_attention_rank=0,
        dp_moe_local_row_offset=0,
        dp_moe_epoch_selected=False,
        moe_epoch_flag0=None,
        moe_epoch_flag1=None,
        moe_epoch_flag2=None,
        moe_epoch_flag3=None,
        num_tokens=2,
        forward_batch=object(),
        input_ids=torch.arange(2),
        input_ids_global=torch.arange(8),
        moe_packed_route=None,
        moe_routed_q_global=None,
        moe_routed_scale_global=None,
        mhc_residual_out=torch.empty((2, 4, 4096), dtype=torch.bfloat16),
    )
    layer = SimpleNamespace(
        _run_moe_ffn_dp_sync=mock.Mock(
            return_value=(
                torch.empty((8, 4096), dtype=torch.bfloat16),
                torch.empty((2, 4096), dtype=torch.bfloat16),
            )
        )
    )
    from sglang.srt.layers.moe.moe_runner import base

    with mock.patch.object(
        base, "moe_output_buffer_ctx", return_value=nullcontext()
    ), pytest.raises(RuntimeError, match="did not honor"):
        dsv4_whole_layer_runtime._run_dp_symmetric_moe_mhc_post(
            layer=layer,
            handle=SimpleNamespace(layer_id=2, dp_moe_sequence_index=-1),
            descriptor=descriptor,
            hidden_states=torch.empty((2, 4096), dtype=torch.bfloat16),
            shared_x_quant=(torch.empty(0), torch.empty(0)),
            routed_x_quant=None,
            use_dp_routed_prequant=False,
            residual=torch.empty((2, 4, 4096), dtype=torch.bfloat16),
            post=torch.empty((2, 4), dtype=torch.float32),
            comb=torch.empty((2, 4, 4), dtype=torch.float32),
        )


class _Experts:
    def __init__(self):
        self.moe_runner_config = SimpleNamespace(inplace=True)
        self.quant_method = object()
        self.calls = []

    def __call__(self, hidden_states, topk_output, prequant=None):
        self.calls.append((hidden_states, topk_output, prequant))
        # FlashInfer returns BF16 even though Stage2 uses the global FP8 tensor
        # only as a shape/device carrier.
        return torch.empty(hidden_states.shape, dtype=torch.bfloat16)


class _RouteAwareMoe:
    """Small DeepseekV2MoE-compatible object used by the V4 layer test."""

    def __init__(self, local_rows: int):
        self.is_hash = False
        self.top_k = 6
        self.layer_id = 2
        self._enable_a2a_moe = False
        self._fuse_shared_experts_inside_sbo = False
        self._shared_expert_tp1 = False
        self.num_fused_shared_experts = 0
        self.routed_scaling_factor = 1.0
        self.tp_size = 1
        self.alt_stream = None
        self.experts = _Experts()
        self._validate_dsv4_huge_local_packed_route = mock.Mock()
        self._get_expert_location_dispatch_info = mock.Mock(return_value=None)

        router_logits = torch.empty((local_rows, 256), dtype=torch.bfloat16)
        local_packed = torch.arange(
            local_rows * self.top_k, dtype=torch.int32
        ).reshape(local_rows, self.top_k)
        local_standard = StandardTopKOutputPacked(
            topk_weights=torch.empty((local_rows, self.top_k)),
            topk_ids=torch.empty((local_rows, self.top_k), dtype=torch.int32),
            router_logits=router_logits,
            packed_topk_ids=local_packed,
        )
        self.gate = mock.Mock(return_value=router_logits)
        self.topk = mock.Mock(return_value=local_standard)

    def build_dsv4_huge_local_packed_route(self, *args, **kwargs):
        return deepseek_v2.DeepseekV2MoE.build_dsv4_huge_local_packed_route(
            self, *args, **kwargs
        )

    def __call__(self, hidden_states, forward_batch, **kwargs):
        return deepseek_v2.DeepseekV2MoE.forward_normal(
            self,
            hidden_states,
            input_ids=kwargs.get("input_ids"),
            input_ids_global=kwargs.get("input_ids_global"),
            # The integration fixture isolates routed-MoE control flow.  The
            # production Attention-DP path computes the TP1 shared expert on
            # the decoder layer before entering this call.
            skip_shared_experts=True,
            shared_x_quant=kwargs.get("shared_x_quant"),
            routed_x_quant=kwargs.get("routed_x_quant"),
            precomputed_topk_output=kwargs.get("precomputed_topk_output"),
        )


class _DenseMlp:
    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, forward_batch, **kwargs):
        self.calls.append((hidden_states, forward_batch, kwargs))
        return hidden_states.clone()


class _TensorRouteMlp(_DenseMlp):
    def build_dsv4_huge_local_packed_route(self, hidden_states, input_ids=None):
        return torch.empty((hidden_states.shape[0], 6), dtype=torch.int32)


def _invoke_dp_moe_path(
    mlp,
    *,
    grouped_side_effect=None,
    packed_route_workspace=None,
    provide_local_routed_quant: bool = True,
    provide_global_routed_quant: bool = True,
    corrupt_local_routed_q: bool = False,
):
    local_hidden = torch.empty((2, 4096), dtype=torch.bfloat16)
    global_hidden = torch.empty((8, 4096), dtype=torch.bfloat16)
    local_routed_q = torch.empty(
        (2, 4096),
        dtype=(torch.bfloat16 if corrupt_local_routed_q else torch.float8_e4m3fn),
    )
    local_routed_scale = torch.empty((2, 128), dtype=torch.uint8)
    global_routed_q = torch.empty((8, 4096), dtype=torch.float8_e4m3fn)
    global_routed_scale = torch.empty((8, 128), dtype=torch.uint8)
    local_output = torch.empty_like(local_hidden)
    packed_route_workspace = (
        torch.empty((8, 6), dtype=torch.int32)
        if packed_route_workspace is None
        else packed_route_workspace
    )
    input_ids = torch.arange(2, dtype=torch.int64)
    input_ids_global = torch.arange(8, dtype=torch.int64)
    forward_batch = SimpleNamespace(dp_padding_mode=None)
    layer = SimpleNamespace(dsa_enable_prefill_cp=False, mlp=mlp)
    parallel = SimpleNamespace(
        attn_dp_size=4,
        attn_tp_size=1,
        tp_size=4,
    )
    moe_backend = SimpleNamespace(is_none=lambda: True)
    server_args = SimpleNamespace(dsv4_worker_backend="huge_kernel")
    forward_context = SimpleNamespace(scoped=lambda **_: nullcontext())
    envs = SimpleNamespace(
        SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER=SimpleNamespace(
            get=lambda: False
        ),
        SGLANG_DP_USE_REDUCE_SCATTER=SimpleNamespace(get=lambda: False),
    )
    tp_group = object()

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(deepseek_v4, "get_parallel", return_value=parallel)
        )
        stack.enter_context(
            mock.patch.object(
                deepseek_v4, "get_moe_a2a_backend", return_value=moe_backend
            )
        )
        stack.enter_context(
            mock.patch.object(deepseek_v4, "get_server_args", return_value=server_args)
        )
        stack.enter_context(
            mock.patch.object(deepseek_v4, "get_tp_group", return_value=tp_group)
        )
        stack.enter_context(
            mock.patch.object(deepseek_v4, "get_forward", return_value=forward_context)
        )
        stack.enter_context(mock.patch.object(deepseek_v4, "envs", envs))
        stack.enter_context(
            mock.patch.object(deepseek_v4, "_SHARED_EXPERT_LOCAL", False)
        )
        stack.enter_context(
            mock.patch.object(deepseek_v4, "is_dp_gatherv_active", return_value=False)
        )
        stack.enter_context(
            mock.patch.object(
                deepseek_v4, "should_use_dp_reduce_scatterv", return_value=False
            )
        )
        stack.enter_context(
            mock.patch.object(
                deepseek_v4,
                "is_dsv4_huge_dp_balanced_max_len_enabled",
                return_value=False,
            )
        )
        global_buffer = stack.enter_context(
            mock.patch.object(
                deepseek_v4, "get_global_dp_buffer", return_value=global_hidden
            )
        )
        stack.enter_context(
            mock.patch.object(
                deepseek_v4, "get_local_dp_buffer", return_value=local_output
            )
        )
        regular_gather = stack.enter_context(
            mock.patch.object(deepseek_v4, "dp_gather_partial")
        )
        grouped_gather = stack.enter_context(
            mock.patch.object(
                deepseek_v4,
                "dp_gather_partial_grouped",
                side_effect=grouped_side_effect,
            )
        )
        scatter = stack.enter_context(mock.patch.object(deepseek_v4, "dp_scatter"))
        stack.enter_context(
            mock.patch.object(
                deepseek_v2,
                "maybe_fuse_routed_scale_and_shared_add",
                side_effect=lambda _experts, routed, _shared, _scale: routed,
            )
        )
        stack.enter_context(
            mock.patch.object(deepseek_v2, "get_server_args", return_value=server_args)
        )

        output = deepseek_v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync(
            layer,
            local_hidden,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
            routed_x_quant=(
                (local_routed_q, local_routed_scale)
                if provide_local_routed_quant
                else None
            ),
            dp_packed_route_output=packed_route_workspace,
            dp_routed_quant_output=(
                (global_routed_q, global_routed_scale)
                if provide_global_routed_quant
                else None
            ),
        )

    return SimpleNamespace(
        output=output,
        local_hidden=local_hidden,
        global_hidden=global_hidden,
        local_routed_q=local_routed_q,
        local_routed_scale=local_routed_scale,
        global_routed_q=global_routed_q,
        global_routed_scale=global_routed_scale,
        local_output=local_output,
        packed_route_workspace=packed_route_workspace,
        forward_batch=forward_batch,
        regular_gather=regular_gather,
        grouped_gather=grouped_gather,
        global_buffer=global_buffer,
        scatter=scatter,
    )


def test_grouped_gather_routes_locally_and_skips_global_gate_and_topk():
    moe = _RouteAwareMoe(local_rows=2)

    def grouped_copy(outputs, inputs, _forward_batch):
        global_q_bytes, global_scale, global_packed = outputs
        local_q_bytes, local_scale, local_packed = inputs
        assert isinstance(local_packed, torch.Tensor)
        assert global_q_bytes.dtype == torch.uint8
        assert local_q_bytes.dtype == torch.uint8
        global_q_bytes[: local_q_bytes.shape[0]].copy_(local_q_bytes)
        global_scale[: local_scale.shape[0]].copy_(local_scale)
        global_packed[: local_packed.shape[0]].copy_(local_packed)

    result = _invoke_dp_moe_path(moe, grouped_side_effect=grouped_copy)

    result.grouped_gather.assert_called_once()
    result.regular_gather.assert_not_called()
    result.global_buffer.assert_not_called()
    assert result.output is result.local_output
    assert moe.gate.call_count == 1
    assert moe.gate.call_args.args[0] is result.local_hidden
    assert moe.topk.call_count == 1
    assert moe.topk.call_args.args[0] is result.local_hidden
    assert len(moe.experts.calls) == 1
    consumed_hidden, consumed_topk, consumed_prequant = moe.experts.calls[0]
    assert consumed_hidden is result.global_routed_q
    assert isinstance(consumed_topk, PackedOnlyTopKOutput)
    assert consumed_topk.packed_topk_ids is result.packed_route_workspace
    assert consumed_prequant[0] is result.global_routed_q
    assert consumed_prequant[1] is result.global_routed_scale


def test_dense_layer_ignores_packed_workspace_and_keeps_legacy_gather():
    dense = _DenseMlp()

    result = _invoke_dp_moe_path(
        dense,
        provide_local_routed_quant=False,
        provide_global_routed_quant=False,
    )

    result.regular_gather.assert_called_once()
    gather_args = result.regular_gather.call_args.args
    assert gather_args[0] is result.global_hidden
    assert gather_args[1] is result.local_hidden
    assert gather_args[2] is result.forward_batch
    result.grouped_gather.assert_not_called()
    assert len(dense.calls) == 1
    assert "precomputed_topk_output" not in dense.calls[0][2]


def test_grouped_gather_exception_is_propagated_without_fallback():
    sentinel = RuntimeError("grouped route gather failed")
    mlp = _TensorRouteMlp()

    with pytest.raises(RuntimeError) as exc_info:
        _invoke_dp_moe_path(mlp, grouped_side_effect=sentinel)

    assert exc_info.value is sentinel
    assert mlp.calls == []


def test_local_routed_quant_without_global_destination_fails_closed():
    moe = _RouteAwareMoe(local_rows=2)

    with pytest.raises(RuntimeError, match="complete local-route MXFP8"):
        _invoke_dp_moe_path(moe, provide_global_routed_quant=False)

    assert moe.gate.call_count == 0
    assert moe.experts.calls == []


def test_invalid_local_routed_quant_workspace_fails_before_collective():
    moe = _RouteAwareMoe(local_rows=2)

    with pytest.raises(RuntimeError, match="invalid DSV4 Huge local/global"):
        _invoke_dp_moe_path(moe, corrupt_local_routed_q=True)

    assert moe.gate.call_count == 0
    assert moe.experts.calls == []


def test_routed_quant_carrier_rejects_nonlocal_shared_expert():
    moe = _RouteAwareMoe(local_rows=2)
    moe.shared_experts = object()

    with pytest.raises(RuntimeError, match="shared experts to stay on the local"):
        _invoke_dp_moe_path(moe)

    assert moe.gate.call_count == 0
    assert moe.experts.calls == []


def test_huge_executor_enables_routed_quant_only_for_formal_local_route():
    source = textwrap.dedent(
        inspect.getsource(dsv4_whole_layer_runtime._execute_common)
    )

    assert "not runtime._attention_dp4 or use_dp_routed_prequant" in source
    assert "routed_x_quant=routed_x_quant" in source
    assert "dp_routed_quant_output=" in source
