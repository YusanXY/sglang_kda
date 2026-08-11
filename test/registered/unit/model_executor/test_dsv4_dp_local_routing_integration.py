"""Integration contracts for DSV4 Huge Attention-DP local routing.

These tests intentionally exercise the Python orchestration boundary with CPU
tensors and mocked collectives.  CUDA kernel correctness is covered elsewhere;
the contracts here pin workspace lifetime, route ownership, and fail-closed
control flow without requiring a distributed process group.
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


def _make_forward_batch(extend_lens: tuple[int, ...], global_rows: int):
    local_rows = sum(extend_lens)
    requests = len(extend_lens)
    return SimpleNamespace(
        forward_mode=_ExtendMode(),
        req_pool_indices=torch.arange(requests, dtype=torch.int32),
        seq_lens=torch.full((requests,), local_rows, dtype=torch.int32),
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=extend_lens,
        out_cache_loc=torch.arange(local_rows, dtype=torch.int32),
        global_dp_buffer_len=global_rows,
        global_num_tokens_cpu=[local_rows] * 4,
        dp_padding_mode=None,
    )


def _make_runtime_with_bound_route_workspace():
    runtime = object.__new__(dsv4_whole_layer_runtime.DSV4WholeLayerRuntime)
    route_storage = torch.empty((131072, 6), dtype=torch.int32)
    scratch = torch.empty(0)

    runtime._active = None
    runtime._handles = (object(),)
    runtime._generation = 1
    runtime._attention_dp4 = True
    runtime._max_local_tokens = 32768
    runtime._use_tp4_token_shard_attention = False
    runtime._use_clustered_mqa = False
    runtime._use_dp_local_routing = True
    runtime._use_tp4_local_wob = False
    runtime._dp_packed_route_workspace = (route_storage.device, route_storage)
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


def _begin_forward(runtime, extend_lens: tuple[int, ...], global_rows: int):
    local_rows = sum(extend_lens)
    forward_batch = _make_forward_batch(extend_lens, global_rows)
    metadata = SimpleNamespace(
        core_attn_metadata=object(),
        indexer_metadata=object(),
        clustered_mqa_metadata=None,
    )
    attn_backend = SimpleNamespace(forward_metadata=metadata)

    with mock.patch.object(
        dsv4_whole_layer_runtime, "get_is_capture_mode", return_value=False
    ), mock.patch.object(
        dsv4_whole_layer_runtime, "get_attn_backend", return_value=attn_backend
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
    assert "route_workspace[1][:moe_num_tokens]" in begin_source


def test_runtime_descriptors_reuse_fixed_packed_route_storage_across_buckets():
    runtime, route_storage = _make_runtime_with_bound_route_workspace()

    req16 = _begin_forward(runtime, (4096,) * 4, 65536)
    assert req16.moe_packed_route.shape == (65536, 6)
    assert req16.moe_packed_route.untyped_storage().data_ptr() == (
        route_storage.untyped_storage().data_ptr()
    )
    runtime.end_forward(req16)

    req32 = _begin_forward(runtime, (4096,) * 8, 131072)
    assert req32.moe_packed_route.shape == (131072, 6)
    assert req32.moe_packed_route.untyped_storage().data_ptr() == (
        route_storage.untyped_storage().data_ptr()
    )
    assert runtime._dp_packed_route_workspace[1] is route_storage
    runtime.end_forward(req32)


@pytest.mark.parametrize(
    ("extend_lens", "global_rows"),
    [
        ((4096,), 16384),
        ((2048, 2048), 65536),
    ],
)
def test_prefix_and_nonbucket_huge_forwards_do_not_enable_local_route(
    extend_lens, global_rows
):
    runtime, _ = _make_runtime_with_bound_route_workspace()

    descriptor = _begin_forward(runtime, extend_lens, global_rows)

    assert descriptor.moe_packed_route is None
    runtime.end_forward(descriptor)


class _Experts:
    def __init__(self):
        self.moe_runner_config = SimpleNamespace(inplace=True)
        self.quant_method = object()
        self.calls = []

    def __call__(self, hidden_states, topk_output, prequant=None):
        self.calls.append((hidden_states, topk_output, prequant))
        return hidden_states.clone()


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
):
    local_hidden = torch.arange(8, dtype=torch.float32).to(torch.bfloat16).reshape(2, 4)
    global_hidden = torch.empty((8, 4), dtype=torch.bfloat16)
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
        stack.enter_context(
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
            dp_packed_route_output=packed_route_workspace,
        )

    return SimpleNamespace(
        output=output,
        local_hidden=local_hidden,
        global_hidden=global_hidden,
        local_output=local_output,
        packed_route_workspace=packed_route_workspace,
        forward_batch=forward_batch,
        regular_gather=regular_gather,
        grouped_gather=grouped_gather,
        scatter=scatter,
    )


def test_grouped_gather_routes_locally_and_skips_global_gate_and_topk():
    moe = _RouteAwareMoe(local_rows=2)

    def grouped_copy(outputs, inputs, _forward_batch):
        global_hidden, global_packed = outputs
        local_hidden, local_packed = inputs
        assert isinstance(local_packed, torch.Tensor)
        global_hidden[: local_hidden.shape[0]].copy_(local_hidden)
        global_packed[: local_packed.shape[0]].copy_(local_packed)

    result = _invoke_dp_moe_path(moe, grouped_side_effect=grouped_copy)

    result.grouped_gather.assert_called_once()
    result.regular_gather.assert_not_called()
    assert result.output is result.local_output
    assert moe.gate.call_count == 1
    assert moe.gate.call_args.args[0] is result.local_hidden
    assert moe.topk.call_count == 1
    assert moe.topk.call_args.args[0] is result.local_hidden
    assert len(moe.experts.calls) == 1
    _, consumed_topk, _ = moe.experts.calls[0]
    assert isinstance(consumed_topk, PackedOnlyTopKOutput)
    assert consumed_topk.packed_topk_ids is result.packed_route_workspace


def test_dense_layer_ignores_packed_workspace_and_keeps_legacy_gather():
    dense = _DenseMlp()

    result = _invoke_dp_moe_path(dense)

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
