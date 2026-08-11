from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.layers.moe.token_dispatcher import standard as standard_module
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import (
    PackedOnlyTopKOutput,
    StandardTopKOutput,
    StandardTopKOutputPacked,
)
from sglang.srt.models import deepseek_v2 as deepseek_v2_module
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


TOP_K = 6


class _A2ABackend:
    def __init__(self, *, is_none: bool):
        self._is_none = is_none

    def is_none(self):
        return self._is_none


class _MoeRunnerBackend:
    def __init__(self, *, is_flashinfer_mxfp4: bool):
        self._is_flashinfer_mxfp4 = is_flashinfer_mxfp4

    def is_flashinfer_mxfp4(self):
        return self._is_flashinfer_mxfp4


def _server_args(*, backend="huge_kernel"):
    return SimpleNamespace(
        dsv4_worker_backend=backend,
        enable_dp_attention=True,
    )


def _parallel():
    return SimpleNamespace(
        tp_size=4,
        attn_dp_size=4,
        attn_tp_size=1,
        attn_dp_rank=2,
    )


def _patch_valid_huge_runtime(monkeypatch, module):
    monkeypatch.setattr(module, "get_server_args", lambda: _server_args())
    monkeypatch.setattr(module, "get_parallel", _parallel)
    monkeypatch.setattr(
        module, "get_moe_a2a_backend", lambda: _A2ABackend(is_none=True)
    )
    if module is deepseek_v2_module:
        monkeypatch.setattr(
            module,
            "get_moe_runner_backend",
            lambda: _MoeRunnerBackend(is_flashinfer_mxfp4=True),
        )


def _bare_moe():
    moe = object.__new__(DeepseekV2MoE)
    torch.nn.Module.__init__(moe)
    moe.top_k = TOP_K
    moe.num_fused_shared_experts = 0
    moe._enable_a2a_moe = False
    return moe


def test_local_packed_route_uses_local_hash_input_ids(monkeypatch):
    _patch_valid_huge_runtime(monkeypatch, deepseek_v2_module)
    moe = _bare_moe()
    moe.is_hash = True
    hidden_states = torch.randn(3, 8)
    local_input_ids = torch.tensor([7, 11, 13])
    router_logits = torch.randn(3, 256)
    packed = torch.arange(3 * TOP_K, dtype=torch.int32).view(3, TOP_K)
    gate = mock.Mock(return_value=router_logits)
    topk = mock.Mock(
        return_value=StandardTopKOutputPacked(
            torch.rand(3, TOP_K),
            torch.zeros((3, TOP_K), dtype=torch.int32),
            router_logits,
            packed,
        )
    )
    dispatch_info = object()
    moe.gate = gate
    moe.topk = topk
    moe._get_expert_location_dispatch_info = lambda: dispatch_info

    result = moe.build_dsv4_huge_local_packed_route(
        hidden_states,
        input_ids=local_input_ids,
    )

    assert result is packed
    gate.assert_called_once_with(hidden_states, None)
    assert topk.call_args.args == (hidden_states, router_logits)
    assert topk.call_args.kwargs["input_ids"] is local_input_ids
    assert topk.call_args.kwargs["expert_location_dispatch_info"] is dispatch_info


def test_local_packed_route_fails_closed_outside_huge(monkeypatch):
    _patch_valid_huge_runtime(monkeypatch, deepseek_v2_module)
    monkeypatch.setattr(
        deepseek_v2_module,
        "get_server_args",
        lambda: _server_args(backend="native"),
    )
    moe = _bare_moe()
    moe.is_hash = False
    moe.gate = mock.Mock(side_effect=AssertionError("gate must not run"))

    with pytest.raises(RuntimeError, match="requires TP4/Attention-DP4"):
        moe.build_dsv4_huge_local_packed_route(torch.randn(2, 8))

    moe.gate.assert_not_called()


class _Experts:
    def __init__(self):
        self.moe_runner_config = SimpleNamespace(inplace=True)
        self.quant_method = object()
        self.calls = []

    def __call__(self, hidden_states, topk_output, *, prequant=None):
        self.calls.append((hidden_states, topk_output, prequant))
        return hidden_states.clone()


def _prepare_forward_normal_moe():
    moe = _bare_moe()
    moe.is_hash = False
    moe.experts = _Experts()
    moe._fuse_shared_experts_inside_sbo = False
    moe._shared_expert_tp1 = False
    moe.routed_scaling_factor = 1.0
    moe.tp_size = 1
    moe.layer_id = 4
    return moe


def test_precomputed_packed_route_skips_global_gate_and_topk(monkeypatch):
    _patch_valid_huge_runtime(monkeypatch, deepseek_v2_module)
    monkeypatch.setattr(
        deepseek_v2_module,
        "maybe_fuse_routed_scale_and_shared_add",
        lambda _experts, routed, _shared, _scale: routed,
    )
    moe = _prepare_forward_normal_moe()
    moe.gate = mock.Mock(side_effect=AssertionError("global gate must not run"))
    moe.topk = mock.Mock(side_effect=AssertionError("global top-k must not run"))
    hidden_states = torch.randn(8, 16)
    carrier = PackedOnlyTopKOutput(
        torch.empty((hidden_states.shape[0], TOP_K), dtype=torch.int32)
    )

    result = moe.forward_normal(
        hidden_states,
        skip_shared_experts=True,
        precomputed_topk_output=carrier,
    )

    assert torch.equal(result, hidden_states)
    moe.gate.assert_not_called()
    moe.topk.assert_not_called()
    assert len(moe.experts.calls) == 1
    called_hidden, called_topk, called_prequant = moe.experts.calls[0]
    assert called_hidden is hidden_states
    assert called_topk is carrier
    assert called_prequant is None


def test_no_precomputed_route_preserves_native_gate_and_topk(monkeypatch):
    monkeypatch.setattr(
        deepseek_v2_module,
        "get_server_args",
        lambda: _server_args(backend="native"),
    )
    monkeypatch.setattr(
        deepseek_v2_module,
        "maybe_fuse_routed_scale_and_shared_add",
        lambda _experts, routed, _shared, _scale: routed,
    )
    moe = _prepare_forward_normal_moe()
    hidden_states = torch.randn(5, 16)
    router_logits = torch.randn(5, 256)
    native_topk_output = StandardTopKOutput(
        torch.rand(5, TOP_K),
        torch.zeros((5, TOP_K), dtype=torch.int32),
        router_logits,
    )
    moe.gate = mock.Mock(return_value=router_logits)
    moe.topk = mock.Mock(return_value=native_topk_output)
    moe._get_expert_location_dispatch_info = lambda: None

    moe.forward_normal(hidden_states, skip_shared_experts=True)

    moe.gate.assert_called_once_with(hidden_states, None)
    moe.topk.assert_called_once_with(
        hidden_states,
        router_logits,
        expert_location_dispatch_info=None,
    )
    assert moe.experts.calls[0][1] is native_topk_output


def _dispatcher(monkeypatch, *, backend="huge_kernel"):
    _patch_valid_huge_runtime(monkeypatch, standard_module)
    monkeypatch.setattr(
        standard_module,
        "get_server_args",
        lambda: _server_args(backend=backend),
    )
    monkeypatch.setattr(
        standard_module,
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
        lambda: False,
    )
    tp_group = SimpleNamespace(
        world_size=4,
        rank_in_group=2,
        all_gatherv=mock.Mock(side_effect=AssertionError("second gather is forbidden")),
    )
    monkeypatch.setattr(standard_module, "get_tp_group", lambda: tp_group)

    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.top_k = TOP_K
    dispatcher.enable_flashinfer_mxfp4_moe = True
    dispatcher.moe_ep_size = 4
    dispatcher.skip_local_expert_mapping = True
    dispatcher.local_expert_mapping = None
    dispatcher.use_aiter_moe_runner = False
    dispatcher.expert_mask_gpu = None
    return dispatcher, tp_group


def test_dispatcher_accepts_only_global_carrier_without_second_gather(monkeypatch):
    dispatcher, tp_group = _dispatcher(monkeypatch)
    hidden_states = torch.randn(8, 16)
    carrier = PackedOnlyTopKOutput(torch.empty((8, TOP_K), dtype=torch.int32))

    output = dispatcher.dispatch(hidden_states, carrier)

    assert output.hidden_states is hidden_states
    assert output.topk_output is carrier
    tp_group.all_gatherv.assert_not_called()


def test_dispatcher_rejects_local_sized_carrier(monkeypatch):
    dispatcher, tp_group = _dispatcher(monkeypatch)
    hidden_states = torch.randn(8, 16)
    local_carrier = PackedOnlyTopKOutput(
        torch.empty((2, TOP_K), dtype=torch.int32)
    )

    with pytest.raises(RuntimeError, match="not globally aligned"):
        dispatcher.dispatch(hidden_states, local_carrier)

    tp_group.all_gatherv.assert_not_called()


def test_dispatcher_rejects_packed_only_carrier_in_native(monkeypatch):
    dispatcher, tp_group = _dispatcher(monkeypatch, backend="native")
    hidden_states = torch.randn(8, 16)
    carrier = PackedOnlyTopKOutput(torch.empty((8, TOP_K), dtype=torch.int32))

    with pytest.raises(RuntimeError, match="requires TP4/Attention-DP4"):
        dispatcher.dispatch(hidden_states, carrier)

    tp_group.all_gatherv.assert_not_called()
