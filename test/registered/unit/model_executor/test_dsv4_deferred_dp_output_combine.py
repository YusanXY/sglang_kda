"""Control-plane contracts for the deferred DSV4 Attention-DP combine.

The Huge runtime can fuse TP partial reduction, local-token slicing, shared
expert addition, and mHC post only when the decoder layer exposes the global
BF16 MoE partial without first materializing SGLang's local DP output buffer.
These CPU-only tests pin that boundary and its fail-closed support matrix.
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

from sglang.srt.models import deepseek_v4


class _Mlp:
    def __init__(self, *, with_local_shared: bool = False):
        self.shared_experts = object() if with_local_shared else None
        self._shared_expert_tp1 = with_local_shared
        self.alt_stream = None
        self.calls = []
        self.shared_calls = []

    def _forward_shared_experts(self, hidden_states, x_quant=None):
        self.shared_calls.append((hidden_states, x_quant))
        return torch.empty_like(hidden_states)

    def __call__(self, hidden_states, forward_batch, **kwargs):
        self.calls.append((hidden_states, forward_batch, kwargs))
        # The routed experts consume global-M rows and produce a BF16 TP
        # partial with the same global row ownership.
        return torch.empty(hidden_states.shape, dtype=torch.bfloat16)


def _invoke(
    *,
    defer: bool,
    backend: str = "huge_kernel",
    attn_dp_size: int = 4,
    attn_tp_size: int = 1,
    a2a_none: bool = True,
    use_cp: bool = False,
    use_a2a_scatter: bool = False,
    use_reduce_scatterv: bool = True,
    with_local_shared: bool = False,
):
    local_hidden = torch.empty((2, 4096), dtype=torch.bfloat16)
    global_hidden = torch.empty((8, 4096), dtype=torch.bfloat16)
    local_reduced = torch.empty_like(local_hidden)
    input_ids = torch.arange(2, dtype=torch.int64)
    input_ids_global = torch.arange(8, dtype=torch.int64)
    forward_batch = SimpleNamespace(dp_padding_mode=None)
    mlp = _Mlp(with_local_shared=with_local_shared)
    layer = SimpleNamespace(
        dsa_enable_prefill_cp=use_cp,
        mlp=mlp,
        _dsv4_huge_dp_shared_expert_local=with_local_shared,
    )
    parallel = SimpleNamespace(
        attn_dp_size=attn_dp_size,
        attn_tp_size=attn_tp_size,
        tp_size=4,
    )
    moe_backend = SimpleNamespace(
        is_none=lambda: a2a_none,
        is_deepep=lambda: not a2a_none,
    )
    server_args = SimpleNamespace(dsv4_worker_backend=backend)
    forward_context = SimpleNamespace(
        scoped=mock.Mock(side_effect=lambda **_: nullcontext())
    )
    envs = SimpleNamespace(
        SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER=SimpleNamespace(
            get=lambda: use_a2a_scatter
        ),
        SGLANG_DP_USE_REDUCE_SCATTER=SimpleNamespace(get=lambda: False),
    )
    tp_group = SimpleNamespace(reduce_scatterv=mock.Mock())

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
            mock.patch.object(
                deepseek_v4, "get_server_args", return_value=server_args
            )
        )
        stack.enter_context(
            mock.patch.object(deepseek_v4, "get_tp_group", return_value=tp_group)
        )
        stack.enter_context(
            mock.patch.object(
                deepseek_v4, "get_forward", return_value=forward_context
            )
        )
        stack.enter_context(mock.patch.object(deepseek_v4, "envs", envs))
        stack.enter_context(
            mock.patch.object(deepseek_v4, "_SHARED_EXPERT_LOCAL", False)
        )
        stack.enter_context(
            mock.patch.object(
                deepseek_v4, "dsa_use_prefill_cp", return_value=use_cp
            )
        )
        stack.enter_context(
            mock.patch.object(
                deepseek_v4, "is_dp_gatherv_active", return_value=False
            )
        )
        use_rsv = stack.enter_context(
            mock.patch.object(
                deepseek_v4,
                "should_use_dp_reduce_scatterv",
                return_value=use_reduce_scatterv,
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
        local_buffer = stack.enter_context(
            mock.patch.object(
                deepseek_v4, "get_local_dp_buffer", return_value=local_reduced
            )
        )
        gather = stack.enter_context(
            mock.patch.object(deepseek_v4, "dp_gather_partial")
        )
        scatter = stack.enter_context(mock.patch.object(deepseek_v4, "dp_scatter"))
        equal_scatter = stack.enter_context(
            mock.patch.object(deepseek_v4, "dp_reduce_scatter_tensor")
        )
        cp_gather = stack.enter_context(
            mock.patch.object(deepseek_v4, "dsa_cp_gather_hidden_states")
        )

        error = None
        output = None
        try:
            output = deepseek_v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync(
                layer,
                local_hidden,
                forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                defer_dp_output_combine=defer,
            )
        except Exception as exc:  # returned for fail-closed assertions
            error = exc

    return SimpleNamespace(
        output=output,
        error=error,
        mlp=mlp,
        local_hidden=local_hidden,
        global_hidden=global_hidden,
        local_reduced=local_reduced,
        forward_context=forward_context,
        tp_group=tp_group,
        use_rsv=use_rsv,
        global_buffer=global_buffer,
        local_buffer=local_buffer,
        gather=gather,
        scatter=scatter,
        equal_scatter=equal_scatter,
        cp_gather=cp_gather,
    )


def _call_names(nodes):
    return {
        node.func.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_deferred_combine_ast_keeps_local_buffer_and_collectives_outside_branch():
    source = textwrap.dedent(
        inspect.getsource(deepseek_v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync)
    )
    tree = ast.parse(source)
    function = tree.body[0]

    kw_defaults = dict(
        zip(
            (arg.arg for arg in function.args.kwonlyargs),
            function.args.kw_defaults,
        )
    )
    assert isinstance(kw_defaults["defer_dp_output_combine"], ast.Constant)
    assert kw_defaults["defer_dp_output_combine"].value is False

    combine_guard = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "defer_dp_output_combine"
    )
    forbidden = {
        "get_local_dp_buffer",
        "dp_scatter",
        "dp_reduce_scatter_tensor",
    }
    assert _call_names(combine_guard.body).isdisjoint(forbidden)
    assert forbidden <= _call_names(combine_guard.orelse)

    mlp_reduce_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mlp_reduce_scatter"
            for target in node.targets
        )
    )
    assert "defer_dp_output_combine" in {
        node.id
        for node in ast.walk(mlp_reduce_assignment)
        if isinstance(node, ast.Name)
    }


def test_deferred_combine_returns_global_partial_without_local_materialization():
    result = _invoke(defer=True)

    assert result.error is None
    global_partial, shared_hidden = result.output
    assert global_partial.shape == (8, 4096)
    assert global_partial.dtype == torch.bfloat16
    assert shared_hidden is None
    result.global_buffer.assert_called_once()
    result.gather.assert_called_once_with(
        result.global_hidden, result.local_hidden, mock.ANY
    )
    result.local_buffer.assert_not_called()
    result.tp_group.reduce_scatterv.assert_not_called()
    result.scatter.assert_not_called()
    result.equal_scatter.assert_not_called()
    result.forward_context.scoped.assert_called_once_with(mlp_reduce_scatter=True)


def test_deferred_combine_keeps_shared_expert_rank_local_and_returns_it():
    result = _invoke(defer=True, with_local_shared=True)

    assert result.error is None
    global_partial, shared_hidden = result.output
    assert global_partial.shape == (8, 4096)
    assert shared_hidden.shape == (2, 4096)
    assert shared_hidden.dtype == torch.bfloat16
    assert len(result.mlp.shared_calls) == 1
    assert result.mlp.shared_calls[0][0] is result.local_hidden
    result.local_buffer.assert_not_called()
    result.tp_group.reduce_scatterv.assert_not_called()


def test_default_path_preserves_existing_reduce_scatterv_and_local_return():
    result = _invoke(defer=False)

    assert result.error is None
    assert result.output is result.local_reduced
    result.local_buffer.assert_called_once()
    result.tp_group.reduce_scatterv.assert_called_once()
    call = result.tp_group.reduce_scatterv.call_args
    assert call.args[0].shape == (8, 4096)
    assert call.kwargs["output"] is result.local_reduced
    result.scatter.assert_not_called()
    result.forward_context.scoped.assert_called_once_with(mlp_reduce_scatter=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": "native"},
        {"attn_dp_size": 2},
        {"use_reduce_scatterv": False},
        {"a2a_none": False},
        {"use_cp": True},
        {
            "a2a_none": False,
            "attn_tp_size": 2,
            "use_a2a_scatter": True,
        },
    ],
    ids=[
        "native-backend",
        "not-attention-dp4",
        "no-reduce-scatterv",
        "moe-a2a",
        "context-parallel",
        "attention-a2a-scatter",
    ],
)
def test_deferred_combine_rejects_unsupported_configuration_without_fallback(
    overrides,
):
    result = _invoke(defer=True, **overrides)

    assert isinstance(result.error, RuntimeError)
    assert "deferred DP output combine requires" in str(result.error)
    assert result.mlp.calls == []
    assert result.mlp.shared_calls == []
    result.global_buffer.assert_not_called()
    result.local_buffer.assert_not_called()
    result.gather.assert_not_called()
    result.tp_group.reduce_scatterv.assert_not_called()
    result.scatter.assert_not_called()
