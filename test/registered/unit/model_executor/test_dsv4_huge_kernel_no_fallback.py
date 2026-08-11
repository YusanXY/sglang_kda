"""No-fallback and low-host-overhead contracts for fusion phases."""

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.model_executor.dsv4_huge_kernel_model_runner import (
    Dsv4HugeKernelModelRunner,
)
from sglang.srt.model_executor.dsv4_huge_kernel_whole_layer_runner import (
    Dsv4HugeKernelWholeLayerRunner,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.models import dsv4_whole_layer_runtime


def test_whole_layer_runner_propagates_cuda_impl_failure_without_native_fallback():
    native = mock.Mock(name="native")
    layer = SimpleNamespace(layer_id=7, _forward_native=native)
    descriptor = SimpleNamespace(
        positions=object(),
        forward_batch=object(),
        input_ids=object(),
        input_ids_global=object(),
    )
    fused = mock.Mock(side_effect=RuntimeError("fused CUDA failure"))
    runtime = mock.Mock(
        active_descriptor=descriptor,
        execute_layer=fused,
    )
    handle = SimpleNamespace(layer_id=7)
    runner = Dsv4HugeKernelWholeLayerRunner(layer, runtime, handle)

    with pytest.raises(RuntimeError, match="fused CUDA failure"):
        runner(
            positions=descriptor.positions,
            forward_batch=descriptor.forward_batch,
            input_ids=descriptor.input_ids,
            input_ids_global=descriptor.input_ids_global,
            hidden_states="hidden",
            prev_residual=None,
            prev_post=None,
            prev_comb=None,
        )

    fused.assert_called_once_with(handle, descriptor, "hidden")
    native.assert_not_called()
    assert not hasattr(runner, "_impl")


def test_whole_layer_dispatch_has_no_exception_fallback_or_host_sync():
    source = textwrap.dedent(
        inspect.getsource(Dsv4HugeKernelWholeLayerRunner.__call__)
    )
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
    forbidden = {"cpu", "item", "tolist", "numpy", "synchronize"}
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (called_attributes & forbidden)


def test_model_runner_rejects_invalid_forward_before_native_runner_executes():
    runner = object.__new__(Dsv4HugeKernelModelRunner)
    runner._huge_kernel_layers_bound = True
    runner.server_args = SimpleNamespace(
        enable_dp_attention=False,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend="disabled", bs=None),
            decode=SimpleNamespace(backend="disabled"),
        )
    )
    invalid = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        global_forward_mode=ForwardMode.DECODE,
        batch_size=1,
        extend_num_tokens=None,
        extend_seq_lens_cpu=None,
    )

    with mock.patch.object(ModelRunner, "forward") as native_forward:
        with pytest.raises(ValueError, match="EXTEND"):
            Dsv4HugeKernelModelRunner.forward(runner, invalid)

    native_forward.assert_not_called()


def test_whole_layer_executor_uses_strict_cuda_mhc_post_boundaries():
    source = textwrap.dedent(
        inspect.getsource(dsv4_whole_layer_runtime._execute_common)
    )
    tree = ast.parse(source)
    call_attrs = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    # Attention input keeps its own hc_pre. Both post boundaries use the
    # strict Huge CUDA primitive and caller-owned ping-pong storage; neither
    # may route through the generic layer hc_post or native decoder fallback.
    assert call_attrs.count("hc_pre") == 1
    assert call_attrs.count("hc_post") == 0
    assert "_huge_mhc_post" in call_names
    assert "_fused_mhc_post_ffn_pre" not in call_names
    assert "_separate_mhc_post_ffn_pre" in call_names
    assert "_forward_native" not in call_attrs

    middle_source = textwrap.dedent(
        inspect.getsource(dsv4_whole_layer_runtime._separate_mhc_post_ffn_pre)
    )
    middle_tree = ast.parse(middle_source)
    middle_names = {
        node.func.id
        for node in ast.walk(middle_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    middle_attrs = [
        node.func.attr
        for node in ast.walk(middle_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "_huge_mhc_post" in middle_names
    assert "hc_post" not in middle_attrs


def test_high_load_huge_runtime_does_not_route_c4_back_to_q1_logits():
    runtime_init = textwrap.dedent(
        inspect.getsource(dsv4_whole_layer_runtime.DSV4WholeLayerRuntime.__init__)
    )
    backend_init = textwrap.dedent(inspect.getsource(DeepseekV4AttnBackend.__init__))

    # Both strict Graph buckets are Q16-aligned.  The M=65536 bucket must keep
    # the same clustered ABI instead of silently selecting the older Q1
    # DeepGEMM path based on max_prefill_tokens.
    assert "self._use_clustered_mqa = True" in runtime_init
    assert "self.dsv4_huge_use_clustered_mqa = self.dsv4_huge_mode" in backend_init
    assert "max_prefill_tokens == 4096" not in runtime_init
    assert "max_prefill_tokens == 4096" not in backend_init


def test_fp8_q_indexer_owns_req16_block_specialization():
    source = (
        Path(__file__).parents[4]
        / "python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh"
    ).read_text()

    # Keep the eight-warp tuning local to the high-load FP8 indexer.  The
    # generic Q and FP4 indexer paths retain their four-warp launch contract.
    assert "kFusedQBlockSize = 128" in source
    assert "kFusedQIndexerBlockSize = 256" in source
    assert "Q_INDEXER_KERNEL void\nfused_q_indexer_rope_hadamard_quant" in source
    assert "Q_KERNEL void\nfused_q_indexer_rope_hadamard_fp4_quant" in source
    assert "blockIdx.x * kFusedQIndexerNumWarps + warp_id" in source
    assert "div_ceil(total_works, kFusedQIndexerNumWarps)" in source
    assert "LaunchKernel(num_blocks, kFusedQIndexerBlockSize" in source


def test_c4_indexer_compute_q_consumes_caller_owned_quant_workspace():
    q_projection = torch.empty(4, 128)
    quant_workspace = (object(), object())
    indexer = SimpleNamespace(
        wq_b=mock.Mock(return_value=(q_projection, None)),
        n_local_heads=2,
        head_dim=128,
        use_fp4_indexer=False,
        weight_scale=0.125,
        freqs_cis=object(),
    )
    fused_result = (object(), object())

    with mock.patch(
        "sglang.srt.layers.attention.dsv4.indexer.get_kda_operator",
        return_value=None,
    ), mock.patch(
        "sglang.srt.layers.attention.dsv4.indexer."
        "fused_q_indexer_rope_hadamard_quant",
        return_value=fused_result,
    ):
        result = C4Indexer.compute_q(
            indexer,
            q_lora=torch.empty(2, 1024),
            positions=torch.zeros(2, dtype=torch.int32),
            weight=torch.empty(2, 2),
            q_quant=quant_workspace,
        )

    indexer.wq_b.assert_called_once_with(quant_workspace)
    assert result is fused_result


def test_huge_q_lora_quant_workspace_reaches_c4_indexer():
    root = Path(__file__).parents[4]
    model_source = (root / "python/sglang/srt/models/deepseek_v4.py").read_text()
    indexer_source = (
        root / "python/sglang/srt/layers/attention/dsv4/indexer.py"
    ).read_text()
    runtime_source = (
        root / "python/sglang/srt/models/dsv4_whole_layer_runtime.py"
    ).read_text()

    assert "q_lora_quant = (q_lora_fp8, q_lora_scale)" in model_source
    assert "q_lora_quant=q_lora_quant" in model_source
    assert "self.wq_b(q_quant if q_quant is not None else q_lora)" in indexer_source
    assert "requires indexer.wq_b block-FP8 [128, 128]" in runtime_source
