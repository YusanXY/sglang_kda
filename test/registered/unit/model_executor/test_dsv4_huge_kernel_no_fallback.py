"""No-fallback and low-host-overhead contracts for fusion phases."""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.model_executor.dsv4_huge_kernel_model_runner import (
    Dsv4HugeKernelModelRunner,
)
from sglang.srt.model_executor.dsv4_huge_kernel_whole_layer_runner import (
    Dsv4HugeKernelWholeLayerRunner,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
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


def test_whole_layer_executor_uses_stable_gpu_attention_to_ffn_boundary():
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

    # Attention input keeps its own hc_pre and the FFN result keeps its final
    # hc_post. The middle boundary remains explicit GPU work until the fused
    # implementation passes the model-level M=4096 cosine gate.
    assert call_attrs.count("hc_pre") == 1
    assert call_attrs.count("hc_post") == 1
    assert "_fused_mhc_post_ffn_pre" not in call_names
    assert "_separate_mhc_post_ffn_pre" in call_names
    assert "_forward_native" not in call_attrs
