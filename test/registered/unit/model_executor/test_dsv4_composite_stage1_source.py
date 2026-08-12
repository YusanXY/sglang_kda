"""CPU/source contracts for the DSV4 composite stage-1 ABI."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    owner = _class(tree, class_name)
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_raw_holder_is_frozen_and_owns_the_complete_future_composite_abi():
    source = _source(
        "python/sglang/srt/layers/quantization/"
        "mxfp4_flashinfer_trtllm_moe.py"
    )
    tree = ast.parse(source)

    request = _class(tree, "Dsv4DpRawMoeRequest")
    output = _class(tree, "Dsv4DpRawMoeOutput")
    for holder in (request, output):
        decorator = next(
            node
            for node in holder.decorator_list
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dataclass"
        )
        frozen = next(
            keyword.value
            for keyword in decorator.keywords
            if keyword.arg == "frozen"
        )
        assert isinstance(frozen, ast.Constant) and frozen.value is True
        equality = next(
            keyword.value
            for keyword in decorator.keywords
            if keyword.arg == "eq"
        )
        assert isinstance(equality, ast.Constant) and equality.value is False

    request_fields = {
        node.target.id
        for node in request.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert request_fields == {
        "x_quant",
        "x_scale",
        "symmetric_slot_anchor",
        "routed_scale",
    }
    output_fields = {
        node.target.id
        for node in output.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert output_fields == {
        "gemm2_out",
        "expanded_to_permuted",
        "packed_topk",
        "symmetric_slot_anchor",
        "top_k",
        "routed_scale",
    }
    assert all(
        not isinstance(node, ast.FunctionDef) or node.name != "record_stream"
        for node in output.body
    )


def test_jit_raw_mode_is_mutually_exclusive_one_shot_and_exception_resettable():
    source = _source("python/sglang/jit_kernel/dsv4_moe_overlap/jit.py")
    assert "thread_local bool dsv4_dp_deferred_raw_active = false" in source
    assert "dsv4_dp_deferred_raw_active = false;" in source
    assert "void dsv4_set_dp_deferred_raw()" in source
    assert "DSV4 DP deferred-raw descriptor was not consumed" in source
    assert "reset_dp_deferred_raw" in source
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv4_set_dp_deferred_raw" in source
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv4_finalize_dp_routed_raw" in source
    assert "DSV4 DP deferred raw requires do_finalize=false" in source
    assert (
        "TVM_FFI_ICHECK_GE(gemm2_output.size(0), state.num_tokens * top_k)"
        in source
    )
    assert "TVM_FFI_ICHECK_NE(gemm2_output.data_ptr(), output.data_ptr())" in source

    launch = source.index("Array<Tensor> result = launcher->run(config, enable_pdl)")
    raw_return = source.index("if (dsv4_defer_dp_raw)", launch)
    routed_finalize = source.index("if (dsv4_fuse_dp_routed)", launch)
    assert launch < raw_return < routed_finalize
    raw_block = source[raw_return:routed_finalize]
    assert "return result" in raw_block
    assert "dsv4_launch_dp_routed_finalize" not in raw_block


def test_huge_raw_forward_never_enters_generic_combine_or_tensor_postprocessing():
    source = _source("python/sglang/srt/layers/moe/fused_moe_triton/layer.py")
    tree = ast.parse(source)
    method = _method(tree, "FusedMoE", "forward_dsv4_deferred_raw")
    attributes = {
        node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "combine" not in attributes
    assert "contiguous" not in attributes
    assert "tensor_model_parallel_all_reduce" not in called_names
    assert "Dsv4DpRawMoeRequest" in source
    assert "Dsv4DpRawMoeOutput" in source
    quant_source = _source(
        "python/sglang/srt/layers/quantization/"
        "mxfp4_flashinfer_trtllm_moe.py"
    )
    assert "do_finalize=dp_raw_request is None" in quant_source
    assert (
        "x_quant.shape != (hidden_states.shape[0], hidden_size)"
        in quant_source
    )
    assert "x_quant.device.type != \"cuda\"" in quant_source
    assert "gemm2_out.shape[0] < num_tokens * top_k" in quant_source
    assert "gemm2_out.shape[1] != out_hidden_size" in quant_source
    assert "gemm2_out.data_ptr() == anchor.data_ptr()" in quant_source
    assert "raw_output_count = len(moe_outputs)" in quant_source
    assert "isinstance(moe_outputs, (tuple, list))" not in quant_source


def test_default_off_gate_preserves_old_path_and_runtime_consumes_raw_in_order():
    env_source = _source("python/sglang/srt/environ.py")
    assert "SGLANG_DSV4_HUGE_DP_MOE_DEFER_RAW = EnvBool(False)" in env_source

    runtime = _source("python/sglang/srt/models/dsv4_whole_layer_runtime.py")
    finalize = runtime.index("finalize_dsv4_dp_routed_raw(")
    shared_join = runtime.index("current_stream.wait_stream(shared_stream)", finalize)
    nvls = runtime.index("return nvls_epoch_post(", finalize)
    assert finalize < shared_join < nvls
    assert "moe_result.record_stream" not in runtime
    assert "cancel_dsv4_finalize()" in runtime[finalize:nvls]

    decoder = _source("python/sglang/srt/models/deepseek_v4.py")
    decoder_tree = ast.parse(decoder)
    method = _method(decoder_tree, "DeepseekV4DecoderLayer", "_run_moe_ffn_dp_sync")
    defaults = dict(
        zip(
            (arg.arg for arg in method.args.kwonlyargs),
            method.args.kw_defaults,
        )
    )
    assert isinstance(defaults["defer_dp_raw_output"], ast.Constant)
    assert defaults["defer_dp_raw_output"].value is False
    # A deferred raw result is an owning holder rather than a Tensor.  The
    # local-shared path must not inspect ``hidden_states.shape`` before the
    # holder and shared stream are returned to the whole-layer consumer.
    assert (
        "if _shared_local is not None and not _defer_shared_expert_add:"
        in decoder
    )
    assert "else _shared_local[: _shared_local.shape[0]]" not in decoder
    raw_join_guard = decoder.index("if not defer_dp_raw_output:")
    decoder_join = decoder.index(
        "current_stream.wait_stream(_shared_local_stream)", raw_join_guard
    )
    assert raw_join_guard < decoder_join
