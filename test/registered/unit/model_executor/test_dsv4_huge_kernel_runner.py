from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.dsv4_huge_kernel_model_runner import (
    validate_dsv4_huge_kernel_bench_args,
    validate_dsv4_huge_kernel_forward,
    validate_dsv4_huge_kernel_startup,
)
from sglang.srt.model_executor.dsv4_huge_kernel_whole_layer_runner import (
    Dsv4HugeKernelWholeLayerRunner,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_factory import get_model_runner_class


def _flash_model_config():
    ratios = [0, 0] + [4, 128] * 20 + [4, 0]
    return SimpleNamespace(
        context_len=73728,
        num_hidden_layers=43,
        hf_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            num_hidden_layers=43,
            q_lora_rank=1024,
            o_groups=8,
            compress_ratios=ratios,
        ),
    )


def _server_args(**overrides):
    values = dict(
        dsv4_worker_backend="huge_kernel",
        device="cuda",
        tp_size=4,
        ep_size=4,
        pp_size=1,
        dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        nnodes=1,
        max_running_requests=1,
        max_total_tokens=73728,
        chunked_prefill_size=4096,
        page_size=256,
        moe_runner_backend="flashinfer_mxfp4",
        enable_two_batch_overlap=False,
        enable_hisparse=False,
        speculative_algorithm=None,
        disable_overlap_schedule=True,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend="disabled"),
            decode=SimpleNamespace(backend="disabled"),
        ),
        enable_mixed_chunk=False,
        enable_lora=False,
    )
    values.update(overrides)
    args = SimpleNamespace(**values)
    args.get_attention_backends = lambda: ("dsv4", "dsv4")
    return args


def test_startup_contract_accepts_flash_b200_tp4_ep4():
    validate_dsv4_huge_kernel_startup(
        server_args=_server_args(),
        model_config=_flash_model_config(),
        gpu_id=0,
        device_name="NVIDIA B200",
        device_capability=(10, 0),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tp_size", 8),
        ("ep_size", 1),
        ("max_running_requests", 2),
        ("disable_overlap_schedule", False),
        ("page_size", 1),
        ("moe_runner_backend", "auto"),
    ],
)
def test_startup_contract_rejects_unsupported_config(field, value):
    with pytest.raises(ValueError):
        validate_dsv4_huge_kernel_startup(
            server_args=_server_args(**{field: value}),
            model_config=_flash_model_config(),
            gpu_id=0,
            device_name="NVIDIA B200",
            device_capability=(10, 0),
        )


def test_startup_contract_rejects_non_b200_without_fallback():
    with pytest.raises(ValueError, match="B200"):
        validate_dsv4_huge_kernel_startup(
            server_args=_server_args(),
            model_config=_flash_model_config(),
            gpu_id=0,
            device_name="NVIDIA H200",
            device_capability=(9, 0),
        )


def test_startup_contract_rejects_any_enabled_cuda_graph_phase():
    args = _server_args()
    args.cuda_graph_config.prefill.backend = "breakable"
    with pytest.raises(ValueError, match="CUDA graphs"):
        validate_dsv4_huge_kernel_startup(
            server_args=args,
            model_config=_flash_model_config(),
            gpu_id=0,
            device_name="NVIDIA B200",
            device_capability=(10, 0),
        )


@pytest.mark.parametrize("num_tokens", [1, 4096])
def test_dynamic_contract_accepts_incremental_cache_build(num_tokens):
    validate_dsv4_huge_kernel_forward(
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            extend_num_tokens=num_tokens,
        )
    )


@pytest.mark.parametrize(
    "batch",
    [
        SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            global_forward_mode=ForwardMode.DECODE,
            batch_size=1,
            extend_num_tokens=None,
        ),
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            extend_num_tokens=4096,
        ),
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            extend_num_tokens=4097,
        ),
    ],
)
def test_dynamic_contract_rejects_decode_batching_and_oversize(batch):
    with pytest.raises(ValueError):
        validate_dsv4_huge_kernel_forward(batch)


def test_bench_contract_requires_single_prefill_ttft_point():
    validate_dsv4_huge_kernel_bench_args(
        SimpleNamespace(
            batch_size=(1,), input_len=(4096,), output_len=(1,), correctness_test=False
        )
    )
    with pytest.raises(ValueError, match="output-len"):
        validate_dsv4_huge_kernel_bench_args(
            SimpleNamespace(
                batch_size=(1,),
                input_len=(4096,),
                output_len=(2,),
                correctness_test=False,
            )
        )


def test_factory_selects_explicit_backend_only():
    assert get_model_runner_class(_server_args()).__name__ == (
        "Dsv4HugeKernelModelRunner"
    )
    assert get_model_runner_class(
        _server_args(dsv4_worker_backend="native")
    ).__name__ == "ModelRunner"
    with pytest.raises(ValueError, match="Unknown"):
        get_model_runner_class(_server_args(dsv4_worker_backend="typo"))


def test_whole_layer_runner_calls_bound_reference_body():
    calls = []

    class FakeLayer:
        layer_id = 7

        def _forward_native(self, **kwargs):
            calls.append(kwargs)
            return "sentinel"

    runner = Dsv4HugeKernelWholeLayerRunner(FakeLayer())
    assert runner(token_count=4096) == "sentinel"
    assert calls == [{"token_count": 4096}]
