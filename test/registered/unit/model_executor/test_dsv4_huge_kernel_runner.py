from types import SimpleNamespace

import pytest

from sglang.srt.layers import dp_attention
from sglang.srt.model_executor.dsv4_huge_kernel_model_runner import (
    DSV4_HUGE_MAX_TOTAL_TOKENS,
    validate_dsv4_huge_kernel_bench_args,
    validate_dsv4_huge_kernel_forward,
    validate_dsv4_huge_kernel_startup,
)
from sglang.srt.model_executor.dsv4_huge_kernel_whole_layer_runner import (
    Dsv4HugeKernelWholeLayerRunner,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_factory import get_model_runner_class


@pytest.mark.parametrize(
    ("token_counts", "expected_mode"),
    [
        ([16384, 16384, 16384, 16384], dp_attention.DpPaddingMode.MAX_LEN),
        ([32768, 32768, 32768, 32768], dp_attention.DpPaddingMode.SUM_LEN),
        ([8192, 16384, 8192, 16384], dp_attention.DpPaddingMode.SUM_LEN),
        ([4096, 0, 0, 0], dp_attention.DpPaddingMode.SUM_LEN),
    ],
)
def test_huge_attention_dp_collective_shape_routing(
    monkeypatch, token_counts, expected_mode
):
    monkeypatch.setattr(dp_attention, "_ATTN_DP_SIZE", 4)
    monkeypatch.setattr(dp_attention, "_FORCE_DSV4_HUGE_BALANCED_MAX_LEN", True)
    monkeypatch.setattr(
        dp_attention, "_DSV4_HUGE_BALANCED_MAX_LEN_MAX_TOKENS", 16384
    )
    assert (
        dp_attention.DpPaddingMode.get_dp_padding_mode(True, token_counts)
        == expected_mode
    )


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
        enable_dp_attention=False,
        attn_cp_size=1,
        dcp_size=1,
        nnodes=1,
        max_running_requests=1,
        max_total_tokens=73728,
        max_prefill_tokens=4096,
        chunked_prefill_size=4096,
        page_size=256,
        moe_runner_backend="flashinfer_mxfp4",
        enable_two_batch_overlap=False,
        enable_hisparse=False,
        speculative_algorithm=None,
        disable_overlap_schedule=True,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend="disabled", bs=None),
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


def test_startup_contract_accepts_flash_b300_tp4_ep4():
    validate_dsv4_huge_kernel_startup(
        server_args=_server_args(),
        model_config=_flash_model_config(),
        gpu_id=0,
        device_name="NVIDIA B300 SXM6 AC",
        device_capability=(10, 3),
    )


def test_startup_contract_accepts_req128_high_load_capacity():
    validate_dsv4_huge_kernel_startup(
        server_args=_server_args(
            max_running_requests=128,
            max_total_tokens=DSV4_HUGE_MAX_TOTAL_TOKENS,
        ),
        model_config=_flash_model_config(),
        gpu_id=0,
        device_name="NVIDIA B200",
        device_capability=(10, 0),
    )


@pytest.mark.parametrize(
    (
        "max_running_requests",
        "max_total_tokens",
        "max_prefill_tokens",
        "chunked_prefill_size",
    ),
    [
        (16, 331776, 65536, 16384),
        (128, DSV4_HUGE_MAX_TOTAL_TOKENS, 131072, 32768),
    ],
)
def test_startup_contract_accepts_attention_dp4_high_load(
    max_running_requests,
    max_total_tokens,
    max_prefill_tokens,
    chunked_prefill_size,
):
    validate_dsv4_huge_kernel_startup(
        server_args=_server_args(
            dp_size=4,
            enable_dp_attention=True,
            max_running_requests=max_running_requests,
            max_total_tokens=max_total_tokens,
            max_prefill_tokens=max_prefill_tokens,
            chunked_prefill_size=chunked_prefill_size,
        ),
        model_config=_flash_model_config(),
        gpu_id=0,
        device_name="NVIDIA B300 SXM6 AC",
        device_capability=(10, 3),
    )


@pytest.mark.parametrize(
    ("dp_size", "enable_dp_attention"),
    [(1, True), (4, False), (2, True)],
)
def test_startup_contract_rejects_mismatched_attention_dp_mode(
    dp_size, enable_dp_attention
):
    with pytest.raises(ValueError, match="exactly TP-only.*attention-DP4"):
        validate_dsv4_huge_kernel_startup(
            server_args=_server_args(
                dp_size=dp_size,
                enable_dp_attention=enable_dp_attention,
            ),
            model_config=_flash_model_config(),
            gpu_id=0,
            device_name="NVIDIA B300 SXM6 AC",
            device_capability=(10, 3),
        )


def test_startup_contract_rejects_attention_dp4_prefill_graph():
    args = _server_args(
        dp_size=4,
        enable_dp_attention=True,
        max_running_requests=16,
        max_total_tokens=331776,
        max_prefill_tokens=65536,
        chunked_prefill_size=16384,
    )
    args.cuda_graph_config.prefill.backend = "breakable"
    args.cuda_graph_config.prefill.bs = [4096, 65536]
    with pytest.raises(ValueError, match="attention-DP4 Huge is Eager-only"):
        validate_dsv4_huge_kernel_startup(
            server_args=args,
            model_config=_flash_model_config(),
            gpu_id=0,
            device_name="NVIDIA B300 SXM6 AC",
            device_capability=(10, 3),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tp_size", 8),
        ("ep_size", 1),
        ("max_running_requests", 129),
        ("max_total_tokens", DSV4_HUGE_MAX_TOTAL_TOKENS + 1),
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


@pytest.mark.parametrize(
    ("device_name", "device_capability"),
    [
        ("NVIDIA H200", (9, 0)),
        ("NVIDIA B200", (10, 3)),
        ("NVIDIA B300 SXM6 AC", (10, 0)),
    ],
)
def test_startup_contract_rejects_unsupported_or_mismatched_gpu_without_fallback(
    device_name, device_capability
):
    with pytest.raises(ValueError, match="B200/SM100 or B300/SM103"):
        validate_dsv4_huge_kernel_startup(
            server_args=_server_args(),
            model_config=_flash_model_config(),
            gpu_id=0,
            device_name=device_name,
            device_capability=device_capability,
        )


def test_startup_contract_accepts_exact_breakable_prefill_graph_bucket():
    args = _server_args()
    args.cuda_graph_config.prefill.backend = "breakable"
    args.cuda_graph_config.prefill.bs = [4096]
    validate_dsv4_huge_kernel_startup(
        server_args=args,
        model_config=_flash_model_config(),
        gpu_id=0,
        device_name="NVIDIA B200",
        device_capability=(10, 0),
    )


def test_startup_contract_accepts_dual_high_load_graph_buckets():
    args = _server_args(
        max_running_requests=16,
        max_prefill_tokens=65536,
        chunked_prefill_size=65536,
        max_total_tokens=331776,
    )
    args.cuda_graph_config.prefill.backend = "breakable"
    args.cuda_graph_config.prefill.bs = [4096, 65536]
    validate_dsv4_huge_kernel_startup(
        server_args=args,
        model_config=_flash_model_config(),
        gpu_id=0,
        device_name="NVIDIA B300 SXM6 AC",
        device_capability=(10, 3),
    )


def test_startup_contract_rejects_non_exact_graph_bucket_without_fallback():
    args = _server_args()
    args.cuda_graph_config.prefill.backend = "breakable"
    args.cuda_graph_config.prefill.bs = [4096, 65536]
    with pytest.raises(ValueError, match="exact buckets"):
        validate_dsv4_huge_kernel_startup(
            server_args=args,
            model_config=_flash_model_config(),
            gpu_id=0,
            device_name="NVIDIA B200",
            device_capability=(10, 0),
        )


def test_startup_contract_rejects_decode_graph():
    args = _server_args()
    args.cuda_graph_config.decode.backend = "piecewise"
    with pytest.raises(ValueError, match="decode CUDA graph"):
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
            extend_seq_lens_cpu=[num_tokens],
        )
    )


@pytest.mark.parametrize(
    ("batch_size", "extend_lens"),
    ((16, [256] * 16), (128, [32] * 128)),
)
def test_dynamic_contract_accepts_high_load_aggregate_m4096(
    batch_size, extend_lens
):
    validate_dsv4_huge_kernel_forward(
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            extend_num_tokens=4096,
            extend_seq_lens_cpu=extend_lens,
        )
    )


def test_dynamic_contract_accepts_true_req16_m65536():
    validate_dsv4_huge_kernel_forward(
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_forward_mode=ForwardMode.EXTEND,
            batch_size=16,
            extend_num_tokens=65536,
            extend_seq_lens_cpu=[4096] * 16,
        )
    )


def test_dynamic_contract_accepts_attention_dp4_collective_idle_rank():
    validate_dsv4_huge_kernel_forward(
        SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            global_forward_mode=None,
            batch_size=0,
            extend_num_tokens=0,
        ),
        attention_dp4=True,
    )


def test_dynamic_contract_rejects_collective_idle_without_attention_dp4():
    with pytest.raises(ValueError, match="only under attention-DP4"):
        validate_dsv4_huge_kernel_forward(
            SimpleNamespace(
                forward_mode=ForwardMode.IDLE,
                global_forward_mode=ForwardMode.EXTEND,
                batch_size=0,
                extend_num_tokens=0,
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
            extend_seq_lens_cpu=None,
        ),
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            extend_num_tokens=4096,
            extend_seq_lens_cpu=[2048],
        ),
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            extend_num_tokens=4097,
            extend_seq_lens_cpu=[4097],
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
    validate_dsv4_huge_kernel_bench_args(
        SimpleNamespace(
            batch_size=(16,), input_len=(4096,), output_len=(1,), correctness_test=False
        )
    )
    validate_dsv4_huge_kernel_bench_args(
        SimpleNamespace(
            batch_size=(16,),
            input_len=(20480,),
            output_len=(1,),
            cache_hit_rate=0.8,
            correctness_test=False,
        )
    )
    validate_dsv4_huge_kernel_bench_args(
        SimpleNamespace(
            batch_size=(128,),
            input_len=(20480,),
            output_len=(1,),
            cache_hit_rate=0.8,
            correctness_test=False,
        )
    )
    with pytest.raises(ValueError, match="1..4096 uncached tokens"):
        validate_dsv4_huge_kernel_bench_args(
            SimpleNamespace(
                batch_size=(128,),
                input_len=(4097,),
                output_len=(1,),
                correctness_test=False,
            )
        )
    with pytest.raises(ValueError, match="batch-size"):
        validate_dsv4_huge_kernel_bench_args(
            SimpleNamespace(
                batch_size=(129,),
                input_len=(4096,),
                output_len=(1,),
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


def test_whole_layer_runner_calls_ratio_handle_not_native_body():
    calls = []

    class FakeLayer:
        layer_id = 7

        def _forward_native(self, **kwargs):
            raise AssertionError("huge runner must never enter _forward_native")

    descriptor = SimpleNamespace(
        positions=object(),
        forward_batch=object(),
        input_ids=object(),
        input_ids_global=object(),
    )

    class FakeRuntime:
        active_descriptor = descriptor

        def execute_layer(self, handle, actual_descriptor, hidden_states):
            calls.append((handle, actual_descriptor, hidden_states))
            return "sentinel"

    runtime = FakeRuntime()
    handle = SimpleNamespace(layer_id=7)
    hidden_states = object()
    runner = Dsv4HugeKernelWholeLayerRunner(FakeLayer(), runtime, handle)
    assert (
        runner(
            positions=descriptor.positions,
            forward_batch=descriptor.forward_batch,
            input_ids=descriptor.input_ids,
            input_ids_global=descriptor.input_ids_global,
            hidden_states=hidden_states,
            prev_residual=None,
            prev_post=None,
            prev_comb=None,
        )
        == "sentinel"
    )
    assert calls == [(handle, descriptor, hidden_states)]
    assert not hasattr(runner, "_impl")
