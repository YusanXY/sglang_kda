"""Strict DeepSeek-V4-Flash whole-layer development runner.

Phase 1 intentionally keeps the mathematically identical layer body behind a
new whole-layer dispatch boundary. Later phases replace that body with fused
CUDA operators without changing Scheduler or benchmark construction paths.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

DSV4_HUGE_CONTEXT_CAPACITY = 73728
DSV4_HUGE_MAX_EXTEND_TOKENS_PER_REQUEST = 4096
# The Eager req128 path may aggregate 32 real requests per ForwardBatch while
# preserving the strict 4096-token per-request cap. This halves the measured
# workload from eight M65536 waves to four M131072 waves without changing
# request semantics. Breakable Graph remains limited to its frozen buckets.
DSV4_HUGE_MAX_EXTEND_TOKENS = 32 * DSV4_HUGE_MAX_EXTEND_TOKENS_PER_REQUEST
DSV4_HUGE_MAX_REQUESTS = 128
DSV4_HUGE_PAGE_SIZE = 256
# req=128 high-load target: 16384 cached + 4096 new tokens per request.
# Include one generated token, then round each request to a complete KV page.
_DSV4_HUGE_HIGH_LOAD_TOKENS_PER_REQUEST = 16384 + 4096 + 1
DSV4_HUGE_MAX_TOTAL_TOKENS = DSV4_HUGE_MAX_REQUESTS * (
    (
        _DSV4_HUGE_HIGH_LOAD_TOKENS_PER_REQUEST
        + DSV4_HUGE_PAGE_SIZE
        - 1
    )
    // DSV4_HUGE_PAGE_SIZE
    * DSV4_HUGE_PAGE_SIZE
)
DSV4_HUGE_TP_SIZE = 4
DSV4_HUGE_EP_SIZE = 4
DSV4_FLASH_COMPRESS_RATIOS = (0, 0) + (4, 128) * 20 + (4,)
DSV4_HUGE_GRAPH_PREFILL_TOKENS = (4096, 65536)
DSV4_HUGE_EAGER_PREFILL_TOKENS = (*DSV4_HUGE_GRAPH_PREFILL_TOKENS, 131072)


def _is_deepseek_v4_flash(model_config) -> bool:
    hf_config = model_config.hf_config
    architectures = tuple(getattr(hf_config, "architectures", ()) or ())
    compress_ratios = tuple(getattr(hf_config, "compress_ratios", ()) or ())
    return (
        architectures == ("DeepseekV4ForCausalLM",)
        and getattr(hf_config, "num_hidden_layers", None) == 43
        and getattr(hf_config, "q_lora_rank", None) == 1024
        and getattr(hf_config, "o_groups", None) == 8
        and compress_ratios[:43] == DSV4_FLASH_COMPRESS_RATIOS
        and compress_ratios[43:] == (0,)
    )


def validate_dsv4_huge_kernel_startup(
    *,
    server_args,
    model_config,
    gpu_id: int,
    is_draft_worker: bool = False,
    device_name: Optional[str] = None,
    device_capability: Optional[tuple[int, int]] = None,
) -> None:
    """Reject every deployment shape outside the phase-1 contract."""

    errors = []
    if not _is_deepseek_v4_flash(model_config):
        errors.append(
            "model must match DeepSeek-V4-Flash "
            "(DeepseekV4ForCausalLM, 43 layers, Flash compression layout)"
        )
    if is_draft_worker:
        errors.append("draft/speculative workers are unsupported")
    if server_args.device != "cuda":
        errors.append(f"device must be cuda, got {server_args.device!r}")
    else:
        resolved_device_name = (
            device_name
            if device_name is not None
            else torch.cuda.get_device_name(gpu_id)
        )
        resolved_capability = (
            device_capability
            if device_capability is not None
            else torch.cuda.get_device_capability(gpu_id)
        )
        device_name_upper = resolved_device_name.upper()
        supported_gpu = (
            "B200" in device_name_upper and resolved_capability == (10, 0)
        ) or ("B300" in device_name_upper and resolved_capability == (10, 3))
        if not supported_gpu:
            errors.append(
                "GPU must be a matching NVIDIA B200/SM100 or B300/SM103, "
                f"got name={resolved_device_name!r}, "
                f"capability={resolved_capability!r}"
            )

    attention_dp4 = bool(server_args.enable_dp_attention)
    valid_parallel_mode = (
        attention_dp4 and server_args.dp_size == 4
    ) or (
        not attention_dp4 and server_args.dp_size == 1
    )
    if not valid_parallel_mode:
        errors.append(
            "Huge supports exactly TP-only (dp_size=1, DP attention disabled) "
            "or attention-DP4 (dp_size=4, --enable-dp-attention); got "
            f"dp_size={server_args.dp_size!r}, "
            f"enable_dp_attention={server_args.enable_dp_attention!r}"
        )

    expected_values = {
        "tp_size": DSV4_HUGE_TP_SIZE,
        "ep_size": DSV4_HUGE_EP_SIZE,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "nnodes": 1,
        "page_size": DSV4_HUGE_PAGE_SIZE,
        "moe_runner_backend": "flashinfer_mxfp4",
        "enable_two_batch_overlap": False,
        "enable_hisparse": False,
    }
    for field, expected in expected_values.items():
        actual = getattr(server_args, field)
        if actual != expected:
            errors.append(f"{field} must be {expected!r}, got {actual!r}")

    if server_args.max_prefill_tokens not in DSV4_HUGE_EAGER_PREFILL_TOKENS:
        errors.append(
            "max_prefill_tokens must select a strict Huge aggregate bucket in "
            f"{DSV4_HUGE_EAGER_PREFILL_TOKENS!r}, got "
            f"{server_args.max_prefill_tokens!r}"
        )
    expected_chunked_prefill = server_args.max_prefill_tokens // (
        4 if attention_dp4 else 1
    )
    if server_args.chunked_prefill_size != expected_chunked_prefill:
        errors.append(
            "chunked_prefill_size must equal the strict per-attention-DP-rank "
            "bucket max_prefill_tokens / attention_dp_size; got "
            f"{server_args.chunked_prefill_size!r} vs expected "
            f"{expected_chunked_prefill!r}"
        )

    if not 1 <= server_args.max_running_requests <= DSV4_HUGE_MAX_REQUESTS:
        errors.append(
            "max_running_requests must be in [1, "
            f"{DSV4_HUGE_MAX_REQUESTS}], got {server_args.max_running_requests!r}"
        )
    if not (
        DSV4_HUGE_CONTEXT_CAPACITY
        <= server_args.max_total_tokens
        <= DSV4_HUGE_MAX_TOTAL_TOKENS
    ):
        errors.append(
            "max_total_tokens must be in "
            f"[{DSV4_HUGE_CONTEXT_CAPACITY}, {DSV4_HUGE_MAX_TOTAL_TOKENS}], "
            f"got {server_args.max_total_tokens!r}"
        )

    if model_config.context_len != DSV4_HUGE_CONTEXT_CAPACITY:
        errors.append(
            "resolved context length must be "
            f"{DSV4_HUGE_CONTEXT_CAPACITY}, got {model_config.context_len}"
        )
    prefill_backend, _ = server_args.get_attention_backends()
    if prefill_backend != "dsv4":
        errors.append(
            f"prefill attention backend must be 'dsv4', got {prefill_backend!r}"
        )
    if server_args.speculative_algorithm is not None:
        errors.append("speculative decoding must be disabled")
    if not server_args.disable_overlap_schedule:
        errors.append("--disable-overlap-schedule is required")
    cuda_graph_config = server_args.cuda_graph_config
    if cuda_graph_config is None:
        errors.append("cuda_graph_config must be resolved")
    else:
        prefill_graph = cuda_graph_config.prefill
        if attention_dp4 and prefill_graph.backend != Backend.DISABLED:
            errors.append(
                "attention-DP4 Huge is Eager-only during the DP tuning phase"
            )
        elif prefill_graph.backend not in (Backend.DISABLED, Backend.BREAKABLE):
            errors.append(
                "prefill CUDA graph backend must be disabled or breakable, "
                f"got {prefill_graph.backend!r}"
            )
        elif (
            server_args.max_prefill_tokens == 131072
            and prefill_graph.backend != Backend.DISABLED
        ):
            errors.append(
                "req32/M131072 is an Eager-only specialization; prefill CUDA "
                "graph must be disabled"
            )
        elif prefill_graph.backend == Backend.BREAKABLE:
            expected_buckets = (
                (4096,)
                if server_args.max_prefill_tokens == 4096
                else (4096, 65536)
            )
            if tuple(prefill_graph.bs or ()) != expected_buckets:
                errors.append(
                    "huge breakable prefill CUDA graph requires exact buckets "
                    f"{list(expected_buckets)}, got {prefill_graph.bs!r}"
                )
        if cuda_graph_config.decode.backend != Backend.DISABLED:
            errors.append("decode CUDA graph must be disabled")
    if server_args.enable_mixed_chunk:
        errors.append("mixed prefill/decode batches are unsupported")
    if server_args.enable_lora:
        errors.append("LoRA is unsupported")

    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(
            "Invalid --dsv4-worker-backend=huge_kernel configuration:\n"
            f"  - {details}"
        )


def validate_dsv4_huge_kernel_forward(
    forward_batch: ForwardBatch,
    *,
    cuda_graph_config=None,
    attention_dp4: bool = False,
) -> None:
    """Validate the dynamic phase-1 batch shape without device synchronizes."""

    if forward_batch.forward_mode is ForwardMode.IDLE:
        if not attention_dp4:
            raise ValueError(
                "dsv4 huge_kernel accepts collective IDLE ranks only under "
                "attention-DP4"
            )
        if forward_batch.global_forward_mode not in (None, ForwardMode.EXTEND):
            raise ValueError(
                "dsv4 huge_kernel attention-DP4 IDLE ranks require an unset "
                "or EXTEND global mode; got "
                f"{forward_batch.global_forward_mode!r}"
            )
        if forward_batch.batch_size != 0:
            raise ValueError(
                "dsv4 huge_kernel attention-DP4 IDLE rank must have an empty "
                f"local batch; got batch_size={forward_batch.batch_size}"
            )
        if forward_batch.extend_num_tokens not in (None, 0):
            raise ValueError(
                "dsv4 huge_kernel attention-DP4 IDLE rank must have zero "
                f"local EXTEND tokens; got {forward_batch.extend_num_tokens!r}"
            )
        return

    if forward_batch.forward_mode is not ForwardMode.EXTEND:
        raise ValueError(
            "dsv4 huge_kernel only accepts ForwardMode.EXTEND; "
            f"got {forward_batch.forward_mode!r}"
        )
    if (
        forward_batch.global_forward_mode is not None
        and forward_batch.global_forward_mode is not ForwardMode.EXTEND
    ):
        raise ValueError(
            "dsv4 huge_kernel only accepts global ForwardMode.EXTEND; "
            f"got {forward_batch.global_forward_mode!r}"
        )
    batch_size = forward_batch.batch_size
    if not 1 <= batch_size <= DSV4_HUGE_MAX_REQUESTS:
        raise ValueError(
            f"dsv4 huge_kernel batch_size must be in [1, "
            f"{DSV4_HUGE_MAX_REQUESTS}]; got {batch_size}"
        )
    num_tokens = forward_batch.extend_num_tokens
    if num_tokens is None or not 1 <= num_tokens <= DSV4_HUGE_MAX_EXTEND_TOKENS:
        raise ValueError(
            "dsv4 huge_kernel aggregate EXTEND M must be in [1, 131072]; "
            f"got {num_tokens!r}"
        )
    extend_lens = forward_batch.extend_seq_lens_cpu
    if extend_lens is None:
        raise ValueError(
            "dsv4 huge_kernel requires the existing host extend-length mirror; "
            "it will not synchronize the GPU length tensor"
        )
    if len(extend_lens) != batch_size:
        raise ValueError(
            f"extend_seq_lens_cpu has {len(extend_lens)} rows for "
            f"batch_size={batch_size}"
        )
    if any(
        not 1 <= int(length) <= DSV4_HUGE_MAX_EXTEND_TOKENS_PER_REQUEST
        for length in extend_lens
    ):
        raise ValueError(
            "every huge-kernel request must contribute 1..4096 EXTEND tokens; "
            f"got {extend_lens!r}"
        )
    if sum(int(length) for length in extend_lens) != num_tokens:
        raise ValueError(
            "sum(extend_seq_lens_cpu) must equal the total EXTEND M without a "
            f"device readback; got lengths={extend_lens!r}, M={num_tokens}"
        )
    if (
        cuda_graph_config is not None
        and cuda_graph_config.prefill.backend == Backend.BREAKABLE
        and (
            num_tokens not in cuda_graph_config.prefill.bs
            or (batch_size, num_tokens) not in ((1, 4096), (16, 65536))
        )
    ):
        raise ValueError(
            "dsv4 huge breakable prefill CUDA graph currently requires the "
            "exact captured shape in {(req=1,M=4096), "
            "(req=16,M=65536)}; no eager fallback is allowed; "
            f"got req={batch_size}, M={num_tokens}"
        )
    if (
        cuda_graph_config is not None
        and cuda_graph_config.prefill.backend == Backend.BREAKABLE
    ):
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None or len(seq_lens_cpu) != batch_size:
            raise ValueError(
                "dsv4 huge Graph requires the existing CPU sequence-length "
                "mirror for every request"
            )
        if any(
            (int(seq_len) - int(extend_len)) % 128 != 0
            for seq_len, extend_len in zip(seq_lens_cpu, extend_lens)
        ):
            raise ValueError(
                "dsv4 huge Graph compact compression plans require every "
                "cached prefix length to be divisible by 128"
            )


def validate_dsv4_huge_kernel_bench_args(bench_args) -> None:
    """Fail in the parent process before bench_one_batch spawns TP workers."""

    errors = []
    # The low-level correctness driver has no prefix-cache option.  Treat that
    # interface as an uncached prefill while the server benchmark supplies the
    # explicit cache-hit ratio used by incremental-TTFT experiments.
    cache_hit_rate = float(getattr(bench_args, "cache_hit_rate", 0.0))
    batch_sizes = tuple(bench_args.batch_size)
    if len(batch_sizes) != 1 or not 1 <= batch_sizes[0] <= DSV4_HUGE_MAX_REQUESTS:
        errors.append(
            f"--batch-size must contain one value in [1, {DSV4_HUGE_MAX_REQUESTS}], "
            f"got {bench_args.batch_size}"
        )
    if tuple(bench_args.output_len) != (1,):
        errors.append(
            "--output-len must be exactly 1 for prefill TTFT, "
            f"got {bench_args.output_len}"
        )
    invalid_new_lens = [
        value - int(value * cache_hit_rate)
        for value in bench_args.input_len
        if not 1
        <= value - int(value * cache_hit_rate)
        <= DSV4_HUGE_MAX_EXTEND_TOKENS_PER_REQUEST
    ]
    if invalid_new_lens:
        errors.append(
            "every request must contribute 1..4096 uncached tokens after "
            f"--cache-hit-rate is applied, got {invalid_new_lens}"
        )
    # ``batch_size * new_per_request`` is the complete benchmark workload,
    # not the shape of one ModelRunner forward.  The Huge scheduler keeps the
    # strict 4096-token per-request cap and admits at most the configured
    # aggregate M=131072 into each Eager ForwardBatch. Req=128 therefore runs
    # as multiple real req=32/M=131072 GPU batches; rejecting the outer 524288
    # token workload here would conflate request semantics with the internal
    # workspace bound.
    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(
            "Invalid bench_one_batch arguments for dsv4 huge_kernel:\n"
            f"  - {details}"
        )


class Dsv4HugeKernelModelRunner(ModelRunner):
    """ModelRunner that installs and exclusively uses whole-layer dispatch."""

    def __init__(self, *args, **kwargs):
        validate_dsv4_huge_kernel_startup(
            server_args=kwargs["server_args"],
            model_config=kwargs["model_config"],
            gpu_id=kwargs["gpu_id"],
            is_draft_worker=kwargs.get("is_draft_worker", False),
        )
        self._huge_kernel_layers_bound = False
        self._dsv4_whole_layer_runtime = None
        super().__init__(*args, **kwargs)

    def init_attention_backends(self):
        super().init_attention_backends()
        if self._huge_kernel_layers_bound:
            return
        from sglang.srt.models.dsv4_whole_layer_runtime import (
            DSV4WholeLayerRuntime,
        )

        runtime = DSV4WholeLayerRuntime(
            config=self.model.config,
            server_args=self.server_args,
        )
        handles = runtime.bind_after_weight_load(
            self.model.model.layers,
            start_layer=self.model.model.start_layer,
            end_layer=self.model.model.end_layer,
        )
        installed = len(handles)
        if installed != self.model_config.num_hidden_layers:
            raise RuntimeError(
                "dsv4 huge_kernel must own every decoder layer; "
                f"installed={installed}, expected={self.model_config.num_hidden_layers}"
            )
        self.model.model._dsv4_whole_layer_runtime = runtime
        self._dsv4_whole_layer_runtime = runtime
        self._huge_kernel_layers_bound = True
        logger.info(
            "Installed strict dsv4 C0/C4/C128 huge dispatch on %d layers",
            installed,
        )

    def forward(self, forward_batch: ForwardBatch, *args, **kwargs):
        if not self._huge_kernel_layers_bound:
            raise RuntimeError(
                "dsv4 huge_kernel forward called before all whole-layer runners "
                "were bound"
            )
        validate_dsv4_huge_kernel_forward(
            forward_batch,
            cuda_graph_config=self.server_args.cuda_graph_config,
            attention_dp4=bool(self.server_args.enable_dp_attention),
        )
        return super().forward(forward_batch, *args, **kwargs)
