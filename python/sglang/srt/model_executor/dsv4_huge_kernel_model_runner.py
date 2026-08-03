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
DSV4_HUGE_MAX_EXTEND_TOKENS = 4096
DSV4_HUGE_TP_SIZE = 4
DSV4_HUGE_EP_SIZE = 4
DSV4_FLASH_COMPRESS_RATIOS = (0, 0) + (4, 128) * 20 + (4,)


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
        if "B200" not in resolved_device_name.upper():
            errors.append(f"GPU must be NVIDIA B200, got {resolved_device_name!r}")
        resolved_capability = (
            device_capability
            if device_capability is not None
            else torch.cuda.get_device_capability(gpu_id)
        )
        if resolved_capability != (10, 0):
            errors.append(
                "GPU compute capability must be (10, 0) for B200, "
                f"got {resolved_capability!r}"
            )

    expected_values = {
        "tp_size": DSV4_HUGE_TP_SIZE,
        "ep_size": DSV4_HUGE_EP_SIZE,
        "pp_size": 1,
        "dp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "nnodes": 1,
        "max_running_requests": 1,
        "max_total_tokens": DSV4_HUGE_CONTEXT_CAPACITY,
        "chunked_prefill_size": DSV4_HUGE_MAX_EXTEND_TOKENS,
        "page_size": 256,
        "moe_runner_backend": "flashinfer_mxfp4",
        "enable_two_batch_overlap": False,
        "enable_hisparse": False,
    }
    for field, expected in expected_values.items():
        actual = getattr(server_args, field)
        if actual != expected:
            errors.append(f"{field} must be {expected!r}, got {actual!r}")

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
    if (
        cuda_graph_config is None
        or cuda_graph_config.prefill.backend != Backend.DISABLED
        or cuda_graph_config.decode.backend != Backend.DISABLED
    ):
        errors.append("prefill and decode CUDA graphs must both be disabled")
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


def validate_dsv4_huge_kernel_forward(forward_batch: ForwardBatch) -> None:
    """Validate the dynamic phase-1 batch shape without device synchronizes."""

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
    if forward_batch.batch_size != 1:
        raise ValueError(
            "dsv4 huge_kernel only accepts batch_size=1; "
            f"got {forward_batch.batch_size}"
        )
    num_tokens = forward_batch.extend_num_tokens
    if num_tokens is None or not 1 <= num_tokens <= DSV4_HUGE_MAX_EXTEND_TOKENS:
        raise ValueError(
            "dsv4 huge_kernel EXTEND size must be in [1, 4096] so the 64K "
            f"cache can be built incrementally; got {num_tokens!r}"
        )


def validate_dsv4_huge_kernel_bench_args(bench_args) -> None:
    """Fail in the parent process before bench_one_batch spawns TP workers."""

    errors = []
    if tuple(bench_args.batch_size) != (1,):
        errors.append(f"--batch-size must be exactly 1, got {bench_args.batch_size}")
    if tuple(bench_args.output_len) != (1,):
        errors.append(
            "--output-len must be exactly 1 for prefill TTFT, "
            f"got {bench_args.output_len}"
        )
    if bench_args.correctness_test:
        errors.append("--correctness-test is unsupported in the prefill-only runner")
    invalid_input_lens = [
        value
        for value in bench_args.input_len
        if not 1 <= value <= DSV4_HUGE_MAX_EXTEND_TOKENS
    ]
    if invalid_input_lens:
        errors.append(
            "every --input-len must be in [1, 4096], got "
            f"{invalid_input_lens}"
        )
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
        super().__init__(*args, **kwargs)

    def init_attention_backends(self):
        super().init_attention_backends()
        if self._huge_kernel_layers_bound:
            return
        from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

        installed = 0
        for module in self.model.modules():
            if isinstance(module, DeepseekV4DecoderLayer):
                module.enable_huge_kernel_runner()
                installed += 1
        if installed != self.model_config.num_hidden_layers:
            raise RuntimeError(
                "dsv4 huge_kernel must own every decoder layer; "
                f"installed={installed}, expected={self.model_config.num_hidden_layers}"
            )
        self._huge_kernel_layers_bound = True
        logger.info("Installed dsv4 huge_kernel dispatch on %d layers", installed)

    def forward(self, forward_batch: ForwardBatch, *args, **kwargs):
        if not self._huge_kernel_layers_bound:
            raise RuntimeError(
                "dsv4 huge_kernel forward called before all whole-layer runners "
                "were bound"
            )
        validate_dsv4_huge_kernel_forward(forward_batch)
        return super().forward(forward_batch, *args, **kwargs)
