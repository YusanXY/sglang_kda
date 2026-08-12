from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.runtime_context import get_forward, get_server_args
from sglang.srt.utils import (
    is_flashinfer_available,
    log_info_on_rank0,
    set_weight_attrs,
)
from sglang.srt.utils.common import is_sm100_supported, next_power_of_2

_MXFP8_QUANTIZE_BACKEND = "cute-dsl" if is_sm100_supported() else "cuda"

if is_flashinfer_available():
    from flashinfer import mxfp8_quantize, shuffle_matrix_a, shuffle_matrix_sf_a
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

from sglang.srt.utils.common import get_bool_env_var

_USE_OFFICIAL_SHUFFLE = get_bool_env_var(
    "SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE", default="true"
)

_DSV4_MOE_OVERLAP_INSTALLED = False
_DSV4_DP_FINALIZE_SCALE_INSTALLED = False
_FLASHINFER_CUBIN_OVERLAY_INSTALLED = False
_DSV4_HUGE_TOP_K = 6


@dataclass(frozen=True, eq=False)
class Dsv4DpRawMoeRequest:
    """Strict Huge-only request for FlashInfer's unfinalized GEMM2 ABI."""

    x_quant: torch.Tensor
    x_scale: torch.Tensor
    symmetric_slot_anchor: torch.Tensor
    routed_scale: float


@dataclass(frozen=True, eq=False)
class Dsv4DpRawMoeOutput:
    """Owning lifetime for the same-main-stream composite input ABI.

    FlashInfer's two internal tensors are also retained by its persistent arena;
    packed routing and the symmetric slot are descriptor-owned.  The whole-layer
    consumer is deliberately submitted on the producer stream before this holder
    is released, so per-layer ``Tensor.record_stream`` host calls are unnecessary.
    """

    gemm2_out: torch.Tensor
    expanded_to_permuted: torch.Tensor
    packed_topk: torch.Tensor
    symmetric_slot_anchor: torch.Tensor
    top_k: int
    routed_scale: float


def _resolve_mxfp4_packed_topk(
    topk_output,
    hidden_states: torch.Tensor,
    *,
    dsv4_worker_backend: str,
    layer_id=None,
) -> tuple[torch.Tensor, int]:
    """Resolve the routing ABI without allowing a Huge-to-native fallback."""
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    is_dsv4_huge = dsv4_worker_backend == "huge_kernel"
    if TopKOutputChecker.format_is_packed_only(topk_output):
        if not is_dsv4_huge:
            raise RuntimeError(
                "packed-only MXFP4 routing is restricted to DSV4 Huge; "
                f"layer_id={layer_id}, backend={dsv4_worker_backend!r}"
            )
        packed_topk = topk_output.packed_topk_ids
    elif TopKOutputChecker.format_is_standard(topk_output):
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        if not is_dsv4_huge:
            # Keep the native path unchanged: standard routing is packed by
            # the existing kernel immediately before FlashInfer MXFP4 MoE.
            return PackTopkIds.execute(topk_ids, topk_weights), topk_ids.shape[1]

        packed_topk = getattr(topk_output, "packed_topk_ids", None)
        if packed_topk is None:
            raise RuntimeError(
                "DSV4 Huge requires fused packed routing for every MoE layer; "
                f"layer_id={layer_id}, got {type(topk_output).__name__}"
            )
    elif TopKOutputChecker.format_is_bypassed(topk_output):
        raise NotImplementedError(
            "the old code in this branch is WRONG. e.g. it does not consider "
            "HashTopK, and may miss args"
        )
    else:
        raise ValueError(f"Unsupported topk output format: {topk_output.format}")

    if not isinstance(packed_topk, torch.Tensor):
        raise RuntimeError(
            "invalid DSV4 Huge packed routing tensor; "
            f"layer_id={layer_id}, got {type(packed_topk).__name__}"
        )

    expected_shape = (hidden_states.shape[0], _DSV4_HUGE_TOP_K)
    if (
        packed_topk.shape != expected_shape
        or packed_topk.dtype != torch.int32
        or not packed_topk.is_contiguous()
        or packed_topk.device != hidden_states.device
        or packed_topk.device.type != "cuda"
    ):
        raise RuntimeError(
            "invalid DSV4 Huge packed routing workspace; "
            f"layer_id={layer_id}, expected shape={expected_shape}, "
            "dtype=torch.int32, "
            f"device={hidden_states.device} (CUDA), contiguous=True; got "
            f"shape={tuple(packed_topk.shape)}, dtype={packed_topk.dtype}, "
            f"device={packed_topk.device}, contiguous={packed_topk.is_contiguous()}"
        )

    return packed_topk, _DSV4_HUGE_TOP_K


def _install_writable_flashinfer_cubin_overlay() -> None:
    """Stage writable JIT include links when flashinfer-cubin is read-only."""
    global _FLASHINFER_CUBIN_OVERLAY_INSTALLED

    if _FLASHINFER_CUBIN_OVERLAY_INSTALLED:
        return

    from flashinfer.jit import env as jit_env
    from flashinfer.jit.cubin_loader import ensure_symlink

    packaged_dir = jit_env.FLASHINFER_CUBIN_DIR
    if os.access(packaged_dir, os.W_OK):
        _FLASHINFER_CUBIN_OVERLAY_INSTALLED = True
        return

    overlay_dir = jit_env.FLASHINFER_WORKSPACE_DIR / "cubin_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for artifact in packaged_dir.iterdir():
        ensure_symlink(overlay_dir / artifact.name, artifact)

    # FlashInfer's generator reads this value dynamically for include staging.
    # Its artifact loader keeps reading the immutable packaged directory.
    jit_env.FLASHINFER_CUBIN_DIR = overlay_dir
    _FLASHINFER_CUBIN_OVERLAY_INSTALLED = True


def _install_dsv4_huge_moe_overlap() -> None:
    """Select the Huge-only TRTLLM MoE module before FlashInfer builds it."""
    global _DSV4_DP_FINALIZE_SCALE_INSTALLED, _DSV4_MOE_OVERLAP_INSTALLED

    _install_writable_flashinfer_cubin_overlay()
    server_args = get_server_args()
    dp_finalize_scale = envs.SGLANG_DSV4_HUGE_DP_MOE_FINALIZE_SCALE.get()
    dp_defer_raw = envs.SGLANG_DSV4_HUGE_DP_MOE_DEFER_RAW.get()
    if dp_defer_raw and not dp_finalize_scale:
        raise RuntimeError(
            "SGLANG_DSV4_HUGE_DP_MOE_DEFER_RAW=1 requires "
            "SGLANG_DSV4_HUGE_DP_MOE_FINALIZE_SCALE=1"
        )
    if (dp_finalize_scale or dp_defer_raw) and (
        server_args.dsv4_worker_backend != "huge_kernel"
        or not server_args.enable_dp_attention
    ):
        raise RuntimeError(
            "SGLANG_DSV4_HUGE_DP_MOE_FINALIZE_SCALE=1 requires the DSV4 "
            "Huge Attention-DP backend"
        )
    if dp_defer_raw and not server_args.disable_flashinfer_autotune:
        raise RuntimeError(
            "SGLANG_DSV4_HUGE_DP_MOE_DEFER_RAW=1 requires "
            "--disable-flashinfer-autotune because FlashInfer tuning forwards "
            "would consume the one-shot raw descriptor before the real call"
        )
    if server_args.dsv4_worker_backend != "huge_kernel":
        return
    if server_args.enable_dp_attention:
        if not dp_finalize_scale:
            # The patched module is otherwise TP-only and creates a second
            # multi-minute SM103 JIT artifact.  Keep the production DP path on
            # the packaged FlashInfer launcher unless the v72 gate is explicit.
            return
        required_dp_stack = (
            "SGLANG_DSV4_HUGE_DP_SYMM_MOE_POST",
            "SGLANG_DSV4_HUGE_DP_MOE_EPOCH",
            "SGLANG_DSV4_HUGE_DP_MOE_EPOCH_COUNTER",
            "SGLANG_DSV4_HUGE_DP_MOE_NVLS",
        )
        missing = tuple(
            name
            for name in required_dp_stack
            if os.environ.get(name, "0") != "1"
        )
        if missing:
            raise RuntimeError(
                "DSV4 Huge DP finalize-scale requires the complete symmetric "
                "NVLS epoch-counter stack; missing " + ", ".join(missing)
            )
        _DSV4_DP_FINALIZE_SCALE_INSTALLED = True
    if _DSV4_MOE_OVERLAP_INSTALLED:
        return

    from flashinfer.fused_moe import core as flashinfer_moe_core

    cache_info = flashinfer_moe_core.get_trtllm_moe_sm100_module.cache_info()
    if cache_info.currsize:
        raise RuntimeError(
            "DSV4 Huge MoE overlap must be installed before the FlashInfer "
            "TRTLLM SM100 module is built"
        )

    from sglang.jit_kernel.dsv4_moe_overlap import (
        gen_dsv4_trtllm_gen_fused_moe_sm100_module,
    )

    flashinfer_moe_core.gen_trtllm_gen_fused_moe_sm100_module = (
        gen_dsv4_trtllm_gen_fused_moe_sm100_module
    )
    _DSV4_MOE_OVERLAP_INSTALLED = True


class Mxfp4FlashinferTrtllmMoEMethod:

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        self.flashinfer_mxfp4_moe_precision = (
            get_server_args().flashinfer_mxfp4_moe_precision
        )

    def create_moe_runner(self, layer, moe_runner_config):
        _install_dsv4_huge_moe_overlap()
        self.moe_runner_config = moe_runner_config

        swiglu_limit = moe_runner_config.swiglu_limit
        assert (
            swiglu_limit is not None
        ), f"swiglu_limit must be non-None for DeepSeek V4 (got {swiglu_limit!r})"
        self._gemm1_clamp_limit_tensor = (
            torch.full(
                (layer.num_local_experts,),
                swiglu_limit,
                dtype=torch.float32,
                device=layer.w13_weight.device,
            )
            if swiglu_limit is not None
            else None
        )

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        fp4_block_k = 32

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.utils import reorder_w1w3_to_w3w1

        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        w13_w, w13_s = reorder_w1w3_to_w3w1(
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        layer.w13_weight = Parameter(w13_w, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_s, requires_grad=False)

        log_info_on_rank0(
            logger,
            f"Shuffling FP4 expert weights for TRT-LLM MxFP4 kernel "
            f"(layer: {self.prefix})...",
        )

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data
        num_experts = w13.shape[0]

        if w13_scale.dtype == torch.float32:
            w13_scale = w13_scale.to(torch.float8_e8m0fnu)
            w2_scale = w2_scale.to(torch.float8_e8m0fnu)

        epilogue_tile_m = 128
        g1_w, g1_s, g2_w, g2_s = [], [], [], []
        if _USE_OFFICIAL_SHUFFLE:
            cache: dict = {}
            for i in range(num_experts):
                w13_u8 = w13[i].view(torch.uint8)
                w13_s_u8 = w13_scale[i].view(torch.uint8)
                w2_u8 = w2[i].view(torch.uint8)
                w2_s_u8 = w2_scale[i].view(torch.uint8)

                perm = _maybe_get_cached_w3_w1_permute_indices(
                    cache,
                    w13_u8,
                    epilogue_tile_m,
                )
                g1_w.append(w13_u8[perm.to(w13_u8.device)].contiguous())
                perm_sf = _maybe_get_cached_w3_w1_permute_indices(
                    cache,
                    w13_s_u8,
                    epilogue_tile_m,
                    num_elts_per_sf=16,
                )
                g1_s.append(
                    block_scale_interleave(
                        w13_s_u8[perm_sf.to(w13_s_u8.device)].contiguous()
                    )
                )

                perm = get_w2_permute_indices_with_cache(
                    cache,
                    w2_u8,
                    epilogue_tile_m,
                )
                g2_w.append(w2_u8[perm.to(w2_u8.device)].contiguous())
                perm_sf = get_w2_permute_indices_with_cache(
                    cache,
                    w2_s_u8,
                    epilogue_tile_m,
                    num_elts_per_sf=16,
                )
                g2_s.append(
                    block_scale_interleave(
                        w2_s_u8[perm_sf.to(w2_s_u8.device)].contiguous()
                    )
                )
        else:
            for i in range(num_experts):
                g1_w.append(shuffle_matrix_a(w13[i].view(torch.uint8), epilogue_tile_m))
                g1_s.append(
                    shuffle_matrix_sf_a(w13_scale[i].view(torch.uint8), epilogue_tile_m)
                )
                g2_w.append(shuffle_matrix_a(w2[i].view(torch.uint8), epilogue_tile_m))
                g2_s.append(
                    shuffle_matrix_sf_a(w2_scale[i].view(torch.uint8), epilogue_tile_m)
                )

        layer.w13_weight = Parameter(torch.stack(g1_w), requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(
            torch.stack(g1_s)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, w13.shape[1], -1),
            requires_grad=False,
        )
        layer.w2_weight = Parameter(torch.stack(g2_w), requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(
            torch.stack(g2_s)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, w2.shape[1], -1),
            requires_grad=False,
        )

        self._register_static_scale_ones(layer)
        torch.cuda.empty_cache()

    def _register_static_scale_ones(self, layer: Module) -> None:
        device = layer.w13_weight.device
        for name in (
            "output1_scale_scalar",
            "output1_scale_gate_scalar",
            "output2_scale_scalar",
        ):
            layer.register_buffer(
                name,
                torch.ones(layer.num_local_experts, device=device, dtype=torch.float32),
                persistent=False,
            )

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
        prequant=None,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        shared_output = None
        routed_scaling_factor = None
        dp_routed_scaling_factor = None
        dp_raw_request = (
            prequant if isinstance(prequant, Dsv4DpRawMoeRequest) else None
        )

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale_inv
        w2_scale = layer.w2_weight_scale_inv

        intermediate_size = w2.shape[2] * 2 if w2.dtype == torch.uint8 else w2.shape[2]
        hidden_size = w13.shape[2] * 2 if w13.dtype == torch.uint8 else w13.shape[2]

        num_local_experts = layer.num_local_experts
        if w13_scale.dim() == 2:
            w13_scale = w13_scale.reshape(num_local_experts, 2 * intermediate_size, -1)
        if w2_scale.dim() == 2:
            w2_scale = w2_scale.reshape(num_local_experts, hidden_size, -1)

        packed_topk, top_k = _resolve_mxfp4_packed_topk(
            topk_output,
            hidden_states,
            dsv4_worker_backend=get_server_args().dsv4_worker_backend,
            layer_id=getattr(layer, "layer_id", None),
        )

        precision = self.flashinfer_mxfp4_moe_precision
        if dp_raw_request is not None and precision != "default":
            raise RuntimeError(
                "DSV4 Huge deferred raw MoE requires default MXFP4/MXFP8 precision"
            )
        if (
            prequant is not None
            and dp_raw_request is None
            and len(prequant) == 3
            and precision != "default"
        ):
            raise RuntimeError(
                "DSV4 Huge DP routed-only finalize-scale requires default "
                "MXFP4/MXFP8 precision"
            )
        if precision == "bf16":
            assert hidden_states.dtype == torch.bfloat16
            x_quant = hidden_states
            x_scale = None
            origin_dim = x_quant.shape[-1]
            if hidden_size != origin_dim:
                x_quant = torch.nn.functional.pad(
                    x_quant,
                    (0, hidden_size - origin_dim),
                    mode="constant",
                    value=0.0,
                )
        elif precision == "default":
            if prequant is None:
                x_quant, x_scale = mxfp8_quantize(
                    hidden_states,
                    False,
                    alignment=hidden_size,
                    backend=_MXFP8_QUANTIZE_BACKEND,
                )
            elif dp_raw_request is not None:
                x_quant = dp_raw_request.x_quant
                x_scale = dp_raw_request.x_scale
                expected_scale_shape = (
                    hidden_states.shape[0],
                    hidden_size // 32,
                )
                if (
                    not isinstance(x_quant, torch.Tensor)
                    or not isinstance(x_scale, torch.Tensor)
                    or hidden_size != 4096
                    or x_quant.shape != (hidden_states.shape[0], hidden_size)
                    or hidden_states.shape != x_quant.shape
                    or x_quant.dtype != torch.float8_e4m3fn
                    or not x_quant.is_contiguous()
                    or x_quant.device != hidden_states.device
                    or x_quant.device.type != "cuda"
                    or x_scale.shape != expected_scale_shape
                    or x_scale.dtype != torch.uint8
                    or not x_scale.is_contiguous()
                    or x_scale.device != hidden_states.device
                ):
                    raise RuntimeError(
                        "invalid DSV4 Huge deferred raw MXFP8 workspace"
                    )
            else:
                if get_server_args().dsv4_worker_backend != "huge_kernel":
                    raise RuntimeError(
                        "routed-MoE prequantization is restricted to DSV4 Huge"
                    )
                if len(prequant) == 2:
                    x_quant, x_scale = prequant
                elif len(prequant) == 3:
                    x_quant, x_scale, dp_routed_scaling_factor = prequant
                    try:
                        dp_routed_scaling_factor = float(
                            dp_routed_scaling_factor
                        )
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "invalid DSV4 Huge DP routed finalize scale"
                        ) from exc
                elif len(prequant) == 4:
                    (
                        x_quant,
                        x_scale,
                        shared_output,
                        routed_scaling_factor,
                    ) = prequant
                else:
                    raise RuntimeError(
                        "DSV4 Huge routed-MoE prequant must have 2, 3, or 4 "
                        "tensors/values"
                    )
                expected_scale_shape = (hidden_states.shape[0], hidden_size // 32)
                if (
                    x_quant.shape != hidden_states.shape
                    or x_quant.dtype != torch.float8_e4m3fn
                    or not x_quant.is_contiguous()
                    or x_quant.device != hidden_states.device
                    or x_scale.shape != expected_scale_shape
                    or x_scale.dtype != torch.uint8
                    or not x_scale.is_contiguous()
                    or x_scale.device != hidden_states.device
                ):
                    raise RuntimeError(
                        "invalid DSV4 Huge routed-MoE MXFP8 prequant workspace"
                    )
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            )
        else:
            raise NotImplementedError(f"Unsupported mxfp4 moe precision: {precision}")

        num_tokens = x_quant.shape[0]
        out_hidden_size = (
            x_quant.shape[-1] * 2
            if x_quant.dtype == torch.uint8
            else x_quant.shape[-1]
        )
        external_symmetric_output = (
            get_forward().moe_output_buffer_external_symmetric
            and get_server_args().dsv4_worker_backend == "huge_kernel"
        )
        if external_symmetric_output:
            symm_output = get_forward().moe_output_buffer
            if (
                symm_output is None
                or symm_output.shape != (num_tokens, out_hidden_size)
                or symm_output.dtype != torch.bfloat16
                or symm_output.device != x_quant.device
                or not symm_output.is_contiguous()
            ):
                raise RuntimeError(
                    "DSV4 Huge MXFP4 MoE requires an exact contiguous "
                    "external symmetric BF16 output buffer"
                )
        else:
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                symm_output = torch.empty(
                    num_tokens,
                    out_hidden_size,
                    dtype=torch.bfloat16,
                    device=x_quant.device,
                )

        if dp_raw_request is not None:
            try:
                dp_raw_scale = float(dp_raw_request.routed_scale)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "invalid DSV4 Huge deferred raw routed scale"
                ) from exc
            anchor = dp_raw_request.symmetric_slot_anchor
            if (
                not _DSV4_DP_FINALIZE_SCALE_INSTALLED
                or get_server_args().dsv4_worker_backend != "huge_kernel"
                or not get_server_args().enable_dp_attention
                or not external_symmetric_output
                or not isinstance(anchor, torch.Tensor)
                or anchor.data_ptr() != symm_output.data_ptr()
                or anchor.shape != symm_output.shape
                or anchor.dtype != torch.bfloat16
                or anchor.device != x_quant.device
                or not anchor.is_contiguous()
                or not math.isfinite(dp_raw_scale)
                or dp_raw_scale != 1.5
            ):
                raise RuntimeError(
                    "invalid DSV4 Huge deferred raw MoE request/slot anchor"
                )
            from sglang.jit_kernel.dsv4_moe_overlap.jit import (
                set_dsv4_dp_deferred_raw,
            )

            set_dsv4_dp_deferred_raw()

        if dp_raw_request is not None:
            pass
        elif dp_routed_scaling_factor is not None:
            if (
                not _DSV4_DP_FINALIZE_SCALE_INSTALLED
                or get_server_args().dsv4_worker_backend != "huge_kernel"
                or not get_server_args().enable_dp_attention
                or not external_symmetric_output
                or not math.isfinite(dp_routed_scaling_factor)
                or dp_routed_scaling_factor <= 0.0
            ):
                raise RuntimeError(
                    "invalid DSV4 Huge DP routed-only finalize-scale descriptor"
                )
            from sglang.jit_kernel.dsv4_moe_overlap.jit import (
                set_dsv4_dp_routed_finalize,
            )

            set_dsv4_dp_routed_finalize(symm_output, dp_routed_scaling_factor)
        elif prequant is not None and shared_output is not None:
            if (
                shared_output.shape != (num_tokens, out_hidden_size)
                or shared_output.dtype != torch.bfloat16
                or not shared_output.is_contiguous()
                or shared_output.device != x_quant.device
                or routed_scaling_factor is None
            ):
                raise RuntimeError(
                    "invalid DSV4 Huge shared-output finalize descriptor"
                )
            from sglang.jit_kernel.dsv4_moe_overlap.jit import (
                set_dsv4_shared_finalize,
            )

            set_dsv4_shared_finalize(shared_output, routed_scaling_factor)

        custom_finalize_armed = (
            dp_raw_request is not None
            or dp_routed_scaling_factor is not None
            or shared_output is not None
        )
        try:
            moe_outputs = trtllm_fp4_block_scale_routed_moe(
                topk_ids=packed_topk,
                routing_bias=None,
                hidden_states=x_quant,
                hidden_states_scale=x_scale,
                gemm1_weights=w13,
                gemm1_weights_scale=w13_scale,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=self._gemm1_clamp_limit_tensor,
                gemm2_weights=w2,
                gemm2_weights_scale=w2_scale,
                gemm2_bias=None,
                output1_scale_scalar=layer.output1_scale_scalar,
                output1_scale_gate_scalar=layer.output1_scale_gate_scalar,
                output2_scale_scalar=layer.output2_scale_scalar,
                num_experts=layer.num_experts,
                top_k=top_k,
                n_group=1,
                topk_group=1,
                intermediate_size=intermediate_size,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=num_local_experts,
                routed_scaling_factor=1.0,
                routing_method_type=int(RoutingMethodType.TopK),
                # FlashInfer's public custom-op exposes the owning GEMM2 and
                # routing intermediates only for do_finalize=False.  The raw
                # one-shot descriptor tells our patched launcher to defer the
                # finalize to the whole-layer executor; every other mode keeps
                # the historical public do_finalize=True ABI.
                do_finalize=dp_raw_request is None,
                tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
                output=symm_output,
            )
        except BaseException:
            if custom_finalize_armed:
                from sglang.jit_kernel.dsv4_moe_overlap.jit import (
                    cancel_dsv4_finalize,
                )

                cancel_dsv4_finalize()
            raise
        if dp_raw_request is not None:
            try:
                raw_output_count = len(moe_outputs)
            except TypeError as exc:
                raise RuntimeError(
                    "DSV4 Huge deferred raw FlashInfer launch returned a "
                    "non-indexable container"
                ) from exc
            if raw_output_count != 3:
                raise RuntimeError(
                    "DSV4 Huge deferred raw FlashInfer launch must return three "
                    f"tensors, got {raw_output_count}"
                )
            gemm2_out = moe_outputs[0]
            expanded_to_permuted = moe_outputs[2]
            if (
                not isinstance(gemm2_out, torch.Tensor)
                or gemm2_out.ndim != 2
                or gemm2_out.shape[0] < num_tokens * top_k
                or gemm2_out.shape[1] != out_hidden_size
                or out_hidden_size != 4096
                or gemm2_out.dtype != torch.bfloat16
                or gemm2_out.device != x_quant.device
                or not gemm2_out.is_contiguous()
                or gemm2_out.data_ptr() == anchor.data_ptr()
                or not isinstance(expanded_to_permuted, torch.Tensor)
                or expanded_to_permuted.shape != (num_tokens * top_k,)
                or expanded_to_permuted.dtype != torch.int32
                or expanded_to_permuted.device != x_quant.device
                or not expanded_to_permuted.is_contiguous()
            ):
                raise RuntimeError(
                    "invalid DSV4 Huge deferred raw FlashInfer tensor ABI"
                )
            output = Dsv4DpRawMoeOutput(
                gemm2_out=gemm2_out,
                expanded_to_permuted=expanded_to_permuted,
                packed_topk=packed_topk,
                symmetric_slot_anchor=anchor,
                top_k=top_k,
                routed_scale=dp_raw_scale,
            )
        elif dp_routed_scaling_factor is not None or shared_output is not None:
            # The patched launcher sets do_finalize=false so FlashInfer exposes
            # GEMM2/routing intermediates to the custom epilogue.  The epilogue
            # writes the caller-owned symmetric buffer; never leak result[0]
            # (the padded GEMM2 intermediate) into the model dataflow.
            output = symm_output
        else:
            output = moe_outputs[0]

        return StandardCombineInput(hidden_states=output)


def maybe_fuse_routed_scale_and_shared_add(
    experts,
    routed: torch.Tensor,
    shared: torch.Tensor | None,
    routed_scaling_factor: float,
) -> torch.Tensor:
    # When MxFP4 fusion is on, the upstream `routed *= scale` is skipped and
    # the scaling is folded into the shared-add via `shared.add_(routed,
    # alpha=scale)`. With no shared output, the missing scale is applied
    # in-place. Otherwise `routed` is already scale-final and we just add
    # `shared` (or pass through if there is none).
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
        Mxfp4FlashinferCutlassMoEMethod,
    )
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
        Mxfp4MarlinMoEMethod,
    )

    fused = isinstance(
        experts.quant_method,
        (
            Mxfp4FlashinferTrtllmMoEMethod,
            Mxfp4FlashinferCutlassMoEMethod,
            Mxfp4MarlinMoEMethod,
        ),
    )
    if fused:
        if shared is not None:
            return shared.add_(routed, alpha=routed_scaling_factor)
        return routed.mul_(routed_scaling_factor)
    if shared is not None:
        routed += shared
    return routed
