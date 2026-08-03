"""Strict whole-layer executor for the DeepSeek-V4-Flash B200 prefill path.

The first implementation is deliberately an orchestration boundary.  It does
not hide a call to ``DeepseekV4DecoderLayer.forward`` (or a renamed native
copy); instead it composes the existing fine-grained primitives.  Individual
boundaries can then be replaced by larger CUDA executors without changing the
model loop ABI.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence

import msgspec
import torch

from sglang.srt.arg_groups.overrides import attention_backends_of, resolved_view
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

CompressRatio = Literal[0, 4, 128]

_MAX_FORWARD_TOKENS = 4096
_MAX_FORWARD_REQUESTS = 16

# This is model identity, not a generic V4 default.  Fail rather than silently
# running a different architecture through a shape-specialized executor.
_DSV4_FLASH_RATIOS: tuple[int, ...] = (
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0,
)


class DSV4ForwardDescriptor(msgspec.Struct, frozen=True, kw_only=True):
    """One host descriptor shared by every local decoder layer.

    Tensor values stay on device.  Construction reads shapes and object
    references only and must not call ``item()``, ``tolist()`` or synchronize.
    ``forward_batch`` remains present while legacy primitives are migrated to
    direct descriptor fields.
    """

    generation: int
    forward_batch: Any
    attn_backend: Any
    forward_metadata: Any
    core_attn_metadata: Any
    positions: torch.Tensor
    input_ids: torch.Tensor
    input_ids_global: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    attention_q_padded: torch.Tensor
    wo_a_output_q: torch.Tensor
    wo_a_output_s_storage: torch.Tensor
    wo_a_gemm_output: torch.Tensor
    num_tokens: int
    batch_size: int


LayerExecutor = Callable[
    [
        "DSV4WholeLayerRuntime",
        "DSV4LayerHandle",
        DSV4ForwardDescriptor,
        torch.Tensor,
    ],
    tuple[torch.Tensor, None, None, None],
]


class DSV4LayerHandle(msgspec.Struct, frozen=True, kw_only=True):
    """Weight-lifetime handle with a statically selected ratio executor."""

    generation: int
    layer_id: int
    compress_ratio: CompressRatio
    layer: Any
    execute: LayerExecutor


class DSV4WholeLayerRuntime:
    """Model-scoped, strict runtime used only by ``huge_kernel`` mode."""

    def __init__(self, *, config: Any, server_args: Any) -> None:
        self._validate_static_config(config, server_args)
        self._config = config
        self._server_args = server_args
        self._generation = 0
        self._handles: tuple[DSV4LayerHandle, ...] = ()
        self._active: Optional[DSV4ForwardDescriptor] = None
        # One fixed-capacity allocation serves both req=1 and req<=16. Views are
        # exact-T and contiguous, so changing the per-request split never calls
        # the CUDA allocator or changes any layer ABI.
        self._wo_a_workspace: Optional[
            tuple[
                torch.device,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = None

    @property
    def handles(self) -> tuple[DSV4LayerHandle, ...]:
        if not self._handles:
            raise RuntimeError(
                "DSV4 huge runtime has no layer handles; bind_after_weight_load "
                "must run after DeepSeek-V4 post_load_weights"
            )
        return self._handles

    @property
    def active_descriptor(self) -> DSV4ForwardDescriptor:
        descriptor = self._active
        if descriptor is None:
            raise RuntimeError(
                "DeepseekV4DecoderLayer entered huge mode outside an active "
                "DSV4 forward descriptor"
            )
        return descriptor

    def bind_after_weight_load(
        self, layers: Sequence[Any], *, start_layer: int, end_layer: int
    ) -> tuple[DSV4LayerHandle, ...]:
        """Bind or refresh handles after initial load and every weight update."""

        if self._active is not None:
            raise RuntimeError("cannot rebind DSV4 layer handles during a forward")
        self._generation += 1
        generation = self._generation
        handles: list[DSV4LayerHandle] = []
        for layer_id in range(start_layer, end_layer):
            layer = layers[layer_id]
            ratio = int(layer.self_attn.compress_ratio)
            try:
                executor = _RATIO_EXECUTORS[ratio]
            except KeyError as exc:
                raise RuntimeError(
                    f"DSV4 huge runtime: layer {layer_id} has unsupported "
                    f"compress_ratio={ratio}"
                ) from exc
            self._validate_layer(layer, layer_id, ratio)
            handles.append(
                DSV4LayerHandle(
                    generation=generation,
                    layer_id=layer_id,
                    compress_ratio=ratio,  # type: ignore[arg-type]
                    layer=layer,
                    execute=executor,
                )
            )
        self._handles = tuple(handles)
        for handle in self._handles:
            # The DecoderLayer.forward boundary owns dispatch.  Re-loading
            # weights refreshes all generation-tagged handles in one pass.
            handle.layer.enable_huge_kernel_runner(self, handle)
        return self._handles

    def begin_forward(
        self,
        *,
        forward_batch: Any,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
    ) -> DSV4ForwardDescriptor:
        if self._active is not None:
            raise RuntimeError("nested DSV4 huge-runtime forwards are unsupported")
        if not self._handles:
            raise RuntimeError("DSV4 huge runtime was not bound after weight loading")
        if get_is_capture_mode():
            raise RuntimeError("DSV4 huge runtime does not support CUDA graph capture")
        mode = forward_batch.forward_mode
        if not mode.is_extend_without_speculative():
            raise RuntimeError(
                "DSV4 huge runtime supports ordinary EXTEND only; "
                f"got forward_mode={mode}"
            )
        batch_size = int(forward_batch.req_pool_indices.shape[0])
        num_tokens = int(positions.shape[0])
        if not 1 <= batch_size <= _MAX_FORWARD_REQUESTS:
            raise RuntimeError(
                "DSV4 huge runtime requires 1..16 requests per EXTEND, "
                f"got {batch_size}"
            )
        if not 1 <= num_tokens <= _MAX_FORWARD_TOKENS:
            raise RuntimeError(
                "DSV4 huge runtime requires aggregate M in 1..4096, "
                f"got {num_tokens}"
            )
        extend_lens = forward_batch.extend_seq_lens_cpu
        if extend_lens is None or len(extend_lens) != batch_size:
            raise RuntimeError(
                "DSV4 huge runtime requires one host-mirrored EXTEND length "
                "per request; GPU length readback is forbidden"
            )
        if sum(int(length) for length in extend_lens) != num_tokens:
            raise RuntimeError(
                "DSV4 huge runtime requires sum(extend_seq_lens_cpu) == M; "
                f"got {extend_lens!r} for M={num_tokens}"
            )
        if input_ids.shape[0] != num_tokens:
            raise RuntimeError(
                "DSV4 huge runtime requires one input id per live position: "
                f"ids={input_ids.shape[0]}, positions={num_tokens}"
            )
        attn_backend = get_attn_backend()
        metadata = attn_backend.forward_metadata
        core = metadata.core_attn_metadata
        (
            attention_q_padded,
            output_q,
            output_s_storage,
            wo_a_gemm_output,
        ) = self._get_wo_a_workspace(
            num_tokens,
            positions.device,
        )
        descriptor = DSV4ForwardDescriptor(
            generation=self._generation,
            forward_batch=forward_batch,
            attn_backend=attn_backend,
            forward_metadata=metadata,
            core_attn_metadata=core,
            positions=positions,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            extend_seq_lens=forward_batch.extend_seq_lens,
            out_cache_loc=forward_batch.out_cache_loc,
            attention_q_padded=attention_q_padded,
            wo_a_output_q=output_q,
            wo_a_output_s_storage=output_s_storage,
            wo_a_gemm_output=wo_a_gemm_output,
            num_tokens=num_tokens,
            batch_size=batch_size,
        )
        self._active = descriptor
        return descriptor

    def _get_wo_a_workspace(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        workspace = self._wo_a_workspace
        if workspace is not None:
            (
                cached_device,
                attention_q_padded,
                output_q,
                output_s_storage,
                wo_a_gemm_output,
            ) = workspace
            if cached_device == device:
                return (
                    attention_q_padded[:num_tokens],
                    output_q[:num_tokens],
                    output_s_storage[: 2 * num_tokens * 32].view(2, num_tokens, 32),
                    wo_a_gemm_output[:num_tokens],
                )
        # Decoder layers execute serially on one stream. Allocate the supported
        # aggregate-M capacity once; exact-T prefix views remain contiguous.
        attention_q_padded = torch.empty(
            (_MAX_FORWARD_TOKENS, 64, 512),
            dtype=torch.bfloat16,
            device=device,
        )
        output_q = torch.empty(
            (_MAX_FORWARD_TOKENS, 2, 4096),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        # Keep this flat so [2,T,32] can be an exact contiguous view for every
        # T. Slicing a [2,capacity,32] tensor on dim 1 would not be contiguous.
        output_s_storage = torch.empty(
            (2 * _MAX_FORWARD_TOKENS * 32,),
            dtype=torch.float32,
            device=device,
        )
        wo_a_gemm_output = torch.empty(
            (_MAX_FORWARD_TOKENS, 2, 1024),
            dtype=torch.bfloat16,
            device=device,
        )
        self._wo_a_workspace = (
            device,
            attention_q_padded,
            output_q,
            output_s_storage,
            wo_a_gemm_output,
        )
        return (
            attention_q_padded[:num_tokens],
            output_q[:num_tokens],
            output_s_storage[: 2 * num_tokens * 32].view(2, num_tokens, 32),
            wo_a_gemm_output[:num_tokens],
        )

    def execute_layer(
        self,
        handle: DSV4LayerHandle,
        descriptor: DSV4ForwardDescriptor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        if descriptor is not self._active:
            raise RuntimeError("stale or foreign DSV4 forward descriptor")
        if handle.generation != self._generation:
            raise RuntimeError(
                f"stale DSV4 layer handle for layer {handle.layer_id}: "
                f"handle generation={handle.generation}, runtime={self._generation}"
            )
        if hidden_states.shape[0] != descriptor.num_tokens:
            raise RuntimeError(
                f"layer {handle.layer_id}: hidden token count changed inside "
                f"huge executor ({hidden_states.shape[0]} != {descriptor.num_tokens})"
            )
        # No condition on ratio here: the handle owns its exact C0/C4/C128 call.
        return handle.execute(self, handle, descriptor, hidden_states)

    def end_forward(self, descriptor: DSV4ForwardDescriptor) -> None:
        if descriptor is not self._active:
            raise RuntimeError("ending a stale or foreign DSV4 forward descriptor")
        self._active = None

    def abort_forward(self, descriptor: DSV4ForwardDescriptor) -> None:
        """Clear lifecycle state while preserving the original exception."""

        if descriptor is self._active:
            self._active = None

    @staticmethod
    def _validate_static_config(config: Any, server_args: Any) -> None:
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if architectures != ("DeepseekV4ForCausalLM",):
            raise RuntimeError(
                "DSV4 huge runtime requires architecture "
                f"DeepseekV4ForCausalLM, got {architectures}"
            )
        actual_ratios = tuple(int(x) for x in config.compress_ratios)
        if actual_ratios != _DSV4_FLASH_RATIOS:
            raise RuntimeError(
                "DSV4 huge runtime requires the exact DeepSeek-V4-Flash "
                "compress_ratios layout: 43 decoder entries plus the model's "
                "trailing C0 sentinel"
            )
        expected_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hc_mult": 4,
        }
        for name, expected in expected_config.items():
            actual = int(getattr(config, name))
            if actual != expected:
                raise RuntimeError(
                    f"DSV4 huge runtime requires config.{name}={expected}, "
                    f"got {actual}"
                )
        view = resolved_view(server_args)
        expected_args = {
            "tp_size": 4,
            "ep_size": 4,
            "pp_size": 1,
            "chunked_prefill_size": 4096,
            "moe_runner_backend": "flashinfer_mxfp4",
            "disable_overlap_schedule": True,
            "enable_dsa_prefill_context_parallel": False,
            "enable_two_batch_overlap": False,
            "speculative_algorithm": None,
        }
        for name, expected in expected_args.items():
            actual = getattr(view, name)
            if actual != expected:
                raise RuntimeError(
                    f"DSV4 huge runtime requires --{name.replace('_', '-')}="
                    f"{expected!r}, got {actual!r}"
                )
        prefill_backend, decode_backend = attention_backends_of(view)
        if (prefill_backend, decode_backend) != ("dsv4", "dsv4"):
            raise RuntimeError(
                "DSV4 huge runtime requires dsv4 for both attention phases, "
                f"got prefill/decode={prefill_backend!r}/{decode_backend!r}"
            )
        if view.cuda_graph_config.prefill.backend != Backend.DISABLED:
            raise RuntimeError("DSV4 huge runtime requires prefill CUDA graph disabled")
        if view.cuda_graph_config.decode.backend != Backend.DISABLED:
            raise RuntimeError("DSV4 huge runtime requires decode CUDA graph disabled")
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("DSV4 huge runtime requires NVIDIA CUDA")
        capability = torch.cuda.get_device_capability()
        if capability not in ((10, 0), (10, 3)):
            raise RuntimeError(
                "DSV4 huge runtime requires B200/SM100 or B300/SM103; got "
                f"sm{capability[0]}{capability[1]}"
            )

    @staticmethod
    def _validate_layer(layer: Any, layer_id: int, ratio: int) -> None:
        attn = layer.self_attn
        if int(attn.layer_id) != layer_id:
            raise RuntimeError(
                f"DSV4 huge runtime layer id mismatch: {attn.layer_id} != {layer_id}"
            )
        if ratio == 0:
            if attn.indexer is not None or attn.compressor is not None:
                raise RuntimeError(f"C0 layer {layer_id} unexpectedly owns compression")
        elif ratio == 4:
            if attn.indexer is None or attn.compressor is None:
                raise RuntimeError(
                    f"C4 layer {layer_id} requires both indexer and compressor"
                )
        elif ratio == 128:
            if attn.indexer is not None or attn.compressor is None:
                raise RuntimeError(
                    f"C128 layer {layer_id} requires compressor and no C4 indexer"
                )
        else:
            raise AssertionError("ratio dispatch validation is incomplete")
        if attn.n_local_heads != 16 or attn.n_local_groups != 2:
            raise RuntimeError(
                f"layer {layer_id}: TP4 specialization requires 16 local heads "
                f"and 2 local output groups, got {attn.n_local_heads}/"
                f"{attn.n_local_groups}"
            )
        if not hasattr(attn.wo_a, "weight_scale_inv"):
            raise RuntimeError(
                f"layer {layer_id}: huge WO_A fusion requires FP8 wo_a weights"
            )
        if layer.use_fused_mhc_post_pre:
            raise RuntimeError(
                "DSV4 huge runtime v1 forbids cross-layer mHC fusion; each "
                "DecoderLayer must remain a self-contained executor"
            )


def _execute_common(
    runtime: DSV4WholeLayerRuntime,
    handle: DSV4LayerHandle,
    descriptor: DSV4ForwardDescriptor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, None, None, None]:
    """Exact non-cross-layer-fused decoder composition.

    Communication remains in ``_run_moe_ffn_dp_sync`` and in the existing
    attention/linear primitives.  This function is intentionally explicit: a
    future fused executor replaces individual calls here, never a hidden
    whole-layer native call.
    """

    del runtime
    layer = handle.layer
    if layer.use_fused_mhc_post_pre:
        raise RuntimeError(
            "DSV4 huge runtime does not support cross-layer mHC fusion in v1"
        )

    residual = hidden_states
    hidden_states, post, comb, norm_fused = layer.hc_pre(
        hidden_states,
        layer.hc_attn_fn,
        layer.hc_attn_scale,
        layer.hc_attn_base,
        norm=layer.input_layernorm,
        forward_batch=descriptor.forward_batch,
    )
    if not norm_fused:
        hidden_states = layer.input_layernorm(hidden_states)

    # MQALayer owns the explicit Q/KV/indexer/compressor/attention/WO_A/WO_B
    # primitives.  The handle authorizes only the strict fused output boundary.
    hidden_states = layer.self_attn(
        x=hidden_states,
        positions=descriptor.positions,
        forward_batch=descriptor.forward_batch,
        x_quant=None,
        e2e_handle=handle,
        e2e_descriptor=descriptor,
    )

    hidden_states = layer.hc_post(hidden_states, residual, post, comb)
    residual = hidden_states
    hidden_states, post, comb, norm_fused = layer.hc_pre(
        hidden_states,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        norm=layer.post_attention_layernorm,
        forward_batch=descriptor.forward_batch,
    )
    if not norm_fused:
        hidden_states = layer.post_attention_layernorm(hidden_states)

    hidden_states = layer._run_moe_ffn_dp_sync(
        hidden_states,
        descriptor.forward_batch,
        input_ids=descriptor.input_ids,
        input_ids_global=descriptor.input_ids_global,
    )
    hidden_states = layer.hc_post(hidden_states, residual, post, comb)
    return hidden_states, None, None, None


def _execute_c0(
    runtime: DSV4WholeLayerRuntime,
    handle: DSV4LayerHandle,
    descriptor: DSV4ForwardDescriptor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, None, None, None]:
    if handle.compress_ratio != 0:
        raise AssertionError("C0 executor received a non-C0 handle")
    return _execute_common(runtime, handle, descriptor, hidden_states)


def _execute_c4(
    runtime: DSV4WholeLayerRuntime,
    handle: DSV4LayerHandle,
    descriptor: DSV4ForwardDescriptor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, None, None, None]:
    if handle.compress_ratio != 4:
        raise AssertionError("C4 executor received a non-C4 handle")
    return _execute_common(runtime, handle, descriptor, hidden_states)


def _execute_c128(
    runtime: DSV4WholeLayerRuntime,
    handle: DSV4LayerHandle,
    descriptor: DSV4ForwardDescriptor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, None, None, None]:
    if handle.compress_ratio != 128:
        raise AssertionError("C128 executor received a non-C128 handle")
    return _execute_common(runtime, handle, descriptor, hidden_states)


_RATIO_EXECUTORS: dict[int, LayerExecutor] = {
    0: _execute_c0,
    4: _execute_c4,
    128: _execute_c128,
}


__all__ = [
    "DSV4ForwardDescriptor",
    "DSV4LayerHandle",
    "DSV4WholeLayerRuntime",
]
