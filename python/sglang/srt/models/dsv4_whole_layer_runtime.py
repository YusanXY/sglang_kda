"""Strict whole-layer executor for the DeepSeek-V4-Flash B200 prefill path.

The first implementation is deliberately an orchestration boundary.  It does
not hide a call to ``DeepseekV4DecoderLayer.forward`` (or a renamed native
copy); instead it composes the existing fine-grained primitives.  Individual
boundaries can then be replaced by larger CUDA executors without changing the
model loop ABI.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional, Sequence

import msgspec
import torch

from sglang.srt.arg_groups.overrides import attention_backends_of, resolved_view
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

CompressRatio = Literal[0, 4, 128]

_MAX_FORWARD_TOKENS = 131072
_MAX_FORWARD_REQUESTS = 128
_MAX_MHC_SPLITS = 64

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
    attention_q_local: Optional[torch.Tensor]
    attention_q_recv: Optional[torch.Tensor]
    attention_q_handle: Any
    attention_q_peer0: Optional[torch.Tensor]
    attention_q_peer1: Optional[torch.Tensor]
    attention_q_peer2: Optional[torch.Tensor]
    attention_q_peer3: Optional[torch.Tensor]
    attention_q_rank: int
    attention_q_direct_route: bool
    moe_partial_local: Optional[torch.Tensor]
    moe_partial_peer0: Optional[torch.Tensor]
    moe_partial_peer1: Optional[torch.Tensor]
    moe_partial_peer2: Optional[torch.Tensor]
    moe_partial_peer3: Optional[torch.Tensor]
    moe_owner_flag0: Optional[torch.Tensor]
    moe_owner_flag1: Optional[torch.Tensor]
    moe_owner_flag2: Optional[torch.Tensor]
    moe_owner_flag3: Optional[torch.Tensor]
    attention_packed_send: Optional[torch.Tensor]
    attention_packed_recv: Optional[torch.Tensor]
    attention_packed_recv_q: Optional[torch.Tensor]
    attention_projected_local: Optional[torch.Tensor]
    attention_projected_gather: Optional[torch.Tensor]
    attention_wob_q: Optional[torch.Tensor]
    attention_wob_s_storage: Optional[torch.Tensor]
    attention_wob_q_chunks: Any
    attention_wob_s_chunks: Any
    attention_wob_partial_chunks: Any
    attention_wob_output_handle: Any
    attention_wob_output_peer0: Optional[torch.Tensor]
    attention_wob_output_peer1: Optional[torch.Tensor]
    attention_wob_output_peer2: Optional[torch.Tensor]
    attention_wob_output_peer3: Optional[torch.Tensor]
    attention_wob_ready_local: Optional[torch.Tensor]
    attention_wob_ready_peer0: Optional[torch.Tensor]
    attention_wob_ready_peer1: Optional[torch.Tensor]
    attention_wob_ready_peer2: Optional[torch.Tensor]
    attention_wob_ready_peer3: Optional[torch.Tensor]
    attention_wob_ready_epoch_base: int
    attention_wob_output_rank: int
    attention_wob_direct_gather: bool
    attention_symm_handle: Any
    attention_symm_peer0: Optional[torch.Tensor]
    attention_symm_peer1: Optional[torch.Tensor]
    attention_symm_peer2: Optional[torch.Tensor]
    attention_symm_peer3: Optional[torch.Tensor]
    attention_symm_recv_q: Optional[torch.Tensor]
    attention_symm_recv_scale: Optional[torch.Tensor]
    attention_symm_rank: int
    attention_symm_direct_push: bool
    tp4_token_shard_attention: bool
    tp4_local_wob: bool
    q_lora_bf16: torch.Tensor
    q_lora_fp8: torch.Tensor
    q_lora_scale: torch.Tensor
    mhc_gemm_mul_storage: torch.Tensor
    mhc_gemm_sq_storage: torch.Tensor
    mhc_residual_mid: torch.Tensor
    mhc_residual_out: torch.Tensor
    mhc_post: torch.Tensor
    mhc_comb: torch.Tensor
    mhc_layer_input: torch.Tensor
    mhc_output_fp8: torch.Tensor
    mhc_output_scale_storage: torch.Tensor
    mhc_output_scale: torch.Tensor
    mhc_routed_output_fp8: torch.Tensor
    mhc_routed_output_scale: torch.Tensor
    shared_down_fp8: torch.Tensor
    shared_down_scale: torch.Tensor
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
        # Both strict buckets are Q16-aligned and fit the clustered kernel's
        # fixed M<=131072 capacity. Keep this selected once per model so the
        # high-load path cannot silently fall back to the Q1 DeepGEMM producer.
        self._use_clustered_mqa = True
        # Experimental eager-only C4 path.  Read the gate once at model
        # construction; layer execution never calls getenv or branches on
        # mutable host state.  Unsupported shapes stay on the already-frozen
        # Huge implementation while the req16/req128 buckets are evaluated.
        self._use_tp4_token_shard_attention = (
            os.environ.get("SGLANG_DSV4_HUGE_TP4_TOKEN_SHARD_ATTN", "0") == "1"
        )
        self._use_tp4_symmetric_wob = (
            os.environ.get("SGLANG_DSV4_HUGE_TP4_SYMM_WOB_A2A", "0") == "1"
        )
        self._use_tp4_direct_push_wob = (
            os.environ.get("SGLANG_DSV4_HUGE_TP4_DIRECT_PUSH_WOB", "0") == "1"
        )
        self._use_tp4_local_wob = (
            os.environ.get("SGLANG_DSV4_HUGE_TP4_LOCAL_WOB", "0") == "1"
        )
        self._use_tp4_local_wob_direct_gather = (
            os.environ.get(
                "SGLANG_DSV4_HUGE_TP4_LOCAL_WOB_DIRECT_GATHER", "0"
            )
            == "1"
        )
        if self._use_tp4_symmetric_wob and not self._use_tp4_token_shard_attention:
            raise RuntimeError(
                "TP4 symmetric WO_B A2A requires TP4 token-sharded attention"
            )
        if self._use_tp4_symmetric_wob != self._use_tp4_direct_push_wob:
            raise RuntimeError(
                "v15 requires symmetric WO_B A2A and direct-push to be enabled together"
            )
        if self._use_tp4_local_wob and not self._use_tp4_token_shard_attention:
            raise RuntimeError(
                "TP4 local WO_B requires TP4 token-sharded attention"
            )
        if self._use_tp4_local_wob and self._use_tp4_symmetric_wob:
            raise RuntimeError(
                "TP4 local WO_B and symmetric/direct-push WO_B are exclusive"
            )
        if (
            self._use_tp4_local_wob_direct_gather
            and not self._use_tp4_local_wob
        ):
            raise RuntimeError(
                "TP4 direct output gather requires the local WO_B path"
            )
        self._generation = 0
        self._handles: tuple[DSV4LayerHandle, ...] = ()
        self._active: Optional[DSV4ForwardDescriptor] = None
        self._clustered_logits_workspace: Optional[torch.Tensor] = None
        # One fixed-capacity allocation serves req=1 through req=128. Views are
        # exact-T and contiguous, so changing the per-request split never calls
        # the CUDA allocator or changes any layer ABI.
        self._wo_a_workspace: Optional[
            tuple[
                torch.device,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = None
        self._q_lora_workspace: Optional[
            tuple[torch.device, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self._tp4_attention_workspace: Optional[
            tuple[
                torch.device,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = None
        self._tp4_symmetric_workspace: Optional[tuple] = None
        self._tp4_symmetric_q_workspace: Optional[tuple] = None
        self._tp4_owner_flags_workspace: Optional[tuple] = None
        self._tp4_local_wob_output_workspace: Optional[tuple] = None
        self._attention_output_forward_epoch = 0
        self._shared_down_workspace: Optional[
            tuple[torch.device, torch.Tensor, torch.Tensor]
        ] = None
        self._mhc_pre_workspace: Optional[tuple] = None

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
        # Compile/load the C4 CUDA module at binding time. The first live
        # request must never encounter a per-layer JIT or a hidden fallback.
        if self._use_clustered_mqa:
            from sglang.jit_kernel.dsv4.clustered_mqa_logits import (
                MAX_C4_CONTEXT,
                MAX_TOTAL_Q,
                load_clustered_mqa_extension,
            )

            load_clustered_mqa_extension()
        from sglang.jit_kernel.dsv4.e2e import (
            load_mhc_post_vec8_extension,
            load_mhc_pre_norm_mxfp8_quant_extension,
            load_tp4_nccl_ring_bf16_reduce_extension,
            load_tp4_moe_mhc_post_extension,
        )

        load_mhc_post_vec8_extension()
        load_mhc_pre_norm_mxfp8_quant_extension(160)
        load_tp4_moe_mhc_post_extension()
        if self._use_tp4_local_wob:
            load_tp4_nccl_ring_bf16_reduce_extension()
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend

        fp8_backend = get_fp8_gemm_runner_backend()
        effective_deep_gemm = fp8_backend.is_deep_gemm() or (
            fp8_backend.is_auto() and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        )
        if not effective_deep_gemm or not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            raise RuntimeError(
                "DSV4 Huge q_lora RMSNorm+block-FP8 fusion requires the "
                "Blackwell DeepGEMM UE8M0 backend; got "
                f"{fp8_backend}"
            )
        if self._use_tp4_token_shard_attention:
            self._bind_tp4_token_projection_weights(
                layers, start_layer=start_layer, end_layer=end_layer
            )
        workspace_device = layers[start_layer].self_attn.wo_a.weight.device
        if self._use_tp4_token_shard_attention:
            self._bind_tp4_symmetric_q_workspace(workspace_device)
            self._bind_tp4_owner_flags_workspace(workspace_device)
        if self._use_tp4_symmetric_wob:
            self._bind_tp4_symmetric_workspace(workspace_device)
        if self._use_tp4_local_wob_direct_gather:
            self._bind_tp4_local_wob_output_workspace(workspace_device)
        if self._use_clustered_mqa and (
            self._clustered_logits_workspace is None
            or self._clustered_logits_workspace.device != workspace_device
        ):
            # Bind the full strict-runtime capacity once.  Prefix construction
            # and the measured incremental request then reuse one GPU address,
            # so neither allocator growth nor a new TMA row stride enters TTFT.
            self._clustered_logits_workspace = torch.empty(
                (MAX_TOTAL_Q, MAX_C4_CONTEXT),
                dtype=torch.float32,
                device=workspace_device,
            )
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
            shared_experts = getattr(handle.layer.mlp, "shared_experts", None)
            if shared_experts is not None:
                # Bind the shared expert to its model-scoped GPU workspace so
                # every layer reuses fixed addresses instead of entering the
                # CUDA allocator from Python.
                shared_experts.bind_dsv4_huge_runtime(self)
        return self._handles

    def _bind_tp4_token_projection_weights(
        self, layers: Sequence[Any], *, start_layer: int, end_layer: int
    ) -> None:
        """Replicate every attention layer's WO weights for token sharding.

        The Q exchange makes every rank the owner of one contiguous
        token quarter and all eight attention output groups. Replicating WO_A
        and the four WO_B shards at load time lets that owner finish projection
        locally, then directly gather the completed token quarters. This same
        layout is valid for C0, C4, and C128.

        DeepGEMM scale tensors are logical transposes over contiguous physical
        storage.  Gather the physical words and restore the documented strides
        without dequantizing or requantizing, preserving every UE8M0 byte.
        """
        from sglang.srt.distributed import get_tp_group

        tp_group = get_tp_group()
        if tp_group.world_size != 4:
            raise RuntimeError(
                "TP4 token-projection specialization requires TP=4, got "
                f"{tp_group.world_size}"
            )

        with torch.no_grad():
            for layer_id in range(start_layer, end_layer):
                attn = layers[layer_id].self_attn
                local_wo_a_weight = attn.wo_a.weight.data
                local_wo_a_scale = attn.wo_a.weight_scale_inv.data
                local_wo_b_weight = attn.wo_b.weight.data
                local_wo_b_scale = attn.wo_b.weight_scale_inv.data
                if tuple(local_wo_a_weight.shape) != (2048, 4096):
                    raise RuntimeError(
                        f"layer {layer_id}: unexpected local WO_A weight shape "
                        f"{tuple(local_wo_a_weight.shape)}"
                    )
                if (
                    tuple(local_wo_a_scale.shape) != (2, 1024, 8)
                    or local_wo_a_scale.dtype != torch.int32
                    or local_wo_a_scale.stride() != (8192, 1, 1024)
                ):
                    raise RuntimeError(
                        f"layer {layer_id}: unexpected local WO_A scale layout "
                        f"shape={tuple(local_wo_a_scale.shape)} "
                        f"dtype={local_wo_a_scale.dtype} "
                        f"stride={local_wo_a_scale.stride()}"
                    )
                if tuple(local_wo_b_weight.shape) != (4096, 2048):
                    raise RuntimeError(
                        f"layer {layer_id}: unexpected local WO_B weight shape "
                        f"{tuple(local_wo_b_weight.shape)}"
                    )
                if (
                    tuple(local_wo_b_scale.shape) != (4096, 4)
                    or local_wo_b_scale.dtype != torch.int32
                    or local_wo_b_scale.stride() != (1, 4096)
                ):
                    raise RuntimeError(
                        f"layer {layer_id}: unexpected local WO_B scale layout "
                        f"shape={tuple(local_wo_b_scale.shape)} "
                        f"dtype={local_wo_b_scale.dtype} "
                        f"stride={local_wo_b_scale.stride()}"
                    )

                attn._dsv4_huge_full_wo_a_weight = tp_group.all_gather(
                    local_wo_a_weight.contiguous(), dim=0
                )
                wo_a_words = local_wo_a_scale.untyped_storage().nbytes() // 4
                local_wo_a_physical = local_wo_a_scale.as_strided(
                    (wo_a_words,), (1,)
                )
                full_wo_a_physical = tp_group.all_gather(
                    local_wo_a_physical, dim=0
                )
                attn._dsv4_huge_full_wo_a_scale = full_wo_a_physical.view(
                    8, 8, 1024
                ).permute(0, 2, 1)

                if self._use_tp4_local_wob:
                    wo_b_weight_chunks = [
                        torch.empty_like(local_wo_b_weight) for _ in range(4)
                    ]
                    tp_group.all_gather(
                        local_wo_b_weight.contiguous(),
                        output_tensor_list=wo_b_weight_chunks,
                    )
                    attn._dsv4_huge_wo_b_weight_chunks = tuple(
                        wo_b_weight_chunks
                    )
                    wo_b_words = (
                        local_wo_b_scale.untyped_storage().nbytes() // 4
                    )
                    local_wo_b_physical = local_wo_b_scale.as_strided(
                        (wo_b_words,), (1,)
                    )
                    wo_b_scale_physical_chunks = [
                        torch.empty_like(local_wo_b_physical) for _ in range(4)
                    ]
                    tp_group.all_gather(
                        local_wo_b_physical,
                        output_tensor_list=wo_b_scale_physical_chunks,
                    )
                    attn._dsv4_huge_wo_b_scale_chunks = tuple(
                        physical.view(4, 4096).transpose(0, 1)
                        for physical in wo_b_scale_physical_chunks
                    )

                if (
                    tuple(attn._dsv4_huge_full_wo_a_weight.shape)
                    != (8192, 4096)
                    or attn._dsv4_huge_full_wo_a_scale.stride()
                    != (8192, 1, 1024)
                ):
                    raise RuntimeError(
                        f"layer {layer_id}: failed to construct full TP4 WO_A layout"
                    )
                if self._use_tp4_local_wob and (
                    any(
                        tuple(weight.shape) != (4096, 2048)
                        or not weight.is_contiguous()
                        for weight in attn._dsv4_huge_wo_b_weight_chunks
                    )
                    or any(
                        tuple(scale.shape) != (4096, 4)
                        or scale.stride() != (1, 4096)
                        for scale in attn._dsv4_huge_wo_b_scale_chunks
                    )
                ):
                    raise RuntimeError(
                        f"layer {layer_id}: failed to construct full TP4 WO_B layouts"
                    )

    def _bind_tp4_symmetric_workspace(self, device: torch.device) -> None:
        """Collectively bind exact-M peer-visible WO_B destination buffers."""
        from torch.distributed import _symmetric_memory as symm_mem

        from sglang.srt.distributed import get_tp_group

        cached = self._tp4_symmetric_workspace
        if cached is not None and cached[0] == device:
            return
        tp_group = get_tp_group()
        if tp_group.world_size != 4:
            raise RuntimeError(
                "TP4 symmetric WO_B A2A requires TP=4, got "
                f"{tp_group.world_size}"
            )
        packed_row_bytes = 2064
        buckets = {}
        for num_tokens in (65536, 131072):
            local_storage = symm_mem.empty(
                (num_tokens * packed_row_bytes,),
                dtype=torch.uint8,
                device=device,
            )
            handle = symm_mem.rendezvous(local_storage, tp_group.device_group)
            peers = tuple(
                handle.get_buffer(
                    source_rank, local_storage.shape, local_storage.dtype
                )
                for source_rank in range(4)
            )
            q_bytes = num_tokens * 2048
            local_q = local_storage[:q_bytes].view(torch.float8_e4m3fn).view(
                num_tokens, 2048
            )
            local_scale = (
                local_storage[q_bytes:]
                .view(torch.int32)
                .view(4, num_tokens)
            )
            buckets[num_tokens] = (
                handle,
                local_storage,
                peers,
                local_q,
                local_scale,
            )
        self._tp4_symmetric_workspace = (
            device,
            buckets,
            tp_group.rank_in_group,
        )

    def _bind_tp4_symmetric_q_workspace(self, device: torch.device) -> None:
        """Bind the peer-visible C4 Q receive tensor used by the fused producer."""
        from torch.distributed import _symmetric_memory as symm_mem

        from sglang.srt.distributed import get_tp_group

        cached = self._tp4_symmetric_q_workspace
        if cached is not None and cached[0] == device:
            return
        tp_group = get_tp_group()
        if tp_group.world_size != 4:
            raise RuntimeError(
                "TP4 direct Q routing requires TP=4, got "
                f"{tp_group.world_size}"
            )
        local_q_recv = symm_mem.empty(
            (_MAX_FORWARD_TOKENS, 16, 512),
            dtype=torch.bfloat16,
            device=device,
        )
        handle = symm_mem.rendezvous(local_q_recv, tp_group.device_group)
        peers = tuple(
            handle.get_buffer(rank, local_q_recv.shape, local_q_recv.dtype)
            for rank in range(4)
        )
        self._tp4_symmetric_q_workspace = (
            device,
            handle,
            local_q_recv,
            peers,
            tp_group.rank_in_group,
        )

    def _bind_tp4_owner_flags_workspace(self, device: torch.device) -> None:
        """Bind four peer-visible GPU protocol words once for all layers."""
        from torch.distributed import _symmetric_memory as symm_mem

        from sglang.srt.distributed import get_tp_group

        cached = self._tp4_owner_flags_workspace
        if cached is not None and cached[0] == device:
            return
        tp_group = get_tp_group()
        if tp_group.world_size != 4:
            raise RuntimeError(
                "TP4 owner GPU synchronization requires TP=4, got "
                f"{tp_group.world_size}"
            )
        local_flags = symm_mem.empty((4,), dtype=torch.int32, device=device)
        local_flags.zero_()
        handle = symm_mem.rendezvous(local_flags, tp_group.device_group)
        peers = tuple(
            handle.get_buffer(rank, local_flags.shape, local_flags.dtype)
            for rank in range(4)
        )
        # Initialization is outside the hot path; publish the zero epoch before
        # any layer can enter the device-resident protocol.
        handle.barrier(channel=0)
        self._tp4_owner_flags_workspace = (
            device,
            handle,
            local_flags,
            peers,
        )

    def _bind_tp4_local_wob_output_workspace(
        self, device: torch.device
    ) -> None:
        """Bind one max-capacity symmetric BF16 output on every TP rank."""
        from torch.distributed import _symmetric_memory as symm_mem

        from sglang.srt.distributed import get_tp_group

        cached = self._tp4_local_wob_output_workspace
        if cached is not None and cached[0] == device:
            return
        tp_group = get_tp_group()
        if tp_group.world_size != 4:
            raise RuntimeError(
                "TP4 direct output gather requires TP=4, got "
                f"{tp_group.world_size}"
            )
        output_bytes = _MAX_FORWARD_TOKENS * 4096 * 2
        ready_groups = _MAX_FORWARD_TOKENS // 2
        ready_bytes = ready_groups * 4
        local_storage = symm_mem.empty(
            (output_bytes + ready_bytes,), dtype=torch.uint8, device=device
        )
        local_output = local_storage[:output_bytes].view(torch.bfloat16).view(
            _MAX_FORWARD_TOKENS, 4096
        )
        local_ready = local_storage[output_bytes:].view(torch.int32)
        local_ready.zero_()
        handle = symm_mem.rendezvous(local_storage, tp_group.device_group)
        peer_storage = tuple(
            handle.get_buffer(rank, local_storage.shape, local_storage.dtype)
            for rank in range(4)
        )
        peers = tuple(
            storage[:output_bytes].view(torch.bfloat16).view(
                _MAX_FORWARD_TOKENS, 4096
            )
            for storage in peer_storage
        )
        ready_peers = tuple(
            storage[output_bytes:].view(torch.int32)
            for storage in peer_storage
        )
        handle.barrier(channel=0)
        self._tp4_local_wob_output_workspace = (
            device,
            handle,
            local_output,
            peers,
            local_ready,
            ready_peers,
            tp_group.rank_in_group,
        )

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
        is_graph_capture = get_is_capture_mode()
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
                "DSV4 huge runtime requires 1..128 requests per EXTEND, "
                f"got {batch_size}"
            )
        if not 1 <= num_tokens <= _MAX_FORWARD_TOKENS:
            raise RuntimeError(
                "DSV4 huge runtime requires aggregate M in 1..131072, "
                f"got {num_tokens}"
            )
        if self._use_tp4_token_shard_attention and is_graph_capture:
            raise RuntimeError(
                "TP4 token-sharded attention is an eager-only experimental path"
            )
        if is_graph_capture and (batch_size, num_tokens) not in (
            (1, 4096),
            (16, 65536),
        ):
            raise RuntimeError(
                "DSV4 huge breakable capture requires the exact static shape "
                "req=1/M=4096 or req=16/M=65536; "
                f"got req={batch_size}, M={num_tokens}"
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
        tp4_token_shard_attention = (
            self._use_tp4_token_shard_attention
            and num_tokens in (65536, 131072)
            and batch_size in (16, 32)
            and all(int(length) == 4096 for length in extend_lens)
        )
        if input_ids.shape[0] != num_tokens:
            raise RuntimeError(
                "DSV4 huge runtime requires one input id per live position: "
                f"ids={input_ids.shape[0]}, positions={num_tokens}"
            )
        attn_backend = get_attn_backend()
        metadata = attn_backend.forward_metadata
        core = metadata.core_attn_metadata
        indexer_metadata = metadata.indexer_metadata
        if indexer_metadata is None:
            raise RuntimeError(
                "DSV4 huge runtime requires C4 indexer metadata for every EXTEND"
            )
        if is_graph_capture:
            sparse_cache = metadata.sparse_prefill_cache
            if sparse_cache is None:
                raise RuntimeError(
                    "Huge Graph capture requires a preallocated sparse cache"
                )
            sparse_cache.rebuild_swa_c0_in_graph_(
                req_to_token=attn_backend.req_to_token,
                full_to_swa=(
                    attn_backend.token_to_kv_pool.full_to_swa_index_mapping
                ),
            )
            sparse_cache.rebuild_c4_in_graph_(page_table=core.page_table)
            # Marks the capture object as graph-owned for SWA/C0/C4. Replay
            # refreshes only the request descriptor and C128 buffers.
            sparse_cache.graph_local_rebuild = True
        from sglang.jit_kernel.dsv4.clustered_mqa_logits import (
            prepare_clustered_mqa_metadata,
        )

        # One GPU schedule and strided page-table view for the whole batch;
        # all 21 C4 layers consume these exact objects without replanning. In
        # Breakable Graph mode the captured object keeps stable addresses and
        # DSV4Metadata refreshes its live schedule in place before every replay.
        if self._use_clustered_mqa:
            tp_rank = None
            tp_size = None
            if tp4_token_shard_attention:
                from sglang.srt.distributed.parallel_state import get_tp_group

                tp_group = get_tp_group()
                tp_rank = tp_group.rank_in_group
                tp_size = tp_group.world_size
            metadata.clustered_mqa_metadata = prepare_clustered_mqa_metadata(
                indexer_metadata=indexer_metadata,
                extend_lens_cpu=extend_lens,
                logits_workspace=self._clustered_logits_workspace,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        else:
            metadata.clustered_mqa_metadata = None
        (
            attention_q_padded,
            output_q,
            output_s_storage,
            wo_a_gemm_output,
        ) = self._get_wo_a_workspace(
            num_tokens,
            positions.device,
        )
        if self._use_tp4_token_shard_attention:
            (
                attention_q_local,
                attention_q_recv,
                attention_packed_send,
                attention_packed_recv,
                attention_packed_recv_q,
                attention_projected_local,
                attention_projected_gather,
                attention_wob_q,
                attention_wob_s_storage,
                attention_wob_q_chunks,
                attention_wob_s_chunks,
                attention_wob_partial_chunks,
            ) = (
                self._get_tp4_attention_workspace(num_tokens, positions.device)
            )
            symmetric_q = self._tp4_symmetric_q_workspace
            if symmetric_q is None or symmetric_q[0] != positions.device:
                raise RuntimeError(
                    "TP4 symmetric Q workspace was not bound on this device"
                )
            (
                _,
                attention_q_handle,
                attention_q_storage,
                attention_q_peers,
                attention_q_rank,
            ) = symmetric_q
            del attention_q_storage
            attention_q_peer0 = attention_q_peers[0]
            attention_q_peer1 = attention_q_peers[1]
            attention_q_peer2 = attention_q_peers[2]
            attention_q_peer3 = attention_q_peers[3]
            attention_q_direct_route = tp4_token_shard_attention
            moe_partial_peers = tuple(
                peer.view(-1)[: num_tokens * 4096].view(num_tokens, 4096)
                for peer in attention_q_peers
            )
            moe_partial_local = moe_partial_peers[attention_q_rank]
            moe_partial_peer0 = moe_partial_peers[0]
            moe_partial_peer1 = moe_partial_peers[1]
            moe_partial_peer2 = moe_partial_peers[2]
            moe_partial_peer3 = moe_partial_peers[3]
            owner_flags = self._tp4_owner_flags_workspace
            if owner_flags is None or owner_flags[0] != positions.device:
                raise RuntimeError(
                    "TP4 owner GPU flags were not bound on this device"
                )
            moe_owner_flag_peers = owner_flags[3]
            moe_owner_flag0 = moe_owner_flag_peers[0]
            moe_owner_flag1 = moe_owner_flag_peers[1]
            moe_owner_flag2 = moe_owner_flag_peers[2]
            moe_owner_flag3 = moe_owner_flag_peers[3]
            if (
                self._use_tp4_local_wob_direct_gather
                and tp4_token_shard_attention
            ):
                output_workspace = self._tp4_local_wob_output_workspace
                if (
                    output_workspace is None
                    or output_workspace[0] != positions.device
                ):
                    raise RuntimeError(
                        "TP4 local WO_B output workspace was not bound"
                    )
                (
                    _,
                    attention_wob_output_handle,
                    attention_wob_output_storage,
                    attention_wob_output_peers,
                    attention_wob_ready_local,
                    attention_wob_ready_peers,
                    attention_wob_output_rank,
                ) = output_workspace
                self._attention_output_forward_epoch = (
                    self._attention_output_forward_epoch + 1
                ) & 0x03FFFFFF
                if self._attention_output_forward_epoch == 0:
                    self._attention_output_forward_epoch = 1
                attention_wob_ready_epoch_base = (
                    self._attention_output_forward_epoch << 6
                )
                owner_tokens = num_tokens // 4
                attention_projected_gather = attention_wob_output_storage[
                    :num_tokens
                ]
                owner_begin = attention_wob_output_rank * owner_tokens
                attention_projected_local = attention_projected_gather[
                    owner_begin : owner_begin + owner_tokens
                ]
                attention_wob_output_peer0 = attention_wob_output_peers[0]
                attention_wob_output_peer1 = attention_wob_output_peers[1]
                attention_wob_output_peer2 = attention_wob_output_peers[2]
                attention_wob_output_peer3 = attention_wob_output_peers[3]
                attention_wob_ready_peer0 = attention_wob_ready_peers[0]
                attention_wob_ready_peer1 = attention_wob_ready_peers[1]
                attention_wob_ready_peer2 = attention_wob_ready_peers[2]
                attention_wob_ready_peer3 = attention_wob_ready_peers[3]
                attention_wob_direct_gather = True
            else:
                attention_wob_output_handle = None
                attention_wob_output_peer0 = None
                attention_wob_output_peer1 = None
                attention_wob_output_peer2 = None
                attention_wob_output_peer3 = None
                attention_wob_ready_local = None
                attention_wob_ready_peer0 = None
                attention_wob_ready_peer1 = None
                attention_wob_ready_peer2 = None
                attention_wob_ready_peer3 = None
                attention_wob_ready_epoch_base = 0
                attention_wob_output_rank = -1
                attention_wob_direct_gather = False
            if self._use_tp4_symmetric_wob and tp4_token_shard_attention:
                symmetric = self._tp4_symmetric_workspace
                if symmetric is None or symmetric[0] != positions.device:
                    raise RuntimeError(
                        "TP4 symmetric WO_B workspace was not bound on this device"
                    )
                (
                    _,
                    attention_symm_buckets,
                    attention_symm_rank,
                ) = symmetric
                try:
                    (
                        attention_symm_handle,
                        _,
                        attention_symm_peers,
                        attention_symm_recv_q,
                        attention_symm_recv_scale,
                    ) = attention_symm_buckets[num_tokens]
                except KeyError as exc:
                    raise RuntimeError(
                        "TP4 direct-push supports exact M=65536 or M=131072"
                    ) from exc
                attention_symm_peer0 = attention_symm_peers[0]
                attention_symm_peer1 = attention_symm_peers[1]
                attention_symm_peer2 = attention_symm_peers[2]
                attention_symm_peer3 = attention_symm_peers[3]
                attention_symm_direct_push = True
            else:
                attention_symm_handle = None
                attention_symm_peer0 = None
                attention_symm_peer1 = None
                attention_symm_peer2 = None
                attention_symm_peer3 = None
                attention_symm_recv_q = None
                attention_symm_recv_scale = None
                attention_symm_rank = -1
                attention_symm_direct_push = False
        else:
            attention_q_local, attention_q_recv = None, None
            attention_q_handle = None
            attention_q_peer0 = None
            attention_q_peer1 = None
            attention_q_peer2 = None
            attention_q_peer3 = None
            attention_q_rank = -1
            attention_q_direct_route = False
            moe_partial_local = None
            moe_partial_peer0 = None
            moe_partial_peer1 = None
            moe_partial_peer2 = None
            moe_partial_peer3 = None
            moe_owner_flag0 = None
            moe_owner_flag1 = None
            moe_owner_flag2 = None
            moe_owner_flag3 = None
            attention_packed_send, attention_packed_recv = None, None
            attention_packed_recv_q = None
            attention_projected_local = None
            attention_projected_gather = None
            attention_wob_q = None
            attention_wob_s_storage = None
            attention_wob_q_chunks = None
            attention_wob_s_chunks = None
            attention_wob_partial_chunks = None
            attention_wob_output_handle = None
            attention_wob_output_peer0 = None
            attention_wob_output_peer1 = None
            attention_wob_output_peer2 = None
            attention_wob_output_peer3 = None
            attention_wob_ready_local = None
            attention_wob_ready_peer0 = None
            attention_wob_ready_peer1 = None
            attention_wob_ready_peer2 = None
            attention_wob_ready_peer3 = None
            attention_wob_ready_epoch_base = 0
            attention_wob_output_rank = -1
            attention_wob_direct_gather = False
            attention_symm_handle = None
            attention_symm_peer0 = None
            attention_symm_peer1 = None
            attention_symm_peer2 = None
            attention_symm_peer3 = None
            attention_symm_recv_q = None
            attention_symm_recv_scale = None
            attention_symm_rank = -1
            attention_symm_direct_push = False
        q_lora_bf16, q_lora_fp8, q_lora_scale = self._get_q_lora_workspace(
            num_tokens,
            positions.device,
        )
        (
            mhc_gemm_mul_storage,
            mhc_gemm_sq_storage,
            mhc_residual_mid,
            mhc_residual_out,
            mhc_post,
            mhc_comb,
            mhc_layer_input,
            mhc_output_fp8,
            mhc_output_scale_storage,
            mhc_output_scale,
            mhc_routed_output_fp8,
            mhc_routed_output_scale,
        ) = self._get_mhc_pre_workspace(num_tokens, positions.device)
        shared_down_fp8, shared_down_scale = self._get_shared_down_workspace(
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
            attention_q_local=attention_q_local,
            attention_q_recv=attention_q_recv,
            attention_q_handle=attention_q_handle,
            attention_q_peer0=attention_q_peer0,
            attention_q_peer1=attention_q_peer1,
            attention_q_peer2=attention_q_peer2,
            attention_q_peer3=attention_q_peer3,
            attention_q_rank=attention_q_rank,
            attention_q_direct_route=attention_q_direct_route,
            moe_partial_local=moe_partial_local,
            moe_partial_peer0=moe_partial_peer0,
            moe_partial_peer1=moe_partial_peer1,
            moe_partial_peer2=moe_partial_peer2,
            moe_partial_peer3=moe_partial_peer3,
            moe_owner_flag0=moe_owner_flag0,
            moe_owner_flag1=moe_owner_flag1,
            moe_owner_flag2=moe_owner_flag2,
            moe_owner_flag3=moe_owner_flag3,
            attention_packed_send=attention_packed_send,
            attention_packed_recv=attention_packed_recv,
            attention_packed_recv_q=attention_packed_recv_q,
            attention_projected_local=attention_projected_local,
            attention_projected_gather=attention_projected_gather,
            attention_wob_q=attention_wob_q,
            attention_wob_s_storage=attention_wob_s_storage,
            attention_wob_q_chunks=attention_wob_q_chunks,
            attention_wob_s_chunks=attention_wob_s_chunks,
            attention_wob_partial_chunks=attention_wob_partial_chunks,
            attention_wob_output_handle=attention_wob_output_handle,
            attention_wob_output_peer0=attention_wob_output_peer0,
            attention_wob_output_peer1=attention_wob_output_peer1,
            attention_wob_output_peer2=attention_wob_output_peer2,
            attention_wob_output_peer3=attention_wob_output_peer3,
            attention_wob_ready_local=attention_wob_ready_local,
            attention_wob_ready_peer0=attention_wob_ready_peer0,
            attention_wob_ready_peer1=attention_wob_ready_peer1,
            attention_wob_ready_peer2=attention_wob_ready_peer2,
            attention_wob_ready_peer3=attention_wob_ready_peer3,
            attention_wob_ready_epoch_base=attention_wob_ready_epoch_base,
            attention_wob_output_rank=attention_wob_output_rank,
            attention_wob_direct_gather=attention_wob_direct_gather,
            attention_symm_handle=attention_symm_handle,
            attention_symm_peer0=attention_symm_peer0,
            attention_symm_peer1=attention_symm_peer1,
            attention_symm_peer2=attention_symm_peer2,
            attention_symm_peer3=attention_symm_peer3,
            attention_symm_recv_q=attention_symm_recv_q,
            attention_symm_recv_scale=attention_symm_recv_scale,
            attention_symm_rank=attention_symm_rank,
            attention_symm_direct_push=attention_symm_direct_push,
            tp4_token_shard_attention=tp4_token_shard_attention,
            tp4_local_wob=(
                self._use_tp4_local_wob and tp4_token_shard_attention
            ),
            q_lora_bf16=q_lora_bf16,
            q_lora_fp8=q_lora_fp8,
            q_lora_scale=q_lora_scale,
            mhc_gemm_mul_storage=mhc_gemm_mul_storage,
            mhc_gemm_sq_storage=mhc_gemm_sq_storage,
            mhc_residual_mid=mhc_residual_mid,
            mhc_residual_out=mhc_residual_out,
            mhc_post=mhc_post,
            mhc_comb=mhc_comb,
            mhc_layer_input=mhc_layer_input,
            mhc_output_fp8=mhc_output_fp8,
            mhc_output_scale_storage=mhc_output_scale_storage,
            mhc_output_scale=mhc_output_scale,
            mhc_routed_output_fp8=mhc_routed_output_fp8,
            mhc_routed_output_scale=mhc_routed_output_scale,
            shared_down_fp8=shared_down_fp8,
            shared_down_scale=shared_down_scale,
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
                    output_s_storage[
                        : 2 * 8 * ((num_tokens + 3) // 4 * 4)
                    ].view(2, 8, (num_tokens + 3) // 4 * 4),
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
        # Keep this flat so each dynamic align(T, 4) gets the exact physical
        # [G, packed-K, aligned-M] stride required by DeepGEMM.
        output_s_storage = torch.empty(
            (2 * 8 * _MAX_FORWARD_TOKENS,),
            dtype=torch.int32,
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
            output_s_storage[
                : 2 * 8 * ((num_tokens + 3) // 4 * 4)
            ].view(2, 8, (num_tokens + 3) // 4 * 4),
            wo_a_gemm_output[:num_tokens],
        )

    def _get_tp4_attention_workspace(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
        Any,
        Any,
    ]:
        workspace = self._tp4_attention_workspace
        if workspace is None or workspace[0] != device:
            # Keep token-sharded C4 traffic disjoint from attention_q_padded.
            # C0/C128's output-only FlashMLA still observes padded heads in its
            # warp-wide online-softmax control; writing NCCL receives into that
            # storage changes their BF16 rounding even after valid heads are
            # overwritten.  These fixed-capacity local16 buffers avoid that
            # cross-layer state without any hot-path allocation.
            q_local = torch.empty(
                (_MAX_FORWARD_TOKENS, 16, 512),
                dtype=torch.bfloat16,
                device=device,
            )
            symmetric_q = self._tp4_symmetric_q_workspace
            if symmetric_q is None or symmetric_q[0] != device:
                raise RuntimeError(
                    "TP4 symmetric Q workspace was not bound on this device"
                )
            q_recv = symmetric_q[2]
            # Peer-Q FlashMLA reads q_recv remotely.  Keep that symmetric
            # producer buffer immutable until the layer's TP4 output-gather
            # barrier orders the next layer, and put all post-attention aliases
            # in a separate local allocation. This avoids a second TP barrier
            # between attention and WO_A/WO_B while preventing a fast rank from
            # clobbering Q that another rank is still consuming.
            q_output_scratch = torch.empty_like(q_recv)
            # After FlashMLA, q_local and q_output_scratch are available for
            # post-WO_A FP8+scale send/receive rows. Each row carries one
            # rank's 2048 WO_B inputs and sixteen scales.
            packed_row_bytes = 2064
            packed_send = q_local.view(torch.uint8).reshape(-1)[
                : _MAX_FORWARD_TOKENS * packed_row_bytes
            ].view(_MAX_FORWARD_TOKENS, packed_row_bytes)
            packed_recv = q_output_scratch.view(torch.uint8).reshape(-1)[
                : _MAX_FORWARD_TOKENS * packed_row_bytes
            ].view(_MAX_FORWARD_TOKENS, packed_row_bytes)
            packed_recv_q = packed_recv[:, :2048].view(
                torch.float8_e4m3fn
            ).view(_MAX_FORWARD_TOKENS, 2048)
            # Retain the alias field for descriptor ABI stability across the
            # profiled v10/v12 experiments; v12 does not launch an all-gather.
            projected_gather = q_local.view(-1)[
                : _MAX_FORWARD_TOKENS * 4096
            ].view(_MAX_FORWARD_TOKENS, 4096)
            projected_local = q_output_scratch.view(-1)[
                : (_MAX_FORWARD_TOKENS // 4) * 4096
            ].view(_MAX_FORWARD_TOKENS // 4, 4096)
            self._tp4_attention_workspace = (
                device,
                q_local,
                q_recv,
                q_output_scratch,
                packed_send,
                packed_recv,
                packed_recv_q,
                projected_local,
                projected_gather,
            )
        else:
            (
                _,
                q_local,
                q_recv,
                q_output_scratch,
                packed_send,
                packed_recv,
                packed_recv_q,
                projected_local,
                projected_gather,
            ) = workspace
        owner_tokens = num_tokens // 4
        q_recv_bytes = q_output_scratch.view(torch.uint8).reshape(-1)
        projected_local_capacity_bytes = (
            (_MAX_FORWARD_TOKENS // 4) * 4096 * 2
        )
        wob_q_capacity_bytes = _MAX_FORWARD_TOKENS * 2048
        wob_q = q_recv_bytes[
            projected_local_capacity_bytes :
            projected_local_capacity_bytes + 4 * owner_tokens * 2048
        ].view(torch.float8_e4m3fn).view(4, owner_tokens, 2048)
        wob_s_offset = projected_local_capacity_bytes + wob_q_capacity_bytes
        wob_s_storage = q_recv_bytes[
            wob_s_offset : wob_s_offset + 4 * 4 * owner_tokens * 4
        ].view(torch.int32).view(4, 4, owner_tokens)
        wob_partial_offset = wob_s_offset + (
            4 * 4 * (_MAX_FORWARD_TOKENS // 4) * 4
        )
        wob_partials = q_recv_bytes[
            wob_partial_offset :
            wob_partial_offset + 4 * owner_tokens * 4096 * 2
        ].view(torch.bfloat16).view(4, owner_tokens, 4096)
        wob_q_chunks = tuple(wob_q[index] for index in range(4))
        wob_s_logical = wob_s_storage.transpose(1, 2)
        wob_s_chunks = tuple(wob_s_logical[index] for index in range(4))
        wob_partial_chunks = tuple(wob_partials[index] for index in range(4))
        return (
            q_local[:num_tokens],
            q_recv[:num_tokens],
            packed_send[:num_tokens],
            packed_recv[:num_tokens],
            packed_recv_q[:num_tokens],
            projected_local[: num_tokens // 4],
            projected_gather[:num_tokens],
            wob_q,
            wob_s_storage,
            wob_q_chunks,
            wob_s_chunks,
            wob_partial_chunks,
        )

    def _get_q_lora_workspace(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        workspace = self._q_lora_workspace
        if workspace is None or workspace[0] != device:
            # q_lora_rank=1024 and block-FP8 group=128 are strict DSV4-Flash
            # invariants validated during runtime construction. Decoder layers
            # execute serially, so one set of addresses serves all 43 layers.
            q_lora_bf16 = torch.empty(
                (_MAX_FORWARD_TOKENS, 1024),
                dtype=torch.bfloat16,
                device=device,
            )
            q_lora_fp8 = torch.empty(
                (_MAX_FORWARD_TOKENS, 1024),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            # DeepGEMM packs four UE8M0 group scales per int32. q_lora has
            # 1024/128=8 groups, hence two packed columns. The one-dimensional
            # backing store yields an exact ceil(M/4)*4 TMA stride per batch
            # without allocating in the per-layer hot path.
            q_lora_scale_storage = torch.empty(
                (_MAX_FORWARD_TOKENS * 2,),
                dtype=torch.int32,
                device=device,
            )
            workspace = (
                device,
                q_lora_bf16,
                q_lora_fp8,
                q_lora_scale_storage,
            )
            self._q_lora_workspace = workspace
        _, q_lora_bf16, q_lora_fp8, q_lora_scale_storage = workspace
        aligned_m = (num_tokens + 3) // 4 * 4
        q_lora_scale = (
            q_lora_scale_storage[: 2 * aligned_m]
            .view(2, aligned_m)
            .transpose(0, 1)[:num_tokens]
        )
        return (
            q_lora_bf16[:num_tokens],
            q_lora_fp8[:num_tokens],
            q_lora_scale,
        )

    def _get_mhc_pre_workspace(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return exact-M views over one model-scoped mHC producer workspace."""

        workspace = self._mhc_pre_workspace
        if workspace is None or workspace[0] != device:
            # The split heuristic is capped at 64 for hc*hidden=16384. Keep the
            # backing arrays flat so every (splits,M,24) view has an exact M
            # stride rather than inheriting MAX_FORWARD_TOKENS as a pitch.
            gemm_mul_storage = torch.empty(
                (_MAX_MHC_SPLITS * _MAX_FORWARD_TOKENS * 24,),
                dtype=torch.float32,
                device=device,
            )
            gemm_sq_storage = torch.empty(
                (_MAX_MHC_SPLITS * _MAX_FORWARD_TOKENS,),
                dtype=torch.float32,
                device=device,
            )
            # Every layer uses the same two fixed addresses: attention post
            # writes ``mid`` and the final post writes ``out``.  The next
            # layer reads ``out`` before overwriting ``mid``, so this is a
            # graph-stable ping-pong without allocator traffic in the hot path.
            residual_mid = torch.empty(
                (_MAX_FORWARD_TOKENS, 4, 4096),
                dtype=torch.bfloat16,
                device=device,
            )
            residual_out = torch.empty_like(residual_mid)
            post = torch.empty(
                (_MAX_FORWARD_TOKENS, 4), dtype=torch.float32, device=device
            )
            comb = torch.empty(
                (_MAX_FORWARD_TOKENS, 4, 4), dtype=torch.float32, device=device
            )
            # This buffer becomes the input/output of the in-place MoE path,
            # so preserve the symmetric allocation property of native mhc_pre.
            from sglang.srt.distributed.device_communicators.pynccl_allocator import (
                use_symmetric_memory,
            )
            from sglang.srt.distributed.parallel_state import get_tp_group
            from sglang.srt.layers.dp_attention import is_allocation_symmetric

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                layer_input = torch.empty(
                    (_MAX_FORWARD_TOKENS, 4096),
                    dtype=torch.bfloat16,
                    device=device,
                )
            output_fp8 = torch.empty(
                (_MAX_FORWARD_TOKENS, 4096),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            output_scale_storage = torch.empty(
                (8 * _MAX_FORWARD_TOKENS,), dtype=torch.int32, device=device
            )
            routed_output_fp8 = torch.empty(
                (_MAX_FORWARD_TOKENS, 4096),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            routed_output_scale = torch.empty(
                (_MAX_FORWARD_TOKENS, 128),
                dtype=torch.uint8,
                device=device,
            )
            workspace = (
                device,
                gemm_mul_storage,
                gemm_sq_storage,
                residual_mid,
                residual_out,
                post,
                comb,
                layer_input,
                output_fp8,
                output_scale_storage,
                routed_output_fp8,
                routed_output_scale,
            )
            self._mhc_pre_workspace = workspace

        (
            _,
            gemm_mul_storage,
            gemm_sq_storage,
            residual_mid,
            residual_out,
            post,
            comb,
            layer_input,
            output_fp8,
            output_scale_storage,
            routed_output_fp8,
            routed_output_scale,
        ) = workspace
        aligned_m = (num_tokens + 3) // 4 * 4
        physical_scale = output_scale_storage[: 8 * aligned_m].view(8, aligned_m)
        logical_scale = physical_scale.transpose(0, 1)[:num_tokens]
        return (
            gemm_mul_storage,
            gemm_sq_storage,
            residual_mid[:num_tokens],
            residual_out[:num_tokens],
            post[:num_tokens],
            comb[:num_tokens],
            layer_input[:num_tokens],
            output_fp8[:num_tokens],
            physical_scale,
            logical_scale,
            routed_output_fp8[:num_tokens],
            routed_output_scale[:num_tokens],
        )

    def _get_shared_down_workspace(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace = self._shared_down_workspace
        if workspace is None or workspace[0] != device:
            # DSV4-Flash has one 512-wide shared expert.  Its group-128 scale
            # row contains four UE8M0 bytes, hence exactly one packed int32.
            shared_down_fp8 = torch.empty(
                (_MAX_FORWARD_TOKENS, 512),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            shared_down_scale_storage = torch.empty(
                (_MAX_FORWARD_TOKENS,),
                dtype=torch.int32,
                device=device,
            )
            workspace = (
                device,
                shared_down_fp8,
                shared_down_scale_storage,
            )
            self._shared_down_workspace = workspace
        _, shared_down_fp8, shared_down_scale_storage = workspace
        aligned_m = (num_tokens + 3) // 4 * 4
        shared_down_scale = (
            shared_down_scale_storage[:aligned_m]
            .view(1, aligned_m)
            .transpose(0, 1)[:num_tokens]
        )
        return shared_down_fp8[:num_tokens], shared_down_scale

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
        if view.max_prefill_tokens not in (4096, 65536, 131072):
            raise RuntimeError(
                "DSV4 huge runtime requires --max-prefill-tokens to select "
                "4096, 65536, or 131072, got "
                f"{view.max_prefill_tokens!r}"
            )
        if view.chunked_prefill_size != view.max_prefill_tokens:
            raise RuntimeError(
                "DSV4 huge runtime requires aggregate chunked-prefill-size "
                "to equal max-prefill-tokens; the scheduler separately caps "
                "each request at 4096, got "
                f"{view.chunked_prefill_size!r} vs {view.max_prefill_tokens!r}"
            )
        prefill_backend, decode_backend = attention_backends_of(view)
        if (prefill_backend, decode_backend) != ("dsv4", "dsv4"):
            raise RuntimeError(
                "DSV4 huge runtime requires dsv4 for both attention phases, "
                f"got prefill/decode={prefill_backend!r}/{decode_backend!r}"
            )
        prefill_graph = view.cuda_graph_config.prefill
        if prefill_graph.backend not in (Backend.DISABLED, Backend.BREAKABLE):
            raise RuntimeError(
                "DSV4 huge runtime requires prefill CUDA graph disabled or "
                f"breakable, got {prefill_graph.backend!r}"
            )
        if (
            view.max_prefill_tokens == 131072
            and prefill_graph.backend != Backend.DISABLED
        ):
            raise RuntimeError(
                "DSV4 huge req32/M131072 specialization is Eager-only; "
                "prefill CUDA graph must be disabled"
            )
        if prefill_graph.backend == Backend.BREAKABLE:
            expected_buckets = (
                (4096,)
                if view.max_prefill_tokens == 4096
                else (4096, 65536)
            )
            if tuple(prefill_graph.bs or ()) != expected_buckets:
                raise RuntimeError(
                    "DSV4 huge breakable prefill CUDA graph requires exact "
                    f"buckets {list(expected_buckets)}, got "
                    f"{prefill_graph.bs!r}"
                )
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
        if attn.q_norm.weight.dtype != torch.bfloat16:
            raise RuntimeError(
                f"layer {layer_id}: Huge q_lora RMSNorm+block-FP8 fusion requires "
                f"BF16 q_norm weights, got {attn.q_norm.weight.dtype}"
            )
        q_b_quant = getattr(attn.wq_b, "quant_method", None)
        q_b_block_size = getattr(q_b_quant, "weight_block_size", None)
        if not getattr(q_b_quant, "block_quant", False) or list(
            q_b_block_size or ()
        ) != [128, 128]:
            raise RuntimeError(
                f"layer {layer_id}: Huge fused q_lora producer requires wq_b "
                f"block-FP8 [128, 128], got {q_b_quant!r} / {q_b_block_size!r}"
            )
        if ratio == 4:
            indexer_q_b_quant = getattr(attn.indexer.wq_b, "quant_method", None)
            indexer_q_b_block_size = getattr(
                indexer_q_b_quant, "weight_block_size", None
            )
            if not getattr(indexer_q_b_quant, "block_quant", False) or list(
                indexer_q_b_block_size or ()
            ) != [128, 128]:
                raise RuntimeError(
                    f"layer {layer_id}: Huge q_lora quant workspace reuse "
                    "requires indexer.wq_b block-FP8 [128, 128], got "
                    f"{indexer_q_b_quant!r} / {indexer_q_b_block_size!r}"
                )
        if layer._post_attention_layernorm_weight_bf16 is None:
            raise RuntimeError(
                f"layer {layer_id}: huge mHC fusion requires the cached BF16 "
                "post-attention RMSNorm weight from post_load_weights"
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
        huge_output_buffers=(
            descriptor.mhc_post,
            descriptor.mhc_comb.view(descriptor.num_tokens, 16),
            descriptor.mhc_layer_input,
        ),
        huge_gemm_workspace=(
            descriptor.mhc_gemm_mul_storage,
            descriptor.mhc_gemm_sq_storage,
        ),
        huge_quant_output=(
            descriptor.mhc_output_fp8,
            descriptor.mhc_output_scale_storage,
            descriptor.mhc_routed_output_fp8,
            descriptor.mhc_routed_output_scale,
            False,
        ),
    )
    if not norm_fused:
        hidden_states = layer.input_layernorm(hidden_states)

    # MQALayer owns the explicit Q/KV/indexer/compressor/attention/WO_A/WO_B
    # primitives.  The handle authorizes only the strict fused output boundary.
    hidden_states = layer.self_attn(
        x=hidden_states,
        positions=descriptor.positions,
        forward_batch=descriptor.forward_batch,
        x_quant=(descriptor.mhc_output_fp8, descriptor.mhc_output_scale),
        e2e_handle=handle,
        e2e_descriptor=descriptor,
    )

    # Consume attention post state, then let the second fused mHC producer
    # directly generate the shared-expert gate input in FP8/UE8M0 form.
    residual, post, comb, hidden_states, shared_x_quant, routed_x_quant = (
        _separate_mhc_post_ffn_pre(
            layer=layer,
            descriptor=descriptor,
            hidden_states=hidden_states,
            residual=residual,
            post=post,
            comb=comb,
        )
    )

    # Prefix construction arrives as legal 1..4096-token EXTEND chunks.  The
    # owner protocol is specialized for the two measured aggregate buckets;
    # keep smaller chunks on the existing Huge CUDA/NCCL path so cache build
    # semantics remain unchanged.  This is not a native-layer fallback: both
    # branches stay inside the Huge whole-layer executor and unsupported
    # aggregate shapes still fail below.
    num_tokens = hidden_states.shape[0]
    if num_tokens not in (65536, 131072):
        if num_tokens > 4096:
            raise RuntimeError(
                "DSV4 Huge owner MoE/mHC supports aggregate M=65536/131072; "
                f"prefix construction is limited to M<=4096, got M={num_tokens}"
            )
        hidden_states = layer._run_moe_ffn_dp_sync(
            hidden_states,
            descriptor.forward_batch,
            input_ids=descriptor.input_ids,
            input_ids_global=descriptor.input_ids_global,
            shared_x_quant=shared_x_quant,
            routed_x_quant=routed_x_quant,
        )
        hidden_states = _huge_mhc_post(
            hidden_states=hidden_states,
            residual=residual,
            post=post,
            comb=comb,
            output=descriptor.mhc_residual_out,
        )
        return hidden_states, None, None, None

    moe_partial_local = descriptor.moe_partial_local
    moe_partials = (
        descriptor.moe_partial_peer0,
        descriptor.moe_partial_peer1,
        descriptor.moe_partial_peer2,
        descriptor.moe_partial_peer3,
    )
    moe_owner_flags = (
        descriptor.moe_owner_flag0,
        descriptor.moe_owner_flag1,
        descriptor.moe_owner_flag2,
        descriptor.moe_owner_flag3,
    )
    if (
        moe_partial_local is None
        or any(partial is None for partial in moe_partials)
        or any(flag is None for flag in moe_owner_flags)
        or descriptor.attention_q_handle is None
    ):
        raise RuntimeError(
            "DSV4 fused MoE/mHC post requires the TP4 symmetric workspace"
        )
    from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx

    with moe_output_buffer_ctx(
        moe_partial_local, external_symmetric=True
    ):
        moe_result = layer._run_moe_ffn_dp_sync(
            hidden_states,
            descriptor.forward_batch,
            input_ids=descriptor.input_ids,
            input_ids_global=descriptor.input_ids_global,
            shared_x_quant=shared_x_quant,
            routed_x_quant=routed_x_quant,
            huge_fused_moe_post=True,
        )
    if moe_result.data_ptr() != moe_partial_local.data_ptr():
        raise RuntimeError(
            "FlashInfer MoE did not honor the Huge symmetric output buffer: "
            f"result_shape={tuple(moe_result.shape)}, "
            f"provided_shape={tuple(moe_partial_local.shape)}, "
            f"result_dtype={moe_result.dtype}, provided_dtype={moe_partial_local.dtype}, "
            f"result_ptr=0x{moe_result.data_ptr():x}, "
            f"provided_ptr=0x{moe_partial_local.data_ptr():x}"
        )
    # The first GPU barrier publishes all four FlashInfer MoE partials.  The
    # second prevents the next layer's Q producer from reusing this symmetric
    # storage while another rank is still reading it.
    barrier_channel = handle.layer_id & 1
    descriptor.attention_q_handle.barrier(channel=barrier_channel)
    hidden_states = _huge_tp4_moe_mhc_post(
        partials=moe_partials,
        flags=moe_owner_flags,
        owner_rank=descriptor.attention_q_rank,
        residual=residual,
        post=post,
        comb=comb,
        output=descriptor.mhc_residual_out,
    )
    descriptor.attention_q_handle.barrier(channel=barrier_channel)
    return hidden_states, None, None, None


def _fused_mhc_post_ffn_pre(
    *,
    layer: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse attention mHC-post with FFN mHC-pre and its RMSNorm.

    The BF16 norm weight is materialized once by ``post_load_weights``.  A
    missing cache is a binding bug, not permission to cast/allocate in the hot
    path or to fall back to the native decoder implementation.
    """

    norm_weight = layer._post_attention_layernorm_weight_bf16
    if norm_weight is None:
        raise RuntimeError(
            f"layer {layer.layer_id}: Huge mHC fusion requires the cached "
            "BF16 post-attention RMSNorm weight"
        )

    from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre

    return mhc_fused_post_pre(
        hidden_states,
        residual,
        post.unsqueeze(-1) if post.ndim == 2 else post,
        comb,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        layer.rms_norm_eps,
        layer.hc_eps,
        layer.hc_eps,
        2.0,
        layer.hc_sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=layer.post_attention_layernorm.variance_epsilon,
    )


def _separate_mhc_post_ffn_pre(
    *,
    layer: Any,
    descriptor: DSV4ForwardDescriptor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Numerically stable Huge CUDA path with graph-stable post storage."""

    residual = _huge_mhc_post(
        hidden_states=hidden_states,
        residual=residual,
        post=post,
        comb=comb,
        output=descriptor.mhc_residual_mid,
        ready_flags=(
            (
                descriptor.attention_wob_ready_peer0,
                descriptor.attention_wob_ready_peer1,
                descriptor.attention_wob_ready_peer2,
                descriptor.attention_wob_ready_peer3,
            )
            if descriptor.num_tokens in (65536, 131072)
            else None
        ),
        ready_epoch=(
            descriptor.attention_wob_ready_epoch_base + layer.layer_id + 1
            if descriptor.num_tokens in (65536, 131072)
            and descriptor.attention_wob_ready_local is not None
            else 0
        ),
    )
    hidden_states, post, comb, norm_fused = layer.hc_pre(
        residual,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        norm=layer.post_attention_layernorm,
        forward_batch=descriptor.forward_batch,
        huge_output_buffers=(
            descriptor.mhc_post,
            descriptor.mhc_comb.view(descriptor.num_tokens, 16),
            descriptor.mhc_layer_input,
        ),
        huge_gemm_workspace=(
            descriptor.mhc_gemm_mul_storage,
            descriptor.mhc_gemm_sq_storage,
        ),
        huge_quant_output=(
            descriptor.mhc_output_fp8,
            descriptor.mhc_output_scale_storage,
            descriptor.mhc_routed_output_fp8,
            descriptor.mhc_routed_output_scale,
            True,
        ),
    )
    if not norm_fused:
        hidden_states = layer.post_attention_layernorm(hidden_states)
    return (
        residual,
        post,
        comb,
        hidden_states,
        (descriptor.mhc_output_fp8, descriptor.mhc_output_scale),
        (
            descriptor.mhc_routed_output_fp8,
            descriptor.mhc_routed_output_scale,
        ),
    )


def _huge_mhc_post(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    output: torch.Tensor,
    ready_flags: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ready_epoch: int = 0,
) -> torch.Tensor:
    """Run the strict CUDA post primitive into graph-stable GPU storage."""

    from sglang.jit_kernel.dsv4.e2e import mhc_post_vec8, mhc_post_vec8_ready

    if ready_flags is not None:
        return mhc_post_vec8_ready(
            hidden_states,
            residual,
            post,
            comb,
            output,
            ready_flags,
            ready_epoch,
        )

    return mhc_post_vec8(hidden_states, residual, post, comb, output)


def _huge_tp4_moe_mhc_post(
    *,
    partials: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    flags: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    owner_rank: int,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run the GPU-synchronized TP4 owner reduction and mHC post."""

    from sglang.jit_kernel.dsv4.e2e import tp4_moe_owner_persistent

    return tp4_moe_owner_persistent(
        partials,
        flags,
        owner_rank,
        residual,
        post,
        comb,
        output,
    )


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
