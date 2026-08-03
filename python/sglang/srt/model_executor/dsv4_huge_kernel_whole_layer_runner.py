"""Strict per-decoder-layer dispatch for the DSV4 huge runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.models.dsv4_whole_layer_runtime import (
        DSV4LayerHandle,
        DSV4WholeLayerRuntime,
    )


class Dsv4HugeKernelWholeLayerRunner:
    """Own a generation-tagged C0/C4/C128 handle, never a native callback."""

    def __init__(
        self,
        layer: DeepseekV4DecoderLayer,
        runtime: DSV4WholeLayerRuntime,
        handle: DSV4LayerHandle,
    ) -> None:
        self.layer_id = layer.layer_id
        self._runtime = runtime
        self._handle = handle

    def rebind(
        self,
        runtime: DSV4WholeLayerRuntime,
        handle: DSV4LayerHandle,
    ) -> None:
        if runtime is not self._runtime or handle.layer_id != self.layer_id:
            raise RuntimeError(
                f"invalid DSV4 huge handle rebind for layer {self.layer_id}"
            )
        self._handle = handle

    def __call__(self, **kwargs: Any):
        descriptor = self._runtime.active_descriptor
        if kwargs["positions"] is not descriptor.positions:
            raise RuntimeError(
                f"layer {self.layer_id}: positions do not belong to the active "
                "DSV4 forward descriptor"
            )
        if kwargs["forward_batch"] is not descriptor.forward_batch:
            raise RuntimeError(
                f"layer {self.layer_id}: ForwardBatch changed inside huge runtime"
            )
        if kwargs["input_ids"] is not descriptor.input_ids:
            raise RuntimeError(
                f"layer {self.layer_id}: input_ids changed inside huge runtime"
            )
        if kwargs["input_ids_global"] is not descriptor.input_ids_global:
            raise RuntimeError(
                f"layer {self.layer_id}: global input_ids changed inside huge runtime"
            )
        for name in ("prev_residual", "prev_post", "prev_comb"):
            if kwargs[name] is not None:
                raise RuntimeError(
                    "DSV4 huge runtime forbids deferred cross-layer mHC state; "
                    f"layer {self.layer_id} received {name}"
                )
        return self._runtime.execute_layer(
            self._handle,
            descriptor,
            kwargs["hidden_states"],
        )
