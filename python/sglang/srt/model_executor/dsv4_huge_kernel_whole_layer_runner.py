"""Per-decoder-layer execution boundary for DSV4 huge-kernel development."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer


class Dsv4HugeKernelWholeLayerRunner:
    """Owns the complete decoder-layer call as one replaceable operation.

    Phase 1 binds the existing numerically proven implementation as the
    mandatory huge-kernel reference body. There is no exception handling or
    runtime fallback. Fusion phases replace ``_impl`` with the CUDA executor.
    """

    def __init__(self, layer: DeepseekV4DecoderLayer):
        self.layer_id = layer.layer_id
        self._impl = layer._forward_native

    def __call__(self, **kwargs):
        return self._impl(**kwargs)
