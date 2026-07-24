"""Lightweight routing for externally supplied KDA kernels."""

from sglang.srt.kda.router import (
    get_kda_operator,
    initialize_kda_router,
    kda_enabled,
)

__all__ = [
    "get_kda_operator",
    "initialize_kda_router",
    "kda_enabled",
]
