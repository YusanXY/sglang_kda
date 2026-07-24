"""Lightweight routing for externally supplied KDA kernels."""

from sglang.srt.kda.router import (
    bind_kda_linear_operators,
    get_kda_operator,
    initialize_kda_router,
    kda_enabled,
)

__all__ = [
    "bind_kda_linear_operators",
    "get_kda_operator",
    "initialize_kda_router",
    "kda_enabled",
]
