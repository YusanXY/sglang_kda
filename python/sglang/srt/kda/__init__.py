"""Lightweight routing for externally supplied KDA kernels."""

from sglang.srt.kda.router import (
    GLM52_ARCHITECTURE,
    bind_kda_linear_operators,
    get_kda_aiter_moe_operator,
    get_kda_moe_operator,
    get_kda_operator,
    get_kda_operator_for_architecture,
    initialize_kda_router,
    kda_enabled,
)

__all__ = [
    "GLM52_ARCHITECTURE",
    "bind_kda_linear_operators",
    "get_kda_aiter_moe_operator",
    "get_kda_moe_operator",
    "get_kda_operator",
    "get_kda_operator_for_architecture",
    "initialize_kda_router",
    "kda_enabled",
]
