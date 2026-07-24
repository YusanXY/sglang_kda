"""Process-local, read-only routing for externally supplied KDA kernels."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_OFF_PROFILE = "off"
_CONFIG_VERSION = 1


@dataclass(frozen=True)
class KdaOperatorRoute:
    """One loaded operator from the selected KDA profile."""

    slot: str
    operator_id: str
    root: Path
    entrypoint: str
    callable: Callable[..., Any]
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class _KdaRouterState:
    profile: str
    architecture: str | None
    routes: Mapping[str, KdaOperatorRoute]


_DISABLED_STATE = _KdaRouterState(
    profile=_OFF_PROFILE,
    architecture=None,
    routes=MappingProxyType({}),
)
_state = _DISABLED_STATE


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _model_architecture(model_config: Any) -> str:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if not isinstance(architectures, (list, tuple)) or not architectures:
        raise ValueError("model_config.hf_config.architectures must not be empty")
    return _require_nonempty_string(
        architectures[0], "model_config.hf_config.architectures[0]"
    )


def _parse_targets(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return tuple(
        _require_nonempty_string(target, f"{name}[{index}]")
        for index, target in enumerate(value)
    )


def _load_entrypoint(
    *,
    slot: str,
    root_value: Any,
    entrypoint_value: Any,
) -> tuple[Path, str, Callable[..., Any]]:
    root_text = _require_nonempty_string(root_value, f"{slot}.root")
    root = Path(root_text)
    if not root.is_absolute():
        raise ValueError(f"{slot}.root must be an absolute path")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{slot}.root must be a directory: {root}")

    entrypoint = _require_nonempty_string(entrypoint_value, f"{slot}.entrypoint")
    relative_file_text, separator, callable_name = entrypoint.partition(":")
    if not separator or not relative_file_text or not callable_name:
        raise ValueError(f"{slot}.entrypoint must use '<relative_file>:<callable>'")
    relative_file = Path(relative_file_text)
    if relative_file.is_absolute():
        raise ValueError(f"{slot}.entrypoint file must be relative to root")

    entrypoint_file = (root / relative_file).resolve(strict=True)
    if not entrypoint_file.is_relative_to(root):
        raise ValueError(f"{slot}.entrypoint file must stay within root")
    if not entrypoint_file.is_file():
        raise ValueError(f"{slot}.entrypoint file must be a file: {entrypoint_file}")

    module_suffix = hashlib.sha256(f"{slot}:{entrypoint_file}".encode()).hexdigest()[
        :16
    ]
    module_name = f"_sglang_kda_{module_suffix}"
    module_spec = importlib.util.spec_from_file_location(module_name, entrypoint_file)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load KDA entrypoint module: {entrypoint_file}")

    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    loaded_callable = getattr(module, callable_name)
    if not callable(loaded_callable):
        raise TypeError(f"{slot}.entrypoint is not callable: {entrypoint}")
    return root, entrypoint, loaded_callable


def initialize_kda_router(server_args: Any, model_config: Any) -> None:
    """Load one immutable KDA profile for the current worker process."""

    global _state

    profile = getattr(server_args, "kda_kernel_profile", _OFF_PROFILE)
    if profile == _OFF_PROFILE:
        _state = _DISABLED_STATE
        return
    profile = _require_nonempty_string(profile, "kda_kernel_profile")

    config_value = getattr(server_args, "kda_kernel_config", None)
    if config_value is None:
        raise ValueError(
            "kda_kernel_config is required when kda_kernel_profile is not 'off'"
        )
    config_path = Path(_require_nonempty_string(config_value, "kda_kernel_config"))
    if not config_path.is_absolute():
        raise ValueError("kda_kernel_config must be an absolute path")

    with config_path.open(encoding="utf-8") as config_file:
        document = _require_mapping(yaml.safe_load(config_file), "KDA configuration")
    if document.get("version") != _CONFIG_VERSION:
        raise ValueError(f"KDA configuration version must be {_CONFIG_VERSION}")

    profiles = _require_mapping(document.get("profiles"), "profiles")
    if profile not in profiles:
        raise ValueError(f"KDA profile is not defined: {profile}")
    selected_profile = _require_mapping(profiles[profile], f"profiles.{profile}")

    expected_architecture = _require_nonempty_string(
        selected_profile.get("architecture"),
        f"profiles.{profile}.architecture",
    )
    actual_architecture = _model_architecture(model_config)
    if actual_architecture != expected_architecture:
        raise RuntimeError(
            "KDA profile architecture mismatch: "
            f"profile={expected_architecture}, model={actual_architecture}"
        )

    operator_configs = _require_mapping(
        selected_profile.get("operators"), f"profiles.{profile}.operators"
    )
    loaded_routes: dict[str, KdaOperatorRoute] = {}
    for slot_value, operator_value in operator_configs.items():
        slot = _require_nonempty_string(slot_value, "operator slot")
        operator = _require_mapping(
            operator_value, f"profiles.{profile}.operators.{slot}"
        )
        operator_id = _require_nonempty_string(
            operator.get("operator_id"), f"{slot}.operator_id"
        )
        root, entrypoint, loaded_callable = _load_entrypoint(
            slot=slot,
            root_value=operator.get("root"),
            entrypoint_value=operator.get("entrypoint"),
        )
        loaded_routes[slot] = KdaOperatorRoute(
            slot=slot,
            operator_id=operator_id,
            root=root,
            entrypoint=entrypoint,
            callable=loaded_callable,
            targets=_parse_targets(operator.get("targets"), f"{slot}.targets"),
        )

    routes = MappingProxyType(loaded_routes)
    _state = _KdaRouterState(
        profile=profile,
        architecture=actual_architecture,
        routes=routes,
    )
    logger.info(
        "KDA kernel routing enabled: profile=%s architecture=%s",
        profile,
        actual_architecture,
    )
    for route in routes.values():
        logger.info(
            "KDA route enabled: slot=%s operator_id=%s root=%s",
            route.slot,
            route.operator_id,
            route.root,
        )


def get_kda_operator(slot: str) -> Callable[..., Any] | None:
    """Return the configured callable for ``slot``, or ``None``."""

    route = _state.routes.get(slot)
    return None if route is None else route.callable


def bind_kda_linear_operators(model: Any) -> None:
    """Bind configured Linear routes to matching modules once after model load."""

    if not kda_enabled():
        return

    linear_routes = tuple(route for route in _state.routes.values() if route.targets)
    if not linear_routes:
        return

    bound_count = 0
    for module in model.modules():
        prefix = getattr(module, "prefix", None)
        if not isinstance(prefix, str):
            continue

        for route in linear_routes:
            if any(fnmatchcase(prefix, target) for target in route.targets):
                module._kda_apply = route.callable
                bound_count += 1
                break

    logger.info(
        "KDA Linear operators bound: profile=%s modules=%d",
        _state.profile,
        bound_count,
    )


def kda_enabled() -> bool:
    """Return whether the current worker has an active KDA profile."""

    return _state.profile != _OFF_PROFILE
