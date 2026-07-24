from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROUTER_PATH = (
    Path(__file__).parents[4] / "python" / "sglang" / "srt" / "kda" / "router.py"
)


@pytest.fixture
def router():
    module_spec = importlib.util.spec_from_file_location(
        "_test_sglang_kda_router", ROUTER_PATH
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _server_args(*, config=None, profile="off"):
    return SimpleNamespace(
        kda_kernel_config=None if config is None else str(config),
        kda_kernel_profile=profile,
    )


def _model_config(architecture="DeepseekV4ForCausalLM"):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))


def _write_config(
    path: Path,
    *,
    profile="deepseek_v4_best",
    architecture="DeepseekV4ForCausalLM",
    operators=None,
):
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "profiles": {
                    profile: {
                        "architecture": architecture,
                        "operators": {} if operators is None else operators,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_off_does_not_read_config_or_import_adapter(router, tmp_path):
    missing_config = tmp_path / "does-not-exist.yaml"

    router.initialize_kda_router(
        _server_args(config=missing_config, profile="off"),
        _model_config(),
    )

    assert not router.kda_enabled()
    assert router.get_kda_operator("deepseek_v4.fp8_gemm_nt") is None


def test_active_profile_requires_config(router):
    with pytest.raises(
        ValueError,
        match="kda_kernel_config is required",
    ):
        router.initialize_kda_router(
            _server_args(profile="deepseek_v4_best"),
            _model_config(),
        )


def test_unknown_profile_is_rejected(router, tmp_path):
    config_path = tmp_path / "routes.yaml"
    _write_config(config_path)

    with pytest.raises(ValueError, match="profile is not defined"):
        router.initialize_kda_router(
            _server_args(config=config_path, profile="missing"),
            _model_config(),
        )


def test_architecture_mismatch_is_rejected(router, tmp_path):
    config_path = tmp_path / "routes.yaml"
    _write_config(config_path, architecture="GlmMoeDsaForCausalLM")

    with pytest.raises(RuntimeError, match="architecture mismatch"):
        router.initialize_kda_router(
            _server_args(config=config_path, profile="deepseek_v4_best"),
            _model_config(),
        )


def test_entrypoint_is_loaded_from_absolute_root(router, tmp_path):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    (operator_root / "sglang_entry.py").write_text(
        "def run(*, value):\n    return value + 1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "routes.yaml"
    _write_config(
        config_path,
        operators={
            "deepseek_v4.fp8_gemm_nt": {
                "operator_id": "deepseek_v4_fp8_gemm_nt",
                "root": str(operator_root),
                "entrypoint": "sglang_entry.py:run",
                "targets": ["model.layers.*.self_attn.wqkv_a"],
            }
        },
    )

    router.initialize_kda_router(
        _server_args(config=config_path, profile="deepseek_v4_best"),
        _model_config(),
    )

    loaded = router.get_kda_operator("deepseek_v4.fp8_gemm_nt")
    assert router.kda_enabled()
    assert loaded is not None
    assert loaded(value=5) == 6
    with pytest.raises(TypeError):
        router._state.routes["other"] = object()


def test_entrypoint_import_failure_is_not_hidden(router, tmp_path):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    (operator_root / "sglang_entry.py").write_text(
        "raise RuntimeError('adapter import failed')\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "routes.yaml"
    _write_config(
        config_path,
        operators={
            "deepseek_v4.fp8_gemm_nt": {
                "operator_id": "deepseek_v4_fp8_gemm_nt",
                "root": str(operator_root),
                "entrypoint": "sglang_entry.py:run",
            }
        },
    )

    with pytest.raises(RuntimeError, match="adapter import failed"):
        router.initialize_kda_router(
            _server_args(config=config_path, profile="deepseek_v4_best"),
            _model_config(),
        )
