from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[4]
ROUTER_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "kda" / "router.py"
HIP_BACKEND_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "deepseek_v4_backend_hip_radix.py"
)
DSA_BACKEND_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "dsa_backend.py"
)


@pytest.fixture
def router():
    module_spec = importlib.util.spec_from_file_location(
        "_test_sglang_kda_mi300x_router", ROUTER_PATH
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _server_args(config: Path):
    return SimpleNamespace(
        kda_kernel_config=str(config),
        kda_kernel_profile="mi300x",
    )


def _model_config(architecture: str):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))


def _write_config(
    path: Path,
    adapter_root: Path,
    *,
    architecture: str = "DeepseekV4ForCausalLM",
    platform: str = "rocm",
    device_arch: str = "gfx942",
    slot: str = "deepseek_v4.aiter_moe",
) -> None:
    adapter_root.mkdir()
    (adapter_root / "sglang_entry.py").write_text(
        "def run(**kwargs):\n    return kwargs\n",
        encoding="utf-8",
    )
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "profiles": {
                    "mi300x": {
                        "architecture": architecture,
                        "platform": platform,
                        "device_arch": device_arch,
                        "operators": {
                            slot: {
                                "operator_id": f"{slot}.mi300x",
                                "root": str(adapter_root),
                                "entrypoint": "sglang_entry.py:run",
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_mi300x_profile_requires_exact_runtime(router, tmp_path, monkeypatch):
    config = tmp_path / "routes.yaml"
    _write_config(config, tmp_path / "adapter")
    monkeypatch.setattr(
        router, "_runtime_platform_and_arch", lambda: ("rocm", "gfx942")
    )

    router.initialize_kda_router(
        _server_args(config),
        _model_config("DeepseekV4ForCausalLM"),
    )

    assert router.get_kda_aiter_moe_operator() is router.get_kda_operator(
        "deepseek_v4.aiter_moe"
    )


def test_glm52_mi300x_profile_selects_glm_aiter_moe(
    router, tmp_path, monkeypatch
):
    config = tmp_path / "routes.yaml"
    _write_config(
        config,
        tmp_path / "adapter",
        architecture="GlmMoeDsaForCausalLM",
        slot="glm52.aiter_moe",
    )
    monkeypatch.setattr(
        router, "_runtime_platform_and_arch", lambda: ("rocm", "gfx942")
    )

    router.initialize_kda_router(
        _server_args(config),
        _model_config("GlmMoeDsaForCausalLM"),
    )

    assert router.get_kda_aiter_moe_operator() is router.get_kda_operator(
        "glm52.aiter_moe"
    )


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        (("cuda", "sm100"), "platform mismatch"),
        (("rocm", "gfx950"), "device architecture mismatch"),
    ],
)
def test_mi300x_profile_rejects_runtime_mismatch(
    router, tmp_path, monkeypatch, runtime, message
):
    config = tmp_path / "routes.yaml"
    _write_config(config, tmp_path / "adapter")
    monkeypatch.setattr(router, "_runtime_platform_and_arch", lambda: runtime)

    with pytest.raises(RuntimeError, match=message):
        router.initialize_kda_router(
            _server_args(config),
            _model_config("DeepseekV4ForCausalLM"),
        )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _attribute_calls(method: ast.FunctionDef, attribute: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == attribute
    ]


def test_deepseek_v4_hip_backend_has_one_direct_paged_attention_route():
    init = _class_method(
        HIP_BACKEND_PATH, "DeepseekV4HipRadixBackend", "__init__"
    )
    forward = _class_method(
        HIP_BACKEND_PATH, "DeepseekV4HipRadixBackend", "forward"
    )
    route_getters = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and _call_name(node) == "get_kda_operator"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "deepseek_v4.hip_paged_attention"
    ]
    adapter_calls = _attribute_calls(forward, "kda_hip_paged_attention")

    assert len(route_getters) == len(adapter_calls) == 1
    assert not adapter_calls[0].args
    assert {
        "q",
        "k_cache",
        "head_dim_v",
        "softmax_scale",
        "indices",
        "topk_length",
        "attention_sink",
        "extra_k_cache",
        "extra_indices",
        "extra_topk_length",
        "scheduler",
        "compress_ratio",
        "forward_mode",
    } == {keyword.arg for keyword in adapter_calls[0].keywords}


@pytest.mark.parametrize("method_name", ["_forward_aiter", "_forward_aiter_extend"])
def test_glm52_aiter_attention_uses_direct_adapter_branch(method_name):
    method = _class_method(
        DSA_BACKEND_PATH, "DeepseekSparseAttnBackend", method_name
    )
    adapter_calls = _attribute_calls(
        method, "kda_aiter_sparse_attention_operator"
    )

    assert len(adapter_calls) == 1
    assert not adapter_calls[0].args
    assert {
        "query",
        "cache",
        "indices",
        "softmax_scale",
        "value_dim",
        "logit_cap",
    } == {keyword.arg for keyword in adapter_calls[0].keywords}
    assert not any(isinstance(node, ast.Try) for node in ast.walk(method))
