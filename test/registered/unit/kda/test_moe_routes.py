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
MOE_RUNNER_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "moe"
    / "moe_runner"
    / "deep_gemm.py"
)
GLOBAL_WRAPPER_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "deep_gemm_wrapper"
    / "entrypoint.py"
)

MOE_SLOT_BY_ARCHITECTURE = {
    "DeepseekV4ForCausalLM": "deepseek_v4.moe",
    "GlmMoeDsaForCausalLM": "glm52.moe_masked_grouped_gemm",
}
EXPECTED_ADAPTER_KEYWORDS = {
    "stage",
    "lhs",
    "rhs",
    "out",
    "routing",
    "expected_m",
}


@pytest.fixture
def router():
    module_spec = importlib.util.spec_from_file_location(
        "_test_sglang_kda_moe_router", ROUTER_PATH
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _server_args(*, config=None, profile="model_best"):
    return SimpleNamespace(
        kda_kernel_config=None if config is None else str(config),
        kda_kernel_profile=profile,
    )


def _model_config(architecture: str):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))


def _write_config(
    path: Path,
    operator_root: Path,
    *,
    architecture: str,
    slot: str,
):
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "profiles": {
                    "model_best": {
                        "architecture": architecture,
                        "operators": {
                            slot: {
                                "operator_id": f"{slot}.test",
                                "root": str(operator_root),
                                "entrypoint": "sglang_entry.py:run",
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("architecture", "slot"),
    MOE_SLOT_BY_ARCHITECTURE.items(),
)
def test_router_selects_moe_slot_once_from_exact_architecture(
    router, tmp_path, architecture, slot
):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    (operator_root / "sglang_entry.py").write_text(
        "def run(**kwargs):\n    return kwargs\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "routes.yaml"
    _write_config(
        config_path,
        operator_root,
        architecture=architecture,
        slot=slot,
    )

    router.initialize_kda_router(
        _server_args(config=config_path),
        _model_config(architecture),
    )

    assert router.get_kda_moe_operator() is router.get_kda_operator(slot)


def test_router_does_not_select_other_architecture_moe_slot(router, tmp_path):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    (operator_root / "sglang_entry.py").write_text(
        "def run(**kwargs):\n    return kwargs\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "routes.yaml"
    _write_config(
        config_path,
        operator_root,
        architecture="DeepseekV4ForCausalLM",
        slot="glm52.moe_masked_grouped_gemm",
    )

    router.initialize_kda_router(
        _server_args(config=config_path),
        _model_config("DeepseekV4ForCausalLM"),
    )

    assert router.get_kda_moe_operator() is None


def test_loaded_moe_adapter_exception_propagates(router, tmp_path):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    (operator_root / "sglang_entry.py").write_text(
        "def run(**kwargs):\n"
        "    raise RuntimeError('moe adapter failed')\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "routes.yaml"
    _write_config(
        config_path,
        operator_root,
        architecture="DeepseekV4ForCausalLM",
        slot="deepseek_v4.moe",
    )
    router.initialize_kda_router(
        _server_args(config=config_path),
        _model_config("DeepseekV4ForCausalLM"),
    )

    adapter = router.get_kda_moe_operator()
    assert adapter is not None
    with pytest.raises(RuntimeError, match="moe adapter failed"):
        adapter(stage="gate_up")


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_moe_presence_check(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and isinstance(node.left.value, ast.Name)
        and node.left.value.id == "self"
        and node.left.attr == "kda_moe_operator"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.IsNot)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def _calls(nodes: list[ast.stmt], name: str) -> list[ast.Call]:
    return [
        node
        for statement in nodes
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def test_masked_runner_uses_direct_adapter_or_native_branches():
    tree = ast.parse(MOE_RUNNER_PATH.read_text(encoding="utf-8"))
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepGemmRunnerCore"
    )
    method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_masked_gemm"
    )
    branches = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If) and _is_moe_presence_check(node.test)
    ]

    assert len(branches) == 2
    adapter_calls = [
        call
        for branch in branches
        for call in _calls(branch.body, "kda_moe_operator")
    ]
    native_calls = [
        call
        for branch in branches
        for call in _calls(
            branch.orelse,
            "grouped_gemm_nt_f8f8bf16_masked",
        )
    ]
    assert len(adapter_calls) == len(native_calls) == 2

    stages = {
        keyword.value.value
        for call in adapter_calls
        for keyword in call.keywords
        if keyword.arg == "stage"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    }
    assert stages == {"gate_up", "down"}
    assert all(not call.args for call in adapter_calls)
    assert all(
        EXPECTED_ADAPTER_KEYWORDS
        <= {keyword.arg for keyword in call.keywords}
        for call in adapter_calls
    )
    assert all(len(call.args) == 5 for call in native_calls)
    assert all(
        {"recipe_a", "recipe_b"}
        <= {keyword.arg for keyword in call.keywords}
        for call in native_calls
    )
    assert any(
        any(keyword.arg is None for keyword in call.keywords)
        for call in adapter_calls
    )
    assert any(
        any(keyword.arg is None for keyword in call.keywords)
        for call in native_calls
    )

    down_branch = next(
        branch
        for branch in branches
        if any(
            any(
                keyword.arg == "stage"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "down"
                for keyword in call.keywords
            )
            for call in _calls(branch.body, "kda_moe_operator")
        )
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "deep_gemm_return_value"
            for target in node.targets
        )
        for statement in down_branch.body
        for node in ast.walk(statement)
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "deep_gemm_return_value"
            for target in node.targets
        )
        for statement in down_branch.orelse
        for node in ast.walk(statement)
    )

    assert not any(isinstance(node, ast.Try) for node in ast.walk(method))
    assert not any(
        isinstance(node, ast.Call)
        and _call_name(node) == "_run_masked_gemm_stage"
        for node in ast.walk(method)
    )
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "down_output"
        for node in ast.walk(method)
    )


def test_runner_captures_fixed_moe_operator_during_construction():
    tree = ast.parse(MOE_RUNNER_PATH.read_text(encoding="utf-8"))
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepGemmRunnerCore"
    )
    init = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "kda_moe_operator"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert (
        isinstance(assignments[0].value, ast.Call)
        and _call_name(assignments[0].value) == "get_kda_moe_operator"
    )


def test_global_deep_gemm_wrapper_has_no_kda_routing():
    tree = ast.parse(GLOBAL_WRAPPER_PATH.read_text(encoding="utf-8"))
    assert not any(
        (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("sglang.srt.kda")
        )
        or (
            isinstance(node, ast.Name)
            and node.id.startswith("get_kda")
        )
        for node in ast.walk(tree)
    )
