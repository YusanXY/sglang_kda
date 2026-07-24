from __future__ import annotations

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[4]
ROUTER_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "kda" / "router.py"
LINEAR_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "layers" / "linear.py"
FP8_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "quantization"
    / "fp8.py"
)
UNQUANT_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "quantization"
    / "unquant.py"
)
MODEL_RUNNER_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "model_executor"
    / "model_runner.py"
)


@pytest.fixture
def router():
    module_spec = importlib.util.spec_from_file_location(
        "_test_sglang_kda_linear_router", ROUTER_PATH
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


def _model_config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    )


def _write_config(path: Path, operator_root: Path, targets: list[str]):
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "profiles": {
                    "deepseek_v4_best": {
                        "architecture": "DeepseekV4ForCausalLM",
                        "operators": {
                            "deepseek_v4.fp8_gemm_nt": {
                                "operator_id": "deepseek_v4_fp8_gemm_nt",
                                "root": str(operator_root),
                                "entrypoint": "sglang_entry.py:run",
                                "targets": targets,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class _FakeModel:
    def __init__(self, *modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


def _enable_router(router, tmp_path, targets):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    (operator_root / "sglang_entry.py").write_text(
        "def run(*, layer, x, bias=None):\n"
        "    return {'layer': layer, 'x': x, 'bias': bias}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "routes.yaml"
    _write_config(config_path, operator_root, targets)
    router.initialize_kda_router(
        _server_args(config=config_path, profile="deepseek_v4_best"),
        _model_config(),
    )
    return router.get_kda_operator("deepseek_v4.fp8_gemm_nt")


def test_binder_matches_prefix_and_leaves_unmatched_module_unbound(router, tmp_path):
    loaded = _enable_router(
        router,
        tmp_path,
        ["model.layers.*.self_attn.q_proj"],
    )
    matched = SimpleNamespace(prefix="model.layers.3.self_attn.q_proj")
    unmatched = SimpleNamespace(prefix="model.layers.3.self_attn.k_proj")
    no_prefix = SimpleNamespace()

    router.bind_kda_linear_operators(_FakeModel(matched, unmatched, no_prefix))

    assert matched._kda_apply is loaded
    assert not hasattr(unmatched, "_kda_apply")
    assert not hasattr(no_prefix, "_kda_apply")


def test_binder_accepts_multiple_targets(router, tmp_path):
    loaded = _enable_router(
        router,
        tmp_path,
        [
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.o_proj",
        ],
    )
    q_proj = SimpleNamespace(prefix="model.layers.0.self_attn.q_proj")
    o_proj = SimpleNamespace(prefix="model.layers.0.self_attn.o_proj")

    router.bind_kda_linear_operators(_FakeModel(q_proj, o_proj))

    assert q_proj._kda_apply is loaded
    assert o_proj._kda_apply is loaded


def test_repeated_binding_reuses_callable_without_callback_chain(router, tmp_path):
    loaded = _enable_router(router, tmp_path, ["model.layers.*.mlp.down_proj"])
    layer = SimpleNamespace(prefix="model.layers.7.mlp.down_proj")
    model = _FakeModel(layer)

    router.bind_kda_linear_operators(model)
    first_binding = layer._kda_apply
    router.bind_kda_linear_operators(model)

    assert first_binding is loaded
    assert layer._kda_apply is first_binding


def test_off_profile_does_not_bind(router):
    layer = SimpleNamespace(prefix="model.layers.0.self_attn.q_proj")
    router.initialize_kda_router(_server_args(profile="off"), _model_config())

    router.bind_kda_linear_operators(_FakeModel(layer))

    assert not hasattr(layer, "_kda_apply")


def _load_apply_method(path: Path, class_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = copy.deepcopy(
        next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "apply"
        )
    )
    method.decorator_list = []
    method.returns = None
    for argument in (
        list(method.args.posonlyargs)
        + list(method.args.args)
        + list(method.args.kwonlyargs)
    ):
        argument.annotation = None
    if method.args.vararg is not None:
        method.args.vararg.annotation = None
    if method.args.kwarg is not None:
        method.args.kwarg.annotation = None

    namespace = {}
    extracted = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(path), "exec"), namespace)  # noqa: S102
    return namespace["apply"]


@pytest.mark.parametrize(
    ("path", "class_name"),
    [
        (FP8_PATH, "Fp8LinearMethod"),
        (UNQUANT_PATH, "UnquantizedLinearMethod"),
    ],
)
def test_quant_method_uses_bound_adapter(path, class_name):
    apply_method = _load_apply_method(path, class_name)
    layer = SimpleNamespace()
    x = object()
    bias = object()

    def adapter(**kwargs):
        return kwargs

    layer._kda_apply = adapter

    result = apply_method(SimpleNamespace(), layer, x, bias)

    assert result == {"layer": layer, "x": x, "bias": bias}


@pytest.mark.parametrize(
    ("path", "class_name"),
    [
        (FP8_PATH, "Fp8LinearMethod"),
        (UNQUANT_PATH, "UnquantizedLinearMethod"),
    ],
)
def test_adapter_exception_propagates(path, class_name):
    apply_method = _load_apply_method(path, class_name)
    layer = SimpleNamespace()

    def adapter(**kwargs):
        raise RuntimeError("adapter failed")

    layer._kda_apply = adapter

    with pytest.raises(RuntimeError, match="adapter failed"):
        apply_method(SimpleNamespace(), layer, object(), None)


def _call_positions(function: ast.FunctionDef, name: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


def test_model_runner_initializes_then_loads_then_binds():
    tree = ast.parse(MODEL_RUNNER_PATH.read_text(encoding="utf-8"))
    model_runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    initialize = next(
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize"
    )

    init_position = _call_positions(initialize, "initialize_kda_router")
    load_position = _call_positions(initialize, "load_model")
    bind_position = _call_positions(initialize, "bind_kda_linear_operators")

    assert len(init_position) == len(load_position) == len(bind_position) == 1
    assert init_position[0] < load_position[0] < bind_position[0]


def test_linear_base_keeps_full_prefix():
    tree = ast.parse(LINEAR_PATH.read_text(encoding="utf-8"))
    linear_base = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LinearBase"
    )
    init = next(
        node
        for node in linear_base.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "prefix"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "prefix"
        for node in ast.walk(init)
    )
