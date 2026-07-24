from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[4]
ROUTER_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "kda" / "router.py"
SERVER_ARGS_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "server_args.py"
ONE_BATCH_PATH = REPO_ROOT / "python" / "sglang" / "benchmark" / "one_batch.py"
OFFLINE_THROUGHPUT_PATH = (
    REPO_ROOT / "python" / "sglang" / "benchmark" / "offline_throughput.py"
)
LAUNCH_SERVER_PATH = REPO_ROOT / "python" / "sglang" / "launch_server.py"
BENCH_SERVING_PATH = REPO_ROOT / "python" / "sglang" / "benchmark" / "serving.py"

DEEPSEEK_ARCHITECTURE = "DeepseekV4ForCausalLM"
GLM52_ARCHITECTURE = "GlmMoeDsaForCausalLM"
DEEPSEEK_SLOTS = {
    "deepseek_v4.fp8_gemm_nt",
    "deepseek_v4.indexer_fp8_quant",
    "deepseek_v4.paged_mqa_logits",
    "deepseek_v4.topk_transform",
    "deepseek_v4.sparse_prefill_attention",
    "deepseek_v4.sparse_decode_attention",
    "deepseek_v4.dense_swa_attention",
    "deepseek_v4.moe",
}
GLM52_SLOTS = {
    "glm52.dsa_projection",
    "glm52.dsa_indexer",
    "glm52.dsa_index_score",
    "glm52.dsa_sparse_attention",
    "glm52.moe_masked_grouped_gemm",
}


@pytest.fixture
def router():
    module_spec = importlib.util.spec_from_file_location(
        "_test_sglang_kda_delivery_router", ROUTER_PATH
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


class _FakeModel:
    def __init__(self, *modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


def _server_args(config: Path | None = None, profile: str = "off"):
    return SimpleNamespace(
        kda_kernel_config=None if config is None else str(config),
        kda_kernel_profile=profile,
    )


def _model_config(architecture: str):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))


def _write_adapter(root: Path, body: str = "return kwargs") -> None:
    root.mkdir()
    (root / "sglang_entry.py").write_text(
        f"def run(**kwargs):\n    {body}\n",
        encoding="utf-8",
    )


def _write_complete_config(
    path: Path,
    *,
    profile: str,
    architecture: str,
    root: Path,
    slots: set[str],
) -> None:
    operators = {
        slot: {
            "operator_id": slot.replace(".", "_"),
            "root": str(root),
            "entrypoint": "sglang_entry.py:run",
        }
        for slot in slots
    }
    if architecture == DEEPSEEK_ARCHITECTURE:
        operators["deepseek_v4.fp8_gemm_nt"]["targets"] = [
            "model.layers.*.self_attn.wqkv_a"
        ]
    else:
        operators["glm52.dsa_projection"]["targets"] = [
            "model.layers.*.self_attn.q_proj"
        ]
        operators["glm52.dsa_indexer"]["targets"] = [
            "model.layers.*.self_attn.indexer.wq_b"
        ]

    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "profiles": {
                    profile: {
                        "architecture": architecture,
                        "operators": operators,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_deepseek_complete_profile_initializes_and_binds_linear(router, tmp_path):
    adapter_root = tmp_path / "adapter"
    _write_adapter(adapter_root)
    config_path = tmp_path / "routes.yaml"
    _write_complete_config(
        config_path,
        profile="deepseek_v4_best",
        architecture=DEEPSEEK_ARCHITECTURE,
        root=adapter_root,
        slots=DEEPSEEK_SLOTS,
    )

    router.initialize_kda_router(
        _server_args(config_path, "deepseek_v4_best"),
        _model_config(DEEPSEEK_ARCHITECTURE),
    )
    linear = SimpleNamespace(prefix="model.layers.3.self_attn.wqkv_a")
    router.bind_kda_linear_operators(_FakeModel(linear))

    assert router.kda_enabled()
    assert set(router._state.routes) == DEEPSEEK_SLOTS
    assert linear._kda_apply is router.get_kda_operator(
        "deepseek_v4.fp8_gemm_nt"
    )
    assert router.get_kda_moe_operator() is router.get_kda_operator(
        "deepseek_v4.moe"
    )


def test_glm52_complete_profile_initializes_exact_routes(router, tmp_path):
    adapter_root = tmp_path / "adapter"
    _write_adapter(adapter_root)
    config_path = tmp_path / "routes.yaml"
    _write_complete_config(
        config_path,
        profile="glm52_best",
        architecture=GLM52_ARCHITECTURE,
        root=adapter_root,
        slots=GLM52_SLOTS,
    )

    router.initialize_kda_router(
        _server_args(config_path, "glm52_best"),
        _model_config(GLM52_ARCHITECTURE),
    )
    projection = SimpleNamespace(prefix="model.layers.1.self_attn.q_proj")
    indexer = SimpleNamespace(prefix="model.layers.1.self_attn.indexer.wq_b")
    router.bind_kda_linear_operators(_FakeModel(projection, indexer))

    assert set(router._state.routes) == GLM52_SLOTS
    assert projection._kda_apply is router.get_kda_operator(
        "glm52.dsa_projection"
    )
    assert indexer._kda_apply is router.get_kda_operator("glm52.dsa_indexer")
    assert router.get_kda_operator_for_architecture(
        "glm52.dsa_index_score", GLM52_ARCHITECTURE
    ) is router.get_kda_operator("glm52.dsa_index_score")
    assert router.get_kda_operator_for_architecture(
        "glm52.dsa_sparse_attention", GLM52_ARCHITECTURE
    ) is router.get_kda_operator("glm52.dsa_sparse_attention")
    assert router.get_kda_moe_operator() is router.get_kda_operator(
        "glm52.moe_masked_grouped_gemm"
    )


def test_architecture_mismatch_fails_before_binding(router, tmp_path):
    adapter_root = tmp_path / "adapter"
    _write_adapter(adapter_root)
    config_path = tmp_path / "routes.yaml"
    _write_complete_config(
        config_path,
        profile="glm52_best",
        architecture=GLM52_ARCHITECTURE,
        root=adapter_root,
        slots=GLM52_SLOTS,
    )

    with pytest.raises(RuntimeError, match="architecture mismatch"):
        router.initialize_kda_router(
            _server_args(config_path, "glm52_best"),
            _model_config(DEEPSEEK_ARCHITECTURE),
        )


def test_adapter_import_and_execution_errors_are_not_hidden(router, tmp_path):
    broken_import_root = tmp_path / "broken_import"
    broken_import_root.mkdir()
    (broken_import_root / "sglang_entry.py").write_text(
        "raise RuntimeError('import exploded')\n",
        encoding="utf-8",
    )
    import_config = tmp_path / "import-routes.yaml"
    _write_complete_config(
        import_config,
        profile="deepseek_v4_best",
        architecture=DEEPSEEK_ARCHITECTURE,
        root=broken_import_root,
        slots=DEEPSEEK_SLOTS,
    )
    with pytest.raises(RuntimeError, match="import exploded"):
        router.initialize_kda_router(
            _server_args(import_config, "deepseek_v4_best"),
            _model_config(DEEPSEEK_ARCHITECTURE),
        )

    broken_call_root = tmp_path / "broken_call"
    _write_adapter(broken_call_root, "raise RuntimeError('execution exploded')")
    call_config = tmp_path / "call-routes.yaml"
    _write_complete_config(
        call_config,
        profile="deepseek_v4_best",
        architecture=DEEPSEEK_ARCHITECTURE,
        root=broken_call_root,
        slots=DEEPSEEK_SLOTS,
    )
    router.initialize_kda_router(
        _server_args(call_config, "deepseek_v4_best"),
        _model_config(DEEPSEEK_ARCHITECTURE),
    )
    with pytest.raises(RuntimeError, match="execution exploded"):
        router.get_kda_operator("deepseek_v4.paged_mqa_logits")(query=object())


def _call_names(function: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name):
                names.add(f"{value.id}.{node.func.attr}")
            else:
                names.add(node.func.attr)
    return names


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_server_args_declares_auto_generated_kda_cli_fields():
    tree = ast.parse(SERVER_ARGS_PATH.read_text(encoding="utf-8"))
    server_args = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in server_args.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"kda_kernel_config", "kda_kernel_profile"}
    }
    add_cli_args = next(
        node
        for node in server_args.body
        if isinstance(node, ast.FunctionDef) and node.name == "add_cli_args"
    )
    from_cli_args = next(
        node
        for node in server_args.body
        if isinstance(node, ast.FunctionDef) and node.name == "from_cli_args"
    )

    assert defaults == {
        "kda_kernel_config": None,
        "kda_kernel_profile": "off",
    }
    assert "add_cli_args_from_dataclass" in _call_names(add_cli_args)
    assert "dataclasses.fields" in _call_names(from_cli_args)


def test_real_server_args_parser_accepts_kda_flags_when_runtime_is_available():
    if importlib.util.find_spec("numpy") is None:
        pytest.skip("full SGLang runtime dependencies are not installed")

    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    native = parser.parse_args(["--model-path", "dummy"])
    routed = parser.parse_args(
        [
            "--model-path",
            "dummy",
            "--kda-kernel-config",
            "/tmp/kda-routes.yaml",
            "--kda-kernel-profile",
            "deepseek_v4_best",
        ]
    )

    assert native.kda_kernel_config is None
    assert native.kda_kernel_profile == "off"
    routed_server_args = ServerArgs.from_cli_args(routed)
    assert routed_server_args.kda_kernel_config == "/tmp/kda-routes.yaml"
    assert routed_server_args.kda_kernel_profile == "deepseek_v4_best"


@pytest.mark.parametrize("path", [ONE_BATCH_PATH, OFFLINE_THROUGHPUT_PATH])
def test_in_process_benchmarks_reuse_server_args_parser(path):
    calls = _call_names(_function(path, "cli_main"))

    assert "ServerArgs.add_cli_args" in calls
    assert "ServerArgs.from_cli_args" in calls


def test_launch_server_reuses_prepare_server_args():
    source = LAUNCH_SERVER_PATH.read_text(encoding="utf-8")
    prepare_calls = _call_names(_function(SERVER_ARGS_PATH, "prepare_server_args"))

    assert "prepare_server_args(sys.argv[1:])" in source
    assert "ServerArgs.add_cli_args" in prepare_calls
    assert "ServerArgs.from_cli_args" in prepare_calls


def test_serving_benchmark_remains_a_client_without_kda_cli():
    source = BENCH_SERVING_PATH.read_text(encoding="utf-8")

    assert "kda_kernel_config" not in source
    assert "kda_kernel_profile" not in source
