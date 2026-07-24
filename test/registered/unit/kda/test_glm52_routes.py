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
MODEL_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "models" / "deepseek_v2.py"
INDEXER_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "dsa"
    / "dsa_indexer.py"
)
BACKEND_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "dsa_backend.py"
)
EXAMPLE_PATH = REPO_ROOT / "examples" / "kda" / "kda-routes.yaml"

GLM52_ARCHITECTURE = "GlmMoeDsaForCausalLM"
GLM52_SLOTS = {
    "glm52.dsa_projection",
    "glm52.dsa_indexer",
    "glm52.dsa_index_score",
    "glm52.dsa_sparse_attention",
    "glm52.moe_masked_grouped_gemm",
}
PROJECTION_TARGETS = {
    "model.layers.*.self_attn.fused_qkv_a_proj_with_mqa",
    "model.layers.*.self_attn.q_b_proj",
    "model.layers.*.self_attn.q_proj",
    "model.layers.*.self_attn.kv_a_proj_with_mqa",
    "model.layers.*.self_attn.kv_b_proj",
    "model.layers.*.self_attn.o_proj",
}
INDEXER_TARGETS = {
    "model.layers.*.self_attn.indexer.wq_b",
    "model.layers.*.self_attn.indexer.wk_weights_proj",
    "model.layers.*.self_attn.indexer.wk",
    "model.layers.*.self_attn.indexer.weights_proj",
}
INDEX_SCORE_KEYWORDS = {
    "phase",
    "query",
    "cache",
    "cache_scale",
    "weights",
    "lengths_start",
    "lengths_end",
    "block_tables",
    "schedule",
    "max_context",
    "q_offset",
}
SPARSE_ATTENTION_KEYWORDS = {
    "query",
    "cache",
    "indices",
    "softmax_scale",
    "value_dim",
}


@pytest.fixture
def router():
    module_spec = importlib.util.spec_from_file_location(
        "_test_sglang_kda_glm52_router", ROUTER_PATH
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
        kda_kernel_profile="glm52_best",
    )


def _model_config(architecture: str):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))


def _write_glm52_config(
    path: Path, operator_root: Path, *, entrypoint_source: str
) -> None:
    (operator_root / "sglang_entry.py").write_text(
        entrypoint_source,
        encoding="utf-8",
    )
    operators = {
        slot: {
            "operator_id": f"{slot}.test",
            "root": str(operator_root),
            "entrypoint": "sglang_entry.py:run",
        }
        for slot in ("glm52.dsa_index_score", "glm52.dsa_sparse_attention")
    }
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "profiles": {
                    "glm52_best": {
                        "architecture": GLM52_ARCHITECTURE,
                        "operators": operators,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_linear_config(
    path: Path,
    operator_root: Path,
    *,
    architecture: str,
    slot: str,
) -> None:
    (operator_root / "sglang_entry.py").write_text(
        "def run(**kwargs):\n    return kwargs\n",
        encoding="utf-8",
    )
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "profiles": {
                    "glm52_best": {
                        "architecture": architecture,
                        "operators": {
                            slot: {
                                "operator_id": f"{slot}.test",
                                "root": str(operator_root),
                                "entrypoint": "sglang_entry.py:run",
                                "targets": [
                                    "model.layers.*.self_attn.q_proj"
                                ],
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


def test_exact_architecture_selects_glm52_operators(router, tmp_path):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    config_path = tmp_path / "routes.yaml"
    _write_glm52_config(
        config_path,
        operator_root,
        entrypoint_source="def run(**kwargs):\n    return kwargs\n",
    )
    router.initialize_kda_router(
        _server_args(config_path),
        _model_config(GLM52_ARCHITECTURE),
    )

    index_score = router.get_kda_operator_for_architecture(
        "glm52.dsa_index_score", GLM52_ARCHITECTURE
    )
    sparse_attention = router.get_kda_operator_for_architecture(
        "glm52.dsa_sparse_attention", GLM52_ARCHITECTURE
    )
    assert index_score is not None
    assert sparse_attention is not None
    assert index_score(phase="decode", query="q") == {
        "phase": "decode",
        "query": "q",
    }
    assert sparse_attention(query="q", cache="kv") == {
        "query": "q",
        "cache": "kv",
    }

    for architecture in (
        "GlmMoeDsaForCausalLMNextN",
        "Glm4MoeForCausalLM",
        "DeepseekV2ForCausalLM",
        None,
    ):
        assert (
            router.get_kda_operator_for_architecture(
                "glm52.dsa_index_score", architecture
            )
            is None
        )


def test_glm52_adapter_exception_is_not_hidden(router, tmp_path):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    config_path = tmp_path / "routes.yaml"
    _write_glm52_config(
        config_path,
        operator_root,
        entrypoint_source=(
            "def run(**kwargs):\n"
            "    raise RuntimeError('glm52 adapter failed')\n"
        ),
    )
    router.initialize_kda_router(
        _server_args(config_path),
        _model_config(GLM52_ARCHITECTURE),
    )
    adapter = router.get_kda_operator_for_architecture(
        "glm52.dsa_sparse_attention", GLM52_ARCHITECTURE
    )

    assert adapter is not None
    with pytest.raises(RuntimeError, match="glm52 adapter failed"):
        adapter(query=object())


@pytest.mark.parametrize(
    ("architecture", "should_bind"),
    [
        (GLM52_ARCHITECTURE, True),
        ("DeepseekV2ForCausalLM", False),
    ],
)
@pytest.mark.parametrize(
    "slot",
    ["glm52.dsa_projection", "glm52.dsa_indexer"],
)
def test_glm52_linear_slots_require_exact_architecture(
    router, tmp_path, architecture, should_bind, slot
):
    operator_root = tmp_path / "operator"
    operator_root.mkdir()
    config_path = tmp_path / "routes.yaml"
    _write_linear_config(
        config_path,
        operator_root,
        architecture=architecture,
        slot=slot,
    )
    router.initialize_kda_router(
        _server_args(config_path),
        _model_config(architecture),
    )
    layer = SimpleNamespace(prefix="model.layers.3.self_attn.q_proj")

    router.bind_kda_linear_operators(_FakeModel(layer))

    assert hasattr(layer, "_kda_apply") is should_bind


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1
    return functions[0]


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
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


def _is_cached_presence_check(node: ast.AST, attribute: str) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and isinstance(node.left.value, ast.Name)
        and node.left.value.id == "self"
        and node.left.attr == attribute
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.IsNot)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def _has_exact_glm52_architecture_check(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Compare)
        and any(
            isinstance(comparator, ast.Name)
            and comparator.id == "GLM52_ARCHITECTURE"
            for comparator in node.comparators
        )
        for node in ast.walk(tree)
    )


def test_model_constructor_excludes_nextn_before_indexer_binding():
    init = _class_method(
        _parse(MODEL_PATH),
        "DeepseekV2AttentionMLA",
        "__init__",
    )
    indexer_call = _calls(init, "Indexer")
    assert indexer_call
    route_calls = _calls(init, "get_kda_operator_for_architecture")
    assert len(route_calls) == 1
    assert any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == "is_nextn"
        for node in ast.walk(init)
    )
    assert _has_exact_glm52_architecture_check(init)
    assert {
        keyword.arg for keyword in indexer_call[0].keywords
    } >= {"kda_index_score_operator"}


def test_index_score_uses_cached_direct_branch_and_native_else():
    method = _function(_parse(INDEXER_PATH), "_get_topk_paged")
    branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and _is_cached_presence_check(node.test, "kda_index_score_operator")
    )
    adapter_calls = _calls(ast.Module(body=branch.body, type_ignores=[]), "kda_index_score_operator")
    assert len(adapter_calls) == 1
    assert not adapter_calls[0].args
    assert {keyword.arg for keyword in adapter_calls[0].keywords} == (
        INDEX_SCORE_KEYWORDS
    )
    phase_keyword = next(
        keyword
        for keyword in adapter_calls[0].keywords
        if keyword.arg == "phase"
    )
    assert (
        isinstance(phase_keyword.value, ast.Attribute)
        and isinstance(phase_keyword.value.value, ast.Name)
        and phase_keyword.value.value.id == "forward_batch"
        and phase_keyword.value.attr == "forward_mode"
    )
    assert any(
        _calls(ast.Module(body=branch.orelse, type_ignores=[]), native_name)
        for native_name in (
            "aiter_paged_mqa_logits",
            "cutedsl_paged_mqa_logits",
            "deepgemm_paged_mqa_logits_native",
            "deepgemm_paged_mqa_logits_split",
        )
    )
    assert not any(isinstance(node, ast.Try) for node in ast.walk(method))


def test_sparse_attention_uses_cached_direct_branch_and_native_else():
    tree = _parse(BACKEND_PATH)
    init = _class_method(tree, "DeepseekSparseAttnBackend", "__init__")
    assert _calls(init, "get_kda_operator_for_architecture")
    assert any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Attribute)
        and node.operand.attr == "is_draft_worker"
        for node in ast.walk(init)
    )
    assert _has_exact_glm52_architecture_check(init)

    method = _function(tree, "_forward_flashmla_sparse")
    branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and _is_cached_presence_check(
            node.test, "kda_sparse_attention_operator"
        )
    )
    adapter_calls = _calls(
        ast.Module(body=branch.body, type_ignores=[]),
        "kda_sparse_attention_operator",
    )
    assert len(adapter_calls) == 1
    assert not adapter_calls[0].args
    assert {keyword.arg for keyword in adapter_calls[0].keywords} == (
        SPARSE_ATTENTION_KEYWORDS
    )
    assert len(
        _calls(
            ast.Module(body=branch.orelse, type_ignores=[]),
            "flash_mla_sparse_fwd",
        )
    ) == 1
    assert not any(isinstance(node, ast.Try) for node in ast.walk(method))


def test_example_uses_exact_architecture_slots_targets_and_placeholder_roots():
    document = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    profile = document["profiles"]["glm52_best"]
    operators = profile["operators"]

    assert document["version"] == 1
    assert profile["architecture"] == GLM52_ARCHITECTURE
    assert set(operators) == GLM52_SLOTS
    assert set(operators["glm52.dsa_projection"]["targets"]) == (
        PROJECTION_TARGETS
    )
    assert set(operators["glm52.dsa_indexer"]["targets"]) == INDEXER_TARGETS
    assert all(
        operator["root"].startswith("/absolute/path/to/")
        for operator in operators.values()
    )


def test_linear_target_components_exist_in_current_sources():
    model_source = MODEL_PATH.read_text(encoding="utf-8")
    indexer_source = INDEXER_PATH.read_text(encoding="utf-8")

    for target in PROJECTION_TARGETS:
        assert f'add_prefix("{target.rsplit(".", 1)[1]}"' in model_source
    for target in INDEXER_TARGETS:
        assert f'add_prefix("{target.rsplit(".", 1)[1]}"' in indexer_source
