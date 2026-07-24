from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
INDEXER_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "dsv4"
    / "indexer.py"
)
BACKEND_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "deepseek_v4_backend.py"
)

EXPECTED_SLOTS = {
    "deepseek_v4.indexer_fp8_quant",
    "deepseek_v4.paged_mqa_logits",
    "deepseek_v4.topk_transform",
    "deepseek_v4.sparse_prefill_attention",
    "deepseek_v4.sparse_decode_attention",
    "deepseek_v4.dense_swa_attention",
}

EXPECTED_KEYWORDS = {
    "kda_indexer_fp8_quant": {
        "q",
        "weight",
        "weight_scale",
        "freqs_cis",
        "positions",
    },
    "kda_paged_mqa_logits": {
        "q",
        "kv_cache",
        "weights",
        "seq_lens",
        "page_table",
        "schedule",
        "max_context",
        "q_offset",
    },
    "kda_topk_transform": {
        "scores",
        "seq_lens",
        "page_tables",
        "output",
        "page_size",
        "metadata",
        "raw_indices",
    },
    "kda_sparse_prefill": {
        "q",
        "kv",
        "indices",
        "softmax_scale",
        "value_dim",
        "attention_sink",
        "topk_length",
    },
}

DENSE_DECODE_KEYWORDS = {
    "q",
    "cache",
    "indices",
    "lengths",
    "attention_sink",
    "scheduler",
}
SPARSE_DECODE_KEYWORDS = {
    "q",
    "swa_cache",
    "swa_indices",
    "swa_lengths",
    "extra_cache",
    "extra_indices",
    "extra_lengths",
    "attention_sink",
    "scheduler",
}


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


def _keyword_names(call: ast.Call) -> set[str | None]:
    return {keyword.arg for keyword in call.keywords}


def _function_containing_call(
    tree: ast.Module, callable_name: str
) -> ast.FunctionDef:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(_call_name(call) == callable_name for call in _calls(node, callable_name))
    ]
    assert len(functions) == 1
    return functions[0]


def _assigns_name(nodes: list[ast.stmt], name: str) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        for statement in nodes
        for node in ast.walk(statement)
    )


def _is_none_check(node: ast.AST, name: str, *, negated: bool) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == name
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.IsNot if negated else ast.Is)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def test_all_deepseek_v4_slots_are_exact_literals():
    slot_literals = set()
    for tree in (_parse(INDEXER_PATH), _parse(BACKEND_PATH)):
        for call in _calls(tree, "get_kda_operator"):
            slot_literals.update(
                node.value
                for node in ast.walk(call)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            )

    assert slot_literals == EXPECTED_SLOTS


def test_paged_mqa_initializes_q_before_fp8_adapter_call():
    tree = _parse(INDEXER_PATH)
    forward = _function_containing_call(tree, "kda_paged_mqa_logits")
    q_layout_branch = next(
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "use_fp4_indexer"
        and _assigns_name(node.body, "q")
        and _assigns_name(node.orelse, "q")
    )
    adapter_call = _calls(forward, "kda_paged_mqa_logits")[0]

    assert q_layout_branch.end_lineno < adapter_call.lineno


def test_paged_mqa_fp4_skips_slot_and_native_selection_is_guarded():
    tree = _parse(INDEXER_PATH)
    forward = _function_containing_call(tree, "kda_paged_mqa_logits")
    route_assignment = next(
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "kda_paged_mqa_logits"
            for target in node.targets
        )
    )

    assert isinstance(route_assignment.value, ast.IfExp)
    assert (
        isinstance(route_assignment.value.test, ast.Name)
        and route_assignment.value.test.id == "use_fp4_indexer"
    )
    assert (
        isinstance(route_assignment.value.body, ast.Constant)
        and route_assignment.value.body.value is None
    )
    assert (
        isinstance(route_assignment.value.orelse, ast.Call)
        and _call_name(route_assignment.value.orelse) == "get_kda_operator"
    )

    native_guard = next(
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and _is_none_check(node.test, "kda_paged_mqa_logits", negated=False)
    )
    fn_definitions = [
        node
        for node in ast.walk(forward)
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "fn"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and any(alias.asname == "fn" for alias in node.names)
        )
    ]

    assert len(fn_definitions) >= 7
    assert all(
        native_guard.lineno <= node.lineno <= native_guard.end_lineno
        for node in fn_definitions
    )


def test_adapters_receive_only_keyword_arguments_with_fixed_contracts():
    trees = (_parse(INDEXER_PATH), _parse(BACKEND_PATH))
    for callable_name, expected_keywords in EXPECTED_KEYWORDS.items():
        calls = [
            call
            for tree in trees
            for call in _calls(tree, callable_name)
        ]
        assert len(calls) == 1
        assert not calls[0].args
        assert _keyword_names(calls[0]) == expected_keywords

    attention_calls = _calls(_parse(BACKEND_PATH), "kda_attention")
    assert len(attention_calls) == 2
    assert all(not call.args for call in attention_calls)
    assert {frozenset(_keyword_names(call)) for call in attention_calls} == {
        frozenset(DENSE_DECODE_KEYWORDS),
        frozenset(SPARSE_DECODE_KEYWORDS),
    }


def test_each_loaded_adapter_is_guarded_by_an_explicit_presence_check():
    for tree in (_parse(INDEXER_PATH), _parse(BACKEND_PATH)):
        assigned_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value) == "get_kda_operator"
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        checked_names = {
            node.left.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.IsNot)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value is None
        }
        assert assigned_names <= checked_names


def test_native_deepseek_v4_calls_remain_available_when_slots_are_absent():
    indexer_tree = _parse(INDEXER_PATH)
    backend_tree = _parse(BACKEND_PATH)

    for native_name in (
        "fused_q_indexer_rope_hadamard_fp4_quant",
        "fused_q_indexer_rope_hadamard_quant",
        "topk_transform_512_pytorch_vectorized",
        "topk_transform_512_v2",
        "topk_transform_512",
    ):
        assert _calls(indexer_tree, native_name)
    assert _calls(indexer_tree, "fn")

    for native_name in (
        "flash_mla_with_kvcache_sm120",
        "flash_mla_with_kvcache",
        "flash_mla_sparse_fwd",
    ):
        assert _calls(backend_tree, native_name)


def test_adapter_exceptions_are_not_caught_at_deepseek_v4_call_sites():
    callables = (
        (_parse(INDEXER_PATH), "kda_indexer_fp8_quant"),
        (_parse(INDEXER_PATH), "kda_paged_mqa_logits"),
        (_parse(INDEXER_PATH), "kda_topk_transform"),
        (_parse(BACKEND_PATH), "kda_attention"),
        (_parse(BACKEND_PATH), "kda_sparse_prefill"),
    )
    for tree, callable_name in callables:
        function = _function_containing_call(tree, callable_name)
        assert not any(isinstance(node, ast.Try) for node in ast.walk(function))
