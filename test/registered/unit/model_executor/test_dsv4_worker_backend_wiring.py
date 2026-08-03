"""Hermetic CLI and construction-path checks for the DSV4 worker backend."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.benchmark.one_batch import (
    BenchArgs,
    _raise_for_failed_workers,
    _reject_stale_correctness_output,
    _save_correctness_output,
    _verify_correctness_output,
    prepare_inputs_for_correctness_test,
    validate_correctness_output_args,
)
from sglang.srt.model_executor.dsv4_huge_kernel_model_runner import (
    validate_dsv4_huge_kernel_bench_args,
)
from sglang.srt.server_args import ServerArgs


REPO_ROOT = Path(__file__).parents[4]
SCHEDULER_PATH = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"
ONE_BATCH_PATH = REPO_ROOT / "python/sglang/benchmark/one_batch.py"


def _calls(path: Path, function_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def test_cli_has_only_native_and_huge_kernel():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    action = next(
        action
        for action in parser._actions
        if "--dsv4-worker-backend" in action.option_strings
    )

    assert action.default == "native"
    assert tuple(action.choices) == ("native", "huge_kernel")
    assert "auto" not in action.choices
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model", "dummy", "--dsv4-worker-backend", "unsupported"]
        )


def test_scheduler_has_dedicated_huge_worker_and_runtime_log_marker():
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    calls = _calls(SCHEDULER_PATH, "init_tp_model_worker")

    assert "Dsv4HugeKernelTpModelWorker" in calls
    assert "TpModelWorker" in calls
    assert "DSV4 worker backend active: %s" in source


def test_low_level_bench_uses_shared_model_runner_factory():
    calls = _calls(ONE_BATCH_PATH, "load_model")

    assert calls.count("create_model_runner") == 1
    assert "ModelRunner" not in calls


def test_correctness_initializes_moe_and_quant_backends_before_model_load():
    tree = ast.parse(ONE_BATCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "correctness_test"
    )
    direct_calls = []
    for statement in function.body:
        value = getattr(statement, "value", None)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            direct_calls.append(value.func.id)

    assert direct_calls[:3] == [
        "initialize_moe_config",
        "initialize_fp8_gemm_config",
        "initialize_fp4_gemm_config",
    ]
    assert direct_calls.index("initialize_fp4_gemm_config") < direct_calls.index(
        "load_model"
    )


def test_correctness_output_parser_and_huge_validator():
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    bench_args = BenchArgs.from_cli_args(
        parser.parse_args(
            [
                "--correctness-test",
                "--correctness-output-file",
                "huge.pt",
                "--batch-size",
                "1",
                "--input-len",
                "4096",
                "--output-len",
                "1",
            ]
        )
    )

    assert bench_args.correctness_output_file == "huge.pt"
    validate_correctness_output_args(bench_args)
    validate_dsv4_huge_kernel_bench_args(bench_args)

    input_ids, reqs = prepare_inputs_for_correctness_test(
        SimpleNamespace(batch_size=(1,), cut_len=1),
        SimpleNamespace(encode=lambda _prompt: [1, 2, 3]),
        custom_prompts=None,
    )
    assert len(input_ids) == len(reqs) == 1

    with pytest.raises(ValueError, match="requires --correctness-test"):
        validate_correctness_output_args(
            SimpleNamespace(
                correctness_test=False,
                correctness_output_file="orphan.pt",
            )
        )
    with pytest.raises(ValueError, match="must end in .pt"):
        validate_correctness_output_args(
            SimpleNamespace(
                correctness_test=True,
                correctness_output_file="wrong.json",
            )
        )


def test_correctness_output_is_cpu_float32_and_atomic(tmp_path):
    output = tmp_path / "nested" / "huge.pt"
    _save_correctness_output(
        output,
        torch.tensor([17], device="cpu"),
        torch.tensor([[1.0, 2.0]], dtype=torch.float16),
    )

    saved = torch.load(output, map_location="cpu", weights_only=True)
    assert torch.equal(saved["next_token_ids"], torch.tensor([17]))
    assert saved["next_token_logits"].device.type == "cpu"
    assert saved["next_token_logits"].dtype == torch.float32
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    source = ONE_BATCH_PATH.read_text(encoding="utf-8")
    assert "if tp_rank == 0 and bench_args.correctness_output_file:" in source


def test_parent_rejects_stale_or_missing_correctness_output(tmp_path):
    output = tmp_path / "correctness.pt"
    bench_args = SimpleNamespace(correctness_output_file=str(output))

    output.write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        _reject_stale_correctness_output(bench_args)

    output.unlink()
    _reject_stale_correctness_output(bench_args)
    with pytest.raises(RuntimeError, match="without producing output"):
        _verify_correctness_output(bench_args)

    output.write_bytes(b"new")
    _verify_correctness_output(bench_args)


def test_parent_raises_for_any_failed_worker_and_never_terminates_joined_workers():
    workers = [
        SimpleNamespace(pid=101, exitcode=0),
        SimpleNamespace(pid=102, exitcode=17),
    ]
    with pytest.raises(RuntimeError, match=r"pid=102, exitcode=17"):
        _raise_for_failed_workers(workers)

    assert "terminate" not in _calls(ONE_BATCH_PATH, "main")
