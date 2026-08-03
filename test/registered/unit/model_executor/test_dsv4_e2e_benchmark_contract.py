"""Hermetic checks for the formal DSV4 paired benchmark harness."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[4]
SCRIPT_DIR = REPO_ROOT / "scripts/dsv4_e2e"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = _load(
    "validate_prefill_ttft_result", SCRIPT_DIR / "validate_prefill_ttft_result.py"
)
summarizer = _load("summarize_prefill_ttft", SCRIPT_DIR / "summarize_prefill_ttft.py")


def _sample(root, ordinal, backend, *, huge_ttft=0.8, hit_rate=0.9412):
    result = root / f"{ordinal:02d}_{backend}.jsonl"
    log = root / f"{ordinal:02d}_{backend}.log"
    ttft = 1.0 if backend == "native" else huge_ttft
    result.write_text(
        json.dumps(
            {
                "batch_size": 1,
                "input_len": 69632,
                "output_len": 1,
                "latency": 1.3,
                "input_throughput": 69632 / ttft,
                "last_ttft": ttft,
                "cache_hit_rate": hit_rate,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log.write_text(
        "Warming up cache with 94.1% hit rate (65536 tokens per request)\n"
        "Cache warmup completed\n"
        "Prefill batch, #new-token: 4096, #cached-token: 65536, cuda graph: False\n"
        f"DSV4 worker backend active: {backend}\n",
        encoding="utf-8",
    )
    return result, log


def _manifest(root, huge_ttfts):
    rows = ["pair\tordinal\tbackend\tresult\tlog"]
    ordinal = 1
    for pair, huge_ttft in enumerate(huge_ttfts, 1):
        for backend in ("native", "huge_kernel"):
            result, log = _sample(root, ordinal, backend, huge_ttft=huge_ttft)
            rows.append(f"{pair}\t{ordinal}\t{backend}\t{result}\t{log}")
            ordinal += 1
    path = root / "manifest.tsv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_validator_requires_scheduler_and_metrics_semantics(tmp_path):
    result, log = _sample(tmp_path, 1, "huge_kernel")
    checked = validator.validate(result, log, "huge_kernel", 0.01)
    assert checked["cached_history"] == 65536
    assert checked["new_chunk"] == 4096
    assert checked["incremental_throughput"] == pytest.approx(5120.0)

    log.write_text(
        "Warming up cache with 94.1% hit rate (65536 tokens per request)\n"
        "Cache warmup completed\n"
        "#new-token: 4096\n"
        "#cached-token: 65536\n"
        "DSV4 worker backend active: huge_kernel\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validator.validate(result, log, "huge_kernel", 0.01)


def test_five_of_five_wins_pass_formal_gate(tmp_path):
    summary = summarizer.summarize(_manifest(tmp_path, [0.8] * 5), 5, 0.01)
    assert summary["status"] == "PASS"
    assert summary["acceptance_gate"]["win_count"] == 5
    assert summary["acceptance_gate"]["required_wins"] == 4
    assert summary["paired_speedup"]["median"] == pytest.approx(1.25)


def test_three_of_five_wins_fail_formal_gate(tmp_path):
    summary = summarizer.summarize(
        _manifest(tmp_path, [0.8, 0.8, 0.8, 1.2, 1.2]), 5, 0.01
    )
    assert summary["status"] == "FAILED"
    assert summary["acceptance_gate"]["win_count"] == 3


def test_runner_pins_exact_workload_and_order():
    source = (SCRIPT_DIR / "run_prefill_ttft_compare.sh").read_text(encoding="utf-8")
    for fragment in (
        "CACHED_HISTORY=65536",
        "NEW_CHUNK=4096",
        "CONTEXT_CAPACITY=73728",
        "CACHE_HIT_RATE=0.9411764705882353",
        "REPO=${REPO_ROOT:-",
        "MODEL=${DSV4_MODEL:-",
        "PYTHON=${PYTHON_BIN:-",
        "GPU_IDS=0,1,2,3",
        'export CUDA_VISIBLE_DEVICES="$GPU_IDS"',
        'nvidia-smi --id="$GPU_IDS" --query-compute-apps=pid',
        '--dsv4-worker-backend "$backend"',
        'run_one "$pair" "$ordinal" native',
        'run_one "$pair" "$ordinal" huge_kernel',
        "PAIRS must be at least 5",
    ):
        assert fragment in source

    nvidia_smi_lines = [
        line for line in source.splitlines() if "nvidia-smi" in line
    ]
    assert nvidia_smi_lines
    assert all('--id="$GPU_IDS"' in line for line in nvidia_smi_lines)
