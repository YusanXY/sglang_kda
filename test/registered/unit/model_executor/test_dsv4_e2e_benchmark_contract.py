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
        "Prefill batch, #new-seq: 1, #new-token: 4096, "
        "#cached-token: 65536, cuda graph: False\n"
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


@pytest.mark.parametrize("batch_size", [16, 128])
def test_high_load_validator_requires_m4096_forward_batches(tmp_path, batch_size):
    cached_per_request = 16384
    new_per_request = 4096
    input_len = cached_per_request + new_per_request
    aggregate_cached = batch_size * cached_per_request
    result = tmp_path / f"req{batch_size}.jsonl"
    log = tmp_path / f"req{batch_size}.log"
    result.write_text(
        json.dumps(
            {
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": 1,
                "latency": 2.0,
                "input_throughput": batch_size * input_len,
                "last_ttft": 1.0,
                "cache_hit_rate": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    marker = {
        "batch_size": batch_size,
        "cached_tokens_per_request": cached_per_request,
        "input_len": input_len,
        "new_tokens_per_request": new_per_request,
        "output_len": 1,
    }
    scheduler_lines = [
        "Prefill batch, "
        f"#new-seq: {batch_size if index == 0 else 0}, "
        "#new-token: 4096, "
        f"#cached-token: {aggregate_cached if index == 0 else 0},"
        for index in range(batch_size)
    ]
    log.write_text(
        "Warming up cache with 80.0% hit rate "
        "(16384 tokens per request)\n"
        "Cache warmup completed\n"
        "Warming exact cached-prefill shape with a non-matching suffix "
        f"({batch_size} requests, {batch_size * new_per_request} new tokens)\n"
        "Cached-prefill shape warmup completed\n"
        "DSV4 worker backend active: huge_kernel\n"
        + validator.MEASURED_REQUEST_BEGIN
        + json.dumps(marker, sort_keys=True)
        + "\n"
        + "\n".join(scheduler_lines)
        + "\nSGLANG_BENCH_MEASURED_REQUEST_END\n",
        encoding="utf-8",
    )

    checked = validator.validate(
        result,
        log,
        "huge_kernel",
        0.01,
        expected_batch_size=batch_size,
        expected_cached_history=cached_per_request,
        expected_new_per_request=new_per_request,
        expected_forward_batch_m=4096,
        require_shape_warmup=True,
    )
    assert checked["new_tokens_per_request"] == 4096
    assert checked["aggregate_new_tokens"] == batch_size * 4096
    assert checked["forward_batch_m"] == 4096
    assert checked["forward_batch_count"] == batch_size

    log.write_text(
        log.read_text(encoding="utf-8").replace(
            "#new-token: 4096", "#new-token: 8192", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ForwardBatch"):
        validator.validate(
            result,
            log,
            "huge_kernel",
            0.01,
            expected_batch_size=batch_size,
            expected_cached_history=cached_per_request,
            expected_new_per_request=new_per_request,
            expected_forward_batch_m=4096,
            require_shape_warmup=True,
        )


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
        "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}",
        "export CUDA_HOME",
        'export CPATH="$CUDA_CCCL_INCLUDE${CPATH:+:$CPATH}"',
        "printf 'cuda_home=%s\\ncuda_cccl_include=%s\\n'",
        'nvidia-smi --id="$GPU_IDS" --query-compute-apps=pid',
        '--dsv4-worker-backend "$backend"',
        "--skip-server-warmup",
        "--skip-warmup",
        'run_one "$pair" "$ordinal" native',
        'run_one "$pair" "$ordinal" huge_kernel',
        "PAIRS must be at least 5",
    ):
        assert fragment in source

    nvidia_smi_lines = [line for line in source.splitlines() if "nvidia-smi" in line]
    assert nvidia_smi_lines
    assert all('--id="$GPU_IDS"' in line for line in nvidia_smi_lines)


def test_high_load_runner_uses_per_request_4k_and_req16_or_req128():
    source = (
        SCRIPT_DIR / "run_high_load_prefill_ttft_compare.sh"
    ).read_text(encoding="utf-8")
    for fragment in (
        "BATCH_SIZE=${DSV4_HIGH_LOAD_REQUESTS:-16}",
        "CACHED_PER_REQUEST=16384",
        "NEW_PER_REQUEST=4096",
        "FORWARD_BATCH_M=4096",
        'EXPECTED_FORWARD_BATCHES=$((AGGREGATE_NEW / FORWARD_BATCH_M))',
        'CHUNKED_PREFILL_SIZE=$FORWARD_BATCH_M',
        "CACHE_HIT_RATE=0.8",
        '[[ "$BATCH_SIZE" != 16 && "$BATCH_SIZE" != 128 ]]',
        'MAX_TOTAL_TOKENS=$((BATCH_SIZE * (INPUT_LEN + 1)))',
        '--max-prefill-tokens "$CHUNKED_PREFILL_SIZE"',
        '--chunked-prefill-size "$CHUNKED_PREFILL_SIZE"',
        '--new-tokens-per-request "$NEW_PER_REQUEST"',
        '--forward-batch-m "$FORWARD_BATCH_M"',
    ):
        assert fragment in source


def test_one_batch_profiler_can_flush_after_prefill_only_request():
    source = (REPO_ROOT / "python/sglang/benchmark/one_batch_server.py").read_text(
        encoding="utf-8"
    )
    assert "--profile-stop-after-request" in source
    assert 'requests.post(url + "/stop_profile"' in source
    assert "profile_stop_after_request=(" in source
    assert "bench_args.profile_stop_after_request" in source
