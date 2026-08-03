#!/usr/bin/env python3
"""Fail closed if a DSV4 cached-prefill benchmark is ambiguous."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_BATCH_SIZE = 1
EXPECTED_CACHED_HISTORY = 65536
EXPECTED_NEW_CHUNK = 4096
EXPECTED_INPUT_LEN = EXPECTED_CACHED_HISTORY + EXPECTED_NEW_CHUNK
EXPECTED_OUTPUT_LEN = 1
EXPECTED_CACHE_HIT_RATE = EXPECTED_CACHED_HISTORY / EXPECTED_INPUT_LEN


def _read_single_result(path: Path) -> dict:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one JSONL row, got {len(rows)}")
    value = json.loads(rows[0])
    if not isinstance(value, dict):
        raise ValueError(f"{path} row must be a JSON object")
    return value


def _positive_finite(result: dict, key: str) -> float:
    value = float(result[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and > 0, got {value!r}")
    return value


def validate(
    result_path: Path,
    log_path: Path,
    backend: str,
    tolerance: float,
    *,
    expected_batch_size: int = EXPECTED_BATCH_SIZE,
    expected_cached_history: int = EXPECTED_CACHED_HISTORY,
    expected_new_per_request: int = EXPECTED_NEW_CHUNK,
    expected_output_len: int = EXPECTED_OUTPUT_LEN,
    require_shape_warmup: bool = False,
) -> dict:
    if min(
        expected_batch_size,
        expected_cached_history,
        expected_new_per_request,
        expected_output_len,
    ) <= 0:
        raise ValueError("all expected benchmark dimensions must be positive")
    expected_input_len = expected_cached_history + expected_new_per_request
    expected_aggregate_new = expected_batch_size * expected_new_per_request
    expected_aggregate_cached = expected_batch_size * expected_cached_history
    expected_cache_hit_rate = expected_cached_history / expected_input_len
    result = _read_single_result(result_path)
    expected = {
        "batch_size": expected_batch_size,
        "input_len": expected_input_len,
        "output_len": expected_output_len,
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            raise ValueError(
                f"{key} semantic mismatch: expected {expected_value}, "
                f"got {result.get(key)!r}"
            )

    last_ttft = _positive_finite(result, "last_ttft")
    latency = _positive_finite(result, "latency")
    _positive_finite(result, "input_throughput")
    if latency + 1e-3 < last_ttft:
        raise ValueError(f"latency {latency} cannot be smaller than TTFT {last_ttft}")

    measured_hit_rate = result.get("cache_hit_rate")
    if measured_hit_rate is None:
        raise ValueError(
            "cache_hit_rate is null; metrics did not prove the requested cache hit"
        )
    measured_hit_rate = float(measured_hit_rate)
    if abs(measured_hit_rate - expected_cache_hit_rate) > tolerance:
        raise ValueError(
            "cache-hit semantic mismatch: expected "
            f"{expected_cache_hit_rate:.8f} +/- {tolerance}, "
            f"got {measured_hit_rate:.8f}"
        )

    log = log_path.read_text(encoding="utf-8", errors="replace")
    required = (
        f"Warming up cache with {expected_cache_hit_rate * 100:.1f}% hit rate "
        f"({expected_cached_history} tokens per request)",
        "Cache warmup completed",
        f"DSV4 worker backend active: {backend}",
    )
    if require_shape_warmup:
        required += (
            "Warming exact cached-prefill shape with a non-matching suffix "
            f"({expected_batch_size} requests, {expected_aggregate_new} new tokens)",
            "Cached-prefill shape warmup completed",
        )
    missing = [marker for marker in required if marker not in log]
    if missing:
        raise ValueError(f"{log_path} is missing required markers: {missing}")
    scheduler_lines = [
        line
        for line in log.splitlines()
        if f"#new-seq: {expected_batch_size}" in line
        and f"#new-token: {expected_aggregate_new}" in line
        and f"#cached-token: {expected_aggregate_cached}" in line
    ]
    if not scheduler_lines:
        raise ValueError(
            f"{log_path} has no scheduler line containing both "
            f"'#new-seq: {expected_batch_size}', "
            f"'#new-token: {expected_aggregate_new}', and "
            f"'#cached-token: {expected_aggregate_cached}'"
        )

    return {
        "backend": backend,
        "batch_size": expected_batch_size,
        # Backward-compatible aliases retained for existing reports/tests.
        "cached_history": expected_cached_history,
        "new_chunk": expected_aggregate_new,
        "cached_history_per_request": expected_cached_history,
        "new_tokens_per_request": expected_new_per_request,
        "aggregate_cached_tokens": expected_aggregate_cached,
        "aggregate_new_tokens": expected_aggregate_new,
        "input_len": expected_input_len,
        "output_len": expected_output_len,
        "cache_hit_rate": measured_hit_rate,
        "last_ttft": last_ttft,
        "incremental_throughput": expected_aggregate_new / last_ttft,
        "latency": latency,
        "scheduler_semantic_line": scheduler_lines[-1],
        "result_path": str(result_path),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("native", "huge_kernel"), required=True
    )
    parser.add_argument("--cache-hit-tolerance", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument(
        "--cached-history-per-request", type=int, default=EXPECTED_CACHED_HISTORY
    )
    parser.add_argument(
        "--new-tokens-per-request", type=int, default=EXPECTED_NEW_CHUNK
    )
    parser.add_argument("--output-len", type=int, default=EXPECTED_OUTPUT_LEN)
    parser.add_argument("--require-shape-warmup", action="store_true")
    args = parser.parse_args()
    checked = validate(
        args.result,
        args.log,
        args.backend,
        args.cache_hit_tolerance,
        expected_batch_size=args.batch_size,
        expected_cached_history=args.cached_history_per_request,
        expected_new_per_request=args.new_tokens_per_request,
        expected_output_len=args.output_len,
        require_shape_warmup=args.require_shape_warmup,
    )
    print(json.dumps(checked, sort_keys=True))


if __name__ == "__main__":
    main()
