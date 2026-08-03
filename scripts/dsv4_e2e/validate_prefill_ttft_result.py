#!/usr/bin/env python3
"""Fail closed if a DSV4 64K-history + 4K-new benchmark is ambiguous."""

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


def validate(result_path: Path, log_path: Path, backend: str, tolerance: float) -> dict:
    result = _read_single_result(result_path)
    expected = {
        "batch_size": EXPECTED_BATCH_SIZE,
        "input_len": EXPECTED_INPUT_LEN,
        "output_len": EXPECTED_OUTPUT_LEN,
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
            "cache_hit_rate is null; metrics did not prove the 65536-token cache hit"
        )
    measured_hit_rate = float(measured_hit_rate)
    if abs(measured_hit_rate - EXPECTED_CACHE_HIT_RATE) > tolerance:
        raise ValueError(
            "cache-hit semantic mismatch: expected "
            f"{EXPECTED_CACHE_HIT_RATE:.8f} +/- {tolerance}, "
            f"got {measured_hit_rate:.8f}"
        )

    log = log_path.read_text(encoding="utf-8", errors="replace")
    required = (
        "Warming up cache with 94.1% hit rate (65536 tokens per request)",
        "Cache warmup completed",
        f"DSV4 worker backend active: {backend}",
    )
    missing = [marker for marker in required if marker not in log]
    if missing:
        raise ValueError(f"{log_path} is missing required markers: {missing}")
    scheduler_lines = [
        line
        for line in log.splitlines()
        if "#new-token: 4096" in line and "#cached-token: 65536" in line
    ]
    if not scheduler_lines:
        raise ValueError(
            f"{log_path} has no scheduler line containing both "
            "'#new-token: 4096' and '#cached-token: 65536'"
        )

    return {
        "backend": backend,
        "batch_size": EXPECTED_BATCH_SIZE,
        "cached_history": EXPECTED_CACHED_HISTORY,
        "new_chunk": EXPECTED_NEW_CHUNK,
        "input_len": EXPECTED_INPUT_LEN,
        "output_len": EXPECTED_OUTPUT_LEN,
        "cache_hit_rate": measured_hit_rate,
        "last_ttft": last_ttft,
        "incremental_throughput": EXPECTED_NEW_CHUNK / last_ttft,
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
    args = parser.parse_args()
    checked = validate(args.result, args.log, args.backend, args.cache_hit_tolerance)
    print(json.dumps(checked, sort_keys=True))


if __name__ == "__main__":
    main()
