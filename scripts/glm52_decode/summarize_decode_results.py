#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from validate_decode_result import validate


def _stats(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": stdev,
        "cv": stdev / mean if mean else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--server-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-moe-runner",
        choices=("auto", "deep_gemm", "flashinfer_trtllm_routed"),
        default="auto",
    )
    parser.add_argument(
        "--expected-shared-expert-parallelism", choices=("tp1", "tp8"), default="tp8"
    )
    args = parser.parse_args()
    samples = [
        validate(
            path,
            args.server_info,
            args.expected_moe_runner,
            require_cached_context=True,
            expected_shared_expert_parallelism=(
                args.expected_shared_expert_parallelism
            ),
        )
        for path in args.result
    ]
    summary = {
        "status": "PASS",
        "semantics": {
            "requests": 64,
            "input_tokens_per_request": 100_000,
            "output_tokens_per_request": 1_000,
            "decode_tokens": 63_936,
            "decode_boundary": "earliest server first token to latest server finish",
            "context_residency": (
                "exactly 99,968 cached prompt tokens per request "
                "(full 64-token KV pages)"
            ),
        },
        "decode_throughput": _stats([x["decode_throughput"] for x in samples]),
        "steady_decode_throughput": _stats(
            [x["steady_decode_throughput"] for x in samples]
        ),
        "decode_duration": _stats([x["decode_duration"] for x in samples]),
        "repeat_input_hash_matches": sum(
            x["input_ids_sha256"] == samples[0]["input_ids_sha256"] for x in samples
        ),
        "exact_batch_output_matches": sum(
            x["output_token_ids_sha256"] == samples[0]["output_token_ids_sha256"]
            for x in samples
        ),
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["decode_throughput"], sort_keys=True))


if __name__ == "__main__":
    main()
