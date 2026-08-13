#!/usr/bin/env python3
"""Validate and summarize repeated DSV4-Pro decode-only samples."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from validate_decode_result import validate


def _stats(values: list[float]) -> dict[str, float | int]:
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


def summarize(results: list[Path], server_info: Path) -> dict:
    samples = [validate(path, server_info) for path in results]
    input_hash_match_count = sum(
        sample["input_ids_sha256"] == samples[0]["input_ids_sha256"]
        for sample in samples
    )
    reference_hashes = samples[0]["output_token_ids_sha256"]
    token_hash_match_count = sum(
        sample["output_token_ids_sha256"] == reference_hashes for sample in samples
    )
    exact_request_matches = [
        sum(
            digest == reference_digest
            for digest, reference_digest in zip(
                sample["output_token_ids_sha256"], reference_hashes
            )
        )
        for sample in samples
    ]
    return {
        "status": "PASS",
        "semantics": {
            "requests": 64,
            "input_tokens_per_request": 100_000,
            "output_tokens_per_request": 1_000,
            "decode_tokens": 63_936,
            "decode_boundary": "earliest first token to final completion",
        },
        "decode_throughput": _stats(
            [sample["decode_throughput"] for sample in samples]
        ),
        "steady_decode_throughput": _stats(
            [sample["steady_decode_throughput"] for sample in samples]
        ),
        "decode_duration": _stats(
            [sample["decode_duration"] for sample in samples]
        ),
        "client_decode_throughput": _stats(
            [sample["client_decode_throughput"] for sample in samples]
        ),
        "first_token_spread": _stats(
            [sample["first_token_spread"] for sample in samples]
        ),
        "server_first_token_spread": _stats(
            [sample["server_first_token_spread"] for sample in samples]
        ),
        "repeat_token_hash_match_count": token_hash_match_count,
        "repeat_token_hash_match_rate": token_hash_match_count / len(samples),
        "exact_request_matches_to_run1": exact_request_matches,
        "exact_request_match_rate_to_run1": [
            count / 64 for count in exact_request_matches
        ],
        "input_hash_match_count": input_hash_match_count,
        "input_hash_match_rate": input_hash_match_count / len(samples),
        "samples": samples,
    }


def _markdown(summary: dict) -> str:
    decode = summary["decode_throughput"]
    steady = summary["steady_decode_throughput"]
    lines = [
        "# DeepSeek V4 Pro decode-only benchmark",
        "",
        "Workload: 64 requests, 100,000 input tokens/request, fixed 1,000 output tokens/request.",
        "The primary numerator is exactly 64 x 999 = 63,936 post-first-token decode tokens, timed from the earliest server first-token timestamp to the latest server finish timestamp.",
        "",
        "| metric | n | mean | median | CV | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| full decode throughput (tok/s) | {decode['count']} | {decode['mean']:.2f} | {decode['median']:.2f} | {decode['cv']:.2%} | {decode['min']:.2f} | {decode['max']:.2f} |",
        f"| steady decode throughput (tok/s) | {steady['count']} | {steady['mean']:.2f} | {steady['median']:.2f} | {steady['cv']:.2%} | {steady['min']:.2f} | {steady['max']:.2f} |",
        "",
        "Repeated output hash matches: "
        f"{summary['repeat_token_hash_match_count']}/{decode['count']}.",
        "Exact request-vector matches to run 1: "
        + ", ".join(
            str(value) for value in summary["exact_request_matches_to_run1"]
        )
        + "/64.",
        "Repeated input hash matches: "
        f"{summary['input_hash_match_count']}/{decode['count']}.",
        "",
        "| run | server decode tok/s | client decode tok/s | steady tok/s | server decode duration (s) | client/server first-token spread (s) | early decode tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, sample in enumerate(summary["samples"], start=1):
        lines.append(
            f"| {index} | {sample['decode_throughput']:.2f} | "
            f"{sample['client_decode_throughput']:.2f} | "
            f"{sample['steady_decode_throughput']:.2f} | "
            f"{sample['decode_duration']:.4f} | "
            f"{sample['first_token_spread']:.4f} / "
            f"{sample['server_first_token_spread']:.4f} | "
            f"{sample['early_decode_tokens']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--server-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.result, args.server_info)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["decode_throughput"], sort_keys=True))


if __name__ == "__main__":
    main()
