#!/usr/bin/env python3
"""Summarize paired native/huge_kernel TTFT samples using only stdlib."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from validate_prefill_ttft_result import (
    EXPECTED_BATCH_SIZE,
    EXPECTED_CACHE_HIT_RATE,
    EXPECTED_CACHED_HISTORY,
    EXPECTED_INPUT_LEN,
    EXPECTED_NEW_CHUNK,
    EXPECTED_OUTPUT_LEN,
    validate,
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError(f"empty manifest: {path}")
    header = lines[0].split("\t")
    expected = ["pair", "ordinal", "backend", "result", "log"]
    if header != expected:
        raise ValueError(f"manifest header must be {expected}, got {header}")
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(header):
            raise ValueError(f"malformed manifest line: {line!r}")
        rows.append(dict(zip(header, values)))
    return rows


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(manifest: Path, min_pairs: int, tolerance: float) -> dict:
    rows = _read_manifest(manifest)
    ordinals = [int(row["ordinal"]) for row in rows]
    if ordinals != list(range(1, len(rows) + 1)):
        raise ValueError(f"ordinals are not contiguous: {ordinals}")
    backends = [row["backend"] for row in rows]
    expected_order = [
        backend
        for _ in range(len(rows) // 2)
        for backend in ("native", "huge_kernel")
    ]
    if backends != expected_order:
        raise ValueError(
            "runs must alternate native,huge_kernel for every pair; "
            f"got {backends}"
        )

    by_pair: dict[int, dict[str, dict]] = defaultdict(dict)
    by_backend: dict[str, list[float]] = defaultdict(list)
    samples = []
    for row in rows:
        checked = validate(
            Path(row["result"]), Path(row["log"]), row["backend"], tolerance
        )
        checked["pair"] = int(row["pair"])
        checked["ordinal"] = int(row["ordinal"])
        by_pair[checked["pair"]][row["backend"]] = checked
        by_backend[row["backend"]].append(checked["last_ttft"])
        samples.append(checked)

    if len(by_pair) < min_pairs:
        raise ValueError(f"need at least {min_pairs} complete pairs, got {len(by_pair)}")
    if any(set(pair) != {"native", "huge_kernel"} for pair in by_pair.values()):
        raise ValueError("every pair must contain exactly native and huge_kernel")

    paired = []
    for pair_id in sorted(by_pair):
        native = by_pair[pair_id]["native"]["last_ttft"]
        huge = by_pair[pair_id]["huge_kernel"]["last_ttft"]
        paired.append(
            {
                "pair": pair_id,
                "native_ttft": native,
                "huge_kernel_ttft": huge,
                "speedup": native / huge,
                "improvement_pct": (native - huge) / native * 100.0,
                "huge_kernel_won": huge < native,
            }
        )

    native_stats = _stats(by_backend["native"])
    huge_stats = _stats(by_backend["huge_kernel"])
    win_count = sum(row["huge_kernel_won"] for row in paired)
    required_wins = (len(paired) * 4 + 4) // 5
    median_gate = huge_stats["median"] < native_stats["median"]
    wins_gate = win_count >= required_wins
    return {
        "status": "PASS" if median_gate and wins_gate else "FAILED",
        "acceptance_gate": {
            "win_count": win_count,
            "pair_count": len(paired),
            "required_wins": required_wins,
            "win_rate": win_count / len(paired),
            "huge_median_below_native": median_gate,
            "wins_gate_passed": wins_gate,
        },
        "semantics": {
            "batch_size": EXPECTED_BATCH_SIZE,
            "cached_history": EXPECTED_CACHED_HISTORY,
            "new_chunk": EXPECTED_NEW_CHUNK,
            "input_len": EXPECTED_INPUT_LEN,
            "output_len": EXPECTED_OUTPUT_LEN,
            "expected_cache_hit_rate": EXPECTED_CACHE_HIT_RATE,
        },
        "backend_ttft": {"native": native_stats, "huge_kernel": huge_stats},
        "incremental_throughput_4096_tokens_per_second": {
            backend: _stats([EXPECTED_NEW_CHUNK / value for value in values])
            for backend, values in by_backend.items()
        },
        "paired_speedup": _stats([row["speedup"] for row in paired]),
        "paired_improvement_pct": _stats(
            [row["improvement_pct"] for row in paired]
        ),
        "pairs": paired,
        "samples": samples,
    }


def _markdown(summary: dict) -> str:
    native = summary["backend_ttft"]["native"]
    huge = summary["backend_ttft"]["huge_kernel"]
    gate = summary["acceptance_gate"]
    speedup = summary["paired_speedup"]
    inc = summary["incremental_throughput_4096_tokens_per_second"]
    lines = [
        "# DSV4 incremental-prefill TTFT comparison",
        "",
        f"Formal status: **{summary['status']}**",
        "",
        "Semantic gate: batch=1, 65536 cached tokens, 4096 new tokens, output=1.",
        "",
        "| backend | n | mean TTFT (s) | median TTFT (s) | median 4K throughput (token/s) |",
        "|---|---:|---:|---:|---:|",
        f"| native | {native['count']} | {native['mean']:.6f} | {native['median']:.6f} | {inc['native']['median']:.2f} |",
        f"| huge_kernel | {huge['count']} | {huge['mean']:.6f} | {huge['median']:.6f} | {inc['huge_kernel']['median']:.2f} |",
        "",
        f"Paired median speedup: **{speedup['median']:.4f}x**  ",
        f"Huge-kernel wins: **{gate['win_count']}/{gate['pair_count']}** (required: {gate['required_wins']})",
        "",
        "| pair | native TTFT (s) | huge TTFT (s) | speedup | huge won |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["pairs"]:
        lines.append(
            f"| {row['pair']} | {row['native_ttft']:.6f} | "
            f"{row['huge_kernel_ttft']:.6f} | {row['speedup']:.4f}x | "
            f"{'yes' if row['huge_kernel_won'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--cache-hit-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    summary = summarize(args.manifest, args.min_pairs, args.cache_hit_tolerance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    if summary["status"] != "PASS":
        gate = summary["acceptance_gate"]
        print(
            "FAILED: formal gate requires at least "
            f"{gate['required_wins']}/{gate['pair_count']} huge-kernel wins and "
            "huge_kernel median TTFT < native median TTFT"
        )
        raise SystemExit(3)
    print("PASS: formal paired TTFT acceptance gate")


if __name__ == "__main__":
    main()
