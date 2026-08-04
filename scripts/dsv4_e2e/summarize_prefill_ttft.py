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


def summarize(
    manifest: Path,
    min_pairs: int,
    tolerance: float,
    *,
    expected_batch_size: int = EXPECTED_BATCH_SIZE,
    expected_cached_history: int = EXPECTED_CACHED_HISTORY,
    expected_new_per_request: int = EXPECTED_NEW_CHUNK,
    expected_output_len: int = EXPECTED_OUTPUT_LEN,
    expected_forward_batch_m: int = EXPECTED_NEW_CHUNK,
    require_shape_warmup: bool = False,
) -> dict:
    expected_input_len = expected_cached_history + expected_new_per_request
    expected_cache_hit_rate = expected_cached_history / expected_input_len
    expected_aggregate_new = expected_batch_size * expected_new_per_request
    rows = _read_manifest(manifest)
    ordinals = [int(row["ordinal"]) for row in rows]
    if ordinals != list(range(1, len(rows) + 1)):
        raise ValueError(f"ordinals are not contiguous: {ordinals}")
    by_pair: dict[int, dict[str, dict]] = defaultdict(dict)
    by_backend: dict[str, list[float]] = defaultdict(list)
    samples = []
    for row in rows:
        if row["backend"] not in {"native", "huge_kernel"}:
            raise ValueError(f"unsupported backend in manifest: {row['backend']}")
        pair_id = int(row["pair"])
        if row["backend"] in by_pair[pair_id]:
            raise ValueError(
                f"duplicate {row['backend']} sample for pair {pair_id}"
            )
        checked = validate(
            Path(row["result"]),
            Path(row["log"]),
            row["backend"],
            tolerance,
            expected_batch_size=expected_batch_size,
            expected_cached_history=expected_cached_history,
            expected_new_per_request=expected_new_per_request,
            expected_output_len=expected_output_len,
            expected_forward_batch_m=expected_forward_batch_m,
            require_shape_warmup=require_shape_warmup,
        )
        checked["pair"] = pair_id
        checked["ordinal"] = int(row["ordinal"])
        by_pair[pair_id][row["backend"]] = checked
        by_backend[row["backend"]].append(checked["last_ttft"])
        samples.append(checked)

    if len(by_pair) < min_pairs:
        raise ValueError(f"need at least {min_pairs} complete pairs, got {len(by_pair)}")
    expected_pair_ids = list(range(1, len(by_pair) + 1))
    if sorted(by_pair) != expected_pair_ids:
        raise ValueError(
            f"pair ids must be contiguous from 1; got {sorted(by_pair)}"
        )
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
            "batch_size": expected_batch_size,
            "cached_history": expected_cached_history,
            "cached_history_per_request": expected_cached_history,
            "new_chunk": expected_new_per_request,
            "new_tokens_per_request": expected_new_per_request,
            "aggregate_new_tokens": expected_aggregate_new,
            "forward_batch_m": expected_forward_batch_m,
            "forward_batch_count": (
                expected_aggregate_new // expected_forward_batch_m
            ),
            "input_len": expected_input_len,
            "output_len": expected_output_len,
            "expected_cache_hit_rate": expected_cache_hit_rate,
        },
        "backend_ttft": {"native": native_stats, "huge_kernel": huge_stats},
        "incremental_throughput_total_new_tokens_per_second": {
            backend: _stats([expected_aggregate_new / value for value in values])
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
    inc = summary["incremental_throughput_total_new_tokens_per_second"]
    semantics = summary["semantics"]
    lines = [
        "# DSV4 incremental-prefill TTFT comparison",
        "",
        f"Formal status: **{summary['status']}**",
        "",
        "Semantic gate: "
        f"batch={semantics['batch_size']}, "
        f"cached/request={semantics['cached_history_per_request']}, "
        f"new/request={semantics['new_tokens_per_request']}, "
        f"aggregate new={semantics['aggregate_new_tokens']}, "
        f"ForwardBatch M={semantics['forward_batch_m']}, "
        f"ForwardBatches={semantics['forward_batch_count']}, "
        f"output={semantics['output_len']}.",
        "",
        "| backend | n | mean TTFT (s) | median TTFT (s) | median incremental throughput (token/s) |",
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
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument(
        "--cached-history-per-request", type=int, default=EXPECTED_CACHED_HISTORY
    )
    parser.add_argument(
        "--new-tokens-per-request", type=int, default=EXPECTED_NEW_CHUNK
    )
    parser.add_argument("--output-len", type=int, default=EXPECTED_OUTPUT_LEN)
    parser.add_argument(
        "--forward-batch-m", type=int, default=EXPECTED_NEW_CHUNK
    )
    parser.add_argument("--require-shape-warmup", action="store_true")
    args = parser.parse_args()
    summary = summarize(
        args.manifest,
        args.min_pairs,
        args.cache_hit_tolerance,
        expected_batch_size=args.batch_size,
        expected_cached_history=args.cached_history_per_request,
        expected_new_per_request=args.new_tokens_per_request,
        expected_output_len=args.output_len,
        expected_forward_batch_m=args.forward_batch_m,
        require_shape_warmup=args.require_shape_warmup,
    )
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
