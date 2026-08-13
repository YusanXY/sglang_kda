#!/usr/bin/env python3
"""Compare GLM-5.2 decode vectors against same-backend run-to-run noise."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path


EXPECTED_BATCH_SIZE = 64
EXPECTED_OUTPUT_LEN = 1000
PREFIX_LENGTHS = (1, 8, 32, 128, EXPECTED_OUTPUT_LEN)


def _read_result(path: Path) -> dict:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one JSONL row")
    row = json.loads(rows[0])
    vectors = row.get("output_token_ids")
    if not isinstance(vectors, list) or len(vectors) != EXPECTED_BATCH_SIZE:
        raise ValueError(f"{path} must contain {EXPECTED_BATCH_SIZE} output vectors")
    if any(len(vector) != EXPECTED_OUTPUT_LEN for vector in vectors):
        raise ValueError(
            f"{path} must contain exactly {EXPECTED_OUTPUT_LEN} tokens per request"
        )
    return row


def _compare(left: dict, right: dict) -> dict:
    if left["input_ids_sha256"] != right["input_ids_sha256"]:
        raise ValueError("input token vectors differ")
    left_vectors = left["output_token_ids"]
    right_vectors = right["output_token_ids"]
    exact_requests = sum(a == b for a, b in zip(left_vectors, right_vectors))
    prefix_match_rates = {}
    for prefix_len in PREFIX_LENGTHS:
        matching_tokens = sum(
            a == b
            for left_vector, right_vector in zip(left_vectors, right_vectors)
            for a, b in zip(
                left_vector[:prefix_len], right_vector[:prefix_len]
            )
        )
        prefix_match_rates[str(prefix_len)] = matching_tokens / (
            EXPECTED_BATCH_SIZE * prefix_len
        )
    matching_tokens = int(
        prefix_match_rates[str(EXPECTED_OUTPUT_LEN)]
        * EXPECTED_BATCH_SIZE
        * EXPECTED_OUTPUT_LEN
    )
    total_tokens = EXPECTED_BATCH_SIZE * EXPECTED_OUTPUT_LEN
    return {
        "exact_request_matches": exact_requests,
        "exact_request_match_rate": exact_requests / EXPECTED_BATCH_SIZE,
        "matching_token_positions": matching_tokens,
        "token_position_match_rate": matching_tokens / total_tokens,
        "prefix_token_position_match_rate": prefix_match_rates,
        "total_token_positions": total_tokens,
    }


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _pair_group(rows: list[dict]) -> dict:
    pairs = [_compare(a, b) for a, b in itertools.combinations(rows, 2)]
    return {
        "pair_count": len(pairs),
        "token_position_match_rate": _summarize(
            [pair["token_position_match_rate"] for pair in pairs]
        ),
        "exact_request_match_rate": _summarize(
            [pair["exact_request_match_rate"] for pair in pairs]
        ),
        "prefix_token_position_match_rate": {
            str(prefix_len): _summarize(
                [
                    pair["prefix_token_position_match_rate"][str(prefix_len)]
                    for pair in pairs
                ]
            )
            for prefix_len in PREFIX_LENGTHS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = [_read_result(path) for path in args.reference]
    candidate = [_read_result(path) for path in args.candidate]
    input_hashes = {
        row["input_ids_sha256"] for row in [*reference, *candidate]
    }
    if len(input_hashes) != 1:
        raise ValueError("all inputs must have the same token-vector hash")

    cross_pairs = [_compare(a, b) for a in reference for b in candidate]
    reference_self = _pair_group(reference)
    candidate_self = _pair_group(candidate)
    cross_prefix = {
        str(prefix_len): _summarize(
            [
                pair["prefix_token_position_match_rate"][str(prefix_len)]
                for pair in cross_pairs
            ]
        )
        for prefix_len in PREFIX_LENGTHS
    }
    # Greedy decode can diverge after a numerically insignificant tie break, so
    # exact 1K-vector equality is not a useful FP8 acceptance rule.  Instead,
    # require the candidate/reference median agreement at every prefix to be no
    # worse than the minimum agreement observed between two native runs.  Also
    # require the candidate's own repeatability to clear that same native floor.
    checks = {}
    for prefix_len in PREFIX_LENGTHS:
        key = str(prefix_len)
        native_floor = reference_self["prefix_token_position_match_rate"][key][
            "min"
        ]
        checks[key] = {
            "native_self_min": native_floor,
            "candidate_self_median": candidate_self[
                "prefix_token_position_match_rate"
            ][key]["median"],
            "cross_median": cross_prefix[key]["median"],
        }
        checks[key]["pass"] = (
            checks[key]["candidate_self_median"] >= native_floor
            and checks[key]["cross_median"] >= native_floor
        )

    summary = {
        "status": "PASS" if all(check["pass"] for check in checks.values()) else "FAIL",
        "input_ids_sha256": next(iter(input_hashes)),
        "acceptance": {
            "rule": (
                "candidate self and cross median prefix agreement must be at "
                "least the minimum native self agreement"
            ),
            "checks": checks,
        },
        "reference_self": reference_self,
        "candidate_self": candidate_self,
        "cross": {
            "pair_count": len(cross_pairs),
            "token_position_match_rate": _summarize(
                [pair["token_position_match_rate"] for pair in cross_pairs]
            ),
            "exact_request_match_rate": _summarize(
                [pair["exact_request_match_rate"] for pair in cross_pairs]
            ),
            "prefix_token_position_match_rate": cross_prefix,
        },
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
