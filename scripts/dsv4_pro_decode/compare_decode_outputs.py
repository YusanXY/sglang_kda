#!/usr/bin/env python3
"""Compare full DSV4-Pro output token vectors after strict workload validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_decode_result import EXPECTED_BATCH_SIZE, EXPECTED_OUTPUT_LEN, validate


def _read_result(path: Path) -> dict:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one JSONL row")
    return json.loads(rows[0])


def compare_vectors(reference: list[list[int]], candidate: list[list[int]]) -> dict:
    if len(reference) != EXPECTED_BATCH_SIZE or len(candidate) != EXPECTED_BATCH_SIZE:
        raise ValueError("both results must contain exactly 64 request vectors")
    exact_requests = 0
    matching_tokens = 0
    first_mismatches: list[dict | None] = []
    for request_index, (ref, cand) in enumerate(zip(reference, candidate)):
        if len(ref) != EXPECTED_OUTPUT_LEN or len(cand) != EXPECTED_OUTPUT_LEN:
            raise ValueError(
                f"request {request_index} does not contain exactly 1000 output tokens"
            )
        exact_requests += ref == cand
        matching_tokens += sum(a == b for a, b in zip(ref, cand))
        mismatch = next(
            (
                {"position": pos, "reference": a, "candidate": b}
                for pos, (a, b) in enumerate(zip(ref, cand))
                if a != b
            ),
            None,
        )
        first_mismatches.append(mismatch)
    total_tokens = EXPECTED_BATCH_SIZE * EXPECTED_OUTPUT_LEN
    return {
        "exact_request_matches": exact_requests,
        "exact_request_match_rate": exact_requests / EXPECTED_BATCH_SIZE,
        "matching_token_positions": matching_tokens,
        "token_position_match_rate": matching_tokens / total_tokens,
        "total_token_positions": total_tokens,
        "first_mismatch_by_request": first_mismatches,
    }


def compare(
    reference_result: Path,
    reference_server_info: Path,
    candidate_result: Path,
    candidate_server_info: Path,
) -> dict:
    reference_checked = validate(reference_result, reference_server_info)
    candidate_checked = validate(candidate_result, candidate_server_info)
    if reference_checked["input_ids_sha256"] != candidate_checked["input_ids_sha256"]:
        raise ValueError("reference and candidate input token vectors differ")
    reference = _read_result(reference_result)
    candidate = _read_result(candidate_result)
    return {
        "status": "PASS",
        "reference_result": str(reference_result),
        "candidate_result": str(candidate_result),
        "input_ids_sha256": reference_checked["input_ids_sha256"],
        **compare_vectors(reference["output_token_ids"], candidate["output_token_ids"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-server-info", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--candidate-server-info", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = compare(
        args.reference_result,
        args.reference_server_info,
        args.candidate_result,
        args.candidate_server_info,
    )
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
