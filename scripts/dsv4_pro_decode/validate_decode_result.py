#!/usr/bin/env python3
"""Fail closed on an ambiguous DSV4-Pro decode-throughput sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_BATCH_SIZE = 64
EXPECTED_INPUT_LEN = 100_000
EXPECTED_INPUT_TOKENS = EXPECTED_BATCH_SIZE * EXPECTED_INPUT_LEN
EXPECTED_OUTPUT_LEN = 1_000
EXPECTED_STREAM_INTERVAL = 64
EXPECTED_DECODE_TOKENS = EXPECTED_BATCH_SIZE * (EXPECTED_OUTPUT_LEN - 1)


def _read_single_result(path: Path) -> dict:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one JSONL row, got {len(rows)}")
    row = json.loads(rows[0])
    if not isinstance(row, dict):
        raise ValueError(f"{path} row must be an object")
    return row


def _finite(result: dict, key: str, *, positive: bool = True) -> float:
    value = float(result[key])
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{key} must be finite and > 0, got {value!r}")
    return value


def validate(result_path: Path, server_info_path: Path) -> dict:
    result = _read_single_result(result_path)
    server_info = json.loads(server_info_path.read_text(encoding="utf-8"))
    if not isinstance(server_info, dict):
        raise ValueError(f"{server_info_path} must contain a JSON object")

    expected_result = {
        "batch_size": EXPECTED_BATCH_SIZE,
        "input_len": EXPECTED_INPUT_LEN,
        "input_len_min": EXPECTED_INPUT_LEN,
        "input_len_max": EXPECTED_INPUT_LEN,
        "input_tokens_total": EXPECTED_INPUT_TOKENS,
        "output_len": EXPECTED_OUTPUT_LEN,
        "stream_interval": EXPECTED_STREAM_INTERVAL,
        "decode_tokens": EXPECTED_DECODE_TOKENS,
        "completed_requests": EXPECTED_BATCH_SIZE,
    }
    for key, expected in expected_result.items():
        if result.get(key) != expected:
            raise ValueError(
                f"{key} mismatch: expected {expected!r}, got {result.get(key)!r}"
            )

    first_ttft = _finite(result, "first_ttft")
    last_ttft = _finite(result, "last_ttft")
    latency = _finite(result, "latency")
    decode_duration = _finite(result, "decode_duration")
    decode_throughput = _finite(result, "decode_throughput")
    client_decode_duration = _finite(result, "client_decode_duration")
    client_decode_throughput = _finite(result, "client_decode_throughput")
    server_first_token_spread = _finite(
        result, "server_first_token_spread", positive=False
    )
    if server_first_token_spread < 0:
        raise ValueError("server_first_token_spread must be non-negative")
    steady_duration = _finite(result, "steady_decode_duration")
    steady_throughput = _finite(result, "steady_decode_throughput")
    if not first_ttft <= last_ttft <= latency:
        raise ValueError(
            f"invalid timing order: {first_ttft=} {last_ttft=} {latency=}"
        )
    server_first_token_ts_min = _finite(result, "server_first_token_ts_min")
    server_finished_ts_max = _finite(result, "server_finished_ts_max")
    if server_finished_ts_max <= server_first_token_ts_min:
        raise ValueError(
            "server finish timestamp must be later than the first-token timestamp"
        )
    if not math.isclose(
        decode_duration,
        server_finished_ts_max - server_first_token_ts_min,
        abs_tol=2e-4,
    ):
        raise ValueError("decode_duration does not match the server timestamp window")
    if not math.isclose(
        client_decode_duration, latency - first_ttft, abs_tol=2e-4
    ):
        raise ValueError(
            "client_decode_duration does not match latency-first_ttft"
        )
    if not math.isclose(steady_duration, latency - last_ttft, abs_tol=2e-4):
        raise ValueError("steady_decode_duration does not match latency-last_ttft")
    if not math.isclose(
        decode_throughput,
        EXPECTED_DECODE_TOKENS / decode_duration,
        abs_tol=0.02,
    ):
        raise ValueError("decode_throughput numerator or duration is inconsistent")
    if not math.isclose(
        client_decode_throughput,
        EXPECTED_DECODE_TOKENS / client_decode_duration,
        abs_tol=0.02,
    ):
        raise ValueError("client_decode_throughput is inconsistent")

    early_tokens = int(result["decode_tokens_before_last_ttft"])
    steady_tokens = int(result["steady_decode_tokens"])
    if early_tokens < 0 or steady_tokens != EXPECTED_DECODE_TOKENS - early_tokens:
        raise ValueError("steady decode token accounting is inconsistent")
    if not math.isclose(
        steady_throughput, steady_tokens / steady_duration, abs_tol=0.02
    ):
        raise ValueError("steady_decode_throughput is inconsistent")
    if result.get("max_retractions") != 0:
        raise ValueError(
            "sample is not a stable one-wave decode: "
            f"max_retractions={result.get('max_retractions')!r}"
        )
    hashes = result.get("output_token_ids_sha256")
    if not isinstance(hashes, list) or len(hashes) != EXPECTED_BATCH_SIZE:
        raise ValueError("missing per-request output token hashes")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
        for value in hashes
    ):
        raise ValueError("invalid output token SHA256 digest")
    output_token_ids = result.get("output_token_ids")
    if not isinstance(output_token_ids, list) or len(output_token_ids) != EXPECTED_BATCH_SIZE:
        raise ValueError("missing per-request output token vectors")
    for index, (token_ids, digest) in enumerate(zip(output_token_ids, hashes)):
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != EXPECTED_OUTPUT_LEN
            or any(not isinstance(token_id, int) for token_id in token_ids)
        ):
            raise ValueError(f"invalid output token vector for request {index}")
        actual_digest = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual_digest != digest:
            raise ValueError(
                f"output token digest mismatch for request {index}: "
                f"expected {digest}, got {actual_digest}"
            )
    input_hash = result.get("input_ids_sha256")
    if (
        not isinstance(input_hash, str)
        or len(input_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in input_hash)
    ):
        raise ValueError("invalid input token SHA256 digest")

    expected_server = {
        "tp_size": 8,
        "dp_size": 8,
        "ep_size": 8,
        "enable_dp_attention": True,
        "moe_a2a_backend": "megamoe",
        "mem_fraction_static": 0.835,
        "swa_full_tokens_ratio": 0.075,
    }
    for key, expected in expected_server.items():
        actual = server_info.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(float(actual), expected, abs_tol=1e-9):
                raise ValueError(
                    f"server {key} mismatch: expected {expected}, got {actual!r}"
                )
        elif actual != expected:
            raise ValueError(
                f"server {key} mismatch: expected {expected!r}, got {actual!r}"
            )
    max_running_requests = int(server_info.get("max_running_requests") or 0)
    if max_running_requests < EXPECTED_BATCH_SIZE:
        raise ValueError(
            "server max_running_requests cannot hold the workload in one wave: "
            f"expected at least {EXPECTED_BATCH_SIZE}, got {max_running_requests}"
        )
    graph_config = server_info.get("cuda_graph_config")
    if not isinstance(graph_config, dict):
        raise ValueError("server_info is missing resolved cuda_graph_config")
    decode_graph = graph_config.get("decode")
    if not isinstance(decode_graph, dict):
        raise ValueError("server_info is missing resolved decode CUDA Graph config")
    if decode_graph.get("max_bs") != 544:
        raise ValueError(
            "resolved decode CUDA Graph max_bs mismatch: expected 544, got "
            f"{decode_graph.get('max_bs')!r}"
        )
    if decode_graph.get("backend") == "disabled":
        raise ValueError("decode CUDA Graph was unexpectedly disabled")
    model_path = str(server_info.get("model_path", ""))
    if not model_path.endswith("/DeepSeek-V4-Pro"):
        raise ValueError(f"unexpected model_path: {model_path!r}")
    if server_info.get("speculative_algorithm") is not None:
        raise ValueError("decode benchmark must not silently enable speculative decode")

    return {
        "status": "PASS",
        "result": str(result_path),
        "server_info": str(server_info_path),
        "decode_throughput": decode_throughput,
        "steady_decode_throughput": steady_throughput,
        "decode_tokens": EXPECTED_DECODE_TOKENS,
        "decode_duration": decode_duration,
        "client_decode_throughput": client_decode_throughput,
        "first_token_spread": float(result["first_token_spread"]),
        "server_first_token_spread": server_first_token_spread,
        "early_decode_tokens": early_tokens,
        "max_retractions": 0,
        "input_ids_sha256": input_hash,
        "output_token_ids_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--server-info", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.result, args.server_info), sort_keys=True))


if __name__ == "__main__":
    main()
