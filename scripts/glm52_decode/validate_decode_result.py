#!/usr/bin/env python3
"""Fail closed on an ambiguous GLM-5.2 Req64 decode sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_BATCH_SIZE = 64
EXPECTED_INPUT_LEN = 100_000
EXPECTED_OUTPUT_LEN = 1_000
EXPECTED_STREAM_INTERVAL = 64
EXPECTED_DECODE_TOKENS = EXPECTED_BATCH_SIZE * (EXPECTED_OUTPUT_LEN - 1)
EXPECTED_DP_RANK_COUNTS = [8] * 8


def _read_single_result(path: Path) -> dict:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one JSONL row, got {len(rows)}")
    row = json.loads(rows[0])
    if not isinstance(row, dict):
        raise ValueError(f"{path} row must be an object")
    return row


def _finite(row: dict, key: str, *, positive: bool = True) -> float:
    value = float(row[key])
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{key} must be finite and > 0, got {value!r}")
    return value


def _validate_output_vectors(
    row: dict, expected_output_len: int = EXPECTED_OUTPUT_LEN
) -> tuple[str, list[str]]:
    input_hash = row.get("input_ids_sha256")
    if not isinstance(input_hash, str) or len(input_hash) != 64:
        raise ValueError("missing input token SHA256")
    hashes = row.get("output_token_ids_sha256")
    vectors = row.get("output_token_ids")
    if not isinstance(hashes, list) or len(hashes) != EXPECTED_BATCH_SIZE:
        raise ValueError("missing per-request output token hashes")
    if not isinstance(vectors, list) or len(vectors) != EXPECTED_BATCH_SIZE:
        raise ValueError("missing per-request output token vectors")
    for index, (token_ids, digest) in enumerate(zip(vectors, hashes)):
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != expected_output_len
            or any(not isinstance(token_id, int) for token_id in token_ids)
        ):
            raise ValueError(f"invalid output token vector for request {index}")
        actual = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual != digest:
            raise ValueError(f"output token digest mismatch for request {index}")
    return input_hash, hashes


def _validate_server(server: dict, expected_moe_runner: str) -> None:
    expected = {
        "tp_size": 8,
        "dp_size": 8,
        "ep_size": 8,
        "enable_dp_attention": True,
        "moe_a2a_backend": "megamoe",
        "mem_fraction_static": 0.835,
        "swa_full_tokens_ratio": 0.075,
        "attention_backend": "dsa",
        "dsa_prefill_backend": "trtllm",
        "dsa_decode_backend": "trtllm",
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 2048,
        "moe_runner_backend": expected_moe_runner,
    }
    for key, wanted in expected.items():
        actual = server.get(key)
        if isinstance(wanted, float):
            if actual is None or not math.isclose(float(actual), wanted, abs_tol=1e-9):
                raise ValueError(f"server {key}: expected {wanted}, got {actual!r}")
        elif actual != wanted:
            raise ValueError(f"server {key}: expected {wanted!r}, got {actual!r}")
    if not str(server.get("model_path", "")).endswith("/GLM-5.2-FP8"):
        raise ValueError(f"unexpected model_path={server.get('model_path')!r}")
    if server.get("speculative_algorithm") is not None:
        raise ValueError("speculative decode is outside the strict workload")

    graph = server.get("cuda_graph_config", {}).get("decode", {})
    if graph.get("backend") == "disabled" or graph.get("max_bs") != 544:
        raise ValueError(f"decode CUDA Graph contract mismatch: {graph!r}")

    fp = server.get("glm52_model_fingerprint")
    if not isinstance(fp, dict):
        raise ValueError("missing GLM-5.2 model fingerprint")
    expected_fp = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "mlp_layer_type_counts": {"dense": 3, "sparse": 75},
        "indexer_type_counts": {"full": 21, "shared": 57},
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "num_attention_heads": 64,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
    }
    for key, wanted in expected_fp.items():
        if fp.get(key) != wanted:
            raise ValueError(f"fingerprint {key}: expected {wanted!r}, got {fp.get(key)!r}")
    quant = fp.get("weight_quantization", {})
    if quant.get("quant_method") != "fp8" or quant.get("fmt") != "e4m3":
        raise ValueError(f"unexpected weight quantization: {quant!r}")
    if quant.get("weight_block_size") != [128, 128]:
        raise ValueError(f"unexpected FP8 weight block: {quant!r}")

    runtime = server.get("glm52_decode_runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("missing GLM-5.2 decode runtime config")
    if runtime.get("mega_moe_kernel_checkpoint_eligible") is not False:
        raise ValueError(f"FP8 checkpoint unexpectedly MegaMoE-eligible: {runtime!r}")
    if runtime.get("custom_all_reduce_impl") != "legacy":
        raise ValueError(
            "formal GLM-5.2 decode requires stable legacy custom AR, got "
            f"{runtime!r}"
        )
    expected_effective = "triton" if expected_moe_runner == "auto" else "deep_gemm"
    if runtime.get("effective_fp8_routed_moe_runner") != expected_effective:
        raise ValueError(
            "effective routed MoE runner mismatch: expected "
            f"{expected_effective!r}, got {runtime!r}"
        )


def validate(
    result_path: Path,
    server_info_path: Path,
    expected_moe_runner: str = "auto",
    require_cached_context: bool = False,
    expected_shared_expert_parallelism: str = "tp8",
    context_build: bool = False,
) -> dict:
    row = _read_single_result(result_path)
    server = json.loads(server_info_path.read_text(encoding="utf-8"))
    _validate_server(server, expected_moe_runner)
    runtime = server["glm52_decode_runtime_config"]
    if runtime.get("shared_expert_parallelism") != expected_shared_expert_parallelism:
        raise ValueError(
            "shared expert parallelism mismatch: expected "
            f"{expected_shared_expert_parallelism!r}, got {runtime!r}"
        )

    expected_output_len = 1 if context_build else EXPECTED_OUTPUT_LEN
    expected_decode_tokens = 0 if context_build else EXPECTED_DECODE_TOKENS
    expected_row = {
        "batch_size": EXPECTED_BATCH_SIZE,
        "input_len": EXPECTED_INPUT_LEN,
        "input_len_min": EXPECTED_INPUT_LEN,
        "input_len_max": EXPECTED_INPUT_LEN,
        "input_tokens_total": EXPECTED_BATCH_SIZE * EXPECTED_INPUT_LEN,
        "output_len": expected_output_len,
        "stream_interval": EXPECTED_STREAM_INTERVAL,
        "decode_tokens": expected_decode_tokens,
        "completed_requests": EXPECTED_BATCH_SIZE,
        "max_retractions": 0,
        "dp_rank_counts": EXPECTED_DP_RANK_COUNTS,
        "requested_dp_ranks": [rank for _ in range(8) for rank in range(8)],
    }
    for key, wanted in expected_row.items():
        if row.get(key) != wanted:
            raise ValueError(f"{key}: expected {wanted!r}, got {row.get(key)!r}")

    cached_tokens = row.get("cached_tokens_per_request")
    if require_cached_context:
        if (
            not isinstance(cached_tokens, list)
            or len(cached_tokens) != EXPECTED_BATCH_SIZE
            or any(
                not isinstance(value, int) or value < EXPECTED_INPUT_LEN - 1
                for value in cached_tokens
            )
        ):
            raise ValueError(
                "decode replay requires at least 99,999 cached prompt tokens "
                f"for every request, got {cached_tokens!r}"
            )

    first_ttft = _finite(row, "first_ttft")
    last_ttft = _finite(row, "last_ttft")
    latency = _finite(row, "latency")
    if not first_ttft <= last_ttft <= latency:
        raise ValueError("invalid first-token/finish timing order")
    server_start = _finite(row, "server_first_token_ts_min")
    server_end = _finite(row, "server_finished_ts_max")
    decode_duration = _finite(row, "decode_duration", positive=not context_build)
    decode_throughput = _finite(row, "decode_throughput", positive=not context_build)
    if not math.isclose(decode_duration, server_end - server_start, abs_tol=2e-4):
        raise ValueError("decode duration does not match server timestamps")
    if context_build:
        if decode_throughput != 0:
            raise ValueError(
                f"context-build decode throughput must be zero, got {decode_throughput}"
            )
    elif not math.isclose(
        decode_throughput, EXPECTED_DECODE_TOKENS / decode_duration, abs_tol=0.02
    ):
        raise ValueError("decode throughput numerator or duration is inconsistent")
    input_hash, hashes = _validate_output_vectors(row, expected_output_len)
    return {
        "status": "PASS",
        "result": str(result_path),
        "decode_throughput": decode_throughput,
        "steady_decode_throughput": _finite(
            row, "steady_decode_throughput", positive=not context_build
        ),
        "decode_duration": decode_duration,
        "input_ids_sha256": input_hash,
        "output_token_ids_sha256": hashes,
        "cached_tokens_per_request": cached_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--server-info", type=Path, required=True)
    parser.add_argument(
        "--expected-moe-runner", choices=("auto", "deep_gemm"), default="auto"
    )
    parser.add_argument("--require-cached-context", action="store_true")
    parser.add_argument(
        "--context-build",
        action="store_true",
        help="Validate the untimed Req64/100K prefix build with output_len=1.",
    )
    parser.add_argument(
        "--expected-shared-expert-parallelism",
        choices=("tp1", "tp8"),
        default="tp8",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.result,
                args.server_info,
                args.expected_moe_runner,
                require_cached_context=args.require_cached_context,
                expected_shared_expert_parallelism=(
                    args.expected_shared_expert_parallelism
                ),
                context_build=args.context_build,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
