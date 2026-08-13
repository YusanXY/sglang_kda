import hashlib
import json
import sys
from pathlib import Path

import pytest

from sglang.benchmark.one_batch_server import (
    BenchArgs,
    calculate_decode_only_metrics,
    validate_profile_cli_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dsv4_pro_decode"))
from validate_decode_result import validate  # noqa: E402
from compare_decode_outputs import compare_vectors  # noqa: E402


def test_first_token_timestamp_survives_coalesced_first_chunk():
    source = (REPO_ROOT / "python/sglang/srt/managers/tokenizer_manager.py").read_text(
        encoding="utf-8"
    )
    first_token_set = "if state.time_stats.first_token_time == 0.0:"
    persistent_export = (
        'meta_info["first_token_ts"] = convert_time_to_realtime(\n'
        "                state.time_stats.first_token_time\n"
        "            )"
    )
    assert first_token_set in source
    assert persistent_export in source
    assert source.index(persistent_export) > source.index(first_token_set)


def test_decode_only_metrics_count_exact_post_first_tokens():
    metrics = calculate_decode_only_metrics(
        batch_size=64,
        output_len=1000,
        latency=112.0,
        first_ttft=12.0,
        last_ttft=12.5,
        decode_tokens_before_last_ttft=32,
    )
    assert metrics == pytest.approx(
        (100.0, 64 * 999, 639.36, 99.5, 64 * 999 - 32, (64 * 999 - 32) / 99.5)
    )


def test_output_vector_comparison_reports_exact_mismatch_position():
    reference = [[index] * 1000 for index in range(64)]
    candidate = [list(ids) for ids in reference]
    candidate[7][321] = -1
    summary = compare_vectors(reference, candidate)
    assert summary["exact_request_matches"] == 63
    assert summary["matching_token_positions"] == 63_999
    assert summary["first_mismatch_by_request"][7] == {
        "position": 321,
        "reference": 7,
        "candidate": -1,
    }
def test_decode_profile_contract_is_fail_closed():
    base = BenchArgs(
        profile=True,
        profile_decode_after_first_token=True,
        profile_activities=("CUDA_PROFILER",),
        profile_steps=200,
    )
    validate_profile_cli_contract(base)

    with pytest.raises(ValueError, match="--profile-only requires --profile"):
        validate_profile_cli_contract(BenchArgs(profile_only=True))

    for broken in (
        BenchArgs(profile_decode_after_first_token=True),
        BenchArgs(
            profile=True,
            profile_decode_after_first_token=True,
            profile_activities=("GPU",),
        ),
        BenchArgs(
            profile=True,
            profile_decode_after_first_token=True,
            profile_activities=("CUDA_PROFILER",),
            profile_by_stage=True,
        ),
        BenchArgs(
            profile=True,
            profile_decode_after_first_token=True,
            profile_activities=("CUDA_PROFILER",),
            profile_steps=0,
        ),
    ):
        with pytest.raises(ValueError):
            validate_profile_cli_contract(broken)


def test_strict_validator_accepts_only_target_contract(tmp_path):
    output_token_ids = [[index] * 1000 for index in range(64)]
    output_token_ids_sha256 = [
        hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for ids in output_token_ids
    ]
    result = {
        "batch_size": 64,
        "input_len": 100000,
        "input_len_min": 100000,
        "input_len_max": 100000,
        "input_tokens_total": 6400000,
        "output_len": 1000,
        "stream_interval": 64,
        "latency": 110.0,
        "first_ttft": 10.0,
        "last_ttft": 10.5,
        "first_token_spread": 0.5,
        "server_first_token_spread": 0.1,
        "server_first_token_ts_min": 1000.0,
        "server_finished_ts_max": 1100.0,
        "decode_duration": 100.0,
        "decode_tokens": 63936,
        "decode_tokens_before_last_ttft": 32,
        "decode_throughput": 639.36,
        "client_decode_duration": 100.0,
        "client_decode_throughput": 639.36,
        "steady_decode_duration": 99.5,
        "steady_decode_tokens": 63904,
        "steady_decode_throughput": 63904 / 99.5,
        "completed_requests": 64,
        "max_retractions": 0,
        "input_ids_sha256": "b" * 64,
        "output_token_ids_sha256": output_token_ids_sha256,
        "output_token_ids": output_token_ids,
    }
    server_info = {
        "model_path": "/var/b300-shared/models/DeepSeek-V4-Pro",
        "tp_size": 8,
        "dp_size": 8,
        "ep_size": 8,
        "enable_dp_attention": True,
        "moe_a2a_backend": "megamoe",
        "mem_fraction_static": 0.835,
        "swa_full_tokens_ratio": 0.075,
        "max_running_requests": 64,
        "cuda_graph_config": {
            "decode": {"backend": "full", "max_bs": 544},
        },
        "speculative_algorithm": None,
    }
    result_path = tmp_path / "result.jsonl"
    info_path = tmp_path / "server_info.json"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    info_path.write_text(json.dumps(server_info), encoding="utf-8")
    assert validate(result_path, info_path)["status"] == "PASS"

    server_info["dp_size"] = 4
    info_path.write_text(json.dumps(server_info), encoding="utf-8")
    with pytest.raises(ValueError, match="dp_size mismatch"):
        validate(result_path, info_path)
