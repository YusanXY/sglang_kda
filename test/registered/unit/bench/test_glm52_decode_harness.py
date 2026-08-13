import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "glm52_decode"))
from validate_decode_result import validate  # noqa: E402


def _server_info():
    return {
        "model_path": "/var/b300-shared/models/GLM-5.2-FP8",
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
        "moe_runner_backend": "auto",
        "cuda_graph_config": {"decode": {"backend": "full", "max_bs": 544}},
        "speculative_algorithm": None,
        "glm52_model_fingerprint": {
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
            "weight_quantization": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            },
        },
        "glm52_decode_runtime_config": {
            "mega_moe_kernel_checkpoint_eligible": False,
            "effective_fp8_routed_moe_runner": "triton",
            "shared_expert_parallelism": "tp8",
        },
    }


def _result(cached_tokens=99_999):
    vectors = [[index] * 1000 for index in range(64)]
    hashes = [
        hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for ids in vectors
    ]
    return {
        "batch_size": 64,
        "input_len": 100_000,
        "input_len_min": 100_000,
        "input_len_max": 100_000,
        "input_tokens_total": 6_400_000,
        "output_len": 1_000,
        "stream_interval": 64,
        "latency": 110.0,
        "first_ttft": 10.0,
        "last_ttft": 10.5,
        "server_first_token_ts_min": 1000.0,
        "server_finished_ts_max": 1100.0,
        "decode_duration": 100.0,
        "decode_tokens": 63_936,
        "decode_throughput": 639.36,
        "steady_decode_throughput": 640.0,
        "completed_requests": 64,
        "max_retractions": 0,
        "dp_rank_counts": [8] * 8,
        "cached_tokens_per_request": [cached_tokens] * 64,
        "input_ids_sha256": "a" * 64,
        "output_token_ids_sha256": hashes,
        "output_token_ids": vectors,
    }


def test_decode_replay_requires_full_cached_context(tmp_path):
    result_path = tmp_path / "result.jsonl"
    info_path = tmp_path / "server_info.json"
    info_path.write_text(json.dumps(_server_info()), encoding="utf-8")
    result_path.write_text(json.dumps(_result()) + "\n", encoding="utf-8")

    checked = validate(
        result_path, info_path, require_cached_context=True
    )
    assert checked["cached_tokens_per_request"] == [99_999] * 64

    result_path.write_text(json.dumps(_result(99_998)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="99,999 cached prompt tokens"):
        validate(result_path, info_path, require_cached_context=True)
