"""
Benchmark the latency of running a single batch with a server.

This script launches a server and uses the HTTP interface.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

Usage:
python3 -m sglang.benchmark.one_batch_server --model meta-llama/Meta-Llama-3.1-8B --batch-size 1 16 64 --input-len 1024 --output-len 8

python3 -m sglang.benchmark.one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8
python3 -m sglang.benchmark.one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --show-report --profile --profile-by-stage
python3 -m sglang.benchmark.one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --result-filename results.jsonl --profile
"""

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import random
import re
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
from pydantic import BaseModel
from tabulate import tabulate
from transformers import AutoProcessor, PreTrainedTokenizer

from sglang.benchmark.datasets import get_dataset
from sglang.benchmark.endpoint import acquire_endpoint
from sglang.benchmark.utils import get_processor, get_tokenizer
from sglang.profiler import run_profile
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_blackwell
from sglang.test.nightly_bench_utils import save_results_as_pydantic_models
from sglang.test.test_utils import is_in_ci, write_github_step_summary

DEFAULT_TIMEOUT = 600


def get_cache_tokens_from_metrics(url: str) -> Optional[tuple]:
    """
    Get cached_tokens_total and prompt_tokens_total from Prometheus /metrics endpoint.
    Returns (cached_tokens_total, prompt_tokens_total) or None if metrics are not available.
    """
    try:
        response = requests.get(url + "/metrics", timeout=5)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            return None

        # Parse Prometheus text format
        # Looking for: sglang:cached_tokens_total{...} <value>
        #              sglang:prompt_tokens_total{...} <value>
        cached_tokens_total = 0.0
        prompt_tokens_total = 0.0

        for line in response.text.split("\n"):
            if line.startswith("sglang:cached_tokens_total{"):
                match = re.search(
                    r"sglang:cached_tokens_total\{[^}]*\}\s+([\d.eE+-]+)", line
                )
                if match:
                    cached_tokens_total += float(match.group(1))
            elif line.startswith("sglang:prompt_tokens_total{"):
                match = re.search(
                    r"sglang:prompt_tokens_total\{[^}]*\}\s+([\d.eE+-]+)", line
                )
                if match:
                    prompt_tokens_total += float(match.group(1))

        return (cached_tokens_total, prompt_tokens_total)
    except Exception as e:
        print(f"Warning: Failed to get cache tokens from metrics: {e}")
        return None


def calculate_cache_hit_rate(
    before: Optional[tuple], after: Optional[tuple]
) -> Optional[float]:
    """
    Calculate cache hit rate from before/after metrics snapshots.
    Returns cached_tokens_delta / prompt_tokens_delta for the benchmark run.
    """
    if before is None or after is None:
        return None

    cached_delta = after[0] - before[0]
    prompt_delta = after[1] - before[1]

    if prompt_delta > 0:
        return cached_delta / prompt_delta
    return None


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    client_stream_interval: int = 1
    input_len_step_percentage: float = 0.0
    base_url: str = ""
    local_tokenizer_path: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_only: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_start_step: Optional[int] = None
    profile_steps: int = 5
    profile_stop_after_request: bool = False
    profile_decode_after_first_token: bool = False
    save_output_token_ids: bool = False
    profile_by_stage: bool = False
    profile_prefix: Optional[str] = None
    profile_output_dir: Optional[str] = None
    dataset_path: str = ""
    dataset_name: str = "random"
    fixed_prompt_file: str = ""
    apply_chat_template: bool = False
    gsp_num_groups: int = 1
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256
    parallel_batch: bool = False
    result_filename: str = "result.jsonl"
    pydantic_result_filename: Optional[str] = None
    append_to_github_summary: bool = True
    seed: int = 42
    cache_hit_rate: float = 0.0
    share_cached_prefix_across_batch: bool = False
    warmup_cached_prefill_shape: bool = False
    backend: str = "sglang"
    fake_prefill: bool = False
    server_args_for_metrics: Optional[List[str]] = None
    lora_name: Optional[List[str]] = None
    lora_request_distribution: str = "uniform"
    lora_zipf_alpha: float = 1.1
    enable_multi_batch: bool = False
    request_timeout: float = DEFAULT_TIMEOUT
    preserve_prefix_cache: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--client-stream-interval",
            type=int,
            default=BenchArgs.client_stream_interval,
        )
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument(
            "--local-tokenizer-path",
            type=str,
            default=BenchArgs.local_tokenizer_path,
            help=(
                "Local tokenizer path to use when benchmarking an external "
                "SGLang server via --base-url. Defaults to the tokenizer path "
                "reported by /server_info."
            ),
        )
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument(
            "--request-timeout",
            type=float,
            default=BenchArgs.request_timeout,
            help=(
                "HTTP request timeout in seconds. Increase this for the first "
                "cold-compilation request of very large models."
            ),
        )
        parser.add_argument(
            "--preserve-prefix-cache",
            action="store_true",
            help=(
                "Do not flush the server prefix cache before the measured case. "
                "This is intended for decode-only replay after an identical "
                "long-context request has populated and validated the KV cache."
            ),
        )
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-only",
            action="store_true",
            help=(
                "Run only the profiled case instead of first executing an "
                "ordinary benchmark copy. Requires --profile. This is useful "
                "for very large prompts and external Nsys capture."
            ),
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=("CPU", "GPU"),
            choices=["CPU", "GPU", "CUDA_PROFILER", "XPU"],
            help=(
                "Profiler activities: CPU, GPU, XPU, CUDA_PROFILER. "
                "CPU/GPU/XPU use torch profiler; CUDA_PROFILER controls an "
                "external profiler capture range."
            ),
        )
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=BenchArgs.profile_start_step,
            help="Start profiling after this many forward steps. Useful for warmup.",
        )
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument(
            "--profile-stop-after-request",
            action="store_true",
            help=(
                "Explicitly stop and flush the profiler after the measured request. "
                "This is required for prefill-only runs that have no later forward "
                "step to trigger the automatic num_steps stop condition."
            ),
        )
        parser.add_argument(
            "--profile-decode-after-first-token",
            action="store_true",
            help=(
                "Arm the profiler only after every request in the batch has "
                "emitted its first token. This excludes prefill from external "
                "CUDA_PROFILER/Nsys captures and is intended for decode-only "
                "analysis. Requires --profile, the sglang backend, and cannot "
                "be combined with --profile-by-stage or --profile-start-step."
            ),
        )
        parser.add_argument(
            "--save-output-token-ids",
            action="store_true",
            help=(
                "Persist the complete per-request output token vectors in the "
                "JSONL result. SHA256 digests are always saved; use this only "
                "when exact cross-version correctness comparison is required."
            ),
        )
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--profile-prefix",
            type=str,
            default=BenchArgs.profile_prefix,
        )
        parser.add_argument(
            "--profile-output-dir",
            type=str,
            default=BenchArgs.profile_output_dir,
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=BenchArgs.dataset_path,
            help="Path to the dataset.",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=BenchArgs.dataset_name,
            choices=["mmmu", "random", "random-ids", "generated-shared-prefix"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--fixed-prompt-file",
            type=str,
            default=BenchArgs.fixed_prompt_file,
            help="Use this file's prompt for every request in the batch, "
            "bypassing --dataset-name.",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Encode the prompt as a single user message through the "
            "model's chat template. Requires --fixed-prompt-file.",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of shared prefix groups. batch_size requests are distributed across groups.",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="Length of the shared system prompt in tokens per group.",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Length of the unique question suffix in tokens per request.",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Output length in tokens for generated-shared-prefix requests.",
        )
        parser.add_argument("--parallel-batch", action="store_true")
        parser.add_argument(
            "--result-filename",
            type=str,
            default=BenchArgs.result_filename,
            help="Store the results line by line in the JSON Line format to this file.",
        )
        parser.add_argument(
            "--pydantic-result-filename",
            type=str,
            default=BenchArgs.pydantic_result_filename,
            help="Store the results as pydantic models in the JSON format to this file.",
        )
        parser.add_argument(
            "--no-append-to-github-summary",
            action="store_false",
            dest="append_to_github_summary",
            help="Disable appending the output of this run to github ci summary",
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
        parser.add_argument(
            "--cache-hit-rate",
            type=float,
            default=BenchArgs.cache_hit_rate,
            help="Cache hit rate for benchmarking (0.0-1.0). "
            "0.0 means no cache hits (flush all), 0.4 means 40%% of input tokens are cached.",
        )
        parser.add_argument(
            "--share-cached-prefix-across-batch",
            action="store_true",
            help=(
                "Make the cache-hit prefix identical across requests while "
                "keeping each request's uncached suffix unchanged. This is "
                "useful for deterministic multi-request cached-prefill shapes."
            ),
        )
        parser.add_argument(
            "--warmup-cached-prefill-shape",
            action="store_true",
            help=(
                "After populating the prefix cache, run one full cached-prefill "
                "shape with a deliberately different first suffix token. The "
                "measured request still matches only the requested prefix."
            ),
        )
        parser.add_argument(
            "--backend",
            type=str,
            default=BenchArgs.backend,
            choices=["sglang", "vllm"],
            help="Backend server type (sglang or vllm).",
        )
        parser.add_argument(
            "--fake-prefill",
            action="store_true",
            default=BenchArgs.fake_prefill,
            help="Enable fake prefill mode for decode-only benchmarking. "
            "Use with a decode server running --disaggregation-transfer-backend fake "
            "to benchmark pure decode performance without a real prefill node.",
        )
        parser.add_argument(
            "--server-args-for-metrics",
            type=str,
            nargs="*",
            default=None,
            help="Server launch arguments to record in metrics output (for tracking configurations).",
        )
        parser.add_argument(
            "--lora-name",
            type=str,
            nargs="*",
            default=BenchArgs.lora_name,
            help="Name(s) of pre-loaded LoRA adapter(s) to apply to the batch "
            "(sent as `lora_path` in the SGLang /generate payload). Requires "
            "the server to be launched with --enable-lora and --lora-paths "
            "<name>=<path> for every name listed here. Pass one name to apply "
            "a single adapter to every prompt, or multiple names to sample a "
            "per-prompt adapter per --lora-request-distribution.",
        )
        parser.add_argument(
            "--lora-request-distribution",
            type=str,
            default=BenchArgs.lora_request_distribution,
            choices=["uniform", "distinct", "skewed"],
            help="How to sample a LoRA adapter per prompt when more than one "
            "is listed in --lora-name. Mirrors serving.py. "
            "'uniform' picks uniformly at random, 'distinct' round-robins so "
            "consecutive prompts get different adapters, 'skewed' samples "
            "from a Zipf distribution over --lora-name (alpha controls the "
            "skew; see --lora-zipf-alpha).",
        )
        parser.add_argument(
            "--lora-zipf-alpha",
            type=float,
            default=BenchArgs.lora_zipf_alpha,
            help="Zipf exponent for 'skewed' LoRA sampling: the number of "
            "requests to adapter i is alpha times the number to adapter i+1. "
            "Must be > 1. Only used when --lora-request-distribution=skewed.",
        )
        parser.add_argument(
            "--enable-multi-batch",
            action="store_true",
            help=(
                "Allow --batch-size to exceed the server's "
                "effective_max_running_requests_per_dp * dp_size. The surplus "
                "requests are queued by the scheduler and promoted as slots "
                "free, so the batch is served as multiple sequential batches "
                "at the running-batch cap. Useful for stabilizing throughput "
                "measurements: driving more total prompts through a "
                "fixed running batch amortizes per-request prefill and "
                "first-step transients into steady-state decode. "
                "NOTE: only `overall_throughput` (= total_tokens / wall_time) "
                "is meaningful in this mode; input_throughput, "
                "output_throughput, last_ttft, and ITL assume one-shot "
                "batching and will be misleading."
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


class BenchOneCaseResult(BaseModel):
    run_name: str
    batch_size: int
    input_len: int
    input_len_min: int
    input_len_max: int
    input_tokens_total: int
    output_len: int
    stream_interval: int
    latency: float
    input_throughput: float
    output_throughput: float
    first_ttft: float
    first_token_spread: float
    server_first_token_spread: float
    server_first_token_ts_min: float
    server_finished_ts_max: float
    decode_duration: float
    decode_tokens: int
    decode_tokens_before_last_ttft: int
    decode_throughput: float
    client_decode_duration: float
    client_decode_throughput: float
    steady_decode_duration: float
    steady_decode_tokens: int
    steady_decode_throughput: float
    completed_requests: int
    max_retractions: int
    input_ids_sha256: str
    output_token_ids_sha256: Optional[List[str]]
    output_token_ids: Optional[List[List[int]]]
    dp_ranks: Optional[List[int]]
    dp_rank_counts: Optional[List[int]]
    cached_tokens_per_request: Optional[List[int]]
    overall_throughput: float
    last_ttft: float
    last_gen_throughput: float
    acc_length: float
    cache_hit_rate: Optional[float] = None
    profile_link: Optional[str] = None

    def dump_to_jsonl(self, result_filename: str):
        with open(result_filename, "a") as fout:
            res = {
                "run_name": self.run_name,
                "batch_size": self.batch_size,
                "input_len": self.input_len,
                "input_len_min": self.input_len_min,
                "input_len_max": self.input_len_max,
                "input_tokens_total": self.input_tokens_total,
                "output_len": self.output_len,
                "stream_interval": self.stream_interval,
                "latency": round(self.latency, 4),
                "input_throughput": round(self.input_throughput, 2),
                "output_throughput": round(self.output_throughput, 2),
                "first_ttft": round(self.first_ttft, 4),
                "first_token_spread": round(self.first_token_spread, 4),
                "server_first_token_spread": round(
                    self.server_first_token_spread, 4
                ),
                "server_first_token_ts_min": round(
                    self.server_first_token_ts_min, 6
                ),
                "server_finished_ts_max": round(self.server_finished_ts_max, 6),
                "decode_duration": round(self.decode_duration, 4),
                "decode_tokens": self.decode_tokens,
                "decode_tokens_before_last_ttft": (
                    self.decode_tokens_before_last_ttft
                ),
                "decode_throughput": round(self.decode_throughput, 2),
                "client_decode_duration": round(self.client_decode_duration, 4),
                "client_decode_throughput": round(
                    self.client_decode_throughput, 2
                ),
                "steady_decode_duration": round(self.steady_decode_duration, 4),
                "steady_decode_tokens": self.steady_decode_tokens,
                "steady_decode_throughput": round(
                    self.steady_decode_throughput, 2
                ),
                "completed_requests": self.completed_requests,
                "max_retractions": self.max_retractions,
                "input_ids_sha256": self.input_ids_sha256,
                "output_token_ids_sha256": self.output_token_ids_sha256,
                "output_token_ids": self.output_token_ids,
                "dp_ranks": self.dp_ranks,
                "dp_rank_counts": self.dp_rank_counts,
                "cached_tokens_per_request": self.cached_tokens_per_request,
                "overall_throughput": round(self.overall_throughput, 2),
                "last_ttft": round(self.last_ttft, 4),
                "last_gen_throughput": round(self.last_gen_throughput, 2),
                "acc_length": round(self.acc_length, 2),
                "cache_hit_rate": (
                    round(self.cache_hit_rate, 4)
                    if self.cache_hit_rate is not None
                    else None
                ),
            }
            fout.write(json.dumps(res) + "\n")


def calculate_decode_only_metrics(
    *,
    batch_size: int,
    output_len: int,
    latency: float,
    first_ttft: float,
    last_ttft: float,
    decode_tokens_before_last_ttft: int = 0,
) -> Tuple[float, int, float, float, int, float]:
    """Return strict full-decode and post-last-first-token metrics.

    The existing ``output_throughput`` metric intentionally keeps its legacy
    definition for compatibility.  It divides ``batch_size * output_len`` by
    the time after the last request receives its first token, even though the
    first token was produced before that boundary.  A decode-only workload
    must instead count exactly ``output_len - 1`` tokens per request.  The
    full-decode rate brackets every decode token from the earliest first-token
    event.  The steady rate starts at the last first-token event and removes
    any decode tokens that earlier requests produced before that boundary.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if output_len <= 0:
        raise ValueError(f"output_len must be positive, got {output_len}")
    if first_ttft <= 0 or last_ttft <= 0:
        raise ValueError(
            "first_ttft/last_ttft was not recorded; every request must emit a first-token "
            "stream event before decode-only metrics can be computed"
        )
    if first_ttft > last_ttft:
        raise ValueError(
            f"first_ttft ({first_ttft}) must not exceed last_ttft ({last_ttft})"
        )
    decode_duration = latency - first_ttft
    steady_decode_duration = latency - last_ttft
    if steady_decode_duration < 0:
        raise ValueError(
            f"latency ({latency}) must not be smaller than last_ttft ({last_ttft})"
        )
    decode_tokens = batch_size * (output_len - 1)
    if not 0 <= decode_tokens_before_last_ttft <= decode_tokens:
        raise ValueError(
            "decode_tokens_before_last_ttft must be in [0, "
            f"{decode_tokens}], got {decode_tokens_before_last_ttft}"
        )
    # DP schedulers may stream an early rank's second token before the final
    # rank's first token reaches the client.  Those early tokens are outside
    # the post-last-first-token timing window and must not remain in its
    # numerator.
    steady_decode_tokens = decode_tokens - decode_tokens_before_last_ttft
    if decode_tokens == 0:
        return decode_duration, 0, 0.0, steady_decode_duration, 0, 0.0
    if decode_duration == 0:
        raise ValueError("non-empty decode span has zero duration")
    if steady_decode_tokens and steady_decode_duration == 0:
        raise ValueError("non-empty steady decode span has zero duration")
    steady_decode_throughput = (
        steady_decode_tokens / steady_decode_duration
        if steady_decode_tokens
        else 0.0
    )
    return (
        decode_duration,
        decode_tokens,
        decode_tokens / decode_duration,
        steady_decode_duration,
        steady_decode_tokens,
        steady_decode_throughput,
    )


def _warmup_cache(
    url: str,
    input_ids: List[List[int]],
    input_len: int,
    cache_hit_rate: float,
    dataset_name: str = "random",
    image_data: Optional[List] = None,
    backend: str = "sglang",
    model_name: Optional[str] = None,
    warmup_cached_prefill_shape: bool = False,
    share_cached_prefix_across_batch: bool = False,
):
    """Warm up the cache by sending prefix tokens to populate the radix/prefix cache.

    Args:
        url: Server URL
        input_ids: List of input token id lists
        input_len: Length of input tokens
        cache_hit_rate: Fraction of input tokens to cache (0.0-1.0)
        dataset_name: Name of the dataset (used to determine if image data should be included)
        image_data: Optional image data for VLM models
        backend: Backend server type ("sglang" or "vllm")
        model_name: Model name (required for vllm backend)
    """
    cached_token_len = int(input_len * cache_hit_rate)
    if cached_token_len <= 0:
        return

    print(
        f"Warming up cache with {cache_hit_rate*100:.1f}% hit rate "
        f"({cached_token_len} tokens per request)"
    )
    # Create prefix input_ids for cache warming
    cache_warmup_input_ids = [ids[:cached_token_len] for ids in input_ids]
    if share_cached_prefix_across_batch:
        shared_prefix = cache_warmup_input_ids[0]
        if any(prefix != shared_prefix for prefix in cache_warmup_input_ids[1:]):
            raise ValueError(
                "shared-prefix cache warmup received non-identical prefixes"
            )
        # One request populates the shared radix entry for the entire batch.
        # Sending all copies makes the scheduler manufacture redundant
        # intermediate shapes (for example req15/M61440), which is both extra
        # work and incompatible with strict exact-bucket CUDA Graph execution.
        cache_warmup_input_ids = [shared_prefix]
        print("Shared prefix detected; warming one canonical cache entry")

    if backend == "vllm":
        cache_warmup_payload = {
            "model": model_name,
            "prompt": cache_warmup_input_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        gen_url = url + "/v1/completions"
    else:
        cache_warmup_payload = {
            "input_ids": cache_warmup_input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,  # Minimal output, just to populate cache
                "ignore_eos": True,
            },
            "stream": False,
        }
        if dataset_name == "mmmu" and image_data is not None:
            # include image data in cache warmup
            cache_warmup_payload["image_data"] = image_data
        gen_url = url + "/generate"

    warmup_response = requests.post(
        gen_url,
        json=cache_warmup_payload,
        timeout=DEFAULT_TIMEOUT,
    )
    warmup_response.raise_for_status()
    print("Cache warmup completed")

    if warmup_cached_prefill_shape:
        if cached_token_len >= min(len(ids) for ids in input_ids):
            raise ValueError(
                "--warmup-cached-prefill-shape requires at least one uncached "
                "suffix token per request"
            )
        final_first_suffix_tokens = {
            ids[cached_token_len] for ids in input_ids
        }
        marker = next(
            (
                token
                for ids in input_ids
                for token in ids[:cached_token_len]
                if token not in final_first_suffix_tokens
            ),
            None,
        )
        if marker is None:
            raise ValueError(
                "could not find a valid prefix token distinct from every "
                "request's first suffix token"
            )
        shape_warmup_input_ids = []
        for ids in input_ids:
            warm_ids = list(ids)
            warm_ids[cached_token_len] = marker
            shape_warmup_input_ids.append(warm_ids)
        shape_warmup_payload = dict(cache_warmup_payload)
        if backend == "vllm":
            shape_warmup_payload["prompt"] = shape_warmup_input_ids
        else:
            shape_warmup_payload["input_ids"] = shape_warmup_input_ids
        print(
            "Warming exact cached-prefill shape with a non-matching suffix "
            f"({len(input_ids)} requests, "
            f"{sum(len(ids) - cached_token_len for ids in input_ids)} new tokens)"
        )
        shape_response = requests.post(
            gen_url,
            json=shape_warmup_payload,
            timeout=DEFAULT_TIMEOUT,
        )
        shape_response.raise_for_status()
        print("Cached-prefill shape warmup completed")


def _flush_cache_with_retry(url: str, endpoint: str, max_retries: int = 3):
    """Post to a cache flush endpoint with retries on failure."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url + endpoint, timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                return
            if attempt >= max_retries - 1:
                response.raise_for_status()
        except requests.RequestException:
            if attempt >= max_retries - 1:
                raise
        time.sleep(2)


@lru_cache(maxsize=None)
def _load_hf_config(name_or_path: str):
    if not name_or_path:
        return None

    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception as e:
        print(
            f"Warning: could not load config for {name_or_path!r} ({e}); "
            "falling back to the HF chat template for --apply-chat-template."
        )
        return None


def _encode_fixed_prompt(
    tok_inner, prompt_text: str, apply_chat_template: bool
) -> List[int]:
    if not apply_chat_template:
        return tok_inner.encode(prompt_text)

    from sglang.srt.entrypoints.openai.chat_encoding import (
        encode_simple_chat,
        resolve_chat_encoding_spec,
    )

    hf_config = _load_hf_config(getattr(tok_inner, "name_or_path", "") or "")
    spec = (
        resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=tok_inner)
        if hf_config is not None
        else None
    )
    return encode_simple_chat(
        tokenizer=tok_inner,
        spec=spec,
        messages=[{"role": "user", "content": prompt_text}],
    )


def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    stream_interval: int,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer: PreTrainedTokenizer | AutoProcessor,
    profile: bool = False,
    profile_activities: Tuple[str] = ("CPU", "GPU"),
    profile_start_step: Optional[int] = None,
    profile_steps: int = BenchArgs.profile_steps,
    profile_stop_after_request: bool = False,
    profile_decode_after_first_token: bool = False,
    profile_by_stage: bool = False,
    profile_prefix: Optional[str] = BenchArgs.profile_prefix,
    profile_output_dir: Optional[str] = BenchArgs.profile_output_dir,
    dataset_name: str = BenchArgs.dataset_name,
    dataset_path: str = BenchArgs.dataset_path,
    parallel_batch: bool = False,
    cache_hit_rate: float = BenchArgs.cache_hit_rate,
    share_cached_prefix_across_batch: bool = False,
    warmup_cached_prefill_shape: bool = False,
    backend: str = "sglang",
    model_name: Optional[str] = None,
    gsp_num_groups: int = BenchArgs.gsp_num_groups,
    gsp_system_prompt_len: int = BenchArgs.gsp_system_prompt_len,
    gsp_question_len: int = BenchArgs.gsp_question_len,
    gsp_output_len: int = BenchArgs.gsp_output_len,
    fake_prefill: bool = False,
    lora_name: Optional[List[str]] = None,
    lora_request_distribution: str = BenchArgs.lora_request_distribution,
    lora_zipf_alpha: float = BenchArgs.lora_zipf_alpha,
    fixed_prompt_file: str = "",
    apply_chat_template: bool = False,
    save_output_token_ids: bool = False,
    seed: int = BenchArgs.seed,
    preserve_prefix_cache: bool = False,
):
    if profile_decode_after_first_token:
        if not profile:
            raise ValueError(
                "--profile-decode-after-first-token requires --profile"
            )
        if backend != "sglang":
            raise ValueError(
                "--profile-decode-after-first-token only supports the sglang backend"
            )
        if profile_by_stage:
            raise ValueError(
                "--profile-decode-after-first-token cannot be combined with "
                "--profile-by-stage"
            )
        if profile_start_step is not None:
            raise ValueError(
                "--profile-decode-after-first-token cannot be combined with "
                "--profile-start-step"
            )
    if not preserve_prefix_cache:
        if backend == "vllm":
            # You need to have export VLLM_SERVER_DEV_MODE=1 in your environment to use this endpoint.
            _flush_cache_with_retry(url, "/reset_prefix_cache")
        else:
            _flush_cache_with_retry(url, "/flush_cache")

    if fixed_prompt_file:
        tok_inner = getattr(tokenizer, "tokenizer", tokenizer)
        with open(fixed_prompt_file) as f:
            prompt_ids = _encode_fixed_prompt(tok_inner, f.read(), apply_chat_template)
        input_ids = [list(prompt_ids) for _ in range(batch_size)]
        input_len = len(prompt_ids)
        image_data = None
    else:
        # Load input token ids via benchmark.datasets.get_dataset
        supported_datasets = ("random", "random-ids", "mmmu", "generated-shared-prefix")
        if dataset_name not in supported_datasets:
            raise ValueError(
                f"Unsupported dataset for batch benchmark: {dataset_name}. "
                f"Supported: {supported_datasets}"
            )

        actual_gsp_groups = min(gsp_num_groups, batch_size)
        dataset_args = SimpleNamespace(
            dataset_name=dataset_name,
            num_prompts=batch_size,
            random_input_len=input_len,
            random_output_len=output_len,
            random_range_ratio=1.0,
            dataset_path=dataset_path,
            tokenize_prompt=dataset_name not in ("mmmu", "generated-shared-prefix"),
            backend=backend,
            seed=seed,
            gsp_num_groups=actual_gsp_groups,
            gsp_prompts_per_group=(batch_size + actual_gsp_groups - 1)
            // actual_gsp_groups,
            gsp_system_prompt_len=gsp_system_prompt_len,
            gsp_question_len=gsp_question_len,
            gsp_output_len=gsp_output_len,
            # The generated-shared-prefix dataset's from_args requires these; the
            # batch-bench path only ever uses the uniform group distribution.
            gsp_group_distribution="uniform",
            gsp_zipf_alpha=None,
        )
        tok_inner = getattr(tokenizer, "tokenizer", tokenizer)
        dataset_model_id = model_name or getattr(tok_inner, "name_or_path", None)
        input_requests = get_dataset(dataset_args, tokenizer, model_id=dataset_model_id)

        if dataset_name == "generated-shared-prefix":
            input_requests = input_requests[:batch_size]
            input_ids = [tokenizer.encode(req.prompt) for req in input_requests]
            input_len = sum(len(ids) for ids in input_ids) // len(input_ids)
            output_len = gsp_output_len
            image_data = None
        elif dataset_name == "mmmu":
            input_ids = [tok_inner.encode(req.prompt) for req in input_requests]
            image_data = [req.image_data for req in input_requests]
        else:
            input_ids = [req.prompt for req in input_requests]
            image_data = None

    if share_cached_prefix_across_batch:
        if not 0.0 < cache_hit_rate <= 1.0:
            raise ValueError(
                "--share-cached-prefix-across-batch requires "
                "--cache-hit-rate in (0, 1]"
            )
        cached_token_len = int(input_len * cache_hit_rate)
        if cached_token_len <= 0:
            raise ValueError("shared cached prefix must contain at least one token")
        if any(len(ids) < cached_token_len for ids in input_ids):
            raise ValueError(
                "every request must be at least as long as the shared cached prefix"
            )
        shared_prefix = list(input_ids[0][:cached_token_len])
        input_ids = [shared_prefix + list(ids[cached_token_len:]) for ids in input_ids]

    # Build payload based on backend
    if backend == "vllm":
        payload = {
            "model": model_name,
            "prompt": input_ids,
            "max_tokens": output_len,
            "temperature": temperature,
            "stream": True,
            "ignore_eos": True,
        }
        if return_logprob:
            payload["logprobs"] = 1
        gen_url = url + "/v1/completions"
    else:
        # Load sampling parameters
        use_structured_outputs = False
        if use_structured_outputs:
            texts = []
            for _ in range(batch_size):
                texts.append(
                    "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                    * 50
                    + "Assistant:"
                )
            json_schema = "$$ANY$$"
        else:
            json_schema = None

        payload = {
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": True,
                "json_schema": json_schema,
                "stream_interval": stream_interval,
            },
            "return_logprob": return_logprob,
            "stream": True,
            **({"parallel_batch": parallel_batch} if parallel_batch else {}),
        }
        payload["input_ids"] = input_ids
        if image_data is not None:
            payload["image_data"] = image_data
        if fake_prefill:
            payload["bootstrap_host"] = FAKE_BOOTSTRAP_HOST
            payload["bootstrap_room"] = 0
        if lora_name:
            # SGLang /generate accepts lora_path as either a string (applied
            # to every prompt) or a list matching the batch size (per-prompt
            # adapter). See io_struct.GenerateReqInput._normalize_lora_path.
            if len(lora_name) == 1:
                payload["lora_path"] = lora_name[0]
            elif lora_request_distribution == "uniform":
                payload["lora_path"] = [
                    random.choice(lora_name) for _ in range(batch_size)
                ]
            elif lora_request_distribution == "distinct":
                payload["lora_path"] = [
                    lora_name[i % len(lora_name)] for i in range(batch_size)
                ]
            elif lora_request_distribution == "skewed":
                weights = np.array([lora_zipf_alpha**-i for i in range(len(lora_name))])
                probs = weights / np.sum(weights)
                payload["lora_path"] = list(
                    np.random.choice(lora_name, size=batch_size, p=probs)
                )
            else:
                raise ValueError(
                    f"Unexpected lora_request_distribution: "
                    f"{lora_request_distribution!r}"
                )
        gen_url = url + "/generate"

    input_hasher = hashlib.sha256()
    input_hasher.update(len(input_ids).to_bytes(8, "little"))
    input_token_lengths = [len(ids) for ids in input_ids]
    if len(input_token_lengths) != batch_size:
        raise RuntimeError(
            f"dataset returned {len(input_token_lengths)} prompts, expected {batch_size}"
        )
    input_len_min = min(input_token_lengths)
    input_len_max = max(input_token_lengths)
    input_tokens_total = sum(input_token_lengths)
    for ids in input_ids:
        input_hasher.update(len(ids).to_bytes(8, "little"))
        input_hasher.update(np.asarray(ids, dtype="<i4").tobytes())
    input_ids_sha256 = input_hasher.hexdigest()
    print(f"SGLANG_BENCH_INPUT_IDS_SHA256 {input_ids_sha256}", flush=True)

    # Warm up cache if cache_hit_rate > 0.0
    if cache_hit_rate > 0.0:
        _warmup_cache(
            url=url,
            input_ids=input_ids,
            input_len=input_len,
            cache_hit_rate=cache_hit_rate,
            dataset_name=dataset_name,
            image_data=image_data,
            backend=backend,
            model_name=model_name,
            warmup_cached_prefill_shape=warmup_cached_prefill_shape,
            share_cached_prefix_across_batch=share_cached_prefix_across_batch,
        )

    # Turn on profiler
    profile_link = None
    if profile and not profile_decode_after_first_token:
        profile_link: str = run_profile(
            url=url,
            num_steps=profile_steps,
            activities=profile_activities,
            output_dir=profile_output_dir,
            profile_by_stage=profile_by_stage,
            profile_prefix=profile_prefix,
            start_step=profile_start_step,
        )

    # Get metrics before the request (for cache hit rate calculation)
    metrics_before = get_cache_tokens_from_metrics(url)

    measured_request = {
        "batch_size": batch_size,
        "cached_tokens_per_request": int(input_len * cache_hit_rate),
        "input_len": input_len,
        "new_tokens_per_request": input_len - int(input_len * cache_hit_rate),
        "output_len": output_len,
    }
    print(
        "SGLANG_BENCH_MEASURED_REQUEST_BEGIN "
        + json.dumps(measured_request, sort_keys=True),
        flush=True,
    )

    # Run the request
    tic = time.perf_counter()
    with requests.post(
        gen_url,
        json=payload,
        stream=True,
        timeout=DEFAULT_TIMEOUT,
    ) as response:
        response.raise_for_status()

        # Get the TTFT of the last request in the batch
        first_ttft = 0.0
        last_ttft = 0.0
        decode_tokens_before_last_ttft = 0
        decode_profile_started = False
        completed_requests = 0
        max_retractions = -1
        output_token_ids_sha256 = None
        persisted_output_token_ids = None
        persisted_dp_ranks = None
        dp_rank_counts = None
        persisted_cached_tokens = None
        if backend == "vllm":
            # Parse OpenAI-compatible streaming format from vLLM
            first_token_indices = set()
            for chunk in response.iter_lines(decode_unicode=False):
                chunk = chunk.decode("utf-8")
                if chunk and chunk.startswith("data:"):
                    data_str = chunk[5:].strip()
                    if data_str == "[DONE]":
                        break
                    data = json.loads(data_str)
                    if "error" in data:
                        raise RuntimeError(f"Request has failed. {data}.")
                    for choice in data.get("choices", []):
                        idx = choice["index"]
                        if idx not in first_token_indices:
                            first_token_indices.add(idx)
                            if first_ttft == 0.0:
                                first_ttft = time.perf_counter() - tic
                            if len(first_token_indices) == batch_size:
                                last_ttft = time.perf_counter() - tic
        else:
            first_token_indices = set()
            finished_indices = set()
            server_first_token_timestamps = {}
            server_finished_timestamps = {}
            latest_completion_tokens = {}
            dp_rank_by_index = {}
            cached_tokens_by_index = {}
            output_token_ids_by_index = [[] for _ in range(batch_size)]
            for chunk in response.iter_lines(decode_unicode=False):
                chunk = chunk.decode("utf-8")
                if chunk and chunk.startswith("data:"):
                    if chunk == "data: [DONE]":
                        break
                    data = json.loads(chunk[5:].strip("\n"))
                    if "error" in data:
                        raise RuntimeError(f"Request has failed. {data}.")

                    assert (
                        data["meta_info"]["finish_reason"] is None
                        or data["meta_info"]["finish_reason"]["type"] == "length"
                    )
                    index = data.get("index")
                    if index is None:
                        if batch_size != 1:
                            raise RuntimeError(
                                "batched SGLang stream event is missing its "
                                "request index; cannot establish the strict "
                                "decode-only boundary"
                            )
                        index = 0
                    if not isinstance(index, int) or not 0 <= index < batch_size:
                        raise RuntimeError(
                            f"invalid request index in stream event: {index!r}"
                        )
                    dp_rank = data["meta_info"].get("dp_rank")
                    if dp_rank is not None:
                        if (
                            isinstance(dp_rank, bool)
                            or not isinstance(dp_rank, int)
                            or dp_rank < 0
                        ):
                            raise RuntimeError(
                                f"request {index} returned invalid dp_rank={dp_rank!r}"
                            )
                        previous_dp_rank = dp_rank_by_index.get(index)
                        if previous_dp_rank is not None and previous_dp_rank != dp_rank:
                            raise RuntimeError(
                                f"request {index} changed DP rank during streaming: "
                                f"{previous_dp_rank} -> {dp_rank}"
                            )
                        dp_rank_by_index[index] = dp_rank
                    cached_tokens = data["meta_info"].get("cached_tokens")
                    if cached_tokens is not None:
                        if (
                            isinstance(cached_tokens, bool)
                            or not isinstance(cached_tokens, int)
                            or cached_tokens < 0
                        ):
                            raise RuntimeError(
                                f"request {index} returned invalid "
                                f"cached_tokens={cached_tokens!r}"
                            )
                        previous_cached_tokens = cached_tokens_by_index.get(index)
                        if (
                            previous_cached_tokens is not None
                            and previous_cached_tokens != cached_tokens
                        ):
                            raise RuntimeError(
                                f"request {index} changed cached_tokens during "
                                f"streaming: {previous_cached_tokens} -> "
                                f"{cached_tokens}"
                            )
                        cached_tokens_by_index[index] = cached_tokens
                    completion_tokens = data["meta_info"]["completion_tokens"]
                    previous_completion_tokens = latest_completion_tokens.get(index, 0)
                    is_finished = data["meta_info"]["finish_reason"] is not None
                    if completion_tokens < previous_completion_tokens or (
                        completion_tokens == previous_completion_tokens
                        and not is_finished
                    ):
                        raise RuntimeError(
                            "completion token count did not advance for request "
                            f"{index}: {previous_completion_tokens} -> {completion_tokens}"
                        )
                    latest_completion_tokens[index] = completion_tokens
                    output_ids = data.get("output_ids")
                    if not isinstance(output_ids, list) or not all(
                        isinstance(token_id, int) for token_id in output_ids
                    ):
                        raise RuntimeError(
                            f"request {index} returned invalid output_ids"
                        )
                    if len(output_ids) == completion_tokens:
                        output_token_ids_by_index[index] = list(output_ids)
                    elif (
                        len(output_token_ids_by_index[index]) + len(output_ids)
                        == completion_tokens
                    ):
                        output_token_ids_by_index[index].extend(output_ids)
                    else:
                        raise RuntimeError(
                            f"request {index} output_ids length does not match "
                            f"completion_tokens={completion_tokens}"
                        )
                    num_retractions = int(
                        data["meta_info"].get("num_retractions", 0)
                    )
                    max_retractions = max(max_retractions, num_retractions)
                    server_first_token_ts = data["meta_info"].get("first_token_ts")
                    if server_first_token_ts is None:
                        raise RuntimeError(
                            f"request {index} stream event is missing server "
                            "first_token_ts"
                        )
                    server_first_token_ts = float(server_first_token_ts)
                    previous_first_token_ts = server_first_token_timestamps.get(index)
                    if previous_first_token_ts is not None and not math.isclose(
                        server_first_token_ts,
                        previous_first_token_ts,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        raise RuntimeError(
                            f"request {index} server first-token timestamp changed: "
                            f"{previous_first_token_ts} -> {server_first_token_ts}"
                        )
                    if index not in first_token_indices:
                        if completion_tokens < 1:
                            raise RuntimeError(
                                f"request {index} first observed stream event has "
                                f"no generated token: {completion_tokens=}"
                            )
                        first_token_indices.add(index)
                        server_first_token_timestamps[index] = server_first_token_ts
                        if first_ttft == 0.0:
                            first_ttft = time.perf_counter() - tic
                        if len(first_token_indices) == batch_size:
                            last_ttft = time.perf_counter() - tic
                            decode_tokens_before_last_ttft = sum(
                                max(0, count - 1)
                                for count in latest_completion_tokens.values()
                            )
                            if profile_decode_after_first_token:
                                profile_link = run_profile(
                                    url=url,
                                    num_steps=profile_steps,
                                    activities=profile_activities,
                                    output_dir=profile_output_dir,
                                    profile_by_stage=False,
                                    profile_prefix=profile_prefix,
                                    start_step=None,
                                )
                                decode_profile_started = True
                        if return_logprob and completion_tokens == 1:
                            first_token_logprobs = data["meta_info"].get(
                                "output_token_logprobs"
                            )
                            print(
                                "SGLANG_BENCH_FIRST_TOKEN_RESULT "
                                + json.dumps(
                                    {
                                        "index": data.get("index"),
                                        "request_id": data["meta_info"].get("id"),
                                        "output_token_logprobs": first_token_logprobs,
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            print(
                                "SGLANG_BENCH_FIRST_TOKEN_LOGPROBS "
                                + json.dumps(first_token_logprobs),
                                flush=True,
                            )

                    if is_finished:
                        if index in finished_indices:
                            raise RuntimeError(
                                f"duplicate finish event for request {index}"
                            )
                        if completion_tokens != output_len:
                            raise RuntimeError(
                                f"request {index} completed with {completion_tokens} "
                                f"tokens, expected {output_len}"
                            )
                        if len(output_token_ids_by_index[index]) != output_len:
                            raise RuntimeError(
                                f"request {index} retained "
                                f"{len(output_token_ids_by_index[index])} token ids, "
                                f"expected {output_len}"
                            )
                        request_finished_ts = data["meta_info"].get(
                            "request_finished_ts"
                        )
                        if request_finished_ts is None:
                            raise RuntimeError(
                                f"request {index} finish event is missing server "
                                "request_finished_ts"
                            )
                        server_finished_timestamps[index] = float(request_finished_ts)
                        finished_indices.add(index)

            if len(first_token_indices) != batch_size:
                missing = sorted(set(range(batch_size)) - first_token_indices)
                raise RuntimeError(
                    "request completed before every first-token boundary was "
                    f"observed; missing request indices: {missing}"
                )
            if profile_decode_after_first_token and not decode_profile_started:
                raise RuntimeError("decode-only profiler was never started")
            if len(finished_indices) != batch_size:
                missing = sorted(set(range(batch_size)) - finished_indices)
                raise RuntimeError(
                    f"missing finish events for request indices: {missing}"
                )
            completed_requests = len(finished_indices)
            if dp_rank_by_index:
                if len(dp_rank_by_index) != batch_size:
                    missing = sorted(set(range(batch_size)) - dp_rank_by_index.keys())
                    raise RuntimeError(
                        "DP rank metadata was only returned for part of the batch; "
                        f"missing request indices: {missing}"
                    )
                persisted_dp_ranks = [dp_rank_by_index[i] for i in range(batch_size)]
                dp_rank_counts = [0] * (max(persisted_dp_ranks) + 1)
                for dp_rank in persisted_dp_ranks:
                    dp_rank_counts[dp_rank] += 1
            if cached_tokens_by_index:
                if len(cached_tokens_by_index) != batch_size:
                    missing = sorted(
                        set(range(batch_size)) - cached_tokens_by_index.keys()
                    )
                    raise RuntimeError(
                        "cached-token metadata was only returned for part of "
                        f"the batch; missing request indices: {missing}"
                    )
                persisted_cached_tokens = [
                    cached_tokens_by_index[i] for i in range(batch_size)
                ]
            server_first_token_spread = (
                max(server_first_token_timestamps.values())
                - min(server_first_token_timestamps.values())
            )
            server_first_token_ts_min = min(server_first_token_timestamps.values())
            server_finished_ts_max = max(server_finished_timestamps.values())
            output_token_ids_sha256 = [
                hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                for token_ids in output_token_ids_by_index
            ]
            if save_output_token_ids:
                persisted_output_token_ids = output_token_ids_by_index

    print("SGLANG_BENCH_MEASURED_REQUEST_END", flush=True)

    if profile and profile_stop_after_request:
        response = requests.post(url + "/stop_profile", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

    # Compute metrics
    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / last_ttft
    output_throughput = batch_size * output_len / (latency - last_ttft)
    (
        client_decode_duration,
        decode_tokens,
        client_decode_throughput,
        steady_decode_duration,
        steady_decode_tokens,
        steady_decode_throughput,
    ) = calculate_decode_only_metrics(
        batch_size=batch_size,
        output_len=output_len,
        latency=latency,
        first_ttft=first_ttft,
        last_ttft=last_ttft,
        decode_tokens_before_last_ttft=decode_tokens_before_last_ttft,
    )
    if backend == "vllm":
        server_first_token_ts_min = -1.0
        server_finished_ts_max = -1.0
        decode_duration = client_decode_duration
        decode_throughput = client_decode_throughput
    else:
        decode_duration = server_finished_ts_max - server_first_token_ts_min
        if decode_tokens and decode_duration <= 0:
            raise RuntimeError(
                "non-empty server decode span must be positive, got "
                f"{decode_duration}"
            )
        decode_throughput = (
            decode_tokens / decode_duration if decode_duration else 0.0
        )
    overall_throughput = batch_size * (input_len + output_len) / latency

    if backend == "vllm":
        # vLLM does not expose these metrics via API
        last_gen_throughput = -1
        acc_length = -1
    else:
        response = requests.get(url + "/server_info", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        server_info = response.json()
        internal_states = server_info.get("internal_states", [])
        acc_length = -1
        last_gen_throughput = -1
        for internal_state in internal_states:
            val_acc = internal_state.get("avg_spec_accept_length")
            if val_acc is not None:
                acc_length = val_acc
            val_thr = internal_state.get("last_gen_throughput")
            if val_thr is not None:
                last_gen_throughput = val_thr

    # Calculate cache hit rate from before/after metrics delta
    metrics_after = get_cache_tokens_from_metrics(url)
    metrics_cache_hit_rate = calculate_cache_hit_rate(metrics_before, metrics_after)

    # Print results
    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")
        print(
            "decode-only throughput: "
            f"{decode_throughput:.2f} tok/s "
            f"({decode_tokens} tokens in {decode_duration:.4f} s, "
            "server timestamp window)"
        )
        print(
            "client-observed decode-only throughput: "
            f"{client_decode_throughput:.2f} tok/s "
            f"({decode_tokens} tokens in {client_decode_duration:.4f} s)"
        )
        print(
            "steady decode-only throughput: "
            f"{steady_decode_throughput:.2f} tok/s "
            f"({steady_decode_tokens} tokens in "
            f"{steady_decode_duration:.4f} s; "
            f"excluded {decode_tokens_before_last_ttft} early decode tokens)"
        )
    print(f"last_ttft: {last_ttft:.2f} s")
    print(f"first-token spread: {last_ttft - first_ttft:.4f} s")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    if acc_length > 0:
        print(f"acc_length: {acc_length:.2f} ")
    if metrics_cache_hit_rate is not None:
        print(f"cache hit rate: {metrics_cache_hit_rate:.4f}")

    # Dump results
    result = BenchOneCaseResult(
        run_name=run_name,
        batch_size=batch_size,
        input_len=input_len,
        input_len_min=input_len_min,
        input_len_max=input_len_max,
        input_tokens_total=input_tokens_total,
        output_len=output_len,
        stream_interval=stream_interval,
        latency=latency,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        first_ttft=first_ttft,
        first_token_spread=last_ttft - first_ttft,
        server_first_token_spread=(
            -1.0 if backend == "vllm" else server_first_token_spread
        ),
        server_first_token_ts_min=server_first_token_ts_min,
        server_finished_ts_max=server_finished_ts_max,
        decode_duration=decode_duration,
        decode_tokens=decode_tokens,
        decode_tokens_before_last_ttft=decode_tokens_before_last_ttft,
        decode_throughput=decode_throughput,
        client_decode_duration=client_decode_duration,
        client_decode_throughput=client_decode_throughput,
        steady_decode_duration=steady_decode_duration,
        steady_decode_tokens=steady_decode_tokens,
        steady_decode_throughput=steady_decode_throughput,
        completed_requests=(batch_size if backend == "vllm" else completed_requests),
        max_retractions=max_retractions,
        input_ids_sha256=input_ids_sha256,
        output_token_ids_sha256=output_token_ids_sha256,
        output_token_ids=persisted_output_token_ids,
        dp_ranks=persisted_dp_ranks,
        dp_rank_counts=dp_rank_counts,
        cached_tokens_per_request=persisted_cached_tokens,
        overall_throughput=overall_throughput,
        last_ttft=last_ttft,
        last_gen_throughput=last_gen_throughput,
        acc_length=acc_length,
        cache_hit_rate=metrics_cache_hit_rate,
        profile_link=profile_link,
    )

    # Save and return the results
    if result_filename:
        result.dump_to_jsonl(result_filename)

    return result


def should_skip_due_to_token_capacity(
    batch_size, input_len, output_len, skip_token_capacity_threshold
):
    if batch_size * (input_len + output_len) > skip_token_capacity_threshold:
        print(
            "=" * 8
            + f"Skip benchmark {batch_size=} * ({input_len=} + {output_len=}) = {batch_size * (input_len + output_len)} > {skip_token_capacity_threshold=} due to kv cache limit."
            + "=" * 8
        )
        return True
    return False


def should_skip_due_to_max_running_requests(
    batch_size, skip_max_running_requests_threshold
):
    if batch_size > skip_max_running_requests_threshold:
        print(
            "=" * 8
            + f"Skip benchmark {batch_size=} > {skip_max_running_requests_threshold=} due to max running requests limit."
            + "=" * 8
        )
        return True
    return False


def get_report_summary(
    results: List[BenchOneCaseResult], bench_args: BenchArgs, server_args: ServerArgs
):
    summary = (
        f"\nInput lens: {bench_args.input_len}. Output lens: {bench_args.output_len}."
    )
    if bench_args.cache_hit_rate > 0.0:
        summary += f" Cache hit rate: {bench_args.cache_hit_rate*100:.1f}%."
    summary += "\n"

    if is_blackwell():
        hourly_cost_per_gpu = 4  # $4/hour for one B200
    else:
        hourly_cost_per_gpu = 2  # $2/hour for one H100
    input_util = 0.7

    # sort result by input_len
    results.sort(key=lambda x: x.input_len)
    rows = []
    headers = [
        "batch size",
        "input len",
        "latency (s)",
        "input throughput (tok/s)",
        "output throughput (tok/s)",
        "decode-only throughput (tok/s)",
        "steady decode-only throughput (tok/s)",
        "acc length",
        "ITL (ms)",
        "input cost ($/1M)",
        "output cost ($/1M)",
        "cache hit rate",
    ]
    if bench_args.profile:
        headers.append("profile")

    for res in results:
        hourly_cost = hourly_cost_per_gpu * server_args.tp_size
        accept_length = f"{res.acc_length:.2f}" if res.acc_length > 0 else "n/a"
        itl_ms = 1000 * res.batch_size / res.output_throughput
        input_cost = 1e6 / (res.input_throughput * input_util) / 3600 * hourly_cost
        output_cost = 1e6 / res.output_throughput / 3600 * hourly_cost
        cache_hit_rate = (
            f"{res.cache_hit_rate:.4f}" if res.cache_hit_rate is not None else "n/a"
        )

        row = [
            res.batch_size,
            res.input_len,
            f"{res.latency:.2f}",
            f"{res.input_throughput:.2f}",
            f"{res.output_throughput:.2f}",
            f"{res.decode_throughput:.2f}",
            f"{res.steady_decode_throughput:.2f}",
            accept_length,
            f"{itl_ms:.2f}",
            f"{input_cost:.2f}",
            f"{output_cost:.2f}",
            cache_hit_rate,
        ]
        if bench_args.profile:
            if res.profile_link:
                row.append(f"[Profile]({res.profile_link})")
            else:
                row.append("n/a")
        rows.append(row)

    summary += tabulate(rows, headers=headers, tablefmt="github")
    summary += "\n"

    return summary


def run_benchmark_internal(
    server_args: ServerArgs,
    bench_args: BenchArgs,
    launch_server_func: Callable = launch_server,
):
    global DEFAULT_TIMEOUT
    validate_profile_cli_contract(bench_args)
    if bench_args.request_timeout <= 0:
        raise ValueError(
            f"--request-timeout must be positive, got {bench_args.request_timeout}"
        )
    DEFAULT_TIMEOUT = bench_args.request_timeout

    # set random seed
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # Resolve the benchmark target: launch a server, or connect to --base-url.
    endpoint = acquire_endpoint(server_args, bench_args.base_url, launch_server_func)
    base_url = endpoint.base_url

    # Get tokenizer and server info
    if bench_args.backend == "vllm":
        # For vLLM, get model name from /v1/models endpoint
        print(f"Connecting to vLLM server at {base_url}...")
        response = requests.get(base_url + "/v1/models", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        model_list = response.json().get("data", [])
        if not model_list:
            raise RuntimeError("No models found on vLLM server via /v1/models")
        model_name = model_list[0]["id"]
        print(f"Found model: {model_name}")
        print(f"Loading tokenizer for {model_name}...")
        if bench_args.dataset_name == "mmmu":
            tokenizer = get_processor(model_name)
        else:
            tokenizer = get_tokenizer(model_name)
        print("Tokenizer loaded.")

        server_info = {"model_name": model_name}
        # vLLM does not expose token capacity or max running requests via API
        skip_token_capacity_threshold = float("inf")
        skip_max_running_requests_threshold = float("inf")
    else:
        model_name = None
        response = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        server_info = response.json()
        if bench_args.local_tokenizer_path:
            tokenizer_path = bench_args.local_tokenizer_path
        elif "tokenizer_path" in server_info:
            tokenizer_path = server_info["tokenizer_path"]
        elif "prefill" in server_info:
            tokenizer_path = server_info["prefill"][0]["tokenizer_path"]
        if bench_args.dataset_name == "mmmu":
            # mmmu implies this is a MLLM
            tokenizer = get_processor(tokenizer_path)
        else:
            tokenizer = get_tokenizer(tokenizer_path)

        internal_states = server_info.get("internal_states", [])
        internal_state = internal_states[0] if internal_states else {}
        dp_size = internal_state.get("dp_size", None) or 1

        # Get effective max running requests
        max_running_requests_per_dp = internal_state.get(
            "effective_max_running_requests_per_dp", -1
        )

        # Get token capacity
        skip_token_capacity_threshold = 0

        for state in internal_states:
            skip_token_capacity_threshold += state.get("memory_usage", {}).get(
                "token_capacity", 1000000000
            )

        assert (
            max_running_requests_per_dp > 0
        ), f"effective_max_running_requests_per_dp is not set, {max_running_requests_per_dp=}"
        skip_max_running_requests_threshold = max_running_requests_per_dp * dp_size

        print(f"{max_running_requests_per_dp=}")
        print(f"{dp_size=}")
        print(f"{skip_max_running_requests_threshold=}")
        print(f"{skip_token_capacity_threshold=}")

    # Under --enable-multi-batch the client intentionally sends more prompts
    # than the server's running cap; surplus requests are queued (no KV
    # reservation) and promoted batch-by-batch. Peak live KV footprint is
    # bounded by the running cap, not by bs, so re-scope both guards:
    #   * max_running_requests: disabled (the whole point of the flag).
    #   * token_capacity: check against min(bs, running_cap) * (il + ol).
    effective_running_cap: Optional[int] = None
    if bench_args.enable_multi_batch:
        if skip_max_running_requests_threshold != float("inf"):
            effective_running_cap = skip_max_running_requests_threshold
        skip_max_running_requests_threshold = float("inf")

        # Multi-batch only kicks in when the client sends strictly more prompts
        # than the server's running cap; otherwise every prompt fits in a
        # single wave and the flag is a no-op for that case (but its metric
        # caveats — misleading input/output throughput and TTFT — still apply).
        # Warn loudly so the user can fix the batch-size sweep.
        if effective_running_cap is not None:
            noop_bs = sorted(
                {bs for bs in bench_args.batch_size if bs <= effective_running_cap}
            )
            if noop_bs:
                print(
                    f"WARNING: --enable-multi-batch is set but batch size(s) "
                    f"{noop_bs} are <= running cap ({effective_running_cap}); "
                    f"those cases will run as a single wave and the flag is a "
                    f"no-op for them. Use batch_size > {effective_running_cap} "
                    f"to actually exercise multi-batch."
                )

    # LoRA distribution args: mirror serving.py semantics so multi-LoRA
    # benchmarks behave consistently across harnesses.
    if bench_args.lora_request_distribution in ("distinct", "skewed"):
        assert bench_args.lora_name is not None and len(bench_args.lora_name) > 1, (
            "--lora-request-distribution=distinct/skewed requires more than "
            "one adapter via --lora-name."
        )
    assert (
        bench_args.lora_zipf_alpha > 1
    ), f"--lora-zipf-alpha must be > 1, got {bench_args.lora_zipf_alpha}"

    if bench_args.apply_chat_template and not bench_args.fixed_prompt_file:
        raise ValueError(
            "--apply-chat-template requires --fixed-prompt-file: the other "
            "datasets generate token ids directly, so there is no prompt text "
            "to run through a chat template."
        )

    gsp_kwargs = dict(
        gsp_num_groups=bench_args.gsp_num_groups,
        gsp_system_prompt_len=bench_args.gsp_system_prompt_len,
        gsp_question_len=bench_args.gsp_question_len,
        gsp_output_len=bench_args.gsp_output_len,
    )

    # Warmup
    if not bench_args.skip_warmup:
        batch_size_unique = list(set(bench_args.batch_size))
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        print(f"Warmup with batch_size={batch_size_unique}")
        for bs in batch_size_unique:
            run_one_case(
                base_url,
                batch_size=bs,
                input_len=1024,
                output_len=16,
                temperature=bench_args.temperature,
                return_logprob=bench_args.return_logprob,
                stream_interval=bench_args.client_stream_interval,
                input_len_step_percentage=bench_args.input_len_step_percentage,
                run_name="",
                result_filename="",
                tokenizer=tokenizer,
                dataset_name=bench_args.dataset_name,
                dataset_path=bench_args.dataset_path,
                parallel_batch=bench_args.parallel_batch,
                backend=bench_args.backend,
                model_name=model_name,
                fake_prefill=bench_args.fake_prefill,
                lora_name=bench_args.lora_name,
                lora_request_distribution=bench_args.lora_request_distribution,
                lora_zipf_alpha=bench_args.lora_zipf_alpha,
                fixed_prompt_file=bench_args.fixed_prompt_file,
                apply_chat_template=bench_args.apply_chat_template,
                seed=bench_args.seed,
                **gsp_kwargs,
            )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    results = []
    profile_results = []
    try:
        # Benchmark all cases. A profile-only run skips this duplicate pass;
        # the profiled rows below become the returned benchmark results.
        if not bench_args.profile_only:
            for bs, il, ol in itertools.product(
                bench_args.batch_size, bench_args.input_len, bench_args.output_len
            ):
                kv_footprint_bs = (
                    bs
                    if effective_running_cap is None
                    else min(bs, effective_running_cap)
                )
                if should_skip_due_to_max_running_requests(
                    bs, skip_max_running_requests_threshold
                ) or should_skip_due_to_token_capacity(
                    kv_footprint_bs, il, ol, skip_token_capacity_threshold
                ):
                    continue
                results.append(
                    run_one_case(
                        base_url,
                        bs,
                        il,
                        ol,
                        temperature=bench_args.temperature,
                        return_logprob=bench_args.return_logprob,
                        stream_interval=bench_args.client_stream_interval,
                        input_len_step_percentage=(
                            bench_args.input_len_step_percentage
                        ),
                        run_name=bench_args.run_name,
                        result_filename=bench_args.result_filename,
                        tokenizer=tokenizer,
                        dataset_name=bench_args.dataset_name,
                        dataset_path=bench_args.dataset_path,
                        parallel_batch=bench_args.parallel_batch,
                        cache_hit_rate=bench_args.cache_hit_rate,
                        share_cached_prefix_across_batch=(
                            bench_args.share_cached_prefix_across_batch
                        ),
                        warmup_cached_prefill_shape=(
                            bench_args.warmup_cached_prefill_shape
                        ),
                        backend=bench_args.backend,
                        model_name=model_name,
                        fake_prefill=bench_args.fake_prefill,
                        lora_name=bench_args.lora_name,
                        lora_request_distribution=(
                            bench_args.lora_request_distribution
                        ),
                        lora_zipf_alpha=bench_args.lora_zipf_alpha,
                        fixed_prompt_file=bench_args.fixed_prompt_file,
                        apply_chat_template=bench_args.apply_chat_template,
                        save_output_token_ids=bench_args.save_output_token_ids,
                        seed=bench_args.seed,
                        preserve_prefix_cache=bench_args.preserve_prefix_cache,
                        **gsp_kwargs,
                    )
                )

        # Profile all cases
        if bench_args.profile:
            try:
                for bs, il, ol in itertools.product(
                    bench_args.batch_size, bench_args.input_len, bench_args.output_len
                ):
                    kv_footprint_bs = (
                        bs
                        if effective_running_cap is None
                        else min(bs, effective_running_cap)
                    )
                    if should_skip_due_to_max_running_requests(
                        bs, skip_max_running_requests_threshold
                    ) or should_skip_due_to_token_capacity(
                        kv_footprint_bs, il, ol, skip_token_capacity_threshold
                    ):
                        continue
                    profile_prefix = (
                        bench_args.profile_prefix or ""
                    ) + f"bs-{bs}-il-{il}"
                    profile_results.append(
                        run_one_case(
                            base_url,
                            bs,
                            il,
                            ol,
                            temperature=bench_args.temperature,
                            return_logprob=bench_args.return_logprob,
                            stream_interval=bench_args.client_stream_interval,
                            input_len_step_percentage=bench_args.input_len_step_percentage,
                            run_name=bench_args.run_name,
                            result_filename=bench_args.result_filename,
                            tokenizer=tokenizer,
                            dataset_name=bench_args.dataset_name,
                            dataset_path=bench_args.dataset_path,
                            parallel_batch=bench_args.parallel_batch,
                            cache_hit_rate=bench_args.cache_hit_rate,
                            share_cached_prefix_across_batch=(
                                bench_args.share_cached_prefix_across_batch
                            ),
                            warmup_cached_prefill_shape=(
                                bench_args.warmup_cached_prefill_shape
                            ),
                            profile=bench_args.profile,
                            profile_activities=bench_args.profile_activities,
                            profile_start_step=bench_args.profile_start_step,
                            profile_steps=bench_args.profile_steps,
                            profile_stop_after_request=(
                                bench_args.profile_stop_after_request
                            ),
                            profile_decode_after_first_token=(
                                bench_args.profile_decode_after_first_token
                            ),
                            save_output_token_ids=bench_args.save_output_token_ids,
                            profile_by_stage=bench_args.profile_by_stage,
                            profile_prefix=profile_prefix,
                            profile_output_dir=bench_args.profile_output_dir,
                            backend=bench_args.backend,
                            model_name=model_name,
                            fake_prefill=bench_args.fake_prefill,
                            lora_name=bench_args.lora_name,
                            lora_request_distribution=bench_args.lora_request_distribution,
                            lora_zipf_alpha=bench_args.lora_zipf_alpha,
                            seed=bench_args.seed,
                            preserve_prefix_cache=bench_args.preserve_prefix_cache,
                            **gsp_kwargs,
                        )
                    )
            except Exception as e:
                if bench_args.profile_only:
                    raise
                print(f"Error profiling, some profile traces may not be dumped: {e}")

            if bench_args.profile_only:
                results = profile_results
            else:
                # Replace the profile link for any successful profile results
                for res, profile_res in zip(results, profile_results, strict=False):
                    res.profile_link = profile_res.profile_link
    finally:
        endpoint.close()

    print(f"\nResults are saved to {bench_args.result_filename}")

    if not bench_args.show_report:
        return results, server_info

    # Print summary
    summary = get_report_summary(results, bench_args, server_args)
    print(summary)

    if is_in_ci() and bench_args.append_to_github_summary:
        write_github_step_summary(summary)

    return results, server_info


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    results, server_info = run_benchmark_internal(server_args, bench_args)

    # Save results as pydantic models in the JSON format
    if bench_args.pydantic_result_filename:
        save_results_as_pydantic_models(
            results,
            pydantic_result_filename=bench_args.pydantic_result_filename,
            model_path=server_args.model_path,
            server_args=bench_args.server_args_for_metrics,
        )

    return results, server_info


def validate_profile_cli_contract(bench_args: BenchArgs) -> None:
    if bench_args.profile_only and not bench_args.profile:
        raise ValueError("--profile-only requires --profile")
    if not bench_args.profile_decode_after_first_token:
        return
    if not bench_args.profile:
        raise ValueError("--profile-decode-after-first-token requires --profile")
    if bench_args.backend != "sglang":
        raise ValueError(
            "--profile-decode-after-first-token only supports the sglang backend"
        )
    if bench_args.profile_by_stage:
        raise ValueError(
            "--profile-decode-after-first-token cannot be combined with "
            "--profile-by-stage"
        )
    if bench_args.profile_start_step is not None:
        raise ValueError(
            "--profile-decode-after-first-token cannot be combined with "
            "--profile-start-step"
        )
    if "CUDA_PROFILER" not in bench_args.profile_activities:
        raise ValueError(
            "--profile-decode-after-first-token requires "
            "--profile-activities CUDA_PROFILER"
        )
    if bench_args.profile_steps <= 0:
        raise ValueError(
            "--profile-decode-after-first-token requires --profile-steps > 0"
        )


def cli_main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)


if __name__ == "__main__":
    cli_main()
