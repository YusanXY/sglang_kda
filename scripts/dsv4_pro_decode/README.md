# DeepSeek V4 Pro decode-only benchmark

This harness measures one fixed serving workload on eight B300 GPUs:

- 64 concurrent requests;
- 100,000 input tokens per request;
- exactly 1,000 generated tokens per request (`ignore_eos=true`);
- temperature 0, random token-id prompts, seed 42;
- stream interval 64 (token 1 is still emitted immediately, then every 64
  tokens, avoiding per-token HTTP/SSE backpressure);
- no speculative decoding.

The server must be launched with the following target configuration:

```bash
python -m sglang.launch_server \
  --model-path /var/b300-shared/models/DeepSeek-V4-Pro \
  --trust-remote-code \
  --tp 8 --dp 8 --ep 8 --enable-dp-attention \
  --moe-a2a-backend megamoe \
  --mem-fraction-static 0.835 \
  --swa-full-tokens-ratio 0.075 \
  --cuda-graph-max-bs-decode 544 \
  --host 127.0.0.1 --port 30000
```

`/server_info` is archived and validated. The harness fails if any of TP, DP,
EP, Attention-DP, MegaMoE, memory fraction, SWA ratio, graph maximum, model
path, or speculative-decoding state differs from the target.

## Timing semantics

The legacy `output_throughput` field is kept for compatibility but is not the
primary result. The primary `decode_throughput` uses exactly `64 * 999 =
63,936` post-first-token tokens and the server timestamp window from the
earliest request's first token to the latest request completion. This covers
every decode token, excludes prefill/first-token work, and excludes HTTP/SSE
transport skew. `client_decode_throughput` retains the equivalent
client-observed window as a transport-overhead diagnostic.

Attention-DP ranks can reach the client at slightly different times. A second
diagnostic, `steady_decode_throughput`, starts at the last request's first
token and subtracts any decode tokens that earlier requests generated before
that boundary. `first_token_spread` and the subtracted token count are always
reported, so DP skew cannot silently bias the numerator.

Every run also requires all 64 requests to finish with exactly 1,000 output
IDs and zero retractions. The Pro scripts save each complete token vector and
a SHA256 digest that the validator recomputes. Five-run repeatability is
summarized by `run_decode_n5.sh`.

`compare_decode_outputs.py` validates a reference and candidate sample before
comparing them. It requires identical input hashes and reports complete-vector
agreement, per-position token agreement, and the first mismatch in each
request. Native-vs-native comparisons establish the numerical noise floor
before applying the same comparison to an optimized build.

## Decode-only Nsys

`run_decode_nsys.sh` uses `--capture-range=cudaProfilerApi` and asks the client
to arm profiling only after all first-token events. It captures 200 decode
forward steps by default and enables CUDA Graph node tracing. This report is
therefore suitable for kernel/API/graph critical-path analysis without 100K
prefill dominating the timeline. `--profile-only` also prevents the normal
benchmark pass from duplicating this expensive workload before the profiled
request.
