# GLM-5.2 decode throughput

Strict workload: 8x B300, TP8/EP8/DP8 with attention DP and MegaMoE, 64
requests, exactly 100,000 input tokens and 1,000 output tokens per request.
The primary throughput numerator is 64 x 999 = 63,936 decode tokens, measured
from the earliest server-side first-token timestamp to the latest server-side
finish timestamp.

Every formal sample must pass `validate_decode_result.py`. It requires an exact
GLM-5.2 model fingerprint, resolved DSA/TRT-LLM attention, enabled decode CUDA
Graph with max batch 544, eight requests on each DP rank, no retractions, and
complete output-token vectors. Unsupported or silently changed configurations
fail instead of falling back into the performance table.

`run_decode_n5.sh` first builds the exact target prefixes with one output token,
then executes the five measured 1K-token samples. The context build is validated
but excluded from the summary: it covers all 6.4M prompt tokens without wasting
another 63,936 untimed decode tokens. Measured samples use the identical seed
and preserve the live prefix cache; with page size 64 every result must report
exactly 99,968 cached tokens for every 100K-token request. The final 32-token
partial page is replayed before first-token timing. Thus the timed forward is
long-context decode replay rather than another 6.4M-token prefill. Set
`REUSE_PREFIX_CACHE=1` only when a separately validated warmup populated those
exact prompts on the same live server.

On the current B300 runtime, `--moe-a2a-backend megamoe` is a requested server
configuration rather than proof that routed experts execute the MegaMoE kernel.
The installed SM100 DeepGEMM MegaMoE supports FP8 activations with FP4 weights;
GLM-5.2 uses FP8 E4M3 expert weights, so stock SGLang leaves
`_mega_moe_weights_built` false and executes the standard EP MoE path. Reports
must call this configuration "MegaMoE-requested" until an FP8-capable fused A2A
implementation is actually bound and profiled.

Launch the stock baseline with:

The fixed launch command uses a 3600-second process-group timeout. The first
full-shape DP8 prefill can JIT-compile FlashInfer/CUTLASS kernels on ranks 1-7
after rank 0 has entered a model-parallel collective; PyTorch's 600-second
default can therefore abort a healthy cold start. This setting is identical
for baseline and optimized runs and does not change measured steady decode.

The launcher also sets `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`. The upstream all-M
precompile hook runs only on rank 0 from inside a model forward; under DP
attention, peer ranks can enter the next collective while rank 0 synchronizes
that sweep and deadlock the first request. Actual shapes are compiled
symmetrically on demand, and an exact Req64/100K context build is still run and
validated before measured samples. This is a common harness fix, not a
throughput optimization.

The formal launcher resolves `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0`, and the
validator requires `/server_info` to report `custom_all_reduce_impl=legacy`.
On this B300 DP8 topology, V2 intermittently leaves eager-prefill ranks spinning
inside one-/two-shot GPU kernels: the same 8x2048-token preflight exceeded 90 s,
whereas legacy custom AR completed in 2.13 s. This remains SGLang's GPU custom
AR, not NCCL, and is held identical for baseline and every optimized sample.

```bash
MOE_RUNNER_BACKEND=auto scripts/glm52_decode/launch_server.sh
```

Launch the rejected DeepGEMM experiment with:

```bash
MOE_RUNNER_BACKEND=deep_gemm scripts/glm52_decode/launch_server.sh
```

Both commands keep TP8/EP8/DP8, DSA attention, requested MegaMoE, memory
capacity and CUDA Graph shapes identical. The only intended execution change is
the FP8 routed-expert runner: stock auto resolves to Triton because the
checkpoint cannot build FP4 MegaMoE weights; the optimized command uses the
standard-dispatch masked DeepGEMM runner.

DeepGEMM improves throughput but is not an accepted implementation for this
checkpoint: it requantizes the non-UE8M0 expert scales and fails the native-noise
output gate. The accepted routed-MoE implementation preserves the checkpoint's
FP8 block scales and uses FlashInfer TRT-LLM behind the same standard dispatcher:

```bash
MOE_RUNNER_BACKEND=flashinfer_trtllm_routed \
  scripts/glm52_decode/launch_server.sh
```

The direct-output candidate additionally lets FlashInfer write its finalized
BF16 result directly into SGLang's caller-owned all-reduce buffer. It removes
one output-sized D2D copy per sparse layer and rank without changing routing,
scales, expert weights, collective order, or graph shapes:

```bash
MOE_RUNNER_BACKEND=flashinfer_trtllm_routed \
GLM52_FLASHINFER_DIRECT_OUTPUT=1 \
  scripts/glm52_decode/launch_server.sh
```

Set the matching strict validator contract while measuring:

```bash
RUNS=5 MOE_RUNNER_BACKEND=flashinfer_trtllm_routed \
EXPECTED_FLASHINFER_DIRECT_OUTPUT=1 \
  scripts/glm52_decode/run_decode_n5.sh
```
