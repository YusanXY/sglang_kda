# GLM-5.2 decode optimization progress

## Fixed experiment contract

- Host: B300-M3, eight B300 GPUs.
- Model: `/var/b300-shared/models/GLM-5.2-FP8`.
- Workload: 64 requests, 100,000 input tokens and 1,000 output tokens per
  request, temperature 0, random-id seed 4199.
- Decode numerator: `64 * (1000 - 1) = 63,936` tokens. Timing starts at the
  last first-token timestamp and ends at the last finished timestamp.
- Parallelism: TP8 / EP8 / DP8 with attention DP enabled.
- Attention: DSA, FP8 E4M3 KV cache, page size 64.
- Capacity: `mem_fraction_static=0.835`, `swa_full_tokens_ratio=0.075`.
- CUDA Graph: full decode graph, maximum batch size 544.

The full target-shape warmup is excluded from measured repetitions. Every
result must retain all 64 output-token vectors and the server runtime
fingerprint before it can enter the performance table.

## Baseline semantics

The command line requests `moe_a2a_backend=megamoe` and leaves the routed MoE
runner at `auto`. This is called **MegaMoE-requested baseline**, not MegaMoE
execution: GLM-5.2 has FP8 E4M3 expert weights, whereas the installed SM100
MegaMoE kernel supports FP8 activations with FP4 weights. SGLang therefore
does not build MegaMoE weights and resolves routed experts to the standard EP
dispatcher plus Triton MoE.

Dense FP8 linear layers continue to use DeepGEMM in both baseline and
optimized candidates.

## Candidate 1: standard EP plus DeepGEMM routed MoE

Commit `6948d7e9f` removes an obsolete weight-loading restriction that only
allowed DeepGEMM routed experts with the DeepEP dispatcher. The repository
already registers the complete standard-dispatch path:

1. global expert IDs are mapped to each rank's 32 local experts;
2. `moe_ep_deepgemm_preprocess` builds expert-major masked input on GPU;
3. DeepGEMM executes the two masked grouped expert GEMMs;
4. `post_reorder_deepgemm` applies routing weights and restores token order;
5. the standard EP combine performs the required cross-rank reduction.

This candidate keeps the topology, DSA attention, capacity and graph shapes
identical. Its expected gain comes only from replacing 75 layers of Triton
routed-expert GEMM execution with the SM100 DeepGEMM path. Accuracy must be
rechecked because DeepGEMM transforms checkpoint block scales to UE8M0.

## Harness and profiling evidence

- Strict workload harness: `scripts/glm52_decode/run_decode_n5.sh`.
- Strict validator: `scripts/glm52_decode/validate_decode_result.py`.
- Server launcher: `scripts/glm52_decode/launch_server.sh`.
- Steady-decode Nsys: `scripts/glm52_decode/run_decode_nsys.sh`.

The Nsys capture begins only after all 64 requests have produced their first
token and records 200 decode steps. CUDA Graph nodes are expanded with
`--cuda-graph-trace=node:host-only`, so the report exposes kernels inside each
graph replay without including the 6.4-million-token prefill.

## Startup findings

Two cold-start failures were diagnosed before collecting performance data:

1. Rank 0 entered a 12,582,912-element model-parallel all-reduce while ranks
   1-7 were still JIT-compiling FlashInfer/CUTLASS kernels. PyTorch's default
   600-second process-group timeout aborted rank 0. Baseline and candidates now
   use the same 3,600-second distributed timeout; the forward watchdog remains
   1,800 seconds.
2. The shared runtime environment overwrote `TRITON_CACHE_DIR` with another
   account's read-only directory. Commit `b0e280ddf` reapplies an explicit
   experiment-owned cache after sourcing the environment. This changes no
   inference kernels and makes warm-cache A/B runs reproducible.
3. SGLang's internal warmup request has a fixed 600-second HTTP timeout, while
   the first all-M DeepGEMM pass took about 11 minutes. Experiments skip that
   internal request and use the strict external workload warmup with a
   14,400-second timeout. DeepGEMM fast warmup still compiles every decode M
   from 1 through 1,024, and only samples larger prefill Ms; this avoids
   replaying all 16K prefill shapes at every restart without reducing decode
   kernel coverage.

Neither startup fix is counted as a decode throughput optimization.

## Results

No performance claim is recorded until the strict full-shape result passes.

| Version | Routed expert path | Steady decode tok/s | Speedup | Accuracy |
|---|---|---:|---:|---|
| Baseline | Standard EP + Triton | pending | 1.000x | pending |
| Candidate 1 | Standard EP + DeepGEMM masked | pending | pending | pending |

## Next evidence-driven candidates

After baseline and Candidate 1 profiles are available, candidates will be
ranked by measured global cost rather than kernel microbenchmarks alone:

- GPU-local DP control broadcast and multi-step decode to remove host/Gloo
  scheduler gaps;
- LM-head DP localization to avoid gathering all DP hidden states before the
  vocabulary projection;
- supported residual/RMSNorm/all-reduce and MoE combine fusion boundaries;
- DSA indexer/top-k, sparse decode attention and routed-MoE kernel internals;
- shared-expert overlap or fusion only after proving its EP8 communication
  dependency and graph compatibility.
