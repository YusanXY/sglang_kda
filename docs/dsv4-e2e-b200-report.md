# DeepSeek V4 Flash huge-kernel B200 report

## Reproducibility

- Branch: `develop_e2e_optimize_yusan`
- Final runtime commit: `6204af92e56ef27e76f03c52197430b831579c80`
- Model: `/mnt/SFS-Shared/guojingyu/models/DeepSeek-V4-Flash`
- Worktree: `/home/gjy/data/agent4kernel/sglang_kda_e2e`
- Hardware: 4 x NVIDIA B200, SM100, 183359 MiB each, NV18 between every pair
- Driver: 610.43.02
- CUDA: `/usr/local/cuda-13.2`
- CCCL: `/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl`
- TP/EP/DP/PP: 4/4/1/1

Formal command:

```bash
cd /home/gjy/data/agent4kernel/sglang_kda_e2e
PAIRS=5 scripts/dsv4_e2e/run_prefill_ttft_compare.sh \
  /home/gjy/data/agent4kernel/sglang_kda_e2e/.runtime/dsv4_e2e_prefill_ttft/b200_formal_6204af92e_v1
```

The driver fixes batch=1, context capacity 73728, input length 69632,
65536 cached tokens, 4096 new tokens, output length 1, page size 256,
chunked prefill size 4096, TP4/EP4, `attention-backend=dsv4`,
`moe-runner-backend=flashinfer_mxfp4`, and disables overlap plus prefill/decode
CUDA graphs. Every accepted sample contains the scheduler line
`#new-token: 4096, #cached-token: 65536`.

## Correctness

- DSV4 CPU contracts: 47 passed.
- Inverse-RoPE/BF16-boundary/UE8M0 FP8 CUDA JIT matrix: 9 passed
  (T=1/17/4096 and int32/int64 positions).
- Final 43-layer TP4 comparison: token IDs exactly equal, logits cosine 1.0,
  max absolute error 0.0, mean absolute error 0.0.
- Comparison artifact:
  `/home/gjy/data/agent4kernel/sglang_kda_e2e/bench_artifacts/b200_correctness/comparison_gpu_scratch_nosync_cuda132.json`

The attempted TileLang MHC fusion was removed rather than hidden behind a
fallback. Its ablations kept the predicted token but failed the required
whole-model cosine gate:

| MHC candidate | logits cosine |
|---|---:|
| in-layer + cross-layer | 0.9989781 |
| in-layer only | 0.9970950 |
| cross-layer only | 0.9978858 |

## Formal TTFT result

Formal status: **PASS**. Huge-kernel won 4 of 5 pairs.

| pair | native TTFT (s) | huge TTFT (s) | speedup | huge won |
|---:|---:|---:|---:|:---:|
| 1 | 0.4164 | 0.2710 | 1.5365x | yes |
| 2 | 0.3194 | 0.3055 | 1.0455x | yes |
| 3 | 0.2844 | 0.3140 | 0.9057x | no |
| 4 | 0.3140 | 0.2844 | 1.1041x | yes |
| 5 | 0.3203 | 0.2710 | 1.1819x | yes |

| backend | mean TTFT (s) | median TTFT (s) | median 4096-token throughput (token/s) |
|---|---:|---:|---:|
| native | 0.330900 | 0.319400 | 12824.05 |
| huge_kernel | 0.289180 | 0.284400 | 14402.25 |

Median TTFT is 10.96% lower and median incremental throughput is 12.31%
higher. Median paired speedup is 1.1041x (median paired improvement 9.43%).
The complete logs, JSONL files, validation JSON, manifest, and summaries are in:

`/home/gjy/data/agent4kernel/sglang_kda_e2e/.runtime/dsv4_e2e_prefill_ttft/b200_formal_6204af92e_v1`

## GPU/Host split audit

The initial whole-layer executor contained too much Python orchestration relative
to new fused CUDA. The final accepted version moves or keeps the following work
on GPU without changing native behavior:

- one CUDA kernel fuses inverse RoPE, the BF16 rounding boundary, and UE8M0 FP8
  quantization before WO_A;
- exact-T attention-Q, WO_A quant, scale, and WO_A GEMM buffers are allocated
  once per ForwardBatch and reused by all 43 sequential layers;
- strict huge batch=1 computes the SWA gather allocation extent from already
  available Host integers, removing the GPU cumsum `.item()` synchronization;
- C4 paged MQA, top-k transform, C4/SWA combine, cache dequantization,
  FlashMLA attention, GEMMs, MoE, and collectives execute in existing CUDA,
  Triton, TileLang, DeepGEMM, FlashInfer, or NCCL kernels.

Python remains the control plane: it creates one batch descriptor and performs
43 layer dispatches. The remaining important optimization boundaries are:

1. C4 paged-MQA still materializes a full logits tensor before a separate
   top-k kernel.
2. Sparse prefill still dequantizes paged FP8 cache into a BF16 flat workspace
   before FlashMLA reads it.
3. WO_A and WO_B remain separate Tensor-Core GEMMs because WO_B depends on the
   globally completed WO_A result.
4. MoE/EP and TP collectives retain required NCCL/FlashInfer synchronization.
5. Prefill CUDA graphs are intentionally disabled by the supported deployment
   contract, so Python launch orchestration is not graph-captured.

The next CUDA work should therefore target fused paged-MQA plus online top-k,
then direct paged-FP8 sparse attention. Rewriting the Worker class itself in
CUDA would not move meaningful tensor compute; eliminating these two
intermediate representations would.
