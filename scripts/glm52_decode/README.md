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

`run_decode_n5.sh` always executes one additional full target-shape batch before
the five measured samples. This warmup is validated but excluded from the
summary so lazy DeepGEMM compilation and first-use allocator work cannot pollute
the throughput distribution.

On the current B300 runtime, `--moe-a2a-backend megamoe` is a requested server
configuration rather than proof that routed experts execute the MegaMoE kernel.
The installed SM100 DeepGEMM MegaMoE supports FP8 activations with FP4 weights;
GLM-5.2 uses FP8 E4M3 expert weights, so stock SGLang leaves
`_mega_moe_weights_built` false and executes the standard EP MoE path. Reports
must call this configuration "MegaMoE-requested" until an FP8-capable fused A2A
implementation is actually bound and profiled.
