#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/b300-shared/home/gjy/sglang_huge}
REPO=${REPO:-$ROOT/.runtime/glm52_decode_worktree}
RUNTIME=${RUNTIME:-/mnt/b300-shared/home/gjy/data/agent4kernel/.runtime/b300}
MODEL=${MODEL:-/var/b300-shared/models/GLM-5.2-FP8}
PORT=${PORT:-30000}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-auto}
GLM52_TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/.runtime/glm52_decode_cache}
GLM52_DEEP_GEMM_CACHE_DIR=${SGLANG_DG_CACHE_DIR:-$ROOT/.runtime/glm52_deep_gemm_cache}
GLM52_SHARED_EXPERT_TP1=${GLM52_SHARED_EXPERT_TP1:-0}
GLM52_CUSTOM_ALL_REDUCE_IMPL=${GLM52_CUSTOM_ALL_REDUCE_IMPL:-legacy}

[[ "$MOE_RUNNER_BACKEND" == auto || "$MOE_RUNNER_BACKEND" == deep_gemm ]] || {
  echo "MOE_RUNNER_BACKEND must be auto or deep_gemm" >&2
  exit 2
}
[[ "$GLM52_SHARED_EXPERT_TP1" == 0 || "$GLM52_SHARED_EXPERT_TP1" == 1 ]] || {
  echo "GLM52_SHARED_EXPERT_TP1 must be 0 or 1" >&2
  exit 2
}
[[ "$GLM52_CUSTOM_ALL_REDUCE_IMPL" == legacy || "$GLM52_CUSTOM_ALL_REDUCE_IMPL" == v2 ]] || {
  echo "GLM52_CUSTOM_ALL_REDUCE_IMPL must be legacy or v2" >&2
  exit 2
}
[[ -d "$REPO/python/sglang" ]] || { echo "missing repo: $REPO" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "missing model: $MODEL" >&2; exit 2; }

source "$RUNTIME/env.sh"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
# env.sh may contain the cache path of the account that created the runtime.
# Re-apply the per-experiment directory after sourcing it so all ranks have a
# writable cache and baseline/candidates reuse the same compiled kernels.
export TRITON_CACHE_DIR="$GLM52_TRITON_CACHE_DIR"
export SGLANG_DG_CACHE_DIR="$GLM52_DEEP_GEMM_CACHE_DIR"
# Never run the rank-0-only all-M DeepGEMM sweep from inside a distributed
# model forward. With DP attention, peer ranks can already be waiting in the
# next collective while rank 0 synchronizes the sweep, which deadlocks the
# first real request. On-demand JIT is symmetric across ranks and the full
# target-shape warmup below the server boundary remains excluded from samples.
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_SHARED_EXPERT_TP1="$GLM52_SHARED_EXPERT_TP1"
# Custom AllReduce V2 can leave B300 DP8 eager prefill ranks spinning inside
# one-/two-shot GPU kernels. Legacy custom AR is still GPU-local (not NCCL),
# is stable for context construction, and remains common to every formal A/B.
if [[ "$GLM52_CUSTOM_ALL_REDUCE_IMPL" == v2 ]]; then
  export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1
else
  export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
fi
mkdir -p "$TRITON_CACHE_DIR" "$SGLANG_DG_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

exec "$RUNTIME/venv/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --skip-server-warmup \
  --tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention \
  --attention-backend dsa --moe-a2a-backend megamoe \
  --moe-runner-backend "$MOE_RUNNER_BACKEND" \
  --fp8-gemm-backend deep_gemm --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.835 --swa-full-tokens-ratio 0.075 \
  --page-size 64 --chunked-prefill-size 16384 \
  --cuda-graph-max-bs-decode 544 --dist-timeout 3600 --watchdog-timeout 1800
