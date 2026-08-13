#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/b300-shared/home/gjy/sglang_huge}
REPO=${REPO:-$ROOT/.runtime/glm52_decode_worktree}
RUNTIME=${RUNTIME:-/mnt/b300-shared/home/gjy/data/agent4kernel/.runtime/b300}
MODEL=${MODEL:-/var/b300-shared/models/GLM-5.2-FP8}
PORT=${PORT:-30000}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-auto}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/.runtime/glm52_decode_cache}

[[ "$MOE_RUNNER_BACKEND" == auto || "$MOE_RUNNER_BACKEND" == deep_gemm ]] || {
  echo "MOE_RUNNER_BACKEND must be auto or deep_gemm" >&2
  exit 2
}
[[ -d "$REPO/python/sglang" ]] || { echo "missing repo: $REPO" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "missing model: $MODEL" >&2; exit 2; }

source "$RUNTIME/env.sh"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

exec "$RUNTIME/venv/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention \
  --attention-backend dsa --moe-a2a-backend megamoe \
  --moe-runner-backend "$MOE_RUNNER_BACKEND" \
  --fp8-gemm-backend deep_gemm --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.835 --swa-full-tokens-ratio 0.075 \
  --page-size 64 --chunked-prefill-size 16384 \
  --cuda-graph-max-bs-decode 544 --dist-timeout 3600 --watchdog-timeout 1800
