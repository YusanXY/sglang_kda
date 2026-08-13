#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/b300-shared/home/gjy/sglang_huge}
REPO=${REPO:-$ROOT/.runtime/dsv4pro_decode_worktree}
RUNTIME=${RUNTIME:-/mnt/b300-shared/home/gjy/data/agent4kernel/.runtime/b300}
MODEL=${MODEL:-/var/b300-shared/models/DeepSeek-V4-Pro}
PYTHON=${PYTHON:-$RUNTIME/venv/bin/python}
CACHE=${CACHE:-$ROOT/.runtime/dsv4pro_decode_cache}

[[ -f "$REPO/python/sglang/__init__.py" ]] || {
  echo "missing SGLang worktree: $REPO" >&2
  exit 2
}
[[ -f "$MODEL/config.json" ]] || { echo "missing model: $MODEL" >&2; exit 2; }
mkdir -p "$CACHE/torch_extensions" "$CACHE/xdg" "$CACHE/triton"

source "$RUNTIME/env.sh"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=1024
export TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions"
export XDG_CACHE_HOME="$CACHE/xdg"
export TRITON_CACHE_DIR="$CACHE/triton"

cd "$REPO"
exec "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --tp 8 --dp 8 --ep 8 --enable-dp-attention \
  --moe-a2a-backend megamoe \
  --mem-fraction-static 0.835 \
  --swa-full-tokens-ratio 0.075 \
  --cuda-graph-max-bs-decode 544 \
  --host 127.0.0.1 --port 30000
