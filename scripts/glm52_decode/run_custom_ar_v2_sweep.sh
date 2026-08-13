#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/b300-shared/home/gjy/sglang_huge}
REPO=${REPO:-$ROOT/.runtime/glm52_decode_worktree}
RUNTIME=${RUNTIME:-/mnt/b300-shared/home/gjy/data/agent4kernel/.runtime/b300}
OUT=${OUT:-$ROOT/.runtime/glm52_custom_ar_v2_sweep_$(date -u +%Y%m%dT%H%M%SZ)}
BLOCKS=${BLOCKS:-16 24 32 40 48 64 96 148}

source "$RUNTIME/env.sh"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p "$OUT"
for blocks in $BLOCKS; do
  "$RUNTIME/venv/bin/torchrun" --standalone --nproc-per-node=8 \
    "$REPO/scripts/glm52_decode/bench_custom_ar_v2.py" \
    --blocks "$blocks" >"$OUT/blocks_${blocks}.jsonl" \
    2>"$OUT/blocks_${blocks}.log"
done
sha256sum "$OUT"/*.jsonl "$OUT"/*.log >"$OUT/SHA256SUMS"
echo "OUT=$OUT"
