#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-baseline}
ROOT=${ROOT:-/mnt/b300-shared/home/gjy/sglang_huge}
REPO=${REPO:-$ROOT/.runtime/dsv4pro_decode_worktree}
RUNTIME=${RUNTIME:-/mnt/b300-shared/home/gjy/data/agent4kernel/.runtime/b300}
MODEL=${MODEL:-/var/b300-shared/models/DeepSeek-V4-Pro}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
RUN_DIR=${RUN_DIR:-$ROOT/.runtime/dsv4pro_decode_nsys}
PYTHON=${PYTHON:-$RUNTIME/venv/bin/python}
PROFILE_STEPS=${PROFILE_STEPS:-200}
PREFIX=$RUN_DIR/req64_100k_1k_${TAG}

source "$RUNTIME/env.sh"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=1024

for path in \
  "$REPO/python/sglang/benchmark/one_batch_server.py" \
  "$REPO/scripts/dsv4_pro_decode/validate_decode_result.py" \
  "$MODEL/config.json"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
for suffix in nsys-rep sqlite log jsonl server_info.json validated.json stats.csv manifest.txt SHA256SUMS; do
  [[ ! -e "$PREFIX.$suffix" ]] || {
    echo "refusing to overwrite $PREFIX.$suffix" >&2
    exit 2
  }
done

{
  echo "host=$(hostname)"
  echo "repo=$REPO"
  echo "commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "model=$MODEL"
  echo "semantics=req64,input_per_request=100000,output_per_request=1000,decode_tokens=63936"
  echo "profile_scope=post_last_first_token,steps=$PROFILE_STEPS"
  nvidia-smi -L
  nvidia-smi topo -m
} >"$PREFIX.manifest.txt"

/usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node:host-only --force-overwrite=true -o "$PREFIX" \
  "$PYTHON" -m sglang.benchmark.one_batch_server \
    --model-path "$MODEL" --trust-remote-code \
    --tp 8 --dp 8 --ep 8 --enable-dp-attention \
    --moe-a2a-backend megamoe \
    --mem-fraction-static 0.835 --swa-full-tokens-ratio 0.075 \
    --cuda-graph-max-bs-decode 544 \
    --batch-size 64 --input-len 100000 --output-len 1000 \
    --temperature 0 --dataset-name random-ids --seed 42 \
    --save-output-token-ids \
    --client-stream-interval 64 --skip-warmup \
    --request-timeout 14400 --no-append-to-github-summary \
    --run-name "dsv4pro_decode_req64_100k_1k_nsys_${TAG}" \
    --profile --profile-only --profile-activities CUDA_PROFILER \
    --profile-decode-after-first-token --profile-steps "$PROFILE_STEPS" \
    --profile-output-dir "$PREFIX.profile_meta" \
    --result-filename "$PREFIX.jsonl" >"$PREFIX.log" 2>&1

mapfile -t server_info_files < <(find "$PREFIX.profile_meta" -name server_args.json -type f)
[[ "${#server_info_files[@]}" -eq 1 ]] || {
  echo "expected exactly one profiler server_args.json, got ${#server_info_files[@]}" >&2
  exit 1
}
cp "${server_info_files[0]}" "$PREFIX.server_info.json"
"$PYTHON" "$REPO/scripts/dsv4_pro_decode/validate_decode_result.py" \
  --result "$PREFIX.jsonl" --server-info "$PREFIX.server_info.json" \
  >"$PREFIX.validated.json"
[[ -s "$PREFIX.nsys-rep" ]] || { echo "missing Nsys report" >&2; exit 1; }
/usr/local/cuda/bin/nsys export --type sqlite --force-overwrite=true \
  --output "$PREFIX.sqlite" "$PREFIX.nsys-rep"
/usr/local/cuda/bin/nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  --format csv "$PREFIX.sqlite" >"$PREFIX.stats.csv"
sha256sum "$PREFIX.nsys-rep" "$PREFIX.sqlite" "$PREFIX.log" \
  "$PREFIX.jsonl" "$PREFIX.server_info.json" "$PREFIX.validated.json" \
  "$PREFIX.stats.csv" "$PREFIX.manifest.txt" >"$PREFIX.SHA256SUMS"
echo "PREFIX=$PREFIX"
