#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-baseline_auto}
ROOT=${ROOT:-/mnt/b300-shared/home/gjy/sglang_huge}
REPO=${REPO:-$ROOT/.runtime/glm52_decode_worktree}
RUNTIME=${RUNTIME:-/mnt/b300-shared/home/gjy/data/agent4kernel/.runtime/b300}
MODEL=${MODEL:-/var/b300-shared/models/GLM-5.2-FP8}
RUN_DIR=${RUN_DIR:-$ROOT/.runtime/glm52_decode_nsys}
PYTHON=${PYTHON:-$RUNTIME/venv/bin/python}
PROFILE_STEPS=${PROFILE_STEPS:-200}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-auto}
GLM52_TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/.runtime/glm52_decode_cache}
PREFIX=$RUN_DIR/req64_100k_1k_${TAG}

[[ "$MOE_RUNNER_BACKEND" == auto || "$MOE_RUNNER_BACKEND" == deep_gemm ]] || {
  echo "MOE_RUNNER_BACKEND must be auto or deep_gemm" >&2
  exit 2
}
source "$RUNTIME/env.sh"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRITON_CACHE_DIR="$GLM52_TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$TRITON_CACHE_DIR" "$RUN_DIR"

for path in \
  "$REPO/python/sglang/benchmark/one_batch_server.py" \
  "$REPO/scripts/glm52_decode/validate_decode_result.py" \
  "$MODEL/config.json"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done
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
  echo "moe_runner_backend=$MOE_RUNNER_BACKEND"
  echo "semantics=req64,input_per_request=100000,output_per_request=1000,decode_tokens=63936"
  echo "profile_scope=post_last_first_token,steps=$PROFILE_STEPS,cuda_graph_nodes=host_only"
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
    --attention-backend dsa --moe-a2a-backend megamoe \
    --moe-runner-backend "$MOE_RUNNER_BACKEND" \
    --fp8-gemm-backend deep_gemm --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.835 --swa-full-tokens-ratio 0.075 \
    --page-size 64 --chunked-prefill-size 16384 \
    --cuda-graph-max-bs-decode 544 --dist-timeout 3600 --watchdog-timeout 1800 \
    --batch-size 64 --input-len 100000 --output-len 1000 \
    --temperature 0 --dataset-name random-ids --seed 4199 \
    --save-output-token-ids --client-stream-interval 64 --skip-warmup \
    --request-timeout 14400 --no-append-to-github-summary \
    --run-name "glm52_decode_req64_100k_1k_nsys_${TAG}" \
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
"$PYTHON" "$REPO/scripts/glm52_decode/validate_decode_result.py" \
  --result "$PREFIX.jsonl" --server-info "$PREFIX.server_info.json" \
  --expected-moe-runner "$MOE_RUNNER_BACKEND" >"$PREFIX.validated.json"
[[ -s "$PREFIX.nsys-rep" ]] || { echo "missing Nsys report" >&2; exit 1; }
/usr/local/cuda/bin/nsys export --type sqlite --force-overwrite=true \
  --output "$PREFIX.sqlite" "$PREFIX.nsys-rep"
/usr/local/cuda/bin/nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  --format csv "$PREFIX.sqlite" >"$PREFIX.stats.csv"
sha256sum "$PREFIX.nsys-rep" "$PREFIX.sqlite" "$PREFIX.log" \
  "$PREFIX.jsonl" "$PREFIX.server_info.json" "$PREFIX.validated.json" \
  "$PREFIX.stats.csv" "$PREFIX.manifest.txt" >"$PREFIX.SHA256SUMS"
echo "PREFIX=$PREFIX"
