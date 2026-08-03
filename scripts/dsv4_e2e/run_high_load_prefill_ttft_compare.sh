#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO_ROOT:-/home/gjy/data/agent4kernel/sglang_kda_e2e}
MODEL=${DSV4_MODEL:-/mnt/SFS-Shared/guojingyu/models/DeepSeek-V4-Flash}
PYTHON=${PYTHON_BIN:-/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops/.runtime/venv/bin/python}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
CUDA_CCCL_INCLUDE=${CUDA_CCCL_INCLUDE:-$CUDA_HOME/targets/x86_64-linux/include/cccl}
PAIRS=${PAIRS:-5}
IDLE_TIMEOUT_SECONDS=${IDLE_TIMEOUT_SECONDS:-3600}
IDLE_STABLE_SECONDS=${IDLE_STABLE_SECONDS:-30}
CACHE_HIT_TOLERANCE=${CACHE_HIT_TOLERANCE:-0.01}
GPU_IDS=0,1,2,3
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=${1:-$REPO/.runtime/dsv4_e2e_high_load_ttft/$STAMP}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALIDATOR=$SCRIPT_DIR/validate_prefill_ttft_result.py
SUMMARIZER=$SCRIPT_DIR/summarize_prefill_ttft.py

BATCH_SIZE=16
CACHED_PER_REQUEST=16384
NEW_PER_REQUEST=256
AGGREGATE_NEW=$((BATCH_SIZE * NEW_PER_REQUEST))
AGGREGATE_CACHED=$((BATCH_SIZE * CACHED_PER_REQUEST))
INPUT_LEN=$((CACHED_PER_REQUEST + NEW_PER_REQUEST))
CONTEXT_CAPACITY=73728
MAX_TOTAL_TOKENS=294912
CACHE_HIT_RATE=0.9846153846153847

if (( PAIRS < 5 )); then
  echo "PAIRS must be at least 5 for a formal comparison; got $PAIRS" >&2
  exit 2
fi
for required in "$REPO" "$MODEL" "$PYTHON" "$VALIDATOR" "$SUMMARIZER" \
  "$CUDA_HOME/bin/nvcc" "$CUDA_CCCL_INCLUDE/cuda/atomic"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path does not exist: $required" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT" "$RUN_ROOT/torch_extensions"
MANIFEST=$RUN_ROOT/manifest.tsv
printf 'pair\tordinal\tbackend\tresult\tlog\n' > "$MANIFEST"

export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_CCCL_INCLUDE${CPATH:+:$CPATH}"
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/torch_extensions"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_DSV4_FP4_EXPERTS=1

{
  printf 'utc_start=%s\n' "$(date -u +%FT%TZ)"
  printf 'hostname=%s\nrepo=%s\n' "$(hostname)" "$REPO"
  printf 'git_head=%s\ngit_branch=%s\n' \
    "$(git -C "$REPO" rev-parse HEAD)" "$(git -C "$REPO" branch --show-current)"
  printf 'cuda_visible_devices=%s\ncuda_home=%s\n' "$GPU_IDS" "$CUDA_HOME"
  printf 'model=%s\npairs=%s\nbatch_size=%s\n' "$MODEL" "$PAIRS" "$BATCH_SIZE"
  printf 'cached_per_request=%s\nnew_per_request=%s\n' "$CACHED_PER_REQUEST" "$NEW_PER_REQUEST"
  printf 'aggregate_cached=%s\naggregate_new=%s\n' "$AGGREGATE_CACHED" "$AGGREGATE_NEW"
  printf 'input_len=%s\noutput_len=1\ncontext_capacity=%s\n' "$INPUT_LEN" "$CONTEXT_CAPACITY"
  printf 'cache_hit_rate=%s\norder=native,huge_kernel repeated by pair\n' "$CACHE_HIT_RATE"
} | tee "$RUN_ROOT/experiment.txt"

gpu_pids() {
  nvidia-smi --id="$GPU_IDS" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
}

wait_gpu_idle() {
  local waited=0 stable=0 pids
  while true; do
    pids=$(gpu_pids)
    if [[ -z "$pids" ]]; then
      stable=$((stable + 5))
      if (( stable >= IDLE_STABLE_SECONDS )); then
        return 0
      fi
    else
      stable=0
    fi
    if (( waited >= IDLE_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for all GPUs to remain idle" >&2
      nvidia-smi --id="$GPU_IDS" >&2 || true
      return 1
    fi
    echo "Waiting for stable GPU idle (${waited}s elapsed, stable=${stable}s; pids: ${pids//$'\n'/,})"
    sleep 5
    waited=$((waited + 5))
  done
}

run_one() {
  local pair=$1 ordinal=$2 backend=$3 stem result log validation
  stem=$(printf '%02d_pair%02d_%s' "$ordinal" "$pair" "$backend")
  result=$RUN_ROOT/$stem.jsonl
  log=$RUN_ROOT/$stem.log
  validation=$RUN_ROOT/$stem.validation.json

  wait_gpu_idle
  nvidia-smi --id="$GPU_IDS" --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits > "$RUN_ROOT/$stem.gpu-before.csv"
  local cmd=(
    "$PYTHON" -m sglang.benchmark.one_batch_server
    --model-path "$MODEL" --trust-remote-code
    --tp-size 4 --ep-size 4
    --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4
    --disable-flashinfer-autotune --dsv4-worker-backend "$backend"
    --context-length "$CONTEXT_CAPACITY" --max-total-tokens "$MAX_TOTAL_TOKENS"
    --max-running-requests "$BATCH_SIZE" --max-prefill-tokens "$AGGREGATE_NEW"
    --mem-fraction-static 0.88 --page-size 256 --swa-full-tokens-ratio 0.1
    --chunked-prefill-size "$AGGREGATE_NEW"
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
    --skip-server-warmup --skip-warmup --disable-overlap-schedule
    --enable-metrics --random-seed 42 --watchdog-timeout 2400
    --request-timeout 2400 --run-name "high_pair${pair}_${backend}"
    --batch-size "$BATCH_SIZE" --input-len "$INPUT_LEN" --output-len 1
    --dataset-name random-ids --cache-hit-rate "$CACHE_HIT_RATE" --seed 42
    --share-cached-prefix-across-batch --warmup-cached-prefill-shape
    --no-append-to-github-summary --result-filename "$result"
  )

  {
    printf 'START pair=%s ordinal=%s backend=%s utc=%s\n' "$pair" "$ordinal" "$backend" "$(date -u +%FT%TZ)"
    printf 'COMMAND'; printf ' %q' "${cmd[@]}"; printf '\n'
  } | tee "$log"
  set +e
  "${cmd[@]}" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  set -e
  printf 'END pair=%s ordinal=%s backend=%s status=%s utc=%s\n' "$pair" "$ordinal" "$backend" "$status" "$(date -u +%FT%TZ)" | tee -a "$log"
  nvidia-smi --id="$GPU_IDS" --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits > "$RUN_ROOT/$stem.gpu-after.csv" || true
  if (( status != 0 )); then
    echo "Benchmark failed: $stem (status=$status)" >&2
    return "$status"
  fi
  "$PYTHON" "$VALIDATOR" --result "$result" --log "$log" --backend "$backend" \
    --cache-hit-tolerance "$CACHE_HIT_TOLERANCE" --batch-size "$BATCH_SIZE" \
    --cached-history-per-request "$CACHED_PER_REQUEST" \
    --new-tokens-per-request "$NEW_PER_REQUEST" --output-len 1 \
    --require-shape-warmup | tee "$validation"
  printf '%s\t%s\t%s\t%s\t%s\n' "$pair" "$ordinal" "$backend" "$result" "$log" >> "$MANIFEST"
  wait_gpu_idle
}

ordinal=1
for pair in $(seq 1 "$PAIRS"); do
  run_one "$pair" "$ordinal" native
  ordinal=$((ordinal + 1))
  run_one "$pair" "$ordinal" huge_kernel
  ordinal=$((ordinal + 1))
done

set +e
"$PYTHON" "$SUMMARIZER" --manifest "$MANIFEST" --output-dir "$RUN_ROOT" \
  --min-pairs "$PAIRS" --cache-hit-tolerance "$CACHE_HIT_TOLERANCE" \
  --batch-size "$BATCH_SIZE" --cached-history-per-request "$CACHED_PER_REQUEST" \
  --new-tokens-per-request "$NEW_PER_REQUEST" --output-len 1 --require-shape-warmup
summary_status=$?
set -e
if (( summary_status != 0 )); then
  printf 'utc_end=%s\nstatus=FAILED\n' "$(date -u +%FT%TZ)" | tee -a "$RUN_ROOT/experiment.txt"
  echo "FAILED: high-load formal comparison gate did not pass; artifacts: $RUN_ROOT" >&2
  exit "$summary_status"
fi
printf 'utc_end=%s\nstatus=PASS\n' "$(date -u +%FT%TZ)" | tee -a "$RUN_ROOT/experiment.txt"
echo "High-load formal comparison complete: $RUN_ROOT"
