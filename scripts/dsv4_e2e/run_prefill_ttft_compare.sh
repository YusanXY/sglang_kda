#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO_ROOT:-/home/gjy/data/agent4kernel/sglang_kda_e2e}
MODEL=${DSV4_MODEL:-/mnt/SFS-Shared/guojingyu/models/DeepSeek-V4-Flash}
PYTHON=${PYTHON_BIN:-/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops/.runtime/venv/bin/python}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
CUDA_CCCL_INCLUDE=${CUDA_CCCL_INCLUDE:-$CUDA_HOME/targets/x86_64-linux/include/cccl}
PAIRS=${PAIRS:-5}
IDLE_TIMEOUT_SECONDS=${IDLE_TIMEOUT_SECONDS:-1800}
CACHE_HIT_TOLERANCE=${CACHE_HIT_TOLERANCE:-0.01}
GPU_IDS=0,1,2,3
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=${1:-$REPO/.runtime/dsv4_e2e_prefill_ttft/$STAMP}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VALIDATOR=$SCRIPT_DIR/validate_prefill_ttft_result.py
SUMMARIZER=$SCRIPT_DIR/summarize_prefill_ttft.py

CACHED_HISTORY=65536
NEW_CHUNK=4096
INPUT_LEN=$((CACHED_HISTORY + NEW_CHUNK))
CONTEXT_CAPACITY=73728
CACHE_HIT_RATE=0.9411764705882353

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
  printf 'hostname=%s\n' "$(hostname)"
  printf 'repo=%s\n' "$REPO"
  printf 'git_head=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
  printf 'git_branch=%s\n' "$(git -C "$REPO" branch --show-current)"
  printf 'cuda_visible_devices=%s\n' "$GPU_IDS"
  printf 'cuda_home=%s\ncuda_cccl_include=%s\n' "$CUDA_HOME" "$CUDA_CCCL_INCLUDE"
  printf 'nvcc_version=%s\n' "$("$CUDA_HOME/bin/nvcc" --version | tail -n 1)"
  printf 'model=%s\npairs=%s\nbatch_size=1\n' "$MODEL" "$PAIRS"
  printf 'cached_history=%s\nnew_chunk=%s\ninput_len=%s\n' "$CACHED_HISTORY" "$NEW_CHUNK" "$INPUT_LEN"
  printf 'output_len=1\ncontext_capacity=%s\ncache_hit_rate=%s\n' "$CONTEXT_CAPACITY" "$CACHE_HIT_RATE"
  printf 'watchdog_timeout_seconds=2400\nrequest_timeout_seconds=2400\n'
  printf 'order=alternating native/huge_kernel first by pair\n'
} | tee "$RUN_ROOT/experiment.txt"

wait_gpu_idle() {
  local waited=0
  while true; do
    local pids
    pids=$(nvidia-smi --id="$GPU_IDS" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    if [[ -z "$pids" ]]; then
      return 0
    fi
    if (( waited >= IDLE_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for all GPUs to become idle" >&2
      nvidia-smi --id="$GPU_IDS" >&2 || true
      return 1
    fi
    echo "Waiting for GPUs to become idle (${waited}s elapsed; pids: ${pids//$'\n'/,})"
    sleep 10
    waited=$((waited + 10))
  done
}

gpu_pids() {
  nvidia-smi --id="$GPU_IDS" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
}

foreign_gpu_processes() {
  local pid owner args
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)
    if [[ -n "$owner" && "$owner" != "$(id -un)" ]]; then
      args=$(ps -o args= -p "$pid" 2>/dev/null || true)
      printf '%s\t%s\t%s\n' "$pid" "$owner" "$args"
    fi
  done < <(
    fuser /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 2>/dev/null \
      | tr ' ' '\n' | sed '/^[[:space:]]*$/d' | sort -u || true
  )
}

run_one() {
  local pair=$1 ordinal=$2 backend=$3 stem result log validation audit
  stem=$(printf '%02d_pair%02d_%s' "$ordinal" "$pair" "$backend")
  result=$RUN_ROOT/$stem.jsonl
  log=$RUN_ROOT/$stem.log
  validation=$RUN_ROOT/$stem.validation.json
  audit=$RUN_ROOT/$stem.gpu-process-audit.tsv

  wait_gpu_idle
  nvidia-smi --id="$GPU_IDS" --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits > "$RUN_ROOT/$stem.gpu-before.csv"
  local cmd=(
    "$PYTHON" -m sglang.benchmark.one_batch_server
    --model-path "$MODEL" --trust-remote-code
    --tp-size 4 --ep-size 4
    --attention-backend dsv4
    --moe-runner-backend flashinfer_mxfp4
    --disable-flashinfer-autotune
    --dsv4-worker-backend "$backend"
    --context-length "$CONTEXT_CAPACITY"
    --max-total-tokens "$CONTEXT_CAPACITY"
    --max-running-requests 1
    --max-prefill-tokens "$NEW_CHUNK"
    --mem-fraction-static 0.75
    --page-size 256 --swa-full-tokens-ratio 0.1
    --chunked-prefill-size "$NEW_CHUNK"
    --cuda-graph-backend-decode disabled
    --cuda-graph-backend-prefill disabled
    --watchdog-timeout 2400 --request-timeout 2400
    --skip-server-warmup
    --disable-overlap-schedule --enable-metrics --random-seed 42
    --run-name "pair${pair}_${backend}"
    --batch-size 1 --input-len "$INPUT_LEN" --output-len 1
    --dataset-name random-ids --cache-hit-rate "$CACHE_HIT_RATE" --seed 42
    --share-cached-prefix-across-batch --warmup-cached-prefill-shape
    --skip-warmup --no-append-to-github-summary
    --result-filename "$result"
  )

  {
    printf 'START pair=%s ordinal=%s backend=%s utc=%s\n' "$pair" "$ordinal" "$backend" "$(date -u +%FT%TZ)"
    printf 'COMMAND'; printf ' %q' "${cmd[@]}"; printf '\n'
  } | tee "$log"
  set +e
  "${cmd[@]}" > >(tee -a "$log") 2>&1 &
  local server_pid=$! audit_failed=0 foreign status
  while kill -0 "$server_pid" 2>/dev/null; do
    foreign=$(foreign_gpu_processes)
    if [[ -n "$foreign" ]]; then
      audit_failed=1
      {
        printf 'utc\tpid\tuser\tcommand\n'
        while IFS= read -r line; do
          printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$line"
        done <<< "$foreign"
      } | tee "$audit" -a "$log"
      kill -TERM "$server_pid" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  wait "$server_pid"
  status=$?
  foreign=$(foreign_gpu_processes)
  if [[ -n "$foreign" ]]; then
    audit_failed=1
    {
      [[ -s "$audit" ]] || printf 'utc\tpid\tuser\tcommand\n'
      while IFS= read -r line; do
        printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$line"
      done <<< "$foreign"
    } | tee -a "$audit" "$log"
  fi
  if (( audit_failed )); then
    status=125
  fi
  set -e
  printf 'END pair=%s ordinal=%s backend=%s status=%s utc=%s\n' "$pair" "$ordinal" "$backend" "$status" "$(date -u +%FT%TZ)" | tee -a "$log"
  nvidia-smi --id="$GPU_IDS" --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits > "$RUN_ROOT/$stem.gpu-after.csv" || true
  if (( status != 0 )); then
    if (( status == 125 )); then
      echo "Benchmark rejected because a foreign GPU process appeared: $stem" >&2
    fi
    echo "Benchmark failed: $stem (status=$status)" >&2
    return "$status"
  fi
  "$PYTHON" "$VALIDATOR" --result "$result" --log "$log" \
    --backend "$backend" --cache-hit-tolerance "$CACHE_HIT_TOLERANCE" \
    --batch-size 1 --cached-history-per-request "$CACHED_HISTORY" \
    --new-tokens-per-request "$NEW_CHUNK" --output-len 1 \
    --forward-batch-m "$NEW_CHUNK" --require-shape-warmup \
    | tee "$validation"
  printf '%s\t%s\t%s\t%s\t%s\n' "$pair" "$ordinal" "$backend" "$result" "$log" >> "$MANIFEST"
  wait_gpu_idle
}

ordinal=1
for pair in $(seq 1 "$PAIRS"); do
  if (( pair % 2 == 1 )); then
    order=(native huge_kernel)
  else
    order=(huge_kernel native)
  fi
  for backend in "${order[@]}"; do
    run_one "$pair" "$ordinal" "$backend"
    ordinal=$((ordinal + 1))
  done
done

set +e
"$PYTHON" "$SUMMARIZER" --manifest "$MANIFEST" --output-dir "$RUN_ROOT" \
  --min-pairs "$PAIRS" --cache-hit-tolerance "$CACHE_HIT_TOLERANCE" \
  --batch-size 1 --cached-history-per-request "$CACHED_HISTORY" \
  --new-tokens-per-request "$NEW_CHUNK" --output-len 1 \
  --forward-batch-m "$NEW_CHUNK" --require-shape-warmup
summary_status=$?
set -e
if (( summary_status != 0 )); then
  printf 'utc_end=%s\nstatus=FAILED\n' "$(date -u +%FT%TZ)" | tee -a "$RUN_ROOT/experiment.txt"
  echo "FAILED: formal comparison gate did not pass; artifacts: $RUN_ROOT" >&2
  exit "$summary_status"
fi
printf 'utc_end=%s\nstatus=PASS\n' "$(date -u +%FT%TZ)" | tee -a "$RUN_ROOT/experiment.txt"
echo "Formal comparison complete: $RUN_ROOT"
