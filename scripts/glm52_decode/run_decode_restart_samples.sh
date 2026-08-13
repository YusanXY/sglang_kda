#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/b300-shared/home/gjy/sglang_huge}
REPO=${REPO:-$ROOT/.runtime/glm52_decode_worktree}
SAMPLE_ROOT=${SAMPLE_ROOT:-$ROOT/.runtime/glm52_decode_restart_samples}
SAMPLES=${SAMPLES:-5}
GLM52_CUSTOM_ALL_REDUCE_IMPL=${GLM52_CUSTOM_ALL_REDUCE_IMPL:-legacy}
GLM52_FP8_GEMM_BACKEND=${GLM52_FP8_GEMM_BACKEND:-deep_gemm}
PORT=${PORT:-30000}
SERVER_WAIT_SECONDS=${SERVER_WAIT_SECONDS:-420}

[[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "SAMPLES must be positive" >&2; exit 2; }
mkdir -p "$SAMPLE_ROOT"

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup_server EXIT INT TERM

for sample in $(seq 1 "$SAMPLES"); do
  sample_dir="$SAMPLE_ROOT/sample_$sample"
  server_log="$SAMPLE_ROOT/server_$sample.log"
  [[ ! -e "$sample_dir" && ! -e "$server_log" ]] || {
    echo "refusing to overwrite sample $sample in $SAMPLE_ROOT" >&2
    exit 2
  }

  # Use the original full-TP control broadcast for repeated cold starts. The
  # local fan-out optimization is not yet reliable before every DP worker has
  # published its first activity snapshot; this one-shot CPU barrier is
  # outside the measured decode window.
  setsid env \
    MOE_RUNNER_BACKEND=flashinfer_trtllm_routed \
    GLM52_COLOCATE_DP_BATCH=1 \
    GLM52_LOCAL_DP_CONTROL=0 \
    GLM52_CUSTOM_ALL_REDUCE_IMPL="$GLM52_CUSTOM_ALL_REDUCE_IMPL" \
    GLM52_FP8_GEMM_BACKEND="$GLM52_FP8_GEMM_BACKEND" \
    GLM52_FLASHINFER_DIRECT_OUTPUT=1 \
    GLM52_FLASHINFER_FUSED_ROUTING_PACK=1 \
    PORT="$PORT" \
    bash "$REPO/scripts/glm52_decode/launch_server.sh" \
    >"$server_log" 2>&1 < /dev/null &
  server_pid=$!

  ready=0
  for _ in $(seq 1 $((SERVER_WAIT_SECONDS / 5))); do
    if curl --connect-timeout 2 --max-time 5 -fsS \
      "http://127.0.0.1:$PORT/server_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -n 160 "$server_log" >&2
      exit 1
    fi
    sleep 5
  done
  [[ "$ready" == 1 ]] || { tail -n 160 "$server_log" >&2; exit 1; }

  RUN_DIR="$sample_dir" RUNS=1 PORT="$PORT" \
    MOE_RUNNER_BACKEND=flashinfer_trtllm_routed \
    EXPECTED_SHARED_EXPERT_PARALLELISM=tp8 \
    EXPECTED_FLASHINFER_DIRECT_OUTPUT=1 \
    EXPECTED_FLASHINFER_FUSED_ROUTING_PACK=1 \
    EXPECTED_CUSTOM_ALL_REDUCE="$GLM52_CUSTOM_ALL_REDUCE_IMPL" \
    EXPECTED_FP8_GEMM_BACKEND="$GLM52_FP8_GEMM_BACKEND" \
    bash "$REPO/scripts/glm52_decode/run_decode_n5.sh"
  cleanup_server
done

echo "SAMPLE_ROOT=$SAMPLE_ROOT"
