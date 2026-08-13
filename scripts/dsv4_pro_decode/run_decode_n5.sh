#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/b300-shared/home/gjy/sglang_huge}
REPO=${REPO:-$ROOT/.runtime/dsv4pro_decode_worktree}
RUNTIME=${RUNTIME:-/mnt/b300-shared/home/gjy/data/agent4kernel/.runtime/b300}
MODEL=${MODEL:-/var/b300-shared/models/DeepSeek-V4-Pro}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
RUN_DIR=${RUN_DIR:-$ROOT/.runtime/dsv4pro_decode_n5}
PYTHON=${PYTHON:-$RUNTIME/venv/bin/python}
RUNS=${RUNS:-5}

source "$RUNTIME/env.sh"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "RUNS must be positive" >&2; exit 2; }
for path in \
  "$REPO/python/sglang/benchmark/one_batch_server.py" \
  "$REPO/scripts/dsv4_pro_decode/validate_decode_result.py" \
  "$REPO/scripts/dsv4_pro_decode/summarize_decode_results.py" \
  "$MODEL/config.json"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }
done

mkdir -p "$RUN_DIR"
SERVER_INFO=$RUN_DIR/server_info.json
curl -fsS "$BASE_URL/server_info" -o "$SERVER_INFO"

result_args=()
for run in $(seq 1 "$RUNS"); do
  result=$RUN_DIR/run_${run}.jsonl
  log=$RUN_DIR/run_${run}.log
  checked=$RUN_DIR/run_${run}.validated.json
  [[ ! -e "$result" && ! -e "$log" && ! -e "$checked" ]] || {
    echo "refusing to overwrite run $run in $RUN_DIR" >&2
    exit 2
  }
  "$PYTHON" -m sglang.benchmark.one_batch_server \
    --model-path None --base-url "$BASE_URL" \
    --local-tokenizer-path "$MODEL" \
    --batch-size 64 --input-len 100000 --output-len 1000 \
    --temperature 0 --dataset-name random-ids --seed 42 \
    --save-output-token-ids \
    --client-stream-interval 64 --skip-warmup \
    --request-timeout 14400 --no-append-to-github-summary \
    --run-name "dsv4pro_decode_req64_100k_1k_run${run}" \
    --result-filename "$result" >"$log" 2>&1
  "$PYTHON" "$REPO/scripts/dsv4_pro_decode/validate_decode_result.py" \
    --result "$result" --server-info "$SERVER_INFO" >"$checked"
  result_args+=(--result "$result")
done

"$PYTHON" "$REPO/scripts/dsv4_pro_decode/summarize_decode_results.py" \
  "${result_args[@]}" --server-info "$SERVER_INFO" --output-dir "$RUN_DIR"
sha256sum "$SERVER_INFO" "$RUN_DIR"/run_*.jsonl "$RUN_DIR"/run_*.log \
  "$RUN_DIR"/run_*.validated.json "$RUN_DIR"/summary.json \
  "$RUN_DIR"/summary.md >"$RUN_DIR/SHA256SUMS"
echo "RUN_DIR=$RUN_DIR"
