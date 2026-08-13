#!/usr/bin/env bash
set -euo pipefail

PREFIX=${1:?usage: finalize_nsys_artifact.sh PREFIX}
REPO=${REPO:-/mnt/b300-shared/home/gjy/sglang_huge/.runtime/glm52_decode_worktree}
RUNTIME=${RUNTIME:-/mnt/b300-shared/home/gjy/data/agent4kernel/.runtime/b300}
EXPECTED_MOE_RUNNER=${EXPECTED_MOE_RUNNER:-flashinfer_trtllm_routed}
EXPECTED_CUSTOM_ALL_REDUCE=${EXPECTED_CUSTOM_ALL_REDUCE:-hybrid_graph_v2}
EXPECTED_FP8_GEMM_BACKEND=${EXPECTED_FP8_GEMM_BACKEND:-deep_gemm}
PYTHON=${PYTHON:-$RUNTIME/venv/bin/python}

mapfile -t server_info_files < <(find "$PREFIX.profile_meta" -name server_args.json -type f)
[[ "${#server_info_files[@]}" -eq 1 ]] || {
  echo "expected exactly one profiler server_args.json, got ${#server_info_files[@]}" >&2
  exit 1
}
cp "${server_info_files[0]}" "$PREFIX.server_info.json"
"$PYTHON" "$REPO/scripts/glm52_decode/validate_decode_result.py" \
  --result "$PREFIX.jsonl" --server-info "$PREFIX.server_info.json" \
  --expected-moe-runner "$EXPECTED_MOE_RUNNER" \
  --expected-custom-all-reduce "$EXPECTED_CUSTOM_ALL_REDUCE" \
  --expected-fp8-gemm-backend "$EXPECTED_FP8_GEMM_BACKEND" \
  --expected-flashinfer-direct-output \
  --expected-flashinfer-fused-routing-pack >"$PREFIX.validated.json"
/usr/local/cuda/bin/nsys export --type sqlite --force-overwrite=true \
  --output "$PREFIX.sqlite" "$PREFIX.nsys-rep"
/usr/local/cuda/bin/nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  --format csv "$PREFIX.sqlite" >"$PREFIX.stats.csv"
sha256sum "$PREFIX.nsys-rep" "$PREFIX.sqlite" "$PREFIX.jsonl" \
  "$PREFIX.server_info.json" "$PREFIX.validated.json" "$PREFIX.stats.csv" \
  "$PREFIX.manifest.txt" >"$PREFIX.SHA256SUMS"
cat "$PREFIX.validated.json"
ls -lh "$PREFIX.nsys-rep" "$PREFIX.sqlite" "$PREFIX.stats.csv"
