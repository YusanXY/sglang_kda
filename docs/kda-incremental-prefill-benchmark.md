# DeepSeek V4 增量 chunked prefill benchmark 策略

本文定义 SGLang KDA 对 DeepSeek V4 长上下文增量 prefill 的端到端对比方法。
目标是在相同模型、输入、并行配置和 cache 状态下，只切换 KDA profile，比较
原生 SGLang 与外部 reference/candidate adapter。

## 1. 固定语义

本文使用的 `context=65536` 是本次 chunk 完成后的总 prompt 长度，不是纯历史
长度：

| 项目 | 值 |
|---|---:|
| 最终 prompt context `L` | 65536 |
| 已缓存历史 `H` | 61440 |
| 新增 chunk `M` | 4096 |
| cache hit rate `H / L` | 0.9375 |
| measured request / batch size | 1 |
| output tokens | 1 |

这与 `deepseek_v4_chunked_mega_mqa_logits` 的 `m=4096` 正式 case 一致：查询
处理结束时 raw context 为 65536。

`--context-length` 是服务端容量上限，不等于被测历史长度。DeepSeek V4 会为
生成和内部状态预留 token，因此本策略将容量设置为 69632；实际输入仍由
`--input-len 65536` 固定。

若目标改成“纯历史为 65536，再追加 4096”，则应改为：

```text
input_len = 65536 + 4096 = 69632
cache_hit_rate = 65536 / 69632 = 0.9411764705882353
context_length >= input_len + 模型预留
```

## 2. 为什么使用 one_batch_server

`python/sglang/benchmark/one_batch.py` 的 latency 路径会把完整 synthetic input
一次交给 `ModelRunner.extend`。它不会根据 `--chunked-prefill-size` 自动构造一段
已经存在的历史 KV。

`python/sglang/benchmark/one_batch_server.py --cache-hit-rate` 会执行以下流程：

1. 清空 radix/KV cache；
2. 生成总长为 `input_len` 的同一份 token IDs；
3. 单独发送前 `int(input_len * cache_hit_rate)` 个 token，建立历史 KV；
4. 在 warmup 请求结束后开始计时；
5. 发送完整 token IDs，prefix hit 后只 extend 未命中的 suffix；
6. 记录端到端 latency 和 TTFT。

因此，本策略的正式请求必须在 server 日志中出现：

```text
#new-seq: 1, #new-token: 4096, #cached-token: 61440
```

cache warmup 是额外的准备请求，不计入 measured request 数。warmup 和正式请求
都生成 1 个 token。

## 3. 前置条件

本文验证过的环境是 Linux、NVIDIA CUDA、4 张 NVIDIA B200：

```bash
export REPO=/home/gjy/data/agent4kernel/sglang_kda
export MODEL=/mnt/SFS-Shared/guojingyu/models/DeepSeek-V4-Flash
export PYTHON=/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops/.runtime/venv/bin/python
export KDA_CONFIG=/absolute/path/to/kda-reference-routes.yaml
export KDA_PROFILE=deepseek_v4_reference_ctx64k

export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$REPO/.runtime/torch_extensions"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_DSV4_FP4_EXPERTS=1
cd "$REPO"
```

运行前确认四张 GPU 没有其他 compute process。两轮必须使用同一个 SGLang
commit、Python 环境、模型目录和 `TORCH_EXTENSIONS_DIR`。

## 4. Baseline

baseline 不传 `--kda-kernel-config`，并显式关闭 KDA profile：

```bash
"$PYTHON" -m sglang.benchmark.one_batch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --tp-size 4 \
  --ep-size 4 \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune \
  --context-length 69632 \
  --max-total-tokens 69632 \
  --max-running-requests 1 \
  --mem-fraction-static 0.75 \
  --page-size 256 \
  --swa-full-tokens-ratio 0.1 \
  --chunked-prefill-size 4096 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --disable-overlap-schedule \
  --kda-kernel-profile off \
  --batch-size 1 \
  --input-len 65536 \
  --output-len 1 \
  --dataset-name random-ids \
  --cache-hit-rate 0.9375 \
  --skip-warmup \
  --no-append-to-github-summary \
  --result-filename baseline.jsonl 2>&1 | tee baseline.log
```

`--skip-warmup` 只跳过 benchmark 客户端额外的通用 case；server 自身 warmup 和
61440-token prefix warmup 仍会执行。

## 5. KDA/reference

只增加 config/profile，其他参数与 baseline 完全相同：

```bash
"$PYTHON" -m sglang.benchmark.one_batch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --tp-size 4 \
  --ep-size 4 \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune \
  --context-length 69632 \
  --max-total-tokens 69632 \
  --max-running-requests 1 \
  --mem-fraction-static 0.75 \
  --page-size 256 \
  --swa-full-tokens-ratio 0.1 \
  --chunked-prefill-size 4096 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --disable-overlap-schedule \
  --kda-kernel-config "$KDA_CONFIG" \
  --kda-kernel-profile "$KDA_PROFILE" \
  --batch-size 1 \
  --input-len 65536 \
  --output-len 1 \
  --dataset-name random-ids \
  --cache-hit-rate 0.9375 \
  --skip-warmup \
  --no-append-to-github-summary \
  --result-filename reference.jsonl 2>&1 | tee reference.log
```

每个 TP rank 都应打印 profile 和 route，例如：

```text
KDA kernel routing enabled: profile=deepseek_v4_reference_ctx64k
KDA route enabled: slot=deepseek_v4.indexer_fp8_quant
KDA route enabled: slot=deepseek_v4.paged_mqa_logits
```

## 6. 指标解释

以 `last_ttft` 作为主要端到端增量 prefill 指标。因为只生成 1 个 token，
`latency` 与 `last_ttft` 应接近。

`one_batch_server` 的 `input_throughput` 使用完整 `input_len=65536` 作为分子，
不能直接解释为 4096-token 增量吞吐。应额外计算：

```text
incremental_throughput = 4096 / last_ttft
speedup = baseline_last_ttft / kda_last_ttft
latency_reduction = 1 - kda_last_ttft / baseline_last_ttft
```

scheduler 日志中的该批次 `input throughput` 只统计 `#new-token=4096`，可以作为
服务端内部观测，但它与客户端 TTFT 的计时边界不同。

## 7. 验收清单

每轮完成后检查：

- 进程退出码为 0；
- `batch size: 1`、`input_len: 65536`、`output_len: 1`；
- warmup 明确使用 61440 tokens；
- 正式请求明确为 `4096 new + 61440 cached`；
- baseline 的 `kda_kernel_profile` 为 `off`；
- KDA 轮四个 TP rank 都加载预期 slot；
- 日志没有 traceback、OOM 或 adapter fallback；
- 两轮结束后 GPU worker 已释放，再启动下一轮；
- 保存完整 command、commit、JSONL 和 server log。

JSONL 中的 `cache_hit_rate` 依赖 `/metrics`。未启用 server metrics 时它可能为
`null`；这时必须使用 scheduler 日志中的 `#cached-token` 验证实际命中。

## 8. 重复与报告

单次运行只用于验证路径。用于性能结论时：

1. baseline 和 KDA 各运行至少 5 次；
2. 交替运行顺序，避免后运行的一方持续受益于 GPU clock/JIT 状态；
3. 每次重启 server 并确认 GPU memory 完全释放；
4. 报告 median、P10/P90、CV 和所有原始样本；
5. 对每个 slot 做单独 profile，区分 adapter/layout 收益与 kernel 收益；
6. 在独立 correctness case 中比较输出或关键中间 tensor。

一次验证快照（commit `3db548f5e`，4×B200）得到：

| 配置 | E2E latency | TTFT | 4096-token 增量吞吐 |
|---|---:|---:|---:|
| native baseline | 0.3831 s | 0.3828 s | 10700 tok/s |
| two-slot reference | 0.2650 s | 0.2647 s | 15474 tok/s |

该快照对应约 1.45× TTFT speedup，但只有一个样本，不能代替多轮统计或
