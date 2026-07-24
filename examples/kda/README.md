# KDA kernel routing

KDA routing is an optional SGLang compatibility layer for externally exported
DeepSeek V4 and GLM-5.2 kernels. SGLang selects a YAML profile once during
worker startup, imports each configured `sglang_entry.py:callable`, and binds
Linear routes once after model loading. Tensor/layout adaptation remains in
the reference or candidate directory.

Copy `kda-routes.yaml` to a deployment-owned location and replace every
`/absolute/path/to/...` root with an absolute directory containing the
corresponding `sglang_entry.py`. Delete slots that do not have an exported
adapter. A configured missing or incompatible adapter fails immediately;
SGLang does not retry or fall back to the native kernel.

The full keyword interfaces are fixed in
`python/sglang/srt/kda/ADAPTER_PROTOCOL.md`. GLM-5.2 routing is wired but waits
for real reference candidate exports; this repository deliberately includes
no placeholder candidate.

## Native/off benchmark

`off` is the default. It does not read the YAML or import adapters, so the
existing benchmark path stays native:

```bash
python -m sglang.benchmark.one_batch \
  --model-path /path/to/model \
  --kda-kernel-profile off \
  <existing one_batch arguments>
```

## DeepSeek V4

Both in-process benchmark entrypoints already use `ServerArgs`, so they accept
the two KDA flags without benchmark-specific code:

```bash
python -m sglang.benchmark.one_batch \
  --model-path /path/to/deepseek-v4 \
  --kda-kernel-config /absolute/path/kda-routes.yaml \
  --kda-kernel-profile deepseek_v4_best \
  <existing one_batch arguments>
```

```bash
python -m sglang.benchmark.offline_throughput \
  --model-path /path/to/deepseek-v4 \
  --kda-kernel-config /absolute/path/kda-routes.yaml \
  --kda-kernel-profile deepseek_v4_best \
  <existing offline_throughput arguments>
```

## GLM-5.2

After the reference team has exported all configured GLM-5.2 adapters:

```bash
python -m sglang.benchmark.one_batch \
  --model-path /path/to/glm-5.2 \
  --kda-kernel-config /absolute/path/kda-routes.yaml \
  --kda-kernel-profile glm52_best \
  <existing one_batch arguments>
```

The profile architecture must be exactly `GlmMoeDsaForCausalLM`; GLM-4,
NextN, and other DeepSeekV2-derived architectures do not inherit these routes.

## Serving benchmark

Pass KDA options to the existing server:

```bash
python -m sglang.launch_server \
  --model-path /path/to/deepseek-v4 \
  --kda-kernel-config /absolute/path/kda-routes.yaml \
  --kda-kernel-profile deepseek_v4_best \
  <existing server arguments>
```

Then run the original client benchmark unchanged:

```bash
python -m sglang.benchmark.serving <existing serving benchmark arguments>
```

`benchmark.serving` is a client and does not load the kernel YAML. Switching
between `off`, `deepseek_v4_best`, and `glm52_best` requires a server restart.
