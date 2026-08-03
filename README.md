# SGLang KDA

SGLang KDA 是一个基于原生 SGLang 的轻量算子路由扩展。它允许在不重写模型、
benchmark 或全局 kernel wrapper 的前提下，通过两个命令行参数和一份 YAML
配置，把 DeepSeek V4 与 GLM-5.2 的指定算子替换为外部
`reference/candidate` 目录提供的实现。

本项目基于 SGLang `dc60f6566123c26d9b781e3d84e0973d01a75364`
开发。KDA 只负责启动期路由、静态绑定和调用协议；算子的 tensor/layout/scale
适配、shape 分派、实现选择及验证仍由算子目录中的 `sglang_entry.py` 和
llm_flops 负责。

## 设计目标

- **最小侵入**：不修改 SGLang benchmark 业务流程，不 monkey patch Torch，
  不替换全局 DeepGEMM wrapper。
- **配置驱动**：使用 `--kda-kernel-config` 和
  `--kda-kernel-profile` 选择一组算子。
- **启动期冻结**：YAML 解析、adapter 导入、architecture 校验和 Linear
  glob 匹配都在模型运行前完成。
- **原生默认**：profile 默认为 `off`；未配置的 slot 继续使用原生 SGLang
  路径。
- **失败透明**：已配置 adapter 的导入或执行失败会直接抛错，不重试、不回退，
  也不静默切换候选。
- **复用原 benchmark**：`one_batch`、`offline_throughput` 和
  `launch_server` 自动继承 KDA 参数；serving benchmark 客户端保持不变。

## 当前支持范围

| 模型 | 精确 architecture | 路由范围 | 状态 |
|---|---|---|---|
| DeepSeek V4 | `DeepseekV4ForCausalLM` | FP8 Linear、C4/indexer、paged MQA logits、top-k transform、稀疏 prefill/decode、dense SWA、MoE | 接入点完整；按实际导出的 adapter 配置 slot |
| GLM-5.2 | `GlmMoeDsaForCausalLM` | DSA projection/indexer Linear、paged index score、统一稀疏 attention、MoE | 接入点和协议完整；等待 reference 团队提供真实 candidate adapter |

GLM-4、`GlmMoeDsaForCausalLMNextN`、draft worker 和其他
DeepSeekV2 派生模型不会自动继承 GLM-5.2 路由。

## 工作方式

```text
原生 CLI / benchmark
        │
        ▼
ServerArgs 中的 KDA 参数
        │
        ▼
启动期读取 YAML、校验 architecture、导入 sglang_entry.py
        │
        ├── 模型加载后按完整 prefix 静态绑定 Linear
        └── 为 C4 / DSA / MoE 固定 callable
                │
                ▼
        推理热路径只做一次 None 判断
                │
                ├── slot 未配置：原生 SGLang kernel
                └── slot 已配置：外部 adapter；异常直接上抛
```

每个外部算子目录建议保持自包含：

```text
operator_export/
├── implementation.py   # llm_flops 正式实现
├── sglang_entry.py     # SGLang 参数适配与 shape/candidate 分派
└── ...                 # implementation 的其他依赖
```

## 快速开始

1. 准备部署配置：

   ```bash
   cp examples/kda/kda-routes.yaml /absolute/path/kda-routes.yaml
   ```

2. 把 YAML 中的 `/absolute/path/to/...` 替换为真实 adapter 目录，并删除尚未
   导出 `sglang_entry.py` 的 operator 条目。不要直接启用仍含占位目录的
   profile。

3. 启动原生 SGLang server：

   ```bash
   python -m sglang.launch_server \
     --model-path /path/to/model \
     --kda-kernel-config /absolute/path/kda-routes.yaml \
     --kda-kernel-profile deepseek_v4_best \
     <其他原生 SGLang 参数>
   ```

4. 使用原 benchmark：

   ```bash
   python -m sglang.benchmark.one_batch \
     --model-path /path/to/model \
     --kda-kernel-config /absolute/path/kda-routes.yaml \
     --kda-kernel-profile deepseek_v4_best \
     <其他 benchmark 参数>

   python -m sglang.benchmark.offline_throughput \
     --model-path /path/to/model \
     --kda-kernel-config /absolute/path/kda-routes.yaml \
     --kda-kernel-profile deepseek_v4_best \
     <其他 benchmark 参数>
   ```

5. 对照原生路径时使用默认值或显式指定：

   ```bash
   --kda-kernel-profile off
   ```

`off` 不读取 YAML，也不会导入 adapter。切换 profile 需要重启 worker/server。

### 长上下文增量 prefill 对比

DeepSeek V4 的长上下文对比使用
`python -m sglang.benchmark.one_batch_server --cache-hit-rate`，先建立历史 KV，
再只计时未命中的 suffix。目标 case 固定为
`65536 cached history + 4096 new = 69632 input_len`、batch size 1、output 1，
对应 `cache_hit_rate=0.9411764705882353`。`--context-length` 必须覆盖完整输入和
模型预留，不能把容量参数误认为历史长度。完整 baseline/KDA 命令、日志验收和
指标计算见 [增量 chunked prefill benchmark 策略](docs/kda-incremental-prefill-benchmark.md)。

## 文档导航

- [KDA 文档索引](docs/README.md)
- [实现结构与接入逻辑](docs/kda-architecture-and-integration.md)
- [增量 chunked prefill benchmark 策略](docs/kda-incremental-prefill-benchmark.md)
- [部署与 benchmark 示例](examples/kda/README.md)
- [Adapter 关键字协议](python/sglang/srt/kda/ADAPTER_PROTOCOL.md)
- [YAML 配置模板](examples/kda/kda-routes.yaml)

## 验证边界

仓库内单元测试覆盖路由、architecture 限定、静态绑定、参数传递、异常传播和
CLI 接线。SGLang KDA 不负责算子数值正确性、容差或性能判定；这些验证仍应在
llm_flops 中完成。示例 YAML 和测试 fake adapter 不能作为正式 candidate。

---

## Upstream SGLang

以下内容保留上游 SGLang 的项目介绍、安装入口和社区信息。

<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Documentation</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Join Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Dev Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

## News
- [2026/06] 🔥 The next generation of speculative decoding: DFlash and Spec V2 ([blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).
- [2026/04] 🔥 DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles ([blog](https://lmsys.org/blog/2026-04-25-deepseek-v4/)).
- [2026/06] SGLang provides day-0 support for latest open models ([Nemotron 3 Ultra](https://lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra/), [Nemotron 3 Super](https://lmsys.org/blog/2026-03-11-run-nvidia-nemotron-3-super/), [Higgs Audio v3 TTS](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)).
- [2026/02] 🔥 Unlocking 25x Inference Performance with SGLang on NVIDIA GB300 NVL72 ([blog](https://lmsys.org/blog/2026-02-20-gb300-inferencex/)).
- [2026/01] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)).
- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/10] SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).

<details>
<summary>More</summary>

- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).

- [2025/11] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)).
- [2025/10] PyTorch Conference 2025 SGLang Talk ([slide](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)).
- [2025/10] SGLang x Nvidia SF Meetup on 10/2 ([recap](https://x.com/lmsysorg/status/1975339501934510231)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark/5090), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.
- **RL & Post-Training Backbone**: SGLang is a proven rollout backend used for training many frontier models, with native RL integrations and adoption by well-known post-training frameworks such as [**AReaL**](https://github.com/inclusionAI/AReaL), [**Miles**](https://github.com/radixark/miles), [**slime**](https://github.com/THUDM/slime), [**Tunix**](https://github.com/google/tunix), [**verl**](https://github.com/volcengine/verl) and more.

## Getting Started
- [Install SGLang](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 rack-scale parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/), [GB300 long context](https://lmsys.org/blog/2026-02-19-gb300-longctx/).

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, Modal, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations.
As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 400,000 GPUs worldwide.
SGLang is currently hosted under the non-profit open-source organization [LMSYS](https://lmsys.org/about/).

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at [sglang@lmsys.org](mailto:sglang@lmsys.org).

Long-term active SGLang contributors are eligible for coding agent sponsorship, such as Cursor, Claude Code, or OpenAI Codex. Email [sglang@lmsys.org](mailto:sglang@lmsys.org) with your most important commits or pull requests.

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
