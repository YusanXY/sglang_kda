# SGLang KDA 文档

本目录记录 SGLang KDA fork 自身的实现与接入约定。上游 SGLang 的完整用户和
开发文档仍位于 `docs_new/`；这里不复制上游文档，只说明 KDA 增量。

## 阅读顺序

1. [根目录 README](../README.md)：项目定位、支持范围和快速开始。
2. [实现结构与接入逻辑](kda-architecture-and-integration.md)：启动生命周期、
   router、Linear 静态绑定、模型专用调用点、失败语义和 adapter 接入步骤。
3. [DeepSeek V4 增量 chunked prefill benchmark](kda-incremental-prefill-benchmark.md)：
   使用 `one_batch_server --cache-hit-rate` 构造历史 KV、固定 4096-token
   增量 chunk，并比较 native 与 KDA profile。
4. [部署与 benchmark 示例](../examples/kda/README.md)：可以直接复用的启动和
   benchmark 命令。
5. [Adapter 关键字协议](../python/sglang/srt/kda/ADAPTER_PROTOCOL.md)：每个
   slot 的精确 callable 参数和返回值约定。
6. [YAML 配置模板](../examples/kda/kda-routes.yaml)：DeepSeek V4 与 GLM-5.2
   profile 示例。

## 核心原则

- KDA 只做兼容和路由，不做算子正确性或性能验证。
- 配置解析、adapter 导入和 Linear 匹配只发生在启动期。
- profile 为 `off` 时保持原生 SGLang 行为。
- slot 未配置时走原生实现；slot 已配置后 adapter 失败直接报错。
- tensor、layout、scale 和 shape/candidate 分派全部放在
  `sglang_entry.py`。
- benchmark、CUDA Graph 开关和模型加载流程继续使用 SGLang 原有机制。

