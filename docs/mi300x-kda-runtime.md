# MI300X KDA 运行说明

本文补充 DeepSeek V4 KDA 路由在 MI300X 上的运行约束。通用路由、YAML 和
adapter 协议见 `kda-architecture-and-integration.md`。

## TP/EP 拓扑

DeepSeek V4 Flash 的 MoE 框架验证应使用 TP8，并将 EP 显式设为 EP4 或
EP8。EP4 的推荐参数为：

```bash
--tp-size 8 \
--ep-size 4 \
--moe-runner-backend aiter \
--moe-a2a-backend none \
--numa-node 0 0 0 0 1 1 1 1
```

EP4 下每个 rank 持有 64 个 local experts。adapter 收到的 `w1`、`w2`
第一维应为 64，同时 `expert_mask` 覆盖 256 个 global experts。adapter
必须把 `expert_mask` 传给 Aiter fused MoE；不得按 EP1 忽略 mask。

## NUMA

MI300X 主机应关闭 automatic NUMA balancing。若当前用户不能修改系统级
`kernel.numa_balancing`，可在启动 server 前启用进程级严格内存策略：

```bash
export SGLANG_NUMA_BIND_V2=0
export SGLANG_STRICT_NUMA_MEMBIND=1
```

`SGLANG_NUMA_BIND_V2=0` 让 scheduler worker 在进程内通过 libnuma 绑定；
`SGLANG_STRICT_NUMA_MEMBIND=1` 将后续 host allocations 设为选定 NUMA
node 的 `MPOL_BIND`，而非仅设置 preferred node。该策略在 model loading
之前应用。

严格绑定是显式 opt-in。若 libnuma 无法分配 node mask 或内核拒绝策略，
启动直接失败，不会静默回退到 preferred policy。

## Benchmark

KDA 开关只放在 server 侧，`bench_serving` 客户端保持原样。A/B 时应保持
以下条件一致：

- 同一模型权重、TP/EP、attention/MoE/GEMM backend；
- 同一随机输入 seed、输入/输出 token 长度、并发和 warmup；
- native 使用 `--kda-kernel-profile off`；
- reference 使用相同 server 参数并增加 KDA YAML/profile；
- 分别记录完成请求数、Req/s、TTFT、TPOT、ITL 和 E2E latency；
- reference adapter 至少记录首次出现的 tensor shape、dtype、stride 和
  device，以核对 EP4 local/global expert ABI。
