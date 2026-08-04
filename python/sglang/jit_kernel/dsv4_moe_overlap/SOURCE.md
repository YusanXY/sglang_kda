# DSV4 Huge TRT-LLM MoE overlap module

This JIT module reuses the installed FlashInfer 0.6.12 TRT-LLM MoE source tree
and changes only the FP4 launch order in
`trtllm_fused_moe_kernel_launcher.cu`. It prepares the MoE runner, workspaces,
and TMA descriptors before submitting the routing kernel, allowing that CPU
work to overlap preceding GPU kernels.

The transformation is enabled only by the `huge_kernel` DSV4 worker backend.
The `native` backend continues to use FlashInfer's original module. There is no
runtime fallback: a preloaded TRT-LLM SM100 module, an unexpected source
layout, or a launcher SHA-256 other than
`bc8ed7c95c18265f4e57607d263263de86eeb754a12d741cdf74fcc23489961b`
raises an error.

The generated source and compiled module live in the normal per-user JIT
caches. Initial compilation is a startup cost and is not part of measured
prefill TTFT.
