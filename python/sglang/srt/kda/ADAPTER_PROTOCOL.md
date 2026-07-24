# KDA adapter protocol

This document fixes the keyword-only call boundary between SGLang and
`sglang_entry.py` for DeepSeek V4 and GLM-5.2 routes. The adapter owns all
tensor, layout, scale, output, and shape-candidate adaptation. SGLang does not
catch adapter exceptions and does not call a native kernel after an adapter
fails.

Profiles must use the exact architecture named by each section. GLM-5.2 routes
are never inherited by `GlmMoeDsaForCausalLMNextN`, GLM-4, or another
DeepSeekV2-derived model.

## Common Linear adapter

The following slots use the static Linear boundary:

- `deepseek_v4.fp8_gemm_nt`
- `glm52.dsa_projection`
- `glm52.dsa_indexer`

```python
def run(*, layer, x, bias=None):
    ...
```

The adapter may inspect `layer.prefix`, `layer.weight`, and the layer's scale
attributes. It owns all activation, weight, scale-layout, output-allocation,
and candidate-specific adaptation, and returns exactly the value expected from
the original quant method. For DeepSeek V4 the deployment template currently
targets `model.layers.*.self_attn.wqkv_a` and
`model.layers.*.self_attn.indexer.wq_b`; deployments should list only prefixes
covered by their exported candidate.

## `deepseek_v4.indexer_fp8_quant`

This slot replaces only the FP8 branch. The FP4 branch remains native.

```python
def run(*, q, weight, weight_scale, freqs_cis, positions):
    ...
```

Return the same `(quantized_query, scales)` structure as
`fused_q_indexer_rope_hadamard_quant`.

## `deepseek_v4.paged_mqa_logits`

This slot replaces only the FP8 paged indexer call. The FP4 and nonpaged
indexers remain native, and SGLang does not load a paged native FP8 backend
when this slot is configured.

```python
def run(
    *,
    q,
    kv_cache,
    weights,
    seq_lens,
    page_table,
    schedule,
    max_context,
    q_offset,
):
    ...
```

Return the logits tensor consumed by the top-k transform. `q_offset` is the
number of live query rows. The adapter may translate the native SGLang layout
to the llm_flops operator ABI.

## `deepseek_v4.topk_transform`

```python
def run(
    *,
    scores,
    seq_lens,
    page_tables,
    output,
    page_size,
    metadata,
    raw_indices,
):
    ...
```

Write selected page indices into `output`. When `raw_indices` is not `None`,
also write the raw token indices required by capture/HiSparse. The SGLang call
site intentionally ignores the return value, matching the native in-place
contract.

## `deepseek_v4.sparse_prefill_attention`

```python
def run(
    *,
    q,
    kv,
    indices,
    softmax_scale,
    value_dim,
    attention_sink,
    topk_length,
):
    ...
```

Return the primary attention output tensor. `attention_sink` and
`topk_length` are part of the current SGLang native call and are passed through
even when a lower-level llm_flops implementation does not require them.

## `deepseek_v4.sparse_decode_attention`

Used when `compress_ratio` is 4 or 128.

```python
def run(
    *,
    q,
    swa_cache,
    swa_indices,
    swa_lengths,
    extra_cache,
    extra_indices,
    extra_lengths,
    attention_sink,
    scheduler,
):
    ...
```

Return the unsqueezed primary attention output. SGLang applies the same final
`squeeze(1)` as on its native path.

## `deepseek_v4.dense_swa_attention`

Used when `compress_ratio` is 0.

```python
def run(
    *,
    q,
    cache,
    indices,
    lengths,
    attention_sink,
    scheduler,
):
    ...
```

Return the unsqueezed primary attention output. SGLang applies the same final
`squeeze(1)` as on its native path.

# GLM-5.2

The profile architecture must be exactly `GlmMoeDsaForCausalLM`. This
integration defines the compatibility boundary only; no real GLM-5.2
candidate adapter is shipped with SGLang.

## `glm52.dsa_projection`

This is a static Linear route and uses the common Linear adapter contract:

```python
def run(*, layer, x, bias=None):
    ...
```

The route targets the complete projection prefixes below:

```text
model.layers.*.self_attn.fused_qkv_a_proj_with_mqa
model.layers.*.self_attn.q_b_proj
model.layers.*.self_attn.q_proj
model.layers.*.self_attn.kv_a_proj_with_mqa
model.layers.*.self_attn.kv_b_proj
model.layers.*.self_attn.o_proj
```

The mutually exclusive `fused_qkv_a_proj_with_mqa` and
`q_proj`/`kv_a_proj_with_mqa` prefixes cover the current
`DeepseekV2AttentionMLA` construction choices. The adapter selects the
llm_flops projection case from `layer.prefix` and owns quantization, scale
layout, output allocation, and candidate selection.

## `glm52.dsa_indexer`

This is also a static Linear route with the same callable:

```python
def run(*, layer, x, bias=None):
    ...
```

Its current prefixes are:

```text
model.layers.*.self_attn.indexer.wq_b
model.layers.*.self_attn.indexer.wk_weights_proj
model.layers.*.self_attn.indexer.wk
model.layers.*.self_attn.indexer.weights_proj
```

`wk_weights_proj` is the fused-indexer alternative to `wk` and
`weights_proj`. Both FP8 and BF16/unquantized Linear methods use this bound
callable.

## `glm52.dsa_index_score`

This slot replaces only the paged score kernel inside
`Indexer._get_topk_paged`. It does not replace the separate ragged/contiguous
prefill score path.

```python
def run(
    *,
    phase,
    query,
    cache,
    cache_scale,
    weights,
    lengths_start,
    lengths_end,
    block_tables,
    schedule,
    max_context,
    q_offset,
):
    ...
```

`phase` is the current native SGLang `forward_batch.forward_mode` object; the
adapter owns its mapping to the llm_flops prefill/decode convention. `query`,
`cache`, `weights`, `lengths_start`, `block_tables`, `schedule`, and
`max_context` are the existing SGLang paged-kernel objects after its native
views. `cache_scale` and `lengths_end` are `None` for the fused paged cache.
`q_offset` preserves the native distinction between live and padded query
rows. The adapter translates this boundary to
`glm5_dsa_index_score/implementation.py`; SGLang performs the existing mask
and top-k transform after the adapter returns logits.

## `glm52.dsa_sparse_attention`

This slot replaces the single `flash_mla_sparse_fwd` invocation inside
`DeepseekSparseAttnBackend._forward_flashmla_sparse`:

```python
def run(
    *,
    query,
    cache,
    indices,
    softmax_scale,
    value_dim,
):
    ...
```

`query` already includes any native head padding, `indices` has the native
`(query_tokens, 1, topk)` view, and SGLang retains its existing output-head
trim after the call. The adapter returns the primary output tensor, not the
native FlashMLA auxiliary tuple.

# Model-scoped masked MoE

The router fixes the active MoE slot once from the profile's exact model
architecture:

| Architecture | Slot |
| --- | --- |
| `DeepseekV4ForCausalLM` | `deepseek_v4.moe` |
| `GlmMoeDsaForCausalLM` | `glm52.moe_masked_grouped_gemm` |

The `DeepGemmRunnerCore` captures that callable during construction. Both
slots use the same keyword-only boundary and replace only the two FP8 masked
grouped GEMMs. BF16 and contiguous GEMMs remain native.

```python
def run(
    *,
    stage,
    lhs,
    rhs,
    out,
    routing,
    expected_m,
    recipe_a,
    recipe_b,
    overlap_args=None,
    max_block_n=256,
):
    ...
```

`stage` is exactly `"gate_up"` for the first grouped GEMM and `"down"` for the
second. `lhs` and `rhs` are `(tensor, scale)` pairs; `out` is the existing
SGLang output buffer; `routing` is the native masked-M tensor.

The gate/up call supplies `recipe_a` and `recipe_b`. The down call supplies
the same recipe keywords and, when native overlap is active, also supplies
`overlap_args` and `max_block_n`. Adapters must write into `out`; SGLang keeps
that buffer as the stage result instead of replacing it with the adapter
return value.

The gate/up adapter return value is ignored, matching the native call. The
down adapter return value has the native DeepGEMM semantics: return
`(block_m, threshold)` when overlap metadata is produced, otherwise return
`None`. SGLang copies the tuple into its existing `meta_overlap_args` mapping.

SGLang does not validate shapes, routing capacity, tensor layout, scales, or
adapter output. It does not catch adapter exceptions and never retries the
native kernel after an adapter failure.
