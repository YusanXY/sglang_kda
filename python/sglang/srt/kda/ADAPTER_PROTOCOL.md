# DeepSeek V4 KDA adapter protocol

This document fixes the keyword-only call boundary between SGLang and
`sglang_entry.py` for the non-Linear DeepSeek V4 routes. The adapter owns all
tensor, layout, scale, output, and shape-candidate adaptation. SGLang does not
catch adapter exceptions and does not call a native kernel after an adapter
fails.

The profile architecture must be exactly `DeepseekV4ForCausalLM`.

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
