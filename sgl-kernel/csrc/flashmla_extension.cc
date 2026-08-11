/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/all.h>
#include <torch/library.h>

#include "api/dense_decode.h"
#include "api/sparse_decode.h"
#include "api/sparse_fwd.h"
#include "sgl_kernel_ops.h"

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>> sgl_sparse_decode_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int64_t d_v,
    double sm_scale) {
  return sparse_attn_decode_interface(
      q,
      kv,
      indices,
      topk_length,
      attn_sink,
      tile_scheduler_metadata,
      num_splits,
      extra_kv,
      extra_indices,
      extra_topk_length,
      static_cast<int>(d_v),
      static_cast<float>(sm_scale));
}

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>> sgl_dense_decode_fwd(
    at::Tensor q,
    const at::Tensor& kcache,
    int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    double softmax_scale,
    bool is_causal,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits) {
  return dense_attn_decode_interface(
      q,
      kcache,
      static_cast<int>(head_size_v),
      seqlens_k,
      block_table,
      static_cast<float>(softmax_scale),
      is_causal,
      tile_scheduler_metadata,
      num_splits);
}

// DSV4 inference only consumes the attention output.  Calling the public
// sparse_prefill_fwd compatibility wrapper also converts max_logits and LSE
// with two pointwise kernels, even when both tensors are immediately dropped.
// Enter the FlashMLA interface directly so Huge mode can omit that dead work
// while keeping the existing three-output API unchanged for all other users.
static at::Tensor sgl_sparse_prefill_fwd_output_impl(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    bool all_heads) {
  using bf16 = cutlass::bfloat16_t;

  Arch arch;
  TORCH_CHECK(arch.is_sm100f(), "sparse_prefill_fwd_output requires an SM100-family GPU");
  KU_CHECK_NDIM(q, 3);
  KU_CHECK_NDIM(kv, 3);
  KU_CHECK_NDIM(indices, 3);
  KU_CHECK_NDIM(attn_sink, 1);
  KU_CHECK_NDIM(topk_length, 1);

  const int s_q = q.size(0);
  const int s_kv = kv.size(0);
  const int h_q = q.size(1);
  const int h_kv = kv.size(1);
  const int d_qk = q.size(2);
  const int topk = indices.size(2);
  const int output_heads = all_heads ? h_q : 16;
  TORCH_CHECK(h_q == 64 && h_kv == 1, "sparse_prefill_fwd_output requires h_q=64 and h_kv=1");
  TORCH_CHECK(d_qk == 512 && d_v == 512, "sparse_prefill_fwd_output requires d_qk=d_v=512");

  KU_CHECK_DEVICE(q);
  KU_CHECK_DEVICE(kv);
  KU_CHECK_DEVICE(indices);
  KU_CHECK_DEVICE(attn_sink);
  KU_CHECK_DEVICE(topk_length);
  KU_CHECK_DTYPE(q, torch::kBFloat16);
  KU_CHECK_DTYPE(kv, torch::kBFloat16);
  KU_CHECK_DTYPE(indices, torch::kInt32);
  KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
  KU_CHECK_DTYPE(topk_length, torch::kInt32);
  KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
  KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
  KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
  KU_CHECK_SHAPE(attn_sink, h_q);
  KU_CHECK_SHAPE(topk_length, s_q);
  KU_CHECK_LAST_DIM_CONTIGUOUS(q);
  KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
  KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
  KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
  KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

  at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
  auto out = torch::empty({s_q, output_heads, d_v}, q.options());
  KU_CHECK_CONTIGUOUS(out);
  SparseAttnFwdParams params = {
      s_q,
      s_kv,
      h_q,
      h_kv,
      d_qk,
      static_cast<int>(d_v),
      topk,
      static_cast<float>(sm_scale),
      static_cast<float>(sm_scale) * LOG_2_E,
      reinterpret_cast<bf16*>(q.data_ptr()),
      reinterpret_cast<bf16*>(kv.data_ptr()),
      indices.data_ptr<int>(),
      ku::get_optional_tensor_ptr<float>(attn_sink),
      ku::get_optional_tensor_ptr<int>(topk_length),
      int64_stride_to_int(q.stride(0)),
      int64_stride_to_int(q.stride(1)),
      int64_stride_to_int(kv.stride(0)),
      int64_stride_to_int(kv.stride(1)),
      int64_stride_to_int(indices.stride(0)),
      int64_stride_to_int(indices.stride(1)),
      reinterpret_cast<bf16*>(out.data_ptr()),
      nullptr,
      nullptr,
      arch.num_sms,
      at::cuda::getCurrentCUDAStream().stream(),
  };
  if (all_heads) {
    sm100::fwd::head64::run_fwd_phase1_all_heads_output_kernel<512>(params);
  } else {
    sm100::fwd::head64::run_fwd_phase1_output_kernel<512>(params);
  }
  return out;
}

static at::Tensor sgl_sparse_prefill_fwd_output(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length) {
  return sgl_sparse_prefill_fwd_output_impl(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, false);
}

static at::Tensor sgl_sparse_prefill_fwd_all_heads_output(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length) {
  return sgl_sparse_prefill_fwd_output_impl(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, true);
}

// Eager TP4 C4-prefill path.  q_sources is the direct NCCL all-to-all receive
// layout [source_rank, token_shard, 16, 512].  FlashMLA reads that hierarchy as
// 64 logical heads and writes [token_shard, source_rank, 16, 512].  The token
// owner can therefore project all eight output groups locally without a return
// all-to-all.
static at::Tensor sgl_sparse_prefill_fwd_tp4_sharded_output(
    const at::Tensor& q_sources,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length) {
  using bf16 = cutlass::bfloat16_t;
  constexpr int kWorldSize = 4;
  constexpr int kLocalHeads = 16;
  constexpr int kHeads = kWorldSize * kLocalHeads;
  constexpr int kHeadDim = 512;

  Arch arch;
  TORCH_CHECK(
      arch.is_sm100f(),
      "sparse_prefill_fwd_tp4_sharded_output requires an SM100-family GPU");
  KU_CHECK_NDIM(q_sources, 4);
  KU_CHECK_NDIM(kv, 3);
  KU_CHECK_NDIM(indices, 3);
  KU_CHECK_NDIM(attn_sink, 1);
  KU_CHECK_NDIM(topk_length, 1);

  const int s_q = q_sources.size(1);
  const int s_kv = kv.size(0);
  const int h_kv = kv.size(1);
  const int topk = indices.size(2);
  TORCH_CHECK(
      q_sources.size(0) == kWorldSize &&
          q_sources.size(2) == kLocalHeads &&
          q_sources.size(3) == kHeadDim,
      "TP4 sharded Q must have shape [4, s_q, 16, 512]");
  TORCH_CHECK(h_kv == 1, "TP4 sharded sparse prefill requires h_kv=1");
  TORCH_CHECK(d_v == kHeadDim, "TP4 sharded sparse prefill requires d_v=512");

  KU_CHECK_DEVICE(q_sources);
  KU_CHECK_DEVICE(kv);
  KU_CHECK_DEVICE(indices);
  KU_CHECK_DEVICE(attn_sink);
  KU_CHECK_DEVICE(topk_length);
  KU_CHECK_DTYPE(q_sources, torch::kBFloat16);
  KU_CHECK_DTYPE(kv, torch::kBFloat16);
  KU_CHECK_DTYPE(indices, torch::kInt32);
  KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
  KU_CHECK_DTYPE(topk_length, torch::kInt32);
  KU_CHECK_SHAPE(kv, s_kv, h_kv, kHeadDim);
  KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
  KU_CHECK_SHAPE(attn_sink, kHeads);
  KU_CHECK_SHAPE(topk_length, s_q);
  KU_CHECK_CONTIGUOUS(q_sources);
  KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
  KU_CHECK_LAST_DIM_CONTIGUOUS(indices);

  at::cuda::CUDAGuard device_guard{
      static_cast<char>(q_sources.get_device())};
  auto out = torch::empty(
      {s_q, kWorldSize, kLocalHeads, kHeadDim}, q_sources.options());
  SparseAttnFwdParams params = {
      s_q,
      s_kv,
      kHeads,
      h_kv,
      kHeadDim,
      kHeadDim,
      topk,
      static_cast<float>(sm_scale),
      static_cast<float>(sm_scale) * LOG_2_E,
      reinterpret_cast<bf16*>(q_sources.data_ptr()),
      reinterpret_cast<bf16*>(kv.data_ptr()),
      indices.data_ptr<int>(),
      ku::get_optional_tensor_ptr<float>(attn_sink),
      ku::get_optional_tensor_ptr<int>(topk_length),
      kLocalHeads * kHeadDim,
      kHeadDim,
      int64_stride_to_int(kv.stride(0)),
      int64_stride_to_int(kv.stride(1)),
      int64_stride_to_int(indices.stride(0)),
      int64_stride_to_int(indices.stride(1)),
      reinterpret_cast<bf16*>(out.data_ptr()),
      nullptr,
      nullptr,
      arch.num_sms,
      at::cuda::getCurrentCUDAStream().stream(),
  };
  sm100::fwd::head64::run_fwd_phase1_tp4_sharded_output_kernel<512>(
      params);
  return out;
}

// TP4 eager path without an explicit Q all-to-all/router.  Each q_peer tensor
// is the peer mapping of one producer rank's symmetric
// [token_capacity, 16, 512] buffer.  FlashMLA directly loads the destination
// token quarter from all four mappings into its 64-head shared-memory tile.
static at::Tensor sgl_sparse_prefill_fwd_tp4_peer_q_output(
    const at::Tensor& q_peer_0,
    const at::Tensor& q_peer_1,
    const at::Tensor& q_peer_2,
    const at::Tensor& q_peer_3,
    int64_t token_shard_rank,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length) {
  using bf16 = cutlass::bfloat16_t;
  constexpr int kWorldSize = 4;
  constexpr int kLocalHeads = 16;
  constexpr int kHeads = kWorldSize * kLocalHeads;
  constexpr int kHeadDim = 512;

  Arch arch;
  TORCH_CHECK(
      arch.is_sm100f(),
      "sparse_prefill_fwd_tp4_peer_q_output requires an SM100-family GPU");
  KU_CHECK_NDIM(q_peer_0, 3);
  KU_CHECK_NDIM(q_peer_1, 3);
  KU_CHECK_NDIM(q_peer_2, 3);
  KU_CHECK_NDIM(q_peer_3, 3);
  KU_CHECK_NDIM(kv, 3);
  KU_CHECK_NDIM(indices, 3);
  KU_CHECK_NDIM(attn_sink, 1);
  KU_CHECK_NDIM(topk_length, 1);

  const int s_q = indices.size(0);
  const int active_s_q = s_q * kWorldSize;
  const int s_kv = kv.size(0);
  const int h_kv = kv.size(1);
  const int topk = indices.size(2);
  TORCH_CHECK(
      token_shard_rank >= 0 && token_shard_rank < kWorldSize,
      "token_shard_rank must be in [0, 4)");
  auto check_peer_shape = [&](const at::Tensor& q_peer, int peer_rank) {
    TORCH_CHECK(
        q_peer.size(0) >= active_s_q &&
            q_peer.size(1) == kLocalHeads &&
            q_peer.size(2) == kHeadDim,
        "TP4 peer Q ", peer_rank,
        " must have shape [capacity>=4*s_q, 16, 512]");
  };
  check_peer_shape(q_peer_0, 0);
  check_peer_shape(q_peer_1, 1);
  check_peer_shape(q_peer_2, 2);
  check_peer_shape(q_peer_3, 3);
  TORCH_CHECK(h_kv == 1, "TP4 peer-Q sparse prefill requires h_kv=1");
  TORCH_CHECK(d_v == kHeadDim, "TP4 peer-Q sparse prefill requires d_v=512");

  KU_CHECK_DEVICE(q_peer_0);
  KU_CHECK_DEVICE(q_peer_1);
  KU_CHECK_DEVICE(q_peer_2);
  KU_CHECK_DEVICE(q_peer_3);
  KU_CHECK_DEVICE(kv);
  KU_CHECK_DEVICE(indices);
  KU_CHECK_DEVICE(attn_sink);
  KU_CHECK_DEVICE(topk_length);
  KU_CHECK_DTYPE(q_peer_0, torch::kBFloat16);
  KU_CHECK_DTYPE(q_peer_1, torch::kBFloat16);
  KU_CHECK_DTYPE(q_peer_2, torch::kBFloat16);
  KU_CHECK_DTYPE(q_peer_3, torch::kBFloat16);
  KU_CHECK_DTYPE(kv, torch::kBFloat16);
  KU_CHECK_DTYPE(indices, torch::kInt32);
  KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
  KU_CHECK_DTYPE(topk_length, torch::kInt32);
  KU_CHECK_CONTIGUOUS(q_peer_0);
  KU_CHECK_CONTIGUOUS(q_peer_1);
  KU_CHECK_CONTIGUOUS(q_peer_2);
  KU_CHECK_CONTIGUOUS(q_peer_3);
  KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
  KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
  KU_CHECK_SHAPE(kv, s_kv, h_kv, kHeadDim);
  KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
  KU_CHECK_SHAPE(attn_sink, kHeads);
  KU_CHECK_SHAPE(topk_length, s_q);

  at::cuda::CUDAGuard device_guard{
      static_cast<char>(q_peer_0.get_device())};
  auto out = torch::empty(
      {s_q, kWorldSize, kLocalHeads, kHeadDim}, q_peer_0.options());
  SparseAttnFwdParams params = {
      s_q,
      s_kv,
      kHeads,
      h_kv,
      kHeadDim,
      kHeadDim,
      topk,
      static_cast<float>(sm_scale),
      static_cast<float>(sm_scale) * LOG_2_E,
      reinterpret_cast<bf16*>(q_peer_0.data_ptr()),
      reinterpret_cast<bf16*>(kv.data_ptr()),
      indices.data_ptr<int>(),
      ku::get_optional_tensor_ptr<float>(attn_sink),
      ku::get_optional_tensor_ptr<int>(topk_length),
      kLocalHeads * kHeadDim,
      kHeadDim,
      int64_stride_to_int(kv.stride(0)),
      int64_stride_to_int(kv.stride(1)),
      int64_stride_to_int(indices.stride(0)),
      int64_stride_to_int(indices.stride(1)),
      reinterpret_cast<bf16*>(out.data_ptr()),
      nullptr,
      nullptr,
      arch.num_sms,
      at::cuda::getCurrentCUDAStream().stream(),
  };
  sm100::fwd::head64::run_fwd_phase1_tp4_peer_q_output_kernel<512>(
      params,
      reinterpret_cast<bf16*>(q_peer_0.data_ptr()),
      reinterpret_cast<bf16*>(q_peer_1.data_ptr()),
      reinterpret_cast<bf16*>(q_peer_2.data_ptr()),
      reinterpret_cast<bf16*>(q_peer_3.data_ptr()),
      active_s_q,
      static_cast<int>(token_shard_rank));
  return out;
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From FlashMLA
   */
  m.def(
      "get_mla_decoding_metadata(Tensor seqlens_k, int num_q_tokens_per_head_k, int h_k, int? h_q, bool "
      "is_fp8_kvcache, int? topk) -> Tensor[]");
  m.impl("get_mla_decoding_metadata", torch::kCUDA, &get_mla_decoding_metadata);

  m.def("get_mla_decoding_metadata_dense_fp8(Tensor seqlens_k, int num_heads_per_head_k, int num_heads_k) -> Tensor[]");
  m.impl("get_mla_decoding_metadata_dense_fp8", torch::kCUDA, &get_mla_decoding_metadata_dense_fp8);

  m.def(
      "fwd_kvcache_mla(Tensor q, Tensor kv_cache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, Tensor num_splits, bool is_fp8, Tensor? indices, "
      "Tensor? attn_sink, Tensor? extra_k_cache, Tensor? extra_indices_in_kvcache, Tensor? topk_length, Tensor? "
      "extra_topk_length) "
      "-> Tensor[]");
  m.impl("fwd_kvcache_mla", torch::kCUDA, &fwd_kvcache_mla);

#ifdef FLASHMLA_ENABLE_SM100
  m.def(
      "dense_prefill_fwd(Tensor workspace_buffer, Tensor q, Tensor k, Tensor v, Tensor cumulative_seqlen_q, Tensor "
      "cumulative_seqlen_kv, Tensor o, Tensor lse, int mask_mode_code, float softmax_scale, int max_seqlen_q, int "
      "max_seqlen_kv, bool is_varlen) -> ()");
  m.impl("dense_prefill_fwd", torch::kCUDA, &FMHACutlassSM100FwdRun);
#endif

  m.def(
      "sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, "
      "Tensor? tile_scheduler_metadata, Tensor? num_splits, Tensor? extra_kv, Tensor? extra_indices, "
      "Tensor? extra_topk_length, int d_v, float sm_scale) -> (Tensor, Tensor, Tensor?, Tensor?)");
  m.impl("sparse_decode_fwd", torch::kCUDA, &sgl_sparse_decode_fwd);

  m.def(
      "dense_decode_fwd(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor? tile_scheduler_metadata, Tensor? num_splits) -> (Tensor, Tensor, "
      "Tensor?, "
      "Tensor?)");
  m.impl("dense_decode_fwd", torch::kCUDA, &sgl_dense_decode_fwd);

  m.def(
      "sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? attn_sink=None, "
      "Tensor? topk_length=None) -> Tensor[]");
  m.impl("sparse_prefill_fwd", torch::kCUDA, &sparse_prefill_fwd);

  m.def(
      "sparse_prefill_fwd_output(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? "
      "attn_sink=None, Tensor? topk_length=None) -> Tensor");
  m.impl("sparse_prefill_fwd_output", torch::kCUDA, &sgl_sparse_prefill_fwd_output);

  m.def(
      "sparse_prefill_fwd_all_heads_output(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? "
      "attn_sink=None, Tensor? topk_length=None) -> Tensor");
  m.impl("sparse_prefill_fwd_all_heads_output", torch::kCUDA, &sgl_sparse_prefill_fwd_all_heads_output);

  m.def(
      "sparse_prefill_fwd_tp4_sharded_output(Tensor q_sources, Tensor kv, Tensor indices, float sm_scale, int d_v, "
      "Tensor? attn_sink=None, Tensor? topk_length=None) -> Tensor");
  m.impl(
      "sparse_prefill_fwd_tp4_sharded_output",
      torch::kCUDA,
      &sgl_sparse_prefill_fwd_tp4_sharded_output);

  m.def(
      "sparse_prefill_fwd_tp4_peer_q_output(Tensor q_peer_0, Tensor q_peer_1, Tensor q_peer_2, Tensor q_peer_3, "
      "int token_shard_rank, Tensor kv, Tensor indices, float sm_scale, int d_v, Tensor? attn_sink=None, "
      "Tensor? topk_length=None) -> Tensor");
  m.impl(
      "sparse_prefill_fwd_tp4_peer_q_output",
      torch::kCUDA,
      &sgl_sparse_prefill_fwd_tp4_peer_q_output);

  m.def(
      "fwd_kvcache_mla_fp8(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, Tensor block_table, float "
      "softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, Tensor num_splits, Tensor? descale_q, Tensor? "
      "descale_k) -> Tensor[]");
  m.impl("fwd_kvcache_mla_fp8", torch::kCUDA, &fwd_kvcache_mla_fp8);
}

REGISTER_EXTENSION(flashmla_ops)
