#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

struct MhcPreNormMxfp8QuantParams {
  const float* __restrict__ gemm_mul;
  const float* __restrict__ gemm_sq;
  const float* __restrict__ hc_scale;
  const float* __restrict__ hc_base;
  const bf16_t* __restrict__ residual;
  const bf16_t* __restrict__ norm_weight;
  float* __restrict__ post_mix;
  float* __restrict__ comb_mix;
  bf16_t* __restrict__ layer_input;
  fp8_e4m3_t* __restrict__ output_fp8;
  uint32_t* __restrict__ output_scale;
  fp8_e4m3_t* __restrict__ routed_output_fp8;
  uint8_t* __restrict__ routed_output_scale;
  float rms_eps;
  float hc_pre_eps;
  float hc_sinkhorn_eps;
  float hc_post_mult;
  float norm_eps;
  uint32_t num_tokens;
  uint32_t num_splits;
  uint32_t sinkhorn_repeat;
  uint32_t scale_group_stride;
  uint32_t emit_routed_quant;
};

__device__ __forceinline__ void mhc_cp_async_16(
    void* shared_dst, const void* global_src) {
  const uint32_t shared_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));
  asm volatile(
      "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
          "r"(shared_addr),
          "l"(global_src));
}

__device__ __forceinline__ void mhc_cp_async_ca_16(
    void* shared_dst, const void* global_src) {
  const uint32_t shared_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16;\n" ::
          "r"(shared_addr),
          "l"(global_src));
}

__device__ __forceinline__ void mhc_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int kPending>
__device__ __forceinline__ void mhc_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(kPending));
}

template <uint32_t kDataThreads>
__device__ __forceinline__ void mhc_data_warps_barrier() {
  static_assert(kDataThreads == 64 || kDataThreads == 128);
  asm volatile("bar.sync 3, %0;\n" :: "n"(kDataThreads) : "memory");
}

template <uint32_t kDataThreads>
__device__ __forceinline__ void mhc_issue_residual_tile(
    const MhcPreNormMxfp8QuantParams& params,
    bf16_t* stage,
    const uint32_t token,
    const uint32_t tile,
    const uint32_t data_tid) {
#pragma unroll
  for (uint32_t route = 0; route < 4; ++route) {
#pragma unroll
    for (uint32_t offset = data_tid * 8; offset < 1024;
         offset += kDataThreads * 8) {
      const bf16_t* src =
          params.residual + token * 16384 + route * 4096 + tile * 1024 +
          offset;
      mhc_cp_async_16(stage + route * 1024 + offset, src);
    }
  }
}

template <uint32_t kDataThreads>
__device__ __forceinline__ void mhc_issue_norm_weight(
    const MhcPreNormMxfp8QuantParams& params,
    bf16_t* stage,
    const uint32_t data_tid) {
#pragma unroll
  for (uint32_t offset = data_tid * 8; offset < 4096;
       offset += kDataThreads * 8) {
    mhc_cp_async_ca_16(stage + offset, params.norm_weight + offset);
  }
}

template <uint32_t kDataThreads>
__device__ __forceinline__ float mhc_process_residual_tile(
    const bf16_t* stage,
    bf16_t* unnormalized,
    const float* pre_mix,
    const uint32_t tile,
    const uint32_t data_tid) {
  float local_sq = 0.0f;
#pragma unroll
  for (uint32_t tile_offset = data_tid * 8; tile_offset < 1024;
       tile_offset += kDataThreads * 8) {
    float route_values[4][8];
#pragma unroll
    for (uint32_t route = 0; route < 4; ++route) {
      const auto* source =
          reinterpret_cast<const bf16x2_t*>(
              stage + route * 1024 + tile_offset);
#pragma unroll
      for (uint32_t pair = 0; pair < 4; ++pair) {
        const fp32x2_t value = device::cast<fp32x2_t>(source[pair]);
        route_values[route][pair * 2] = value.x;
        route_values[route][pair * 2 + 1] = value.y;
      }
    }

    float output[8];
#pragma unroll
    for (uint32_t element = 0; element < 8; ++element) {
      float value = 0.0f;
#pragma unroll
      for (uint32_t route = 0; route < 4; ++route) {
        value = fmaf(pre_mix[route], route_values[route][element], value);
      }
      output[element] = value;
      local_sq = fmaf(value, value, local_sq);
    }

    alignas(16) bf16x2_t rounded[4];
#pragma unroll
    for (uint32_t pair = 0; pair < 4; ++pair) {
      const uint32_t element = pair * 2;
      rounded[pair] = device::cast<bf16x2_t>(
          make_float2(output[element], output[element + 1]));
    }
    const uint32_t offset = tile * 1024 + tile_offset;
    reinterpret_cast<uint4*>(unnormalized + offset)[0] =
        reinterpret_cast<const uint4*>(rounded)[0];
  }
  return local_sq;
}

template <uint32_t kDataThreads, bool kEmitRoutedQuant>
__device__ __forceinline__ void mhc_process_norm_quant_tile(
    const MhcPreNormMxfp8QuantParams& params,
    const bf16_t* unnormalized,
    const bf16_t* weight_stage,
    const float norm_factor,
    const uint32_t token,
    const uint32_t tile,
    const uint32_t data_tid,
    const uint32_t lane) {
#pragma unroll
  for (uint32_t tile_offset = data_tid * 8; tile_offset < 1024;
       tile_offset += kDataThreads * 8) {
    const uint32_t offset = tile * 1024 + tile_offset;
    const auto* source =
        reinterpret_cast<const bf16x2_t*>(unnormalized + offset);
    const auto* weight =
        reinterpret_cast<const bf16x2_t*>(weight_stage + tile_offset);
    float values[8];
    alignas(16) bf16x2_t rounded[4];
    float local_absmax = 1e-10f;
#pragma unroll
    for (uint32_t pair = 0; pair < 4; ++pair) {
      const fp32x2_t source_value = device::cast<fp32x2_t>(source[pair]);
      const fp32x2_t weight_value = device::cast<fp32x2_t>(weight[pair]);
      const fp32x2_t value = make_float2(
          source_value.x * norm_factor * weight_value.x,
          source_value.y * norm_factor * weight_value.y);
      rounded[pair] = device::cast<bf16x2_t>(value);
      const fp32x2_t rounded_value =
          device::cast<fp32x2_t>(rounded[pair]);
      values[pair * 2] = rounded_value.x;
      values[pair * 2 + 1] = rounded_value.y;
      local_absmax = fmaxf(local_absmax, fabsf(rounded_value.x));
      local_absmax = fmaxf(local_absmax, fabsf(rounded_value.y));
    }
    reinterpret_cast<uint4*>(params.layer_input + token * 4096 + offset)[0] =
        reinterpret_cast<const uint4*>(rounded)[0];

    const float thread_absmax = local_absmax;
#pragma unroll
    for (uint32_t delta = 8; delta > 0; delta >>= 1) {
      local_absmax = fmaxf(
          local_absmax,
          __shfl_down_sync(0xffffffffu, local_absmax, delta, 16));
    }
    int32_t exponent = 0;
    if ((lane & 15) == 0)
      exponent =
          cast_to_ue8m0(local_absmax / device::math::FP8_E4M3_MAX);
    exponent = __shfl_sync(
        0xffffffffu, exponent, static_cast<int>(lane & ~15u));
    const float inv_scale = inv_scale_ue8m0(exponent);

    alignas(8) fp8x2_e4m3_t packed_output[4];
#pragma unroll
    for (uint32_t pair = 0; pair < 4; ++pair) {
      packed_output[pair] = pack_fp8(
          values[pair * 2] * inv_scale,
          values[pair * 2 + 1] * inv_scale);
    }
    reinterpret_cast<uint64_t*>(
        params.output_fp8 + token * 4096 + offset)[0] =
        reinterpret_cast<const uint64_t*>(packed_output)[0];

    if constexpr (kEmitRoutedQuant) {
      float routed_absmax = thread_absmax;
      routed_absmax = fmaxf(
          routed_absmax,
          __shfl_down_sync(0xffffffffu, routed_absmax, 2, 4));
      routed_absmax = fmaxf(
          routed_absmax,
          __shfl_down_sync(0xffffffffu, routed_absmax, 1, 4));
      int32_t routed_exponent = 0;
      if ((lane & 3) == 0) {
        routed_exponent = cast_to_ue8m0(
            routed_absmax / device::math::FP8_E4M3_MAX);
      }
      routed_exponent = __shfl_sync(
          0xffffffffu, routed_exponent, static_cast<int>(lane & ~3u));
      const float routed_inv_scale = inv_scale_ue8m0(routed_exponent);
      alignas(8) fp8x2_e4m3_t packed_routed_output[4];
#pragma unroll
      for (uint32_t pair = 0; pair < 4; ++pair) {
        packed_routed_output[pair] = pack_fp8(
            values[pair * 2] * routed_inv_scale,
            values[pair * 2 + 1] * routed_inv_scale);
      }
      reinterpret_cast<uint64_t*>(
          params.routed_output_fp8 + token * 4096 + offset)[0] =
          reinterpret_cast<const uint64_t*>(packed_routed_output)[0];
      if ((lane & 3) == 0) {
        const uint32_t routed_group = (tile * 1024 + tile_offset) / 32;
        params.routed_output_scale[token * 128 + routed_group] =
            static_cast<uint8_t>(routed_exponent);
      }
    }

    if ((lane & 15) == 0) {
      const uint32_t group = (tile * 1024 + tile_offset) / 128;
      auto* scale_bytes = reinterpret_cast<uint8_t*>(params.output_scale);
      scale_bytes[
          ((group / 4) * params.scale_group_stride + token) * 4 + group % 4] =
          static_cast<uint8_t>(exponent);
    }
  }
}

// Fixed-shape SM100/SM103 path for the DSV4 Flash MHC boundary.  Warp 0 owns
// the tiny routing head while the data warps independently execute the
// residual/RMSNorm/FP8 pipeline.  The residual sweep uses two async stages;
// the 8 KiB norm weight stays L1-resident and is read directly, avoiding four
// extra cp.async groups, shared-memory writes, and named-barrier hand-offs.
// Quantization is folded into the second sweep so the normalized BF16 tensor
// is never reread by another kernel.
template <uint32_t kDataWarps, bool kEmitRoutedQuant, bool kUsePDL>
__global__ __launch_bounds__((kDataWarps + 1) * 32, 1)
void mhc_pre_norm_mxfp8_quant_pipelined_kernel(
    const MhcPreNormMxfp8QuantParams __grid_constant__ params) {
  using namespace device;
  constexpr uint32_t kDataThreads = kDataWarps * 32;
  static_assert(kDataWarps == 2 || kDataWarps == 4);

  __shared__ float mixes[24];
  __shared__ float pre_mix[4];
  __shared__ bf16_t residual_stages[2][4 * 1024];
  __shared__ bf16_t unnormalized[4096];
  __shared__ float data_warp_sums[kDataWarps];
  __shared__ float norm_factor;

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31;

  PDLWaitPrimary<kUsePDL>();

  // Only warp 0 consumes the 24 routing outputs.  The previous version used
  // ``tid % 24`` for all 96 threads and discarded the two data warps' copies,
  // redundantly issuing three extra split reductions for every output.  Keep
  // the arithmetic order for the 24 live lanes while leaving the data warps
  // free to enter the residual pipeline after the block-wide hand-off.
  if (tid < 24) {
    float routing_sq = 0.0f;
    float routing_mix = 0.0f;
    for (uint32_t split = 0; split < params.num_splits; ++split) {
      routing_sq += params.gemm_sq[split * params.num_tokens + token];
      routing_mix += params.gemm_mul[
          (split * params.num_tokens + token) * 24 + tid];
    }
    mixes[tid] = routing_mix * math::rsqrt(
        routing_sq * (1.0f / 16384.0f) + params.rms_eps);
  }
  __syncthreads();

  if (tid < 32) {
    if (lane < 4) {
      params.post_mix[token * 4 + lane] =
          params.hc_post_mult /
          (1.0f + expf(-(mixes[4 + lane] * params.hc_scale[1] +
                          params.hc_base[4 + lane])));
    }
    if (lane < 16) {
      constexpr uint32_t kMatrixMask = 0x0000ffffu;
      float cm = mixes[8 + lane] * params.hc_scale[2] +
                 params.hc_base[8 + lane];
      float row_max = cm;
      row_max = fmaxf(
          row_max, __shfl_xor_sync(kMatrixMask, row_max, 1, 4));
      row_max = fmaxf(
          row_max, __shfl_xor_sync(kMatrixMask, row_max, 2, 4));
      cm = expf(cm - row_max);
      float row_sum = cm;
      row_sum += __shfl_xor_sync(kMatrixMask, cm, 1, 4);
      row_sum += __shfl_xor_sync(kMatrixMask, cm, 2, 4);
      row_sum += __shfl_xor_sync(kMatrixMask, cm, 3, 4);
      cm = cm / row_sum + params.hc_sinkhorn_eps;

      float col_sum = cm;
      col_sum += __shfl_xor_sync(kMatrixMask, cm, 4, 16);
      col_sum += __shfl_xor_sync(kMatrixMask, cm, 8, 16);
      col_sum += __shfl_xor_sync(kMatrixMask, cm, 12, 16);
      cm /= col_sum + params.hc_sinkhorn_eps;

      for (uint32_t repeat = 1; repeat < params.sinkhorn_repeat; ++repeat) {
        row_sum = cm;
        row_sum += __shfl_xor_sync(kMatrixMask, cm, 1, 4);
        row_sum += __shfl_xor_sync(kMatrixMask, cm, 2, 4);
        row_sum += __shfl_xor_sync(kMatrixMask, cm, 3, 4);
        cm /= row_sum + params.hc_sinkhorn_eps;

        col_sum = cm;
        col_sum += __shfl_xor_sync(kMatrixMask, cm, 4, 16);
        col_sum += __shfl_xor_sync(kMatrixMask, cm, 8, 16);
        col_sum += __shfl_xor_sync(kMatrixMask, cm, 12, 16);
        cm /= col_sum + params.hc_sinkhorn_eps;
      }
      params.comb_mix[token * 16 + lane] = cm;
    }
  } else {
    const uint32_t data_tid = tid - 32;
    const uint32_t data_warp = data_tid >> 5;

    // Warp 1 computes pre_mix independently of warp 0's Sinkhorn work.  The
    // named 64-thread barrier makes it visible to both data warps without
    // coupling them to the routing warp.
    if (data_tid < 4) {
      pre_mix[data_tid] =
          1.0f /
              (1.0f + expf(-(mixes[data_tid] * params.hc_scale[0] +
                              params.hc_base[data_tid]))) +
          params.hc_pre_eps;
    }
    mhc_data_warps_barrier<kDataThreads>();

    mhc_issue_residual_tile<kDataThreads>(
        params, residual_stages[0], token, 0, data_tid);
    mhc_cp_async_commit();
    mhc_issue_residual_tile<kDataThreads>(
        params, residual_stages[1], token, 1, data_tid);
    mhc_cp_async_commit();
    float local_sq = 0.0f;
    mhc_cp_async_wait<1>();
    mhc_data_warps_barrier<kDataThreads>();
    local_sq += mhc_process_residual_tile<kDataThreads>(
        residual_stages[0], unnormalized, pre_mix, 0, data_tid);
    mhc_data_warps_barrier<kDataThreads>();
    mhc_issue_residual_tile<kDataThreads>(
        params, residual_stages[0], token, 2, data_tid);
    mhc_cp_async_commit();

    mhc_cp_async_wait<1>();
    mhc_data_warps_barrier<kDataThreads>();
    local_sq += mhc_process_residual_tile<kDataThreads>(
        residual_stages[1], unnormalized, pre_mix, 1, data_tid);
    mhc_data_warps_barrier<kDataThreads>();
    mhc_issue_residual_tile<kDataThreads>(
        params, residual_stages[1], token, 3, data_tid);
    mhc_cp_async_commit();

    mhc_cp_async_wait<1>();
    mhc_data_warps_barrier<kDataThreads>();
    local_sq += mhc_process_residual_tile<kDataThreads>(
        residual_stages[0], unnormalized, pre_mix, 2, data_tid);

    // Once residual tile 2 has been consumed, stage 0 is dead.  The
    // attention-DP specialization reuses those 8 KiB to prefetch all
    // RMSNorm weights while tile 3 is processed.  This hides the global-load
    // scoreboard exposed after routed quantization was compiled out without
    // increasing static shared memory or changing occupancy.
    if constexpr (!kEmitRoutedQuant) {
      mhc_issue_norm_weight<kDataThreads>(
          params, residual_stages[0], data_tid);
      mhc_cp_async_commit();
      // Wait for the older residual-tile-3 group while leaving the new norm
      // weight group in flight.
      mhc_cp_async_wait<1>();
    } else {
      mhc_cp_async_wait<0>();
    }
    mhc_data_warps_barrier<kDataThreads>();
    local_sq += mhc_process_residual_tile<kDataThreads>(
        residual_stages[1], unnormalized, pre_mix, 3, data_tid);
    if constexpr (!kEmitRoutedQuant) {
      mhc_cp_async_wait<0>();
    }

    const float warp_sq = warp::reduce_sum(local_sq);
    if (lane == 0) data_warp_sums[data_warp] = warp_sq;
    mhc_data_warps_barrier<kDataThreads>();
    if (data_tid == 0) {
      float total_sq = 0.0f;
#pragma unroll
      for (uint32_t warp = 0; warp < kDataWarps; ++warp) {
        total_sq += data_warp_sums[warp];
      }
      norm_factor = math::rsqrt(
          total_sq * (1.0f / 4096.0f) + params.norm_eps);
    }
    mhc_data_warps_barrier<kDataThreads>();

    const bf16_t* norm_weight =
        kEmitRoutedQuant ? params.norm_weight : residual_stages[0];
    mhc_process_norm_quant_tile<kDataThreads, kEmitRoutedQuant>(
        params,
        unnormalized,
        norm_weight,
        norm_factor,
        token,
        0,
        data_tid,
        lane);
    mhc_process_norm_quant_tile<kDataThreads, kEmitRoutedQuant>(
        params,
        unnormalized,
        norm_weight + 1024,
        norm_factor,
        token,
        1,
        data_tid,
        lane);
    mhc_process_norm_quant_tile<kDataThreads, kEmitRoutedQuant>(
        params,
        unnormalized,
        norm_weight + 2048,
        norm_factor,
        token,
        2,
        data_tid,
        lane);
    mhc_process_norm_quant_tile<kDataThreads, kEmitRoutedQuant>(
        params,
        unnormalized,
        norm_weight + 3072,
        norm_factor,
        token,
        3,
        data_tid,
        lane);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
__global__ __launch_bounds__(kThreads, 1)
void mhc_pre_norm_mxfp8_quant_kernel(
    const MhcPreNormMxfp8QuantParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kHC = 4;
  constexpr uint32_t kHidden = 4096;
  constexpr uint32_t kFlattened = kHC * kHidden;
  constexpr uint32_t kMixes = kHC * (2 + kHC);
  constexpr uint32_t kGroup = 128;
  constexpr uint32_t kGroups = kHidden / kGroup;
  constexpr uint32_t kPackedGroups = kGroups / 4;
  constexpr uint32_t kNumWarps = kThreads / kWarpThreads;
  constexpr uint32_t kGroupsPerWarp = kGroups / kNumWarps;
  static_assert(kThreads == 64 || kThreads == 128 || kThreads == 256 ||
                kThreads == 512 || kThreads == 1024);
  static_assert(kGroups % kNumWarps == 0);

  __shared__ bf16_t unnormalized[kHidden];
  __shared__ float routing_mixes[kMixes];
  __shared__ float routing_cm[kHC * kHC];
  __shared__ float routing_rms;
  __shared__ float pre_mix[kHC];
  __shared__ float warp_sums[kNumWarps];
  __shared__ float norm_factor;
  __shared__ uint8_t group_exponents[kGroups];

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % kWarpThreads;
  const uint32_t warp_id = tid / kWarpThreads;

  PDLWaitPrimary<kUsePDL>();

  // Warp 0 computes the 24-value routing head cooperatively.  The four row
  // lanes then perform Sinkhorn row/column steps in shared memory, preserving
  // the reference arithmetic order without making the entire CTA wait for a
  // single lane's 24-way reduction.
  if (warp_id == 0) {
    float mix = 0.0f;
    if (lane < kMixes) {
      for (uint32_t split = 0; split < params.num_splits; ++split) {
        mix += params.gemm_mul[
            (split * params.num_tokens + token) * kMixes + lane];
      }
    }
    if (lane == 0) {
      float sq = 0.0f;
      for (uint32_t split = 0; split < params.num_splits; ++split) {
        sq += params.gemm_sq[split * params.num_tokens + token];
      }
      routing_rms = math::rsqrt(
          sq / static_cast<float>(kFlattened) + params.rms_eps);
    }
    __syncwarp();
    if (lane < kMixes) routing_mixes[lane] = mix * routing_rms;
    __syncwarp();

    if (lane < kHC) {
      pre_mix[lane] =
          1.0f /
              (1.0f + expf(-(routing_mixes[lane] * params.hc_scale[0] +
                              params.hc_base[lane]))) +
          params.hc_pre_eps;
      params.post_mix[token * kHC + lane] =
          params.hc_post_mult /
          (1.0f +
           expf(-(routing_mixes[kHC + lane] * params.hc_scale[1] +
                  params.hc_base[kHC + lane])));

      const uint32_t row = lane;
      float row_values[kHC];
      float row_max = -3.402823466e+38F;
#pragma unroll
      for (uint32_t col = 0; col < kHC; ++col) {
        const uint32_t offset = kHC * 2 + row * kHC + col;
        row_values[col] = routing_mixes[offset] * params.hc_scale[2] +
                          params.hc_base[offset];
        row_max = fmaxf(row_max, row_values[col]);
      }
      float row_sum = 0.0f;
#pragma unroll
      for (uint32_t col = 0; col < kHC; ++col) {
        row_values[col] = expf(row_values[col] - row_max);
        row_sum += row_values[col];
      }
#pragma unroll
      for (uint32_t col = 0; col < kHC; ++col) {
        routing_cm[row * kHC + col] =
            row_values[col] / row_sum + params.hc_sinkhorn_eps;
      }
    }
    __syncwarp();

    if (lane < kHC) {
      const uint32_t col = lane;
      float col_sum = 0.0f;
#pragma unroll
      for (uint32_t row = 0; row < kHC; ++row)
        col_sum += routing_cm[row * kHC + col];
#pragma unroll
      for (uint32_t row = 0; row < kHC; ++row)
        routing_cm[row * kHC + col] /=
            col_sum + params.hc_sinkhorn_eps;
    }
    __syncwarp();

    for (uint32_t repeat = 1; repeat < params.sinkhorn_repeat; ++repeat) {
      if (lane < kHC) {
        const uint32_t row = lane;
        float row_sum = 0.0f;
#pragma unroll
        for (uint32_t col = 0; col < kHC; ++col)
          row_sum += routing_cm[row * kHC + col];
#pragma unroll
        for (uint32_t col = 0; col < kHC; ++col)
          routing_cm[row * kHC + col] /=
              row_sum + params.hc_sinkhorn_eps;
      }
      __syncwarp();
      if (lane < kHC) {
        const uint32_t col = lane;
        float col_sum = 0.0f;
#pragma unroll
        for (uint32_t row = 0; row < kHC; ++row)
          col_sum += routing_cm[row * kHC + col];
#pragma unroll
        for (uint32_t row = 0; row < kHC; ++row)
          routing_cm[row * kHC + col] /=
              col_sum + params.hc_sinkhorn_eps;
      }
      __syncwarp();
    }
    if (lane < kHC * kHC) {
      params.comb_mix[token * kHC * kHC + lane] = routing_cm[lane];
    }
  }
  __syncthreads();

  float local_sq = 0.0f;
#pragma unroll 1
  for (uint32_t group_iter = 0; group_iter < kGroupsPerWarp; ++group_iter) {
    const uint32_t group = warp_id + group_iter * kNumWarps;
    const uint32_t h01 = group * kGroup + lane * 2;
    const uint32_t h23 = h01 + kGroup / 2;
    float value0 = 0.0f;
    float value1 = 0.0f;
    float value2 = 0.0f;
    float value3 = 0.0f;
#pragma unroll
    for (uint32_t route = 0; route < kHC; ++route) {
      const auto* route_ptr =
          params.residual + (token * kHC + route) * kHidden;
      const fp32x2_t route01 = cast<fp32x2_t>(
          reinterpret_cast<const bf16x2_t*>(route_ptr + h01)[0]);
      const fp32x2_t route23 = cast<fp32x2_t>(
          reinterpret_cast<const bf16x2_t*>(route_ptr + h23)[0]);
      value0 = fmaf(pre_mix[route], route01.x, value0);
      value1 = fmaf(pre_mix[route], route01.y, value1);
      value2 = fmaf(pre_mix[route], route23.x, value2);
      value3 = fmaf(pre_mix[route], route23.y, value3);
    }
    local_sq = fmaf(value0, value0, local_sq);
    local_sq = fmaf(value1, value1, local_sq);
    local_sq = fmaf(value2, value2, local_sq);
    local_sq = fmaf(value3, value3, local_sq);
    reinterpret_cast<bf16x2_t*>(unnormalized + h01)[0] =
        cast<bf16x2_t>(make_float2(value0, value1));
    reinterpret_cast<bf16x2_t*>(unnormalized + h23)[0] =
        cast<bf16x2_t>(make_float2(value2, value3));
  }

  const float warp_sq = warp::reduce_sum(local_sq);
  if (lane == 0) warp_sums[warp_id] = warp_sq;
  __syncthreads();
  if (warp_id == 0) {
    const float warp_value = lane < kNumWarps ? warp_sums[lane] : 0.0f;
    const float block_sq = warp::reduce_sum<kNumWarps>(warp_value);
    if (lane == 0) {
      norm_factor = math::rsqrt(
          block_sq / static_cast<float>(kHidden) + params.norm_eps);
    }
  }
  __syncthreads();

  // Each warp owns one or more complete 128-value quant groups.  This keeps
  // group reductions warp-local for every thread-count specialization.
#pragma unroll 1
  for (uint32_t group_iter = 0; group_iter < kGroupsPerWarp; ++group_iter) {
    const uint32_t group = warp_id + group_iter * kNumWarps;
    const uint32_t h01 = group * kGroup + lane * 2;
    const uint32_t h23 = h01 + kGroup / 2;
    const fp32x2_t source01 = cast<fp32x2_t>(
        reinterpret_cast<const bf16x2_t*>(unnormalized + h01)[0]);
    const fp32x2_t source23 = cast<fp32x2_t>(
        reinterpret_cast<const bf16x2_t*>(unnormalized + h23)[0]);
    const fp32x2_t weight01 = cast<fp32x2_t>(
        reinterpret_cast<const bf16x2_t*>(params.norm_weight + h01)[0]);
    const fp32x2_t weight23 = cast<fp32x2_t>(
        reinterpret_cast<const bf16x2_t*>(params.norm_weight + h23)[0]);
    float values[4] = {
        source01.x * norm_factor * weight01.x,
        source01.y * norm_factor * weight01.y,
        source23.x * norm_factor * weight23.x,
        source23.y * norm_factor * weight23.y,
    };
    const bf16x2_t output01 =
        cast<bf16x2_t>(make_float2(values[0], values[1]));
    const bf16x2_t output23 =
        cast<bf16x2_t>(make_float2(values[2], values[3]));
    reinterpret_cast<bf16x2_t*>(params.layer_input + token * kHidden + h01)[0] =
        output01;
    reinterpret_cast<bf16x2_t*>(params.layer_input + token * kHidden + h23)[0] =
        output23;
    const fp32x2_t rounded01 = cast<fp32x2_t>(output01);
    const fp32x2_t rounded23 = cast<fp32x2_t>(output23);
    values[0] = rounded01.x;
    values[1] = rounded01.y;
    values[2] = rounded23.x;
    values[3] = rounded23.y;
    float local_absmax = 1e-10f;
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      local_absmax = fmaxf(local_absmax, fabsf(values[e]));
    }

    const float group_absmax = warp::reduce_max(local_absmax);
    int32_t exponent = 0;
    if (lane == 0) {
      exponent = cast_to_ue8m0(group_absmax / math::FP8_E4M3_MAX);
      group_exponents[group] = static_cast<uint8_t>(exponent);
    }
    exponent = __shfl_sync(0xffffffffu, exponent, 0);
    const float inv_scale = inv_scale_ue8m0(exponent);
    const uint32_t output_offset = token * kHidden;
    reinterpret_cast<fp8x2_e4m3_t*>(params.output_fp8 + output_offset + h01)[0] =
        pack_fp8(values[0] * inv_scale, values[1] * inv_scale);
    reinterpret_cast<fp8x2_e4m3_t*>(params.output_fp8 + output_offset + h23)[0] =
        pack_fp8(values[2] * inv_scale, values[3] * inv_scale);
  }
  __syncthreads();

  if (tid < kPackedGroups) {
    const uint32_t group = tid * 4;
    const uint32_t packed =
        static_cast<uint32_t>(group_exponents[group]) |
        (static_cast<uint32_t>(group_exponents[group + 1]) << 8) |
        (static_cast<uint32_t>(group_exponents[group + 2]) << 16) |
        (static_cast<uint32_t>(group_exponents[group + 3]) << 24);
    params.output_scale[token + tid * params.scale_group_stride] = packed;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
struct MhcPreNormMxfp8QuantKernel {
  static void run(
      const tvm::ffi::TensorView gemm_mul,
      const tvm::ffi::TensorView gemm_sq,
      const tvm::ffi::TensorView hc_scale,
      const tvm::ffi::TensorView hc_base,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView layer_input,
      const tvm::ffi::TensorView output_fp8,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView routed_output_fp8,
      const tvm::ffi::TensorView routed_output_scale,
      const double rms_eps,
      const double hc_pre_eps,
      const double hc_sinkhorn_eps,
      const double hc_post_mult,
      const int64_t sinkhorn_repeat,
      const double norm_eps,
      const bool emit_routed_quant) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto S = SymbolicSize{"num_splits"};
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"packed_groups"};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, M, 24}).with_strides({-1, 24, 1}).with_dtype<float>().with_device(device).verify(gemm_mul);
    TensorMatcher({S, M}).with_strides({-1, 1}).with_dtype<float>().with_device(device).verify(gemm_sq);
    TensorMatcher({3}).with_strides({1}).with_dtype<float>().with_device(device).verify(hc_scale);
    TensorMatcher({24}).with_strides({1}).with_dtype<float>().with_device(device).verify(hc_base);
    TensorMatcher({M, 4, 4096}).with_strides({16384, 4096, 1}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({4096}).with_strides({1}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    TensorMatcher({M, 4}).with_strides({4, 1}).with_dtype<float>().with_device(device).verify(post_mix);
    TensorMatcher({M, 4, 4}).with_strides({16, 4, 1}).with_dtype<float>().with_device(device).verify(comb_mix);
    TensorMatcher({M, 4096}).with_strides({4096, 1}).with_dtype<bf16_t>().with_device(device).verify(layer_input);
    TensorMatcher({M, 4096}).with_strides({4096, 1}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_fp8);
    TensorMatcher({P, M}).with_dtype<int32_t>().with_device(device).verify(output_scale);
    TensorMatcher({M, 4096}).with_strides({4096, 1}).with_dtype<fp8_e4m3_t>().with_device(device).verify(routed_output_fp8);
    TensorMatcher({M, 128}).with_strides({128, 1}).with_dtype<uint8_t>().with_device(device).verify(routed_output_scale);

    RuntimeCheck(P.unwrap() == 8, "mHC FP8 packed scale width must be 8");
    RuntimeCheck(output_scale.stride(0) >= M.unwrap(), "mHC scale groups overlap");
    RuntimeCheck(sinkhorn_repeat > 0, "sinkhorn_repeat must be positive");
    if (M.unwrap() == 0) return;
    const auto params = MhcPreNormMxfp8QuantParams{
        .gemm_mul = static_cast<const float*>(gemm_mul.data_ptr()),
        .gemm_sq = static_cast<const float*>(gemm_sq.data_ptr()),
        .hc_scale = static_cast<const float*>(hc_scale.data_ptr()),
        .hc_base = static_cast<const float*>(hc_base.data_ptr()),
        .residual = static_cast<const bf16_t*>(residual.data_ptr()),
        .norm_weight = static_cast<const bf16_t*>(norm_weight.data_ptr()),
        .post_mix = static_cast<float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<float*>(comb_mix.data_ptr()),
        .layer_input = static_cast<bf16_t*>(layer_input.data_ptr()),
        .output_fp8 = static_cast<fp8_e4m3_t*>(output_fp8.data_ptr()),
        .output_scale = static_cast<uint32_t*>(output_scale.data_ptr()),
        .routed_output_fp8 = static_cast<fp8_e4m3_t*>(routed_output_fp8.data_ptr()),
        .routed_output_scale = static_cast<uint8_t*>(routed_output_scale.data_ptr()),
        .rms_eps = static_cast<float>(rms_eps),
        .hc_pre_eps = static_cast<float>(hc_pre_eps),
        .hc_sinkhorn_eps = static_cast<float>(hc_sinkhorn_eps),
        .hc_post_mult = static_cast<float>(hc_post_mult),
        .norm_eps = static_cast<float>(norm_eps),
        .num_tokens = static_cast<uint32_t>(M.unwrap()),
        .num_splits = static_cast<uint32_t>(S.unwrap()),
        .sinkhorn_repeat = static_cast<uint32_t>(sinkhorn_repeat),
        .scale_group_stride = static_cast<uint32_t>(output_scale.stride(0)),
        .emit_routed_quant = static_cast<uint32_t>(emit_routed_quant),
    };
    if constexpr (kThreads == 96) {
      if (emit_routed_quant) {
        LaunchKernel(M.unwrap(), 96, device.unwrap())
            .enable_pdl(kUsePDL)(
                mhc_pre_norm_mxfp8_quant_pipelined_kernel<2, true, kUsePDL>,
                params);
      } else {
        LaunchKernel(M.unwrap(), 96, device.unwrap())
            .enable_pdl(kUsePDL)(
                mhc_pre_norm_mxfp8_quant_pipelined_kernel<2, false, kUsePDL>,
                params);
      }
    } else if constexpr (kThreads == 160) {
      if (emit_routed_quant) {
        LaunchKernel(M.unwrap(), 160, device.unwrap())
            .enable_pdl(kUsePDL)(
                mhc_pre_norm_mxfp8_quant_pipelined_kernel<4, true, kUsePDL>,
                params);
      } else {
        LaunchKernel(M.unwrap(), 160, device.unwrap())
            .enable_pdl(kUsePDL)(
                mhc_pre_norm_mxfp8_quant_pipelined_kernel<4, false, kUsePDL>,
                params);
      }
    } else {
      LaunchKernel(M.unwrap(), kThreads, device.unwrap())
          .enable_pdl(kUsePDL)(
              mhc_pre_norm_mxfp8_quant_kernel<kThreads, kUsePDL>, params);
    }
  }
};

}  // namespace
