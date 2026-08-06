// DeepSeek-V4-Flash Huge mHC post specialization.
//
// The generic TileLang kernel stages one 4096-wide tile through shared memory
// before writing it back.  This fixed HC=4/H=4096 implementation instead
// loads and stores eight BF16 values per 128-bit transaction, keeps the four
// route vectors in registers, and writes the rounded BF16 state directly.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_bf16.h>

namespace {

constexpr uint32_t kMhcPostHC = 4;
constexpr uint32_t kMhcPostHidden = 4096;
constexpr uint32_t kMhcPostVec = 8;
constexpr uint32_t kMhcPostThreads = 256;

struct MhcPostVec8Params {
  const __nv_bfloat16* __restrict__ hidden_in;
  const __nv_bfloat16* __restrict__ residual;
  const float* __restrict__ post_mix;
  const float* __restrict__ comb_mix;
  __nv_bfloat16* __restrict__ output;
  uint32_t num_tokens;
};

union MhcPostBf16PairBits {
  uint32_t raw;
  __nv_bfloat162 value;
};

SGL_DEVICE __nv_bfloat162 mhc_post_uint_to_bf16x2(const uint32_t raw) {
  MhcPostBf16PairBits bits;
  bits.raw = raw;
  return bits.value;
}

SGL_DEVICE uint32_t mhc_post_bf16x2_to_uint(const __nv_bfloat162 value) {
  MhcPostBf16PairBits bits;
  bits.value = value;
  return bits.raw;
}

SGL_DEVICE float2 mhc_post_fma2(
    const float coefficient, const float2 value, float2 accumulator) {
  accumulator.x = fmaf(coefficient, value.x, accumulator.x);
  accumulator.y = fmaf(coefficient, value.y, accumulator.y);
  return accumulator;
}

template <bool kUsePDL>
__global__ void mhc_post_vec8_kernel(
    const MhcPostVec8Params __grid_constant__ params) {
  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  __shared__ float coefficients[kMhcPostHC + kMhcPostHC * kMhcPostHC];
  if (tid < kMhcPostHC) {
    coefficients[tid] = params.post_mix[token * kMhcPostHC + tid];
  }
  if (tid < kMhcPostHC * kMhcPostHC) {
    coefficients[kMhcPostHC + tid] =
        params.comb_mix[token * kMhcPostHC * kMhcPostHC + tid];
  }
  __syncthreads();

  constexpr uint32_t kChunksPerToken = kMhcPostHidden / kMhcPostVec;
  const auto* hidden_chunks = reinterpret_cast<const uint4*>(
      params.hidden_in + static_cast<uint64_t>(token) * kMhcPostHidden);
  auto* output_chunks = reinterpret_cast<uint4*>(
      params.output +
      static_cast<uint64_t>(token) * kMhcPostHC * kMhcPostHidden);
  const uint64_t residual_base =
      static_cast<uint64_t>(token) * kMhcPostHC * kMhcPostHidden;

  for (uint32_t chunk = tid; chunk < kChunksPerToken;
       chunk += kMhcPostThreads) {
    const uint4 hidden_raw = hidden_chunks[chunk];
    const float2 hidden_values[kMhcPostVec / 2] = {
        __bfloat1622float2(mhc_post_uint_to_bf16x2(hidden_raw.x)),
        __bfloat1622float2(mhc_post_uint_to_bf16x2(hidden_raw.y)),
        __bfloat1622float2(mhc_post_uint_to_bf16x2(hidden_raw.z)),
        __bfloat1622float2(mhc_post_uint_to_bf16x2(hidden_raw.w)),
    };

    float2 residual_values[kMhcPostHC][kMhcPostVec / 2];
#pragma unroll
    for (uint32_t input_route = 0; input_route < kMhcPostHC;
         ++input_route) {
      const auto* route_chunks = reinterpret_cast<const uint4*>(
          params.residual + residual_base +
          static_cast<uint64_t>(input_route) * kMhcPostHidden);
      const uint4 residual_raw = route_chunks[chunk];
      residual_values[input_route][0] = __bfloat1622float2(
          mhc_post_uint_to_bf16x2(residual_raw.x));
      residual_values[input_route][1] = __bfloat1622float2(
          mhc_post_uint_to_bf16x2(residual_raw.y));
      residual_values[input_route][2] = __bfloat1622float2(
          mhc_post_uint_to_bf16x2(residual_raw.z));
      residual_values[input_route][3] = __bfloat1622float2(
          mhc_post_uint_to_bf16x2(residual_raw.w));
    }

#pragma unroll
    for (uint32_t output_route = 0; output_route < kMhcPostHC;
         ++output_route) {
      __nv_bfloat162 rounded[kMhcPostVec / 2];
#pragma unroll
      for (uint32_t pair = 0; pair < kMhcPostVec / 2; ++pair) {
        float2 value = make_float2(
            coefficients[output_route] * hidden_values[pair].x,
            coefficients[output_route] * hidden_values[pair].y);
#pragma unroll
        for (uint32_t input_route = 0; input_route < kMhcPostHC;
             ++input_route) {
          value = mhc_post_fma2(
              coefficients[
                  kMhcPostHC + input_route * kMhcPostHC + output_route],
              residual_values[input_route][pair],
              value);
        }
        rounded[pair] = __float22bfloat162_rn(value);
      }
      output_chunks[output_route * kChunksPerToken + chunk] = make_uint4(
          mhc_post_bf16x2_to_uint(rounded[0]),
          mhc_post_bf16x2_to_uint(rounded[1]),
          mhc_post_bf16x2_to_uint(rounded[2]),
          mhc_post_bf16x2_to_uint(rounded[3]));
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct MhcPostVec8Kernel {
  static void run(
      const tvm::ffi::TensorView hidden_in,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kMhcPostHidden})
        .with_strides({kMhcPostHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(hidden_in);
    TensorMatcher({M, kMhcPostHC, kMhcPostHidden})
        .with_strides({kMhcPostHC * kMhcPostHidden, kMhcPostHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({M, kMhcPostHC})
        .with_strides({kMhcPostHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({M, kMhcPostHC, kMhcPostHC})
        .with_strides({kMhcPostHC * kMhcPostHC, kMhcPostHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({M, kMhcPostHC, kMhcPostHidden})
        .with_strides({kMhcPostHC * kMhcPostHidden, kMhcPostHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);

    if (M.unwrap() == 0) return;
    const auto params = MhcPostVec8Params{
        .hidden_in = reinterpret_cast<const __nv_bfloat16*>(hidden_in.data_ptr()),
        .residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .num_tokens = static_cast<uint32_t>(M.unwrap()),
    };
    LaunchKernel(M.unwrap(), kMhcPostThreads, device.unwrap())
        .enable_pdl(kUsePDL)(mhc_post_vec8_kernel<kUsePDL>, params);
  }
};

}  // namespace
