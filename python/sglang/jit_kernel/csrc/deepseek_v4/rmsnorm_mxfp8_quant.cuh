#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>
#include <cuda_fp8.h>
#include <type_traits>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

struct RmsnormMxfp8QuantParams {
  const bf16_t* __restrict__ input;
  const bf16_t* __restrict__ weight;
  bf16_t* __restrict__ output_bf16;
  fp8_e4m3_t* __restrict__ output_fp8;
  uint32_t* __restrict__ output_scale;
  float eps;
  uint32_t num_tokens;
  uint32_t input_stride;
  uint32_t scale_group_stride;
};

template <int64_t kHiddenSize, int64_t kGroupSize, bool kUsePDL>
__global__ __launch_bounds__(kHiddenSize, 1) void rmsnorm_mxfp8_quant_kernel(
    const RmsnormMxfp8QuantParams __grid_constant__ params) {
  using namespace device;

  static_assert(kHiddenSize == 1024, "DSV4 Flash q_lora specialization expects K=1024");
  static_assert(kGroupSize == 128, "DSV4 q_lora block-FP8 group must be 128");
  constexpr uint32_t kNumWarps = kHiddenSize / kWarpThreads;
  constexpr uint32_t kWarpsPerGroup = kGroupSize / kWarpThreads;
  constexpr uint32_t kNumGroups = kHiddenSize / kGroupSize;
  constexpr uint32_t kPackedScaleGroups = kNumGroups / 4;

  __shared__ float warp_sums[kNumWarps];
  __shared__ float warp_absmax[kNumWarps];
  __shared__ float group_inv_scales[kNumGroups];
  __shared__ uint8_t group_exponents[kNumGroups];
  __shared__ float norm_factor;

  const uint32_t token_id = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid % kWarpThreads;
  const uint32_t warp_id = tid / kWarpThreads;
  const uint32_t input_row_offset = token_id * params.input_stride;
  const uint32_t output_row_offset = token_id * kHiddenSize;

  PDLWaitPrimary<kUsePDL>();

  const float x = cast<float>(params.input[input_row_offset + tid]);
  const float local_sq = x * x;
  const float warp_sq = warp::reduce_sum(local_sq);
  if (lane_id == 0) warp_sums[warp_id] = warp_sq;
  __syncthreads();

  if (warp_id == 0) {
    const float block_sq = warp::reduce_sum<kNumWarps>(warp_sums[lane_id]);
    if (lane_id == 0) {
      norm_factor = math::rsqrt(block_sq / static_cast<float>(kHiddenSize) + params.eps);
    }
  }
  __syncthreads();

  // Preserve the original RMSNorm -> quant numerical boundary in registers:
  // q_lora is first rounded to BF16, then that BF16 value is quantized.
  const float w = cast<float>(params.weight[tid]);
  const bf16_t y_bf16 = cast<bf16_t>(x * norm_factor * w);
  params.output_bf16[output_row_offset + tid] = y_bf16;
  const float y = cast<float>(y_bf16);

  const float local_warp_absmax = warp::reduce_max(fabsf(y));
  if (lane_id == 0) warp_absmax[warp_id] = local_warp_absmax;
  __syncthreads();

  const uint32_t group_id = tid / kGroupSize;
  if (lane_id == 0 && (warp_id % kWarpsPerGroup) == 0) {
    float group_absmax = 1e-10f;
#pragma unroll
    for (uint32_t i = 0; i < kWarpsPerGroup; ++i) {
      group_absmax = fmaxf(group_absmax, warp_absmax[warp_id + i]);
    }
    const int32_t ue8m0_exp =
        cast_to_ue8m0(group_absmax / math::FP8_E4M3_MAX);
    group_exponents[group_id] = static_cast<uint8_t>(ue8m0_exp);
    group_inv_scales[group_id] = inv_scale_ue8m0(ue8m0_exp);
  }
  __syncthreads();

  // DeepGEMM's Blackwell ABI stores four UE8M0 exponents in each int32,
  // with logical [M, K/128/4] and token-contiguous/TMA-aligned strides.
  if (tid < kPackedScaleGroups) {
    const uint32_t first_group = tid * 4;
    const uint32_t packed =
        static_cast<uint32_t>(group_exponents[first_group]) |
        (static_cast<uint32_t>(group_exponents[first_group + 1]) << 8) |
        (static_cast<uint32_t>(group_exponents[first_group + 2]) << 16) |
        (static_cast<uint32_t>(group_exponents[first_group + 3]) << 24);
    params.output_scale[token_id + tid * params.scale_group_stride] = packed;
  }

  const float inv_scale = group_inv_scales[group_id];

  const float y_next = __shfl_down_sync(0xffffffffu, y, 1);
  if ((lane_id & 1u) == 0) {
    reinterpret_cast<fp8x2_e4m3_t*>(params.output_fp8 + output_row_offset)[tid / 2] =
        pack_fp8(y * inv_scale, y_next * inv_scale);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <typename DType, int64_t kHiddenSize, int64_t kGroupSize, bool kUsePDL>
struct RmsnormMxfp8QuantKernel {
  static_assert(std::is_same_v<DType, bf16_t>);
  static constexpr auto kernel = rmsnorm_mxfp8_quant_kernel<kHiddenSize, kGroupSize, kUsePDL>;

  static void run(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output_bf16,
      const tvm::ffi::TensorView output_fp8,
      const tvm::ffi::TensorView output_scale,
      const double eps) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_size"};
    auto P = SymbolicSize{"packed scale groups"};
    device.set_options<kDLCUDA>();
    K.set_value(kHiddenSize);

    TensorMatcher({M, K})
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({M, K})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output_bf16);
    TensorMatcher({K}).with_dtype<bf16_t>().with_device(device).verify(weight);
    TensorMatcher({M, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_fp8);
    TensorMatcher({M, P})
        .with_strides({1, -1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(output_scale);

    const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
    constexpr uint32_t kNumGroups = kHiddenSize / kGroupSize;
    constexpr uint32_t kPackedScaleGroups = kNumGroups / 4;
    RuntimeCheck(input.stride(0) >= kHiddenSize, "q_lora input rows overlap");
    RuntimeCheck(
        input.stride(0) <= static_cast<int64_t>(UINT32_MAX),
        "q_lora input stride exceeds uint32 range");
    RuntimeCheck(P.unwrap() == kPackedScaleGroups, "invalid packed q_lora scale width");
    RuntimeCheck(
        output_scale.stride(1) >= num_tokens,
        "packed q_lora scale groups overlap");

    const auto params = RmsnormMxfp8QuantParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .output_bf16 = static_cast<bf16_t*>(output_bf16.data_ptr()),
        .output_fp8 = static_cast<fp8_e4m3_t*>(output_fp8.data_ptr()),
        .output_scale = static_cast<uint32_t*>(output_scale.data_ptr()),
        .eps = static_cast<float>(eps),
        .num_tokens = num_tokens,
        .input_stride = static_cast<uint32_t>(input.stride(0)),
        .scale_group_stride = static_cast<uint32_t>(output_scale.stride(1)),
    };
    LaunchKernel(num_tokens, kHiddenSize, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
