// DeepSeek-V4-Flash TP4 attention-output fusion:
//   inverse RoPE (tail 64 of every 512-value head)
//   + BF16 rounding boundary
//   + FP8-E4M3 UE8M0 group-128 activation quantization for WO_A.
//
// The old path launches the in-place RoPE kernel, writes the rotated BF16
// tail, then launches fp8_wo_a_group_major_quant_ue8m0.  This kernel reads the
// attention output once and never writes the rotated BF16 intermediate.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

constexpr float kLocalAbsmaxFloor = 1e-10f;
constexpr uint32_t kQuantGroup = 128;
constexpr uint32_t kThreadsPerGroup = 8;
constexpr uint32_t kGroupsPerBlock = 16;
constexpr uint32_t kInputVecBytes = 32;
constexpr uint32_t kInputInt4Count = kInputVecBytes / sizeof(int4);

template <int kThreads>
SGL_DEVICE float subgroup_reduce_max(float value) {
  static_assert(kThreads == 8);
  constexpr device::warp::mask_t kSubMask =
      (device::warp::mask_t{1} << kThreads) - 1;
  const auto subgroup = (threadIdx.x % 32) / kThreads;
  const auto mask = kSubMask << (subgroup * kThreads);
  return device::warp::reduce_max<kThreads>(value, mask);
}

template <typename T, int kHeadDim, int kRopeDim, bool kUsePDL, typename PosT>
__global__ void inverse_rope_fp8_wo_a_kernel(
    const T* __restrict__ input,
    const float* __restrict__ freqs_real,
    const PosT* __restrict__ positions,
    fp8_e4m3_t* __restrict__ output_q,
    uint32_t* __restrict__ output_s,
    int64_t total_scale_groups,
    int64_t num_tokens,
    int hidden_groups,
    int outer_groups,
    int64_t aligned_num_tokens,
    int64_t input_stride_t) {
  static_assert(kHeadDim == 512);
  static_assert(kRopeDim == 64);
  static_assert(kHeadDim % kQuantGroup == 0);
  static_assert(kRopeDim % 2 == 0);

  device::PDLWaitPrimary<kUsePDL>();

  const int64_t local_group = threadIdx.x / kThreadsPerGroup;
  const int lane = threadIdx.x % kThreadsPerGroup;
  // A 512-wide head contains four 128-value quant groups, and only its final
  // group owns RoPE values.  Arrange each warp by group phase across four
  // heads: warps 0..2 are wholly NOPE and warp 3 is wholly RoPE.  The former
  // contiguous mapping put one RoPE subgroup in every warp and predicated off
  // 24/32 lanes throughout the complex multiply.
  constexpr int kGroupsPerHead = kHeadDim / kQuantGroup;
  constexpr int kSubgroupsPerWarp = 32 / kThreadsPerGroup;
  constexpr int kHeadsPerBlock = kGroupsPerBlock / kGroupsPerHead;
  const int blocks_per_token_outer = hidden_groups / kGroupsPerBlock;
  const int block_in_token_outer = blockIdx.x % blocks_per_token_outer;
  const int64_t token_outer = blockIdx.x / blocks_per_token_outer;
  const int group_phase = local_group / kSubgroupsPerWarp;
  const int head_in_block = local_group % kSubgroupsPerWarp;
  const int hidden_group =
      (block_in_token_outer * kHeadsPerBlock + head_in_block) *
          kGroupsPerHead +
      group_phase;
  const int64_t scale_group = token_outer * hidden_groups + hidden_group;
  if (scale_group < total_scale_groups) {
    const int outer = token_outer % outer_groups;
    const int64_t token = token_outer / outer_groups;

    constexpr int kVec = kInputVecBytes / sizeof(T);
    static_assert(kVec * kThreadsPerGroup == kQuantGroup);

    const int64_t group_in_outer = static_cast<int64_t>(hidden_group) * kQuantGroup;
    const int64_t input_offset =
        token * input_stride_t +
        static_cast<int64_t>(outer) * hidden_groups * kQuantGroup +
        group_in_outer;
    const int64_t output_offset = scale_group * kQuantGroup;

    int4 raw[kInputInt4Count];
    T* values_t = reinterpret_cast<T*>(raw);
#pragma unroll
    for (uint32_t i = 0; i < kInputInt4Count; ++i) {
      raw[i] = reinterpret_cast<const int4*>(
          input + input_offset + lane * kVec)[i];
    }

    float values[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      values[i] = static_cast<float>(values_t[i]);
    }

    // kQuantGroup and kHeadDim are aligned, so a group never crosses a head.
    // The final 128-value group owns 64 NOPE values followed by all 64 RoPE
    // values.  A lane owns 16 consecutive elements, hence both values of every
    // complex pair are local to one lane.
    const int group_head_offset =
        static_cast<int>(group_in_outer % kHeadDim);
    const int lane_head_offset = group_head_offset + lane * kVec;
    const int rope_begin = kHeadDim - kRopeDim;
    if (lane_head_offset >= rope_begin) {
      const int32_t position = static_cast<int32_t>(positions[token]);
      const auto* freq_pairs = reinterpret_cast<const fp32x2_t*>(
          freqs_real + static_cast<int64_t>(position) * kRopeDim);
#pragma unroll
      for (int i = 0; i < kVec; i += 2) {
        const int rope_element = lane_head_offset + i - rope_begin;
        const auto [freq_real, freq_imag] = freq_pairs[rope_element / 2];
        const float x_real = values[i];
        const float x_imag = values[i + 1];
        // inverse: (a + bi) * (c - di)
        const float y_real = x_real * freq_real + x_imag * freq_imag;
        const float y_imag = x_imag * freq_real - x_real * freq_imag;
        // Preserve the original two-kernel numerical boundary: inverse RoPE
        // stores BF16, and the quantizer reloads that rounded value.
        values[i] = static_cast<float>(device::cast<T>(y_real));
        values[i + 1] = static_cast<float>(device::cast<T>(y_imag));
      }
    }

    float local_absmax = kLocalAbsmaxFloor;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      local_absmax = fmaxf(local_absmax, fabsf(values[i]));
    }
    const float absmax = subgroup_reduce_max<kThreadsPerGroup>(local_absmax);
    constexpr float kFP8MaxInv = 1.0f / kFP8E4M3Max;
    const int32_t scale_ue8m0 = cast_to_ue8m0(absmax * kFP8MaxInv);
    const float quant_mul = inv_scale_ue8m0(scale_ue8m0);
    int4 packed;
    auto* packed_pairs = reinterpret_cast<fp8x2_e4m3_t*>(&packed);
#pragma unroll
    for (int i = 0; i < kVec; i += 2) {
      packed_pairs[i / 2] =
          pack_fp8(values[i] * quant_mul, values[i + 1] * quant_mul);
    }
    *reinterpret_cast<int4*>(
        output_q + output_offset + lane * kVec) = packed;

    // DeepGEMM consumes four UE8M0 exponents packed in one int32.  Each
    // subgroup owns a distinct byte of that word; byte-granular global stores
    // preserve the final ABI without shuffles, a barrier, or a pack launch.
    if (lane == 0) {
      constexpr int kScalesPerWord = 4;
      const int packed_hidden_groups = hidden_groups / kScalesPerWord;
      const int packed_hidden_group = hidden_group / kScalesPerWord;
      const int64_t packed_word =
          (static_cast<int64_t>(outer) * packed_hidden_groups +
           packed_hidden_group) *
              aligned_num_tokens +
          token;
      reinterpret_cast<uint8_t*>(output_s)[
          packed_word * sizeof(uint32_t) + hidden_group % kScalesPerWord] =
          static_cast<uint8_t>(scale_ue8m0);
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, int kHeadDim, int kRopeDim, bool kUsePDL>
struct InverseRopeFP8WoAQuantUE8M0Kernel {
  template <typename PosT>
  static constexpr auto kernel =
      inverse_rope_fp8_wo_a_kernel<T, kHeadDim, kRopeDim, kUsePDL, PosT>;

  static void run(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView freqs_real,
      tvm::ffi::TensorView positions,
      tvm::ffi::TensorView output_q,
      tvm::ffi::TensorView output_s) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto TSize = SymbolicSize{"num_tokens"};
    auto GSize = SymbolicSize{"num_outer_groups"};
    auto DSize = SymbolicSize{"hidden_dim"};
    auto PSize = SymbolicSize{"packed_hidden_groups"};
    auto ASize = SymbolicSize{"aligned_num_tokens"};

    TensorMatcher({TSize, GSize, DSize})
        .with_strides({-1, DSize, 1})
        .with_dtype<T>()
        .with_device(device)
        .verify(input);
    TensorMatcher({-1, kRopeDim})
        .with_strides({kRopeDim, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(freqs_real);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({TSize})
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device)
        .verify(positions);
    TensorMatcher({TSize, GSize, DSize})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output_q);
    TensorMatcher({GSize, PSize, ASize})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(output_s);

    const int64_t num_tokens = TSize.unwrap();
    const int outer_groups = static_cast<int>(GSize.unwrap());
    const int hidden = static_cast<int>(DSize.unwrap());
    const int hidden_groups = hidden / static_cast<int>(kQuantGroup);
    const int packed_hidden_groups = static_cast<int>(PSize.unwrap());
    const int64_t aligned_num_tokens = ASize.unwrap();
    const int64_t input_stride_t = input.stride(0);

    RuntimeCheck(hidden == 4096, "TP4 fused WO_A hidden dim must be 4096");
    RuntimeCheck(outer_groups == 2, "TP4 fused WO_A outer groups must be 2");
    RuntimeCheck(hidden % kHeadDim == 0, "group hidden must contain whole heads");
    RuntimeCheck(hidden % kQuantGroup == 0, "hidden must be divisible by 128");
    RuntimeCheck(
        packed_hidden_groups == hidden_groups / 4,
        "packed output scale hidden-group mismatch");
    RuntimeCheck(
        aligned_num_tokens >= num_tokens && aligned_num_tokens % 4 == 0,
        "packed output scale token extent must be align(T, 4)");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % sizeof(int4) == 0,
        "input pointer must be 16-byte aligned");
    RuntimeCheck(
        num_tokens <= 1 ||
            input_stride_t % (sizeof(int4) / sizeof(T)) == 0,
        "input token stride must preserve vector alignment");

    const int64_t total_scale_groups =
        num_tokens * outer_groups * hidden_groups;
    if (total_scale_groups == 0) return;
    const dim3 grid(
        (total_scale_groups + kGroupsPerBlock - 1) / kGroupsPerBlock);
    const dim3 block(kGroupsPerBlock * kThreadsPerGroup);
    const auto kernel_i32 = kernel<int32_t>;
    const auto kernel_i64 = kernel<int64_t>;
    if (pos_dtype.is_type<int32_t>()) {
      LaunchKernel(grid, block, device.unwrap())
          .enable_pdl(kUsePDL)(
              kernel_i32,
              static_cast<const T*>(input.data_ptr()),
              static_cast<const float*>(freqs_real.data_ptr()),
              static_cast<const int32_t*>(positions.data_ptr()),
              static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
              static_cast<uint32_t*>(output_s.data_ptr()),
              total_scale_groups,
              num_tokens,
              hidden_groups,
              outer_groups,
              aligned_num_tokens,
              input_stride_t);
    } else {
      LaunchKernel(grid, block, device.unwrap())
          .enable_pdl(kUsePDL)(
              kernel_i64,
              static_cast<const T*>(input.data_ptr()),
              static_cast<const float*>(freqs_real.data_ptr()),
              static_cast<const int64_t*>(positions.data_ptr()),
              static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
              static_cast<uint32_t*>(output_s.data_ptr()),
              total_scale_groups,
              num_tokens,
              hidden_groups,
              outer_groups,
              aligned_num_tokens,
              input_stride_t);
    }
  }
};

}  // namespace
