// DeepSeek-V4-Flash Huge TP4 MoE reduction + mHC post specialization.
//
// FlashInfer writes each rank's BF16 MoE partial directly into a peer-visible
// symmetric buffer.  One kernel then reproduces the measured NCCL ring BF16
// accumulation order and immediately consumes the reduced value in mHC post,
// avoiding a reduced [M,4096] intermediate and a separate post launch.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace {

constexpr uint32_t kTp4MoeMhcHC = 4;
constexpr uint32_t kTp4MoeMhcHidden = 4096;
constexpr uint32_t kTp4MoeMhcVec = 8;
constexpr uint32_t kTp4MoeMhcThreads = 256;
constexpr uint32_t kTp4OwnerPostTokensPerCTA = 4;
// B300/SM103 has 148 SMs.  Four 256-thread CTAs per SM matches the
// launch-bounds constrained owner-post occupancy and guarantees that the
// whole persistent grid can remain resident while it performs a software
// grid barrier.  This path is intentionally specialized to the strict B300
// deployment matrix.
constexpr uint32_t kTp4OwnerPersistentBlocks = 148 * 4;
constexpr uint32_t kTp4OwnerFlagArrived = 0;
constexpr uint32_t kTp4OwnerFlagProducedEpoch = 1;
constexpr uint32_t kTp4OwnerFlagPeerReadyEpoch = 2;
constexpr uint32_t kTp4OwnerFlagCount = 4;
constexpr uint32_t kTp4MoeEpochReadyBase = 0;
constexpr uint32_t kTp4MoeEpochDoneBase = 2;
constexpr uint32_t kTp4MoeEpochFlagCount = 4;
constexpr uint64_t kTp4MoeEpochLocalReadyBit = 1ULL << 31;
constexpr uint64_t kTp4MoeEpochLocalCountMask =
    kTp4MoeEpochLocalReadyBit - 1;
constexpr uint64_t kNCCLChannelGroupElements = 1ULL << 20;
constexpr uint64_t kNCCLPeriodElements = 4 * kNCCLChannelGroupElements;

struct Tp4MoeMhcPostParams {
  const __nv_bfloat16* __restrict__ input0;
  const __nv_bfloat16* __restrict__ input1;
  const __nv_bfloat16* __restrict__ input2;
  const __nv_bfloat16* __restrict__ input3;
  const __nv_bfloat16* __restrict__ multicast_input;
  const __nv_bfloat16* __restrict__ residual;
  const float* __restrict__ post_mix;
  const float* __restrict__ comb_mix;
  __nv_bfloat16* __restrict__ output;
  uint32_t num_tokens;
};

// Attention-DP combine specialization.  Each peer exposes its TP-sharded
// MoE output as one symmetric [global_M, H] buffer.  A rank consumes only its
// contiguous local row interval, adds the independently computed local shared
// expert with the same BF16 rounding boundary as torch.add, and immediately
// applies mHC post.  No reduced [local_M, H] tensor is materialized.
struct Tp4MoeLocalSliceMhcPostParams {
  const __nv_bfloat16* __restrict__ input0;
  const __nv_bfloat16* __restrict__ input1;
  const __nv_bfloat16* __restrict__ input2;
  const __nv_bfloat16* __restrict__ input3;
  const __nv_bfloat16* __restrict__ multicast_input;
  const __nv_bfloat16* __restrict__ shared_hidden;
  const __nv_bfloat16* __restrict__ residual;
  const float* __restrict__ post_mix;
  const float* __restrict__ comb_mix;
  __nv_bfloat16* __restrict__ output;
  uint32_t local_num_tokens;
  uint32_t local_row_offset;
};

struct Tp4MoeLocalSliceEpochParams {
  Tp4MoeLocalSliceMhcPostParams post;
  uint32_t* flags0;
  uint32_t* flags1;
  uint32_t* flags2;
  uint32_t* flags3;
  uint32_t rank;
  uint32_t slot;
  uint32_t epoch;
};

struct Tp4MoeEpochControlParams {
  uint32_t* flags0;
  uint32_t* flags1;
  uint32_t* flags2;
  uint32_t* flags3;
  uint32_t rank;
  uint32_t slot;
  uint32_t epoch;
};

struct Tp4MoeLocalSliceEpochCounterParams {
  Tp4MoeLocalSliceMhcPostParams post;
  uint32_t* flags0;
  uint32_t* flags1;
  uint32_t* flags2;
  uint32_t* flags3;
  unsigned long long* completion_state;
  uint32_t rank;
  uint32_t slot;
  uint32_t epoch;
};

struct Tp4MoeWaitSlotParams {
  const uint32_t* flags0;
  const uint32_t* flags1;
  const uint32_t* flags2;
  const uint32_t* flags3;
  uint32_t slot;
  uint32_t expected_done_epoch;
};

struct Tp4MoeOwnerPersistentParams {
  Tp4MoeMhcPostParams post;
  uint32_t* flags0;
  uint32_t* flags1;
  uint32_t* flags2;
  uint32_t* flags3;
  uint32_t owner_rank;
};

union Tp4MoeMhcBf16PairBits {
  uint32_t raw;
  __nv_bfloat162 value;
};

SGL_DEVICE __nv_bfloat162 tp4_moe_mhc_uint_to_bf16x2(const uint32_t raw) {
  Tp4MoeMhcBf16PairBits bits;
  bits.raw = raw;
  return bits.value;
}

SGL_DEVICE uint32_t tp4_moe_mhc_bf16x2_to_uint(
    const __nv_bfloat162 value) {
  Tp4MoeMhcBf16PairBits bits;
  bits.value = value;
  return bits.raw;
}

SGL_DEVICE uint4 tp4_moe_mhc_multimem_reduce_bf16x8(
    const __nv_bfloat16* multicast_ptr) {
  uint4 reduced;
  asm volatile(
      "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
      "{%0,%1,%2,%3}, [%4];"
      : "=r"(reduced.x),
        "=r"(reduced.y),
        "=r"(reduced.z),
        "=r"(reduced.w)
      : "l"(multicast_ptr)
      : "memory");
  return reduced;
}

template <uint32_t kPair>
SGL_DEVICE uint32_t tp4_moe_mhc_uint4_pair(const uint4 value) {
  static_assert(kPair < kTp4MoeMhcVec / 2);
  if constexpr (kPair == 0) {
    return value.x;
  } else if constexpr (kPair == 1) {
    return value.y;
  } else if constexpr (kPair == 2) {
    return value.z;
  } else {
    return value.w;
  }
}

SGL_DEVICE __nv_bfloat162 tp4_moe_mhc_add4_ordered(
    __nv_bfloat162 a,
    __nv_bfloat162 b,
    __nv_bfloat162 c,
    __nv_bfloat162 d,
    const uint32_t group) {
  switch (group) {
    case 0:
      return __hadd2(__hadd2(__hadd2(b, c), d), a);
    case 1:
      return __hadd2(__hadd2(__hadd2(c, d), a), b);
    case 2:
      return __hadd2(__hadd2(__hadd2(a, d), b), c);
    default:
      return __hadd2(__hadd2(__hadd2(a, b), c), d);
  }
}

SGL_DEVICE float2 tp4_moe_mhc_fma2(
    const float coefficient, const float2 value, float2 accumulator) {
  accumulator.x = fmaf(coefficient, value.x, accumulator.x);
  accumulator.y = fmaf(coefficient, value.y, accumulator.y);
  return accumulator;
}

SGL_DEVICE void tp4_moe_mhc_cp_async_16(
    void* shared_dst, const void* global_src) {
  const uint32_t shared_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));
  asm volatile(
      "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
          "r"(shared_addr),
      "l"(global_src));
}

SGL_DEVICE void tp4_moe_mhc_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

SGL_DEVICE uint32_t tp4_moe_mhc_load_acquire_sys(
    const uint32_t* ptr) {
  uint32_t value;
  asm volatile(
      "ld.acquire.sys.global.u32 %0, [%1];"
      : "=r"(value)
      : "l"(ptr)
      : "memory");
  return value;
}

SGL_DEVICE void tp4_moe_mhc_store_release_sys(
    uint32_t* ptr, const uint32_t value) {
  asm volatile(
      "st.release.sys.global.u32 [%0], %1;" ::
          "l"(ptr),
      "r"(value)
      : "memory");
}

SGL_DEVICE uint32_t tp4_moe_mhc_atomic_add_release_sys(
    uint32_t* ptr, const uint32_t value) {
  uint32_t previous;
  asm volatile(
      "atom.release.sys.global.add.u32 %0, [%1], %2;"
      : "=r"(previous)
      : "l"(ptr), "r"(value)
      : "memory");
  return previous;
}

SGL_DEVICE unsigned long long tp4_moe_mhc_atomic_add_acq_rel_gpu(
    unsigned long long* ptr, const unsigned long long value) {
  unsigned long long previous;
  asm volatile(
      "atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;"
      : "=l"(previous)
      : "l"(ptr), "l"(value)
      : "memory");
  return previous;
}

SGL_DEVICE unsigned long long tp4_moe_mhc_load_acquire_gpu_u64(
    const unsigned long long* ptr) {
  unsigned long long value;
  asm volatile(
      "ld.acquire.gpu.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(ptr)
      : "memory");
  return value;
}

SGL_DEVICE void tp4_moe_mhc_fence_proxy_alias() {
  // FlashInfer produces through each rank's unicast mapping, whereas NVLS
  // consumes the same bytes through their multicast alias.  The epoch
  // acquire orders producer completion; this proxy fence makes that ordering
  // visible across the two virtual aliases before any multimem load.
  asm volatile("fence.proxy.alias;" : : : "memory");
}

template <int kPending>
SGL_DEVICE void tp4_moe_mhc_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(kPending));
}

template <bool kUsePDL, bool kUseMultimem>
__global__ void tp4_moe_mhc_post_kernel(
    const Tp4MoeMhcPostParams __grid_constant__ params) {
  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t token = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  __shared__ float coefficients[
      kTp4MoeMhcHC + kTp4MoeMhcHC * kTp4MoeMhcHC];
  __shared__ __align__(16) uint4 peer_stages[2][4][kTp4MoeMhcThreads];
  if (tid < kTp4MoeMhcHC) {
    coefficients[tid] = params.post_mix[token * kTp4MoeMhcHC + tid];
  }
  if (tid < kTp4MoeMhcHC * kTp4MoeMhcHC) {
    coefficients[kTp4MoeMhcHC + tid] =
        params.comb_mix[
            token * kTp4MoeMhcHC * kTp4MoeMhcHC + tid];
  }
  __syncthreads();

  constexpr uint32_t kChunksPerToken =
      kTp4MoeMhcHidden / kTp4MoeMhcVec;
  const uint64_t hidden_base =
      static_cast<uint64_t>(token) * kTp4MoeMhcHidden;
  auto* output_chunks = reinterpret_cast<uint4*>(
      params.output +
      static_cast<uint64_t>(token) * kTp4MoeMhcHC * kTp4MoeMhcHidden);
  const uint64_t residual_base =
      static_cast<uint64_t>(token) * kTp4MoeMhcHC * kTp4MoeMhcHidden;

  if constexpr (!kUseMultimem) {
#pragma unroll
    for (uint32_t stage = 0; stage < 2; ++stage) {
      const uint32_t chunk = tid + stage * kTp4MoeMhcThreads;
      const uint64_t element_offset =
          hidden_base + static_cast<uint64_t>(chunk) * kTp4MoeMhcVec;
      tp4_moe_mhc_cp_async_16(
          &peer_stages[stage][0][tid], params.input0 + element_offset);
      tp4_moe_mhc_cp_async_16(
          &peer_stages[stage][1][tid], params.input1 + element_offset);
      tp4_moe_mhc_cp_async_16(
          &peer_stages[stage][2][tid], params.input2 + element_offset);
      tp4_moe_mhc_cp_async_16(
          &peer_stages[stage][3][tid], params.input3 + element_offset);
      tp4_moe_mhc_cp_async_commit();
    }
  }

  for (uint32_t chunk = tid; chunk < kChunksPerToken;
       chunk += kTp4MoeMhcThreads) {
    uint4 reduced_raw;
    if constexpr (kUseMultimem) {
      reduced_raw = tp4_moe_mhc_multimem_reduce_bf16x8(
          params.multicast_input + hidden_base +
          static_cast<uint64_t>(chunk) * kTp4MoeMhcVec);
    } else {
      const uint32_t stage = chunk / kTp4MoeMhcThreads;
      if (stage == 0) {
        tp4_moe_mhc_cp_async_wait<1>();
      } else {
        tp4_moe_mhc_cp_async_wait<0>();
      }
      const uint4 input0_raw = peer_stages[stage][0][tid];
      const uint4 input1_raw = peer_stages[stage][1][tid];
      const uint4 input2_raw = peer_stages[stage][2][tid];
      const uint4 input3_raw = peer_stages[stage][3][tid];
      const auto* input0_pairs =
          reinterpret_cast<const __nv_bfloat162*>(&input0_raw);
      const auto* input1_pairs =
          reinterpret_cast<const __nv_bfloat162*>(&input1_raw);
      const auto* input2_pairs =
          reinterpret_cast<const __nv_bfloat162*>(&input2_raw);
      const auto* input3_pairs =
          reinterpret_cast<const __nv_bfloat162*>(&input3_raw);
      auto* reduced_pairs =
          reinterpret_cast<__nv_bfloat162*>(&reduced_raw);
      const uint64_t first_element =
          hidden_base + static_cast<uint64_t>(chunk) * kTp4MoeMhcVec;
      const uint32_t group = static_cast<uint32_t>(
          (first_element % kNCCLPeriodElements) /
          kNCCLChannelGroupElements);
#pragma unroll
      for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
        reduced_pairs[pair] = tp4_moe_mhc_add4_ordered(
            input0_pairs[pair],
            input1_pairs[pair],
            input2_pairs[pair],
            input3_pairs[pair],
            group);
      }
    }
    const auto* reduced_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&reduced_raw);
    float2 hidden_values[kTp4MoeMhcVec / 2];
#pragma unroll
    for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
      hidden_values[pair] = __bfloat1622float2(reduced_pairs[pair]);
    }

    float2 residual_values[kTp4MoeMhcHC][kTp4MoeMhcVec / 2];
#pragma unroll
    for (uint32_t input_route = 0; input_route < kTp4MoeMhcHC;
         ++input_route) {
      const auto* route_chunks = reinterpret_cast<const uint4*>(
          params.residual + residual_base +
          static_cast<uint64_t>(input_route) * kTp4MoeMhcHidden);
      const uint4 residual_raw = route_chunks[chunk];
      residual_values[input_route][0] = __bfloat1622float2(
          tp4_moe_mhc_uint_to_bf16x2(residual_raw.x));
      residual_values[input_route][1] = __bfloat1622float2(
          tp4_moe_mhc_uint_to_bf16x2(residual_raw.y));
      residual_values[input_route][2] = __bfloat1622float2(
          tp4_moe_mhc_uint_to_bf16x2(residual_raw.z));
      residual_values[input_route][3] = __bfloat1622float2(
          tp4_moe_mhc_uint_to_bf16x2(residual_raw.w));
    }

#pragma unroll
    for (uint32_t output_route = 0; output_route < kTp4MoeMhcHC;
         ++output_route) {
      __nv_bfloat162 rounded[kTp4MoeMhcVec / 2];
#pragma unroll
      for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
        float2 value = make_float2(
            coefficients[output_route] * hidden_values[pair].x,
            coefficients[output_route] * hidden_values[pair].y);
#pragma unroll
        for (uint32_t input_route = 0; input_route < kTp4MoeMhcHC;
             ++input_route) {
          value = tp4_moe_mhc_fma2(
              coefficients[
                  kTp4MoeMhcHC +
                  input_route * kTp4MoeMhcHC + output_route],
              residual_values[input_route][pair],
              value);
        }
        rounded[pair] = __float22bfloat162_rn(value);
      }
      output_chunks[output_route * kChunksPerToken + chunk] = make_uint4(
          tp4_moe_mhc_bf16x2_to_uint(rounded[0]),
          tp4_moe_mhc_bf16x2_to_uint(rounded[1]),
          tp4_moe_mhc_bf16x2_to_uint(rounded[2]),
          tp4_moe_mhc_bf16x2_to_uint(rounded[3]));
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

// Bring-up kernel for the Attention-DP output boundary.  The peer inputs are
// global symmetric buffers, whereas every other tensor is the local token
// slice.  The global element offset is deliberately retained in the channel
// group calculation: changing it to a local offset changes BF16 addition
// order relative to NCCL reduce_scatterv at 1M-element channel boundaries.
SGL_DEVICE void tp4_moe_local_slice_shared_mhc_post_token(
    const Tp4MoeLocalSliceMhcPostParams& params,
    const uint32_t local_token,
    float* coefficients,
    uint4 peer_stages[2][4][kTp4MoeMhcThreads]) {
  const uint32_t global_token = params.local_row_offset + local_token;
  const uint32_t tid = threadIdx.x;
  if (tid < kTp4MoeMhcHC) {
    coefficients[tid] =
        params.post_mix[local_token * kTp4MoeMhcHC + tid];
  }
  if (tid < kTp4MoeMhcHC * kTp4MoeMhcHC) {
    coefficients[kTp4MoeMhcHC + tid] =
        params.comb_mix[
            local_token * kTp4MoeMhcHC * kTp4MoeMhcHC + tid];
  }
  __syncthreads();

  constexpr uint32_t kChunksPerToken =
      kTp4MoeMhcHidden / kTp4MoeMhcVec;
  const uint64_t global_hidden_base =
      static_cast<uint64_t>(global_token) * kTp4MoeMhcHidden;
  const uint64_t local_hidden_base =
      static_cast<uint64_t>(local_token) * kTp4MoeMhcHidden;
  const uint64_t residual_base =
      static_cast<uint64_t>(local_token) *
      kTp4MoeMhcHC * kTp4MoeMhcHidden;
  auto* output_chunks = reinterpret_cast<uint4*>(
      params.output + residual_base);

#pragma unroll
  for (uint32_t stage = 0; stage < 2; ++stage) {
    const uint32_t chunk = tid + stage * kTp4MoeMhcThreads;
    const uint64_t element =
        global_hidden_base +
        static_cast<uint64_t>(chunk) * kTp4MoeMhcVec;
    tp4_moe_mhc_cp_async_16(
        &peer_stages[stage][0][tid], params.input0 + element);
    tp4_moe_mhc_cp_async_16(
        &peer_stages[stage][1][tid], params.input1 + element);
    tp4_moe_mhc_cp_async_16(
        &peer_stages[stage][2][tid], params.input2 + element);
    tp4_moe_mhc_cp_async_16(
        &peer_stages[stage][3][tid], params.input3 + element);
    tp4_moe_mhc_cp_async_commit();
  }

  for (uint32_t chunk = tid; chunk < kChunksPerToken;
       chunk += kTp4MoeMhcThreads) {
    const uint32_t stage = chunk / kTp4MoeMhcThreads;
    if (stage == 0) {
      tp4_moe_mhc_cp_async_wait<1>();
    } else {
      tp4_moe_mhc_cp_async_wait<0>();
    }
    const uint4 input0_raw = peer_stages[stage][0][tid];
    const uint4 input1_raw = peer_stages[stage][1][tid];
    const uint4 input2_raw = peer_stages[stage][2][tid];
    const uint4 input3_raw = peer_stages[stage][3][tid];
    const auto* input0_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input0_raw);
    const auto* input1_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input1_raw);
    const auto* input2_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input2_raw);
    const auto* input3_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input3_raw);
    __nv_bfloat162 reduced_pairs[kTp4MoeMhcVec / 2];
    const uint64_t first_element =
        global_hidden_base +
        static_cast<uint64_t>(chunk) * kTp4MoeMhcVec;
    const uint32_t group = static_cast<uint32_t>(
        (first_element % kNCCLPeriodElements) /
        kNCCLChannelGroupElements);
#pragma unroll
    for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
      reduced_pairs[pair] = tp4_moe_mhc_add4_ordered(
          input0_pairs[pair],
          input1_pairs[pair],
          input2_pairs[pair],
          input3_pairs[pair],
          group);
    }

    const auto* shared_chunks = reinterpret_cast<const uint4*>(
        params.shared_hidden + local_hidden_base);
    const uint4 shared_raw = shared_chunks[chunk];
    const auto* shared_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&shared_raw);
    float2 hidden_values[kTp4MoeMhcVec / 2];
#pragma unroll
    for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
      const float2 reduced_value =
          __bfloat1622float2(reduced_pairs[pair]);
      const float2 shared_value =
          __bfloat1622float2(shared_pairs[pair]);
      // Preserve the former reduce_scatterv -> BF16 torch.add boundary.
      const __nv_bfloat162 rounded_sum = __float22bfloat162_rn(make_float2(
          reduced_value.x + shared_value.x,
          reduced_value.y + shared_value.y));
      hidden_values[pair] = __bfloat1622float2(rounded_sum);
    }

    float2 residual_values[kTp4MoeMhcHC][kTp4MoeMhcVec / 2];
#pragma unroll
    for (uint32_t input_route = 0; input_route < kTp4MoeMhcHC;
         ++input_route) {
      const auto* route_chunks = reinterpret_cast<const uint4*>(
          params.residual + residual_base +
          static_cast<uint64_t>(input_route) * kTp4MoeMhcHidden);
      const uint4 residual_raw = route_chunks[chunk];
      residual_values[input_route][0] = __bfloat1622float2(
          tp4_moe_mhc_uint_to_bf16x2(residual_raw.x));
      residual_values[input_route][1] = __bfloat1622float2(
          tp4_moe_mhc_uint_to_bf16x2(residual_raw.y));
      residual_values[input_route][2] = __bfloat1622float2(
          tp4_moe_mhc_uint_to_bf16x2(residual_raw.z));
      residual_values[input_route][3] = __bfloat1622float2(
          tp4_moe_mhc_uint_to_bf16x2(residual_raw.w));
    }

#pragma unroll
    for (uint32_t output_route = 0; output_route < kTp4MoeMhcHC;
         ++output_route) {
      __nv_bfloat162 rounded[kTp4MoeMhcVec / 2];
#pragma unroll
      for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
        float2 value = make_float2(
            coefficients[output_route] * hidden_values[pair].x,
            coefficients[output_route] * hidden_values[pair].y);
#pragma unroll
        for (uint32_t input_route = 0; input_route < kTp4MoeMhcHC;
             ++input_route) {
          value = tp4_moe_mhc_fma2(
              coefficients[
                  kTp4MoeMhcHC +
                  input_route * kTp4MoeMhcHC + output_route],
              residual_values[input_route][pair],
              value);
        }
        rounded[pair] = __float22bfloat162_rn(value);
      }
      output_chunks[output_route * kChunksPerToken + chunk] = make_uint4(
          tp4_moe_mhc_bf16x2_to_uint(rounded[0]),
          tp4_moe_mhc_bf16x2_to_uint(rounded[1]),
          tp4_moe_mhc_bf16x2_to_uint(rounded[2]),
          tp4_moe_mhc_bf16x2_to_uint(rounded[3]));
    }
  }

}

// NVLS specialization of the same Attention-DP boundary.  The multicast VA
// is pre-offset to this rank's local row interval and selected double-buffer
// slot on the host descriptor path.  One switch-side reduction replaces four
// peer loads, the 32 KiB peer staging allocation, and the software add chain.
// ``multimem.ld_reduce ... acc::f32 ... bf16x2`` returns BF16 pairs; the
// following shared-expert add is deliberately rounded back to BF16 so the
// former reduce-scatter -> torch.add boundary remains explicit.
//
// Keep the six BF16 uint4 inputs packed while computing one pair and one
// output route at a time.  The former float2[4] + float2[4][4] live ranges
// exceeded the 64-register launch-bound budget and generated 80 B/thread of
// local stack.  Recompute the cheap BF16-to-FP32 unpack for each output route
// so only one pair is live at a time, then assemble all four pairs into one
// uint4 store.  This preserves every output route's input0->3 FMA order and
// both BF16 rounding boundaries while recovering the original 128-bit global
// store without rematerializing the spilled arrays.
template <uint32_t kPair>
SGL_DEVICE uint32_t tp4_moe_local_slice_multimem_shared_mhc_post_pair(
    const uint4 reduced_raw,
    const uint4 shared_raw,
    const uint4 residual_raw0,
    const uint4 residual_raw1,
    const uint4 residual_raw2,
    const uint4 residual_raw3,
    const float* coefficients,
    const uint32_t output_route) {
  const float2 reduced_value = __bfloat1622float2(
      tp4_moe_mhc_uint_to_bf16x2(
          tp4_moe_mhc_uint4_pair<kPair>(reduced_raw)));
  const float2 shared_value = __bfloat1622float2(
      tp4_moe_mhc_uint_to_bf16x2(
          tp4_moe_mhc_uint4_pair<kPair>(shared_raw)));
  const __nv_bfloat162 rounded_sum = __float22bfloat162_rn(make_float2(
      reduced_value.x + shared_value.x,
      reduced_value.y + shared_value.y));
  const float2 hidden_value = __bfloat1622float2(rounded_sum);

  const float2 residual_value0 = __bfloat1622float2(
      tp4_moe_mhc_uint_to_bf16x2(
          tp4_moe_mhc_uint4_pair<kPair>(residual_raw0)));
  const float2 residual_value1 = __bfloat1622float2(
      tp4_moe_mhc_uint_to_bf16x2(
          tp4_moe_mhc_uint4_pair<kPair>(residual_raw1)));
  const float2 residual_value2 = __bfloat1622float2(
      tp4_moe_mhc_uint_to_bf16x2(
          tp4_moe_mhc_uint4_pair<kPair>(residual_raw2)));
  const float2 residual_value3 = __bfloat1622float2(
      tp4_moe_mhc_uint_to_bf16x2(
          tp4_moe_mhc_uint4_pair<kPair>(residual_raw3)));

  float2 value = make_float2(
      coefficients[output_route] * hidden_value.x,
      coefficients[output_route] * hidden_value.y);
  value = tp4_moe_mhc_fma2(
      coefficients[kTp4MoeMhcHC + output_route],
      residual_value0,
      value);
  value = tp4_moe_mhc_fma2(
      coefficients[2 * kTp4MoeMhcHC + output_route],
      residual_value1,
      value);
  value = tp4_moe_mhc_fma2(
      coefficients[3 * kTp4MoeMhcHC + output_route],
      residual_value2,
      value);
  value = tp4_moe_mhc_fma2(
      coefficients[4 * kTp4MoeMhcHC + output_route],
      residual_value3,
      value);
  return tp4_moe_mhc_bf16x2_to_uint(__float22bfloat162_rn(value));
}

SGL_DEVICE void tp4_moe_local_slice_multimem_shared_mhc_post_token(
    const Tp4MoeLocalSliceMhcPostParams& params,
    const uint32_t local_token,
    float* coefficients) {
  const uint32_t tid = threadIdx.x;
  if (tid < kTp4MoeMhcHC) {
    coefficients[tid] =
        params.post_mix[local_token * kTp4MoeMhcHC + tid];
  }
  if (tid < kTp4MoeMhcHC * kTp4MoeMhcHC) {
    coefficients[kTp4MoeMhcHC + tid] =
        params.comb_mix[
            local_token * kTp4MoeMhcHC * kTp4MoeMhcHC + tid];
  }
  __syncthreads();

  constexpr uint32_t kChunksPerToken =
      kTp4MoeMhcHidden / kTp4MoeMhcVec;
  const uint64_t local_hidden_base =
      static_cast<uint64_t>(local_token) * kTp4MoeMhcHidden;
  const uint64_t residual_base =
      static_cast<uint64_t>(local_token) *
      kTp4MoeMhcHC * kTp4MoeMhcHidden;
  auto* output_chunks = reinterpret_cast<uint4*>(
      params.output + residual_base);

  for (uint32_t chunk = tid; chunk < kChunksPerToken;
       chunk += kTp4MoeMhcThreads) {
    const uint64_t element =
        local_hidden_base +
        static_cast<uint64_t>(chunk) * kTp4MoeMhcVec;
    const uint4 reduced_raw = tp4_moe_mhc_multimem_reduce_bf16x8(
        params.multicast_input + element);

    const auto* shared_chunks = reinterpret_cast<const uint4*>(
        params.shared_hidden + local_hidden_base);
    const uint4 shared_raw = shared_chunks[chunk];

    const auto* residual_chunks = reinterpret_cast<const uint4*>(
        params.residual + residual_base);
    const uint4 residual_raw0 = residual_chunks[chunk];
    const uint4 residual_raw1 =
        residual_chunks[kChunksPerToken + chunk];
    const uint4 residual_raw2 =
        residual_chunks[2 * kChunksPerToken + chunk];
    const uint4 residual_raw3 =
        residual_chunks[3 * kChunksPerToken + chunk];

#pragma unroll 1
    for (uint32_t output_route = 0; output_route < kTp4MoeMhcHC;
         ++output_route) {
      uint4 output_raw;
      output_raw.x =
          tp4_moe_local_slice_multimem_shared_mhc_post_pair<0>(
              reduced_raw,
              shared_raw,
              residual_raw0,
              residual_raw1,
              residual_raw2,
              residual_raw3,
              coefficients,
              output_route);
      output_raw.y =
          tp4_moe_local_slice_multimem_shared_mhc_post_pair<1>(
              reduced_raw,
              shared_raw,
              residual_raw0,
              residual_raw1,
              residual_raw2,
              residual_raw3,
              coefficients,
              output_route);
      output_raw.z =
          tp4_moe_local_slice_multimem_shared_mhc_post_pair<2>(
              reduced_raw,
              shared_raw,
              residual_raw0,
              residual_raw1,
              residual_raw2,
              residual_raw3,
              coefficients,
              output_route);
      output_raw.w =
          tp4_moe_local_slice_multimem_shared_mhc_post_pair<3>(
              reduced_raw,
              shared_raw,
              residual_raw0,
              residual_raw1,
              residual_raw2,
              residual_raw3,
              coefficients,
              output_route);
      output_chunks[output_route * kChunksPerToken + chunk] = output_raw;
    }
  }
}

template <bool kUsePDL>
__global__ void tp4_moe_local_slice_shared_mhc_post_kernel(
    const Tp4MoeLocalSliceMhcPostParams __grid_constant__ params) {
  device::PDLWaitPrimary<kUsePDL>();
  __shared__ float coefficients[
      kTp4MoeMhcHC + kTp4MoeMhcHC * kTp4MoeMhcHC];
  __shared__ __align__(16) uint4 peer_stages[2][4][kTp4MoeMhcThreads];
  tp4_moe_local_slice_shared_mhc_post_token(
      params, blockIdx.x, coefficients, peer_stages);
  device::PDLTriggerSecondary<kUsePDL>();
}

// Wait before reusing one double-buffered symmetric producer slot.  Epoch
// zero denotes a slot that has never been published and therefore needs no
// peer wait.  A single system-acquire polling thread is sufficient; stream
// ordering holds the following producer until this CTA completes.
__global__ void tp4_moe_wait_slot_reusable_kernel(
    const Tp4MoeWaitSlotParams __grid_constant__ params) {
  if (threadIdx.x != 0 || params.expected_done_epoch == 0) {
    return;
  }
  bool reusable = false;
  while (!reusable) {
    const uint32_t done_index = kTp4MoeEpochDoneBase + params.slot;
    reusable =
        tp4_moe_mhc_load_acquire_sys(params.flags0 + done_index) ==
            params.expected_done_epoch &&
        tp4_moe_mhc_load_acquire_sys(params.flags1 + done_index) ==
            params.expected_done_epoch &&
        tp4_moe_mhc_load_acquire_sys(params.flags2 + done_index) ==
            params.expected_done_epoch &&
        tp4_moe_mhc_load_acquire_sys(params.flags3 + done_index) ==
            params.expected_done_epoch;
    if (!reusable) {
      __nanosleep(128);
    }
  }
}

// Non-cooperative epoch control.  The FFI wrapper launches ready/wait, the
// high-throughput one-CTA-per-token fused data kernel, and done publication on
// one CUDA stream.  Keeping the three launches inside one custom op preserves
// a single Python boundary while avoiding the fixed resident grid and its two
// expensive grid-wide barriers.
__global__ void tp4_moe_publish_ready_wait_kernel(
    const Tp4MoeEpochControlParams __grid_constant__ params) {
  if (threadIdx.x != 0) {
    return;
  }
  uint32_t* local_flags = params.flags0;
  if (params.rank == 1) {
    local_flags = params.flags1;
  } else if (params.rank == 2) {
    local_flags = params.flags2;
  } else if (params.rank == 3) {
    local_flags = params.flags3;
  }
  __threadfence_system();
  tp4_moe_mhc_store_release_sys(
      local_flags + kTp4MoeEpochReadyBase + params.slot, params.epoch);
  bool ready = false;
  while (!ready) {
    const uint32_t ready_index = kTp4MoeEpochReadyBase + params.slot;
    ready =
        tp4_moe_mhc_load_acquire_sys(params.flags0 + ready_index) ==
            params.epoch &&
        tp4_moe_mhc_load_acquire_sys(params.flags1 + ready_index) ==
            params.epoch &&
        tp4_moe_mhc_load_acquire_sys(params.flags2 + ready_index) ==
            params.epoch &&
        tp4_moe_mhc_load_acquire_sys(params.flags3 + ready_index) ==
            params.epoch;
    if (!ready) {
      __nanosleep(128);
    }
  }
}

__global__ void tp4_moe_publish_done_kernel(
    const Tp4MoeEpochControlParams __grid_constant__ params) {
  if (threadIdx.x != 0) {
    return;
  }
  uint32_t* local_flags = params.flags0;
  if (params.rank == 1) {
    local_flags = params.flags1;
  } else if (params.rank == 2) {
    local_flags = params.flags2;
  } else if (params.rank == 3) {
    local_flags = params.flags3;
  }
  __threadfence_system();
  tp4_moe_mhc_store_release_sys(
      local_flags + kTp4MoeEpochDoneBase + params.slot, params.epoch);
}

// Double-buffered, non-cooperative epoch consumer.  A per-slot 64-bit local
// state packs {epoch, local_ready_bit | completed_CTA_count}.  The CAS winner
// alone polls the four system-scope peer flags, then releases a cheap local
// gate.  All CTAs consume peer partials after acquiring that gate, and the
// last CTA publishes done.
// Before it exits, that CTA proves the *other* slot's previous epoch is done on
// every peer, so the next layer can overwrite that slot without a standalone
// wait kernel.  No CTA residency assumption or grid-wide barrier is required.
template <bool kUseMultimem>
__global__ __launch_bounds__(kTp4MoeMhcThreads, 4)
void tp4_moe_local_slice_shared_mhc_post_epoch_counter_kernel(
    const Tp4MoeLocalSliceEpochCounterParams __grid_constant__ params) {
  const uint32_t tid = threadIdx.x;
  uint32_t* local_flags = params.flags0;
  if (params.rank == 1) {
    local_flags = params.flags1;
  } else if (params.rank == 2) {
    local_flags = params.flags2;
  } else if (params.rank == 3) {
    local_flags = params.flags3;
  }

  if (tid == 0) {
    const uint64_t expected_previous =
        params.epoch <= 2
        ? 0
        : (static_cast<uint64_t>(params.epoch - 2) << 32) |
              kTp4MoeEpochLocalReadyBit |
              static_cast<uint64_t>(gridDim.x);
    const uint64_t initialized = static_cast<uint64_t>(params.epoch) << 32;
    auto* state = params.completion_state + params.slot;
    uint64_t observed = atomicAdd(state, 0ULL);
    bool initializer = false;
    while (static_cast<uint32_t>(observed >> 32) != params.epoch) {
      if (observed != expected_previous) {
        __trap();
      }
      const uint64_t prior = atomicCAS(state, expected_previous, initialized);
      if (prior == expected_previous) {
        initializer = true;
        observed = initialized;
        break;
      }
      observed = prior;
    }
    if (initializer) {
      if constexpr (kUseMultimem) {
        // Bridge the producer's generic/unicast writes into the multicast
        // proxy before this rank publishes its ready epoch.
        tp4_moe_mhc_fence_proxy_alias();
      }
      __threadfence_system();
      tp4_moe_mhc_store_release_sys(
          local_flags + kTp4MoeEpochReadyBase + params.slot,
          params.epoch);

      bool ready = false;
      while (!ready) {
        const uint32_t ready_index = kTp4MoeEpochReadyBase + params.slot;
        ready =
            tp4_moe_mhc_load_acquire_sys(params.flags0 + ready_index) ==
                params.epoch &&
            tp4_moe_mhc_load_acquire_sys(params.flags1 + ready_index) ==
                params.epoch &&
            tp4_moe_mhc_load_acquire_sys(params.flags2 + ready_index) ==
                params.epoch &&
            tp4_moe_mhc_load_acquire_sys(params.flags3 + ready_index) ==
                params.epoch;
        if (!ready) {
          __nanosleep(128);
        }
      }
      tp4_moe_mhc_atomic_add_acq_rel_gpu(
          state, kTp4MoeEpochLocalReadyBit);
    } else {
      bool locally_ready = false;
      while (!locally_ready) {
        const uint64_t local_state =
            tp4_moe_mhc_load_acquire_gpu_u64(state);
        locally_ready =
            static_cast<uint32_t>(local_state >> 32) == params.epoch &&
            (local_state & kTp4MoeEpochLocalReadyBit) != 0;
        if (!locally_ready) {
          __nanosleep(128);
        }
      }
    }
  }
  __syncthreads();

  __shared__ float coefficients[
      kTp4MoeMhcHC + kTp4MoeMhcHC * kTp4MoeMhcHC];
  if constexpr (kUseMultimem) {
    tp4_moe_mhc_fence_proxy_alias();
    for (uint32_t local_token = blockIdx.x;
         local_token < params.post.local_num_tokens;
         local_token += gridDim.x) {
      tp4_moe_local_slice_multimem_shared_mhc_post_token(
          params.post, local_token, coefficients);
      __syncthreads();
    }
  } else {
    __shared__ __align__(16) uint4
        peer_stages[2][4][kTp4MoeMhcThreads];
    for (uint32_t local_token = blockIdx.x;
         local_token < params.post.local_num_tokens;
         local_token += gridDim.x) {
      tp4_moe_local_slice_shared_mhc_post_token(
          params.post, local_token, coefficients, peer_stages);
      __syncthreads();
    }
  }

  if (tid == 0) {
    auto* state = params.completion_state + params.slot;
    __threadfence();
    // Every CTA releases completion of its peer reads through this single
    // atomic modification order.  The last CTA's acquire observes that
    // release sequence before it publishes the system-scope done epoch, so a
    // producer cannot overwrite the slot while a non-last CTA still has
    // outstanding reads.
    const uint64_t prior =
        tp4_moe_mhc_atomic_add_acq_rel_gpu(state, 1ULL);
    const uint32_t prior_count = static_cast<uint32_t>(
        prior & kTp4MoeEpochLocalCountMask);
    if (static_cast<uint32_t>(prior >> 32) != params.epoch ||
        (prior & kTp4MoeEpochLocalReadyBit) == 0 ||
        prior_count >= gridDim.x) {
      __trap();
    }
    if (prior_count + 1 == gridDim.x) {
      __threadfence_system();
      tp4_moe_mhc_store_release_sys(
          local_flags + kTp4MoeEpochDoneBase + params.slot, params.epoch);

      const uint32_t expected_reuse_epoch =
          params.epoch == 1 ? 0 : params.epoch - 1;
      if (expected_reuse_epoch != 0) {
        const uint32_t next_slot = params.slot ^ 1;
        const uint32_t done_index = kTp4MoeEpochDoneBase + next_slot;
        bool reusable = false;
        while (!reusable) {
          reusable =
              tp4_moe_mhc_load_acquire_sys(params.flags0 + done_index) ==
                  expected_reuse_epoch &&
              tp4_moe_mhc_load_acquire_sys(params.flags1 + done_index) ==
                  expected_reuse_epoch &&
              tp4_moe_mhc_load_acquire_sys(params.flags2 + done_index) ==
                  expected_reuse_epoch &&
              tp4_moe_mhc_load_acquire_sys(params.flags3 + done_index) ==
                  expected_reuse_epoch;
          if (!reusable) {
            __nanosleep(128);
          }
        }
      }
    }
  }
}

// The producer has completed on this stream before this launch.  One leader
// publishes that rank's ready epoch, waits for all four peer producers, and
// releases the resident grid.  Every CTA then grid-strides local rows through
// the same ordered TP reduction + BF16 shared-add + mHC body as the bring-up
// ABI.  A final grid barrier lets the leader publish slot completion without
// a Host barrier or a second CUDA launch.
__global__ __launch_bounds__(kTp4MoeMhcThreads, 4)
void tp4_moe_local_slice_shared_mhc_post_epoch_kernel(
    const Tp4MoeLocalSliceEpochParams __grid_constant__ params) {
  const uint32_t tid = threadIdx.x;
  uint32_t* local_flags = params.flags0;
  if (params.rank == 1) {
    local_flags = params.flags1;
  } else if (params.rank == 2) {
    local_flags = params.flags2;
  } else if (params.rank == 3) {
    local_flags = params.flags3;
  }

  auto grid = cooperative_groups::this_grid();
  if (blockIdx.x == 0 && tid == 0) {
    __threadfence_system();
    tp4_moe_mhc_store_release_sys(
        local_flags + kTp4MoeEpochReadyBase + params.slot,
        params.epoch);
    bool ready = false;
    while (!ready) {
      const uint32_t ready_index = kTp4MoeEpochReadyBase + params.slot;
      ready =
          tp4_moe_mhc_load_acquire_sys(params.flags0 + ready_index) ==
              params.epoch &&
          tp4_moe_mhc_load_acquire_sys(params.flags1 + ready_index) ==
              params.epoch &&
          tp4_moe_mhc_load_acquire_sys(params.flags2 + ready_index) ==
              params.epoch &&
          tp4_moe_mhc_load_acquire_sys(params.flags3 + ready_index) ==
              params.epoch;
      if (!ready) {
        __nanosleep(128);
      }
    }
  }
  grid.sync();

  __shared__ float coefficients[
      kTp4MoeMhcHC + kTp4MoeMhcHC * kTp4MoeMhcHC];
  __shared__ __align__(16) uint4 peer_stages[2][4][kTp4MoeMhcThreads];
  for (uint32_t local_token = blockIdx.x;
       local_token < params.post.local_num_tokens;
       local_token += gridDim.x) {
    tp4_moe_local_slice_shared_mhc_post_token(
        params.post, local_token, coefficients, peer_stages);
    // Do not let an early warp reuse the staged peer/coefficient storage for
    // its next grid-stride row while another warp still consumes this row.
    __syncthreads();
  }

  grid.sync();
  if (blockIdx.x == 0 && tid == 0) {
    __threadfence_system();
    tp4_moe_mhc_store_release_sys(
        local_flags + kTp4MoeEpochDoneBase + params.slot,
        params.epoch);
  }
}

// Two-stage owner protocol for the strict no-Graph TP4 prefill buckets.
// Stage 1 assigns one contiguous token quarter to every rank.  A rank reads
// that quarter from all four MoE partial buffers and overwrites only the same
// quarter in its local symmetric buffer with the exact NCCL-ordered result.
// The four owner quarters are disjoint, so this in-place write cannot clobber
// data another rank still needs.  Stage 2 reads every token from its owner and
// immediately applies mHC post without materializing a replicated [M,H]
// all-reduce result.
template <bool kUsePDL>
__global__ void tp4_moe_owner_reduce_kernel(
    const __nv_bfloat16* __restrict__ input0,
    const __nv_bfloat16* __restrict__ input1,
    const __nv_bfloat16* __restrict__ input2,
    const __nv_bfloat16* __restrict__ input3,
    __nv_bfloat16* __restrict__ local_output,
    uint64_t owner_first,
    uint64_t owner_elements) {
  device::PDLWaitPrimary<kUsePDL>();
  constexpr uint32_t kPairsPerVector = kTp4MoeMhcVec / 2;
  const uint64_t vector_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t num_vectors = owner_elements / kTp4MoeMhcVec;
  const uint64_t vector_stride =
      static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t vector = vector_index; vector < num_vectors;
       vector += vector_stride) {
    const uint64_t element = owner_first + vector * kTp4MoeMhcVec;
    const uint4 input0_raw =
        *reinterpret_cast<const uint4*>(input0 + element);
    const uint4 input1_raw =
        *reinterpret_cast<const uint4*>(input1 + element);
    const uint4 input2_raw =
        *reinterpret_cast<const uint4*>(input2 + element);
    const uint4 input3_raw =
        *reinterpret_cast<const uint4*>(input3 + element);
    uint4 reduced_raw;
    const auto* input0_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input0_raw);
    const auto* input1_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input1_raw);
    const auto* input2_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input2_raw);
    const auto* input3_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input3_raw);
    auto* reduced_pairs = reinterpret_cast<__nv_bfloat162*>(&reduced_raw);
    const uint32_t group = static_cast<uint32_t>(
        (element % kNCCLPeriodElements) / kNCCLChannelGroupElements);
#pragma unroll
    for (uint32_t pair = 0; pair < kPairsPerVector; ++pair) {
      reduced_pairs[pair] = tp4_moe_mhc_add4_ordered(
          input0_pairs[pair],
          input1_pairs[pair],
          input2_pairs[pair],
          input3_pairs[pair],
          group);
    }
    *reinterpret_cast<uint4*>(local_output + element) = reduced_raw;
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
__global__ __launch_bounds__(kTp4MoeMhcThreads, 4)
void tp4_moe_owner_mhc_post_kernel(
    const Tp4MoeMhcPostParams __grid_constant__ params) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t tid = threadIdx.x;
  const uint32_t owner_tokens = params.num_tokens / 4;
  const uint32_t owner = blockIdx.x & 3;
  const uint32_t owner_block = blockIdx.x >> 2;
  const uint32_t first_token =
      owner * owner_tokens + owner_block * kTp4OwnerPostTokensPerCTA;
  const __nv_bfloat16* owner_input = params.input0;
  if (owner == 1) {
    owner_input = params.input1;
  } else if (owner == 2) {
    owner_input = params.input2;
  } else if (owner == 3) {
    owner_input = params.input3;
  }
  __shared__ float coefficients[
      kTp4MoeMhcHC + kTp4MoeMhcHC * kTp4MoeMhcHC];
  constexpr uint32_t kChunksPerToken =
      kTp4MoeMhcHidden / kTp4MoeMhcVec;
  #pragma unroll
  for (uint32_t token_offset = 0;
       token_offset < kTp4OwnerPostTokensPerCTA; ++token_offset) {
    const uint32_t token = first_token + token_offset;
    if (tid < kTp4MoeMhcHC) {
      coefficients[tid] = params.post_mix[token * kTp4MoeMhcHC + tid];
    }
    if (tid < kTp4MoeMhcHC * kTp4MoeMhcHC) {
      coefficients[kTp4MoeMhcHC + tid] =
          params.comb_mix[
              token * kTp4MoeMhcHC * kTp4MoeMhcHC + tid];
    }
    __syncthreads();

    const uint64_t hidden_base =
        static_cast<uint64_t>(token) * kTp4MoeMhcHidden;
    const uint64_t residual_base =
        static_cast<uint64_t>(token) * kTp4MoeMhcHC * kTp4MoeMhcHidden;
    auto* output_chunks = reinterpret_cast<uint4*>(
        params.output + residual_base);

    for (uint32_t chunk = tid; chunk < kChunksPerToken;
         chunk += kTp4MoeMhcThreads) {
      const uint4 hidden_raw = *reinterpret_cast<const uint4*>(
          owner_input + hidden_base +
          static_cast<uint64_t>(chunk) * kTp4MoeMhcVec);
      const auto* hidden_pairs =
          reinterpret_cast<const __nv_bfloat162*>(&hidden_raw);
      float2 hidden_values[kTp4MoeMhcVec / 2];
#pragma unroll
      for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
        hidden_values[pair] = __bfloat1622float2(hidden_pairs[pair]);
      }

      float2 residual_values[kTp4MoeMhcHC][kTp4MoeMhcVec / 2];
#pragma unroll
      for (uint32_t input_route = 0; input_route < kTp4MoeMhcHC;
           ++input_route) {
        const auto* route_chunks = reinterpret_cast<const uint4*>(
            params.residual + residual_base +
            static_cast<uint64_t>(input_route) * kTp4MoeMhcHidden);
        const uint4 residual_raw = route_chunks[chunk];
        residual_values[input_route][0] = __bfloat1622float2(
            tp4_moe_mhc_uint_to_bf16x2(residual_raw.x));
        residual_values[input_route][1] = __bfloat1622float2(
            tp4_moe_mhc_uint_to_bf16x2(residual_raw.y));
        residual_values[input_route][2] = __bfloat1622float2(
            tp4_moe_mhc_uint_to_bf16x2(residual_raw.z));
        residual_values[input_route][3] = __bfloat1622float2(
            tp4_moe_mhc_uint_to_bf16x2(residual_raw.w));
      }

#pragma unroll
      for (uint32_t output_route = 0; output_route < kTp4MoeMhcHC;
           ++output_route) {
        __nv_bfloat162 rounded[kTp4MoeMhcVec / 2];
#pragma unroll
        for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
          float2 value = make_float2(
              coefficients[output_route] * hidden_values[pair].x,
              coefficients[output_route] * hidden_values[pair].y);
#pragma unroll
          for (uint32_t input_route = 0; input_route < kTp4MoeMhcHC;
               ++input_route) {
            value = tp4_moe_mhc_fma2(
                coefficients[
                    kTp4MoeMhcHC +
                    input_route * kTp4MoeMhcHC + output_route],
                residual_values[input_route][pair],
                value);
          }
          rounded[pair] = __float22bfloat162_rn(value);
        }
        output_chunks[output_route * kChunksPerToken + chunk] = make_uint4(
            tp4_moe_mhc_bf16x2_to_uint(rounded[0]),
            tp4_moe_mhc_bf16x2_to_uint(rounded[1]),
            tp4_moe_mhc_bf16x2_to_uint(rounded[2]),
            tp4_moe_mhc_bf16x2_to_uint(rounded[3]));
      }
    }
    __syncthreads();
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// Persistent owner protocol.  The first phase reduces the rank-owned token
// quarter.  A software grid barrier publishes completion through peer-visible
// system-scope flags.  Once all four ranks have published the same epoch, the
// resident grid consumes the owner quarters and applies mHC post.  The caller
// retains the barriers before and after this kernel for producer publication
// and workspace-reuse safety; the expensive middle symmetric-memory barrier
// and one launch boundary disappear.
template <bool kUsePDL>
__global__ __launch_bounds__(kTp4MoeMhcThreads, 4)
void tp4_moe_owner_persistent_kernel(
    const Tp4MoeOwnerPersistentParams __grid_constant__ params) {
  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t tid = threadIdx.x;
  const uint32_t owner_rank = params.owner_rank;
  uint32_t* local_flags = params.flags0;
  if (owner_rank == 1) {
    local_flags = params.flags1;
  } else if (owner_rank == 2) {
    local_flags = params.flags2;
  } else if (owner_rank == 3) {
    local_flags = params.flags3;
  }

  __nv_bfloat16* local_output =
      const_cast<__nv_bfloat16*>(params.post.input0);
  if (owner_rank == 1) {
    local_output = const_cast<__nv_bfloat16*>(params.post.input1);
  } else if (owner_rank == 2) {
    local_output = const_cast<__nv_bfloat16*>(params.post.input2);
  } else if (owner_rank == 3) {
    local_output = const_cast<__nv_bfloat16*>(params.post.input3);
  }
  const uint64_t owner_elements =
      static_cast<uint64_t>(params.post.num_tokens / 4) *
      kTp4MoeMhcHidden;
  const uint64_t owner_first =
      static_cast<uint64_t>(owner_rank) * owner_elements;
  const uint64_t num_vectors = owner_elements / kTp4MoeMhcVec;
  const uint64_t vector_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
  const uint64_t vector_stride =
      static_cast<uint64_t>(gridDim.x) * blockDim.x;
  constexpr uint32_t kPairsPerVector = kTp4MoeMhcVec / 2;

  for (uint64_t vector = vector_index; vector < num_vectors;
       vector += vector_stride) {
    const uint64_t element = owner_first + vector * kTp4MoeMhcVec;
    const uint4 input0_raw =
        *reinterpret_cast<const uint4*>(params.post.input0 + element);
    const uint4 input1_raw =
        *reinterpret_cast<const uint4*>(params.post.input1 + element);
    const uint4 input2_raw =
        *reinterpret_cast<const uint4*>(params.post.input2 + element);
    const uint4 input3_raw =
        *reinterpret_cast<const uint4*>(params.post.input3 + element);
    uint4 reduced_raw;
    const auto* input0_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input0_raw);
    const auto* input1_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input1_raw);
    const auto* input2_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input2_raw);
    const auto* input3_pairs =
        reinterpret_cast<const __nv_bfloat162*>(&input3_raw);
    auto* reduced_pairs = reinterpret_cast<__nv_bfloat162*>(&reduced_raw);
    const uint32_t group = static_cast<uint32_t>(
        (element % kNCCLPeriodElements) / kNCCLChannelGroupElements);
#pragma unroll
    for (uint32_t pair = 0; pair < kPairsPerVector; ++pair) {
      reduced_pairs[pair] = tp4_moe_mhc_add4_ordered(
          input0_pairs[pair],
          input1_pairs[pair],
          input2_pairs[pair],
          input3_pairs[pair],
          group);
    }
    *reinterpret_cast<uint4*>(local_output + element) = reduced_raw;
  }

  // A cooperative grid barrier replaces one system-scope arrival atomic per
  // CTA.  Only the grid leader touches the system-scope epoch: the previous
  // implementation made all 592*256 threads issue the same acquire load.
  // The leader can publish and immediately poll the peer leaders because its
  // own reduction grid is already quiescent at this point.
  auto grid = cooperative_groups::this_grid();
  grid.sync();
  if (blockIdx.x == 0 && tid == 0) {
    const uint32_t next_epoch =
        tp4_moe_mhc_load_acquire_sys(
            local_flags + kTp4OwnerFlagProducedEpoch) +
        1;
    __threadfence_system();
    tp4_moe_mhc_store_release_sys(
        local_flags + kTp4OwnerFlagProducedEpoch, next_epoch);
    bool ready = false;
    while (!ready) {
      ready =
          tp4_moe_mhc_load_acquire_sys(
              params.flags0 + kTp4OwnerFlagProducedEpoch) == next_epoch &&
          tp4_moe_mhc_load_acquire_sys(
              params.flags1 + kTp4OwnerFlagProducedEpoch) == next_epoch &&
          tp4_moe_mhc_load_acquire_sys(
              params.flags2 + kTp4OwnerFlagProducedEpoch) == next_epoch &&
          tp4_moe_mhc_load_acquire_sys(
              params.flags3 + kTp4OwnerFlagProducedEpoch) == next_epoch;
      if (!ready) {
        __nanosleep(128);
      }
    }
  }
  grid.sync();

  // v58b split: finish the GPU-resident cross-rank dependency here.  The
  // wrapper immediately launches the occupancy-tuned owner-post kernel on the
  // same stream, so no Host synchronization or symmetric barrier is needed.
  device::PDLTriggerSecondary<kUsePDL>();
  return;

  __shared__ float coefficients[
      kTp4MoeMhcHC + kTp4MoeMhcHC * kTp4MoeMhcHC];
  constexpr uint32_t kChunksPerToken =
      kTp4MoeMhcHidden / kTp4MoeMhcVec;
  const uint32_t owner_tokens = params.post.num_tokens / 4;
  const uint32_t post_owner = blockIdx.x & 3;
  const uint32_t owner_block = blockIdx.x >> 2;
  const uint32_t owner_block_stride = gridDim.x >> 2;
  const uint32_t owner_token_end = (post_owner + 1) * owner_tokens;
  const __nv_bfloat16* owner_input = params.post.input0;
  if (post_owner == 1) {
    owner_input = params.post.input1;
  } else if (post_owner == 2) {
    owner_input = params.post.input2;
  } else if (post_owner == 3) {
    owner_input = params.post.input3;
  }
  for (uint32_t token = post_owner * owner_tokens + owner_block;
       token < owner_token_end; token += owner_block_stride) {
    if (tid < kTp4MoeMhcHC) {
      coefficients[tid] =
          params.post.post_mix[token * kTp4MoeMhcHC + tid];
    }
    if (tid < kTp4MoeMhcHC * kTp4MoeMhcHC) {
      coefficients[kTp4MoeMhcHC + tid] =
          params.post.comb_mix[
              token * kTp4MoeMhcHC * kTp4MoeMhcHC + tid];
    }
    __syncthreads();

    const uint64_t hidden_base =
        static_cast<uint64_t>(token) * kTp4MoeMhcHidden;
    const uint64_t residual_base =
        static_cast<uint64_t>(token) * kTp4MoeMhcHC * kTp4MoeMhcHidden;
    auto* output_chunks = reinterpret_cast<uint4*>(
        params.post.output + residual_base);

    for (uint32_t chunk = tid; chunk < kChunksPerToken;
         chunk += kTp4MoeMhcThreads) {
      const uint4 hidden_raw = *reinterpret_cast<const uint4*>(
          owner_input + hidden_base +
          static_cast<uint64_t>(chunk) * kTp4MoeMhcVec);
      const auto* hidden_pairs =
          reinterpret_cast<const __nv_bfloat162*>(&hidden_raw);
      float2 hidden_values[kTp4MoeMhcVec / 2];
#pragma unroll
      for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
        hidden_values[pair] = __bfloat1622float2(hidden_pairs[pair]);
      }

      float2 residual_values[kTp4MoeMhcHC][kTp4MoeMhcVec / 2];
#pragma unroll
      for (uint32_t input_route = 0; input_route < kTp4MoeMhcHC;
           ++input_route) {
        const auto* route_chunks = reinterpret_cast<const uint4*>(
            params.post.residual + residual_base +
            static_cast<uint64_t>(input_route) * kTp4MoeMhcHidden);
        const uint4 residual_raw = route_chunks[chunk];
        residual_values[input_route][0] = __bfloat1622float2(
            tp4_moe_mhc_uint_to_bf16x2(residual_raw.x));
        residual_values[input_route][1] = __bfloat1622float2(
            tp4_moe_mhc_uint_to_bf16x2(residual_raw.y));
        residual_values[input_route][2] = __bfloat1622float2(
            tp4_moe_mhc_uint_to_bf16x2(residual_raw.z));
        residual_values[input_route][3] = __bfloat1622float2(
            tp4_moe_mhc_uint_to_bf16x2(residual_raw.w));
      }

#pragma unroll
      for (uint32_t output_route = 0; output_route < kTp4MoeMhcHC;
           ++output_route) {
        __nv_bfloat162 rounded[kTp4MoeMhcVec / 2];
#pragma unroll
        for (uint32_t pair = 0; pair < kTp4MoeMhcVec / 2; ++pair) {
          float2 value = make_float2(
              coefficients[output_route] * hidden_values[pair].x,
              coefficients[output_route] * hidden_values[pair].y);
#pragma unroll
          for (uint32_t input_route = 0; input_route < kTp4MoeMhcHC;
               ++input_route) {
            value = tp4_moe_mhc_fma2(
                coefficients[
                    kTp4MoeMhcHC +
                    input_route * kTp4MoeMhcHC + output_route],
                residual_values[input_route][pair],
                value);
          }
          rounded[pair] = __float22bfloat162_rn(value);
        }
        output_chunks[output_route * kChunksPerToken + chunk] = make_uint4(
            tp4_moe_mhc_bf16x2_to_uint(rounded[0]),
            tp4_moe_mhc_bf16x2_to_uint(rounded[1]),
            tp4_moe_mhc_bf16x2_to_uint(rounded[2]),
            tp4_moe_mhc_bf16x2_to_uint(rounded[3]));
      }
    }
    __syncthreads();
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct Tp4MoeMhcPostKernel {
  static void run(
      const tvm::ffi::TensorView input0,
      const tvm::ffi::TensorView input1,
      const tvm::ffi::TensorView input2,
      const tvm::ffi::TensorView input3,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input0);
    TensorMatcher({M, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input1);
    TensorMatcher({M, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input2);
    TensorMatcher({M, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input3);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({M, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);

    RuntimeCheck(
        M.unwrap() == 65536 || M.unwrap() == 131072,
        "TP4 fused MoE/mHC post only supports M=65536 or M=131072");
    const auto params = Tp4MoeMhcPostParams{
        .input0 = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr()),
        .input1 = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
        .input2 = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
        .input3 = reinterpret_cast<const __nv_bfloat16*>(input3.data_ptr()),
        .multicast_input = nullptr,
        .residual =
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .num_tokens = static_cast<uint32_t>(M.unwrap()),
    };
    LaunchKernel(M.unwrap(), kTp4MoeMhcThreads, device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_moe_mhc_post_kernel<kUsePDL, false>, params);
  }

  static void run_local_slice_shared(
      const tvm::ffi::TensorView input0,
      const tvm::ffi::TensorView input1,
      const tvm::ffi::TensorView input2,
      const tvm::ffi::TensorView input3,
      int64_t local_row_offset,
      const tvm::ffi::TensorView shared_hidden,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto G = SymbolicSize{"global_num_tokens"};
    auto L = SymbolicSize{"local_num_tokens"};
    device.set_options<kDLCUDA>();

    for (const auto tensor : {input0, input1, input2, input3}) {
      TensorMatcher({G, kTp4MoeMhcHidden})
          .with_strides({kTp4MoeMhcHidden, 1})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(tensor);
    }
    TensorMatcher({L, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(shared_hidden);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({L, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);

    RuntimeCheck(
        G.unwrap() >= 4096 && G.unwrap() <= 131072 &&
            G.unwrap() % 4096 == 0,
        "TP4 local-slice MoE/mHC requires global_M in [4096,131072] "
        "and divisible by 4096");
    RuntimeCheck(
        L.unwrap() >= 4096 && L.unwrap() <= 32768 &&
            L.unwrap() % 4096 == 0,
        "TP4 local-slice MoE/mHC requires local_M in [4096,32768] "
        "and divisible by 4096");
    RuntimeCheck(
        local_row_offset >= 0 && local_row_offset <= G.unwrap() &&
            local_row_offset % 4096 == 0,
        "TP4 local-slice MoE/mHC requires a 4096-row-aligned in-range "
        "local offset");
    RuntimeCheck(
        L.unwrap() <= G.unwrap() - local_row_offset,
        "TP4 local-slice MoE/mHC local interval exceeds global partials");

    const auto params = Tp4MoeLocalSliceMhcPostParams{
        .input0 = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr()),
        .input1 = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
        .input2 = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
        .input3 = reinterpret_cast<const __nv_bfloat16*>(input3.data_ptr()),
        .multicast_input = nullptr,
        .shared_hidden = reinterpret_cast<const __nv_bfloat16*>(
            shared_hidden.data_ptr()),
        .residual =
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .local_num_tokens = static_cast<uint32_t>(L.unwrap()),
        .local_row_offset = static_cast<uint32_t>(local_row_offset),
    };
    LaunchKernel(L.unwrap(), kTp4MoeMhcThreads, device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_moe_local_slice_shared_mhc_post_kernel<kUsePDL>, params);
  }

  static void run_wait_slot_reusable(
      const tvm::ffi::TensorView flags0,
      const tvm::ffi::TensorView flags1,
      const tvm::ffi::TensorView flags2,
      const tvm::ffi::TensorView flags3,
      int64_t slot,
      int64_t expected_done_epoch) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    for (const auto flags : {flags0, flags1, flags2, flags3}) {
      TensorMatcher({kTp4MoeEpochFlagCount})
          .with_strides({1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(flags);
    }
    RuntimeCheck(
        slot == 0 || slot == 1,
        "TP4 MoE epoch slot must be 0 or 1");
    RuntimeCheck(
        expected_done_epoch >= 0 && expected_done_epoch <= 0xffffffffLL,
        "TP4 MoE expected done epoch must fit uint32");
    RuntimeCheck(
        !kUsePDL,
        "TP4 MoE epoch synchronization does not support PDL");

    const auto params = Tp4MoeWaitSlotParams{
        .flags0 = reinterpret_cast<const uint32_t*>(flags0.data_ptr()),
        .flags1 = reinterpret_cast<const uint32_t*>(flags1.data_ptr()),
        .flags2 = reinterpret_cast<const uint32_t*>(flags2.data_ptr()),
        .flags3 = reinterpret_cast<const uint32_t*>(flags3.data_ptr()),
        .slot = static_cast<uint32_t>(slot),
        .expected_done_epoch =
            static_cast<uint32_t>(expected_done_epoch),
    };
    LaunchKernel(1, 32, device.unwrap())(
        tp4_moe_wait_slot_reusable_kernel, params);
  }

  static void run_local_slice_shared_epoch(
      const tvm::ffi::TensorView input0,
      const tvm::ffi::TensorView input1,
      const tvm::ffi::TensorView input2,
      const tvm::ffi::TensorView input3,
      int64_t local_row_offset,
      const tvm::ffi::TensorView shared_hidden,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView flags0,
      const tvm::ffi::TensorView flags1,
      const tvm::ffi::TensorView flags2,
      const tvm::ffi::TensorView flags3,
      int64_t rank,
      int64_t slot,
      int64_t epoch) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto G = SymbolicSize{"global_num_tokens"};
    auto L = SymbolicSize{"local_num_tokens"};
    device.set_options<kDLCUDA>();

    for (const auto tensor : {input0, input1, input2, input3}) {
      TensorMatcher({G, kTp4MoeMhcHidden})
          .with_strides({kTp4MoeMhcHidden, 1})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(tensor);
    }
    TensorMatcher({L, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(shared_hidden);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({L, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    for (const auto flags : {flags0, flags1, flags2, flags3}) {
      TensorMatcher({kTp4MoeEpochFlagCount})
          .with_strides({1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(flags);
    }

    RuntimeCheck(
        G.unwrap() >= 4096 && G.unwrap() <= 131072 &&
            G.unwrap() % 4096 == 0,
        "TP4 epoch local-slice MoE/mHC requires global_M in "
        "[4096,131072] and divisible by 4096");
    RuntimeCheck(
        L.unwrap() >= 4096 && L.unwrap() <= 32768 &&
            L.unwrap() % 4096 == 0,
        "TP4 epoch local-slice MoE/mHC requires local_M in "
        "[4096,32768] and divisible by 4096");
    RuntimeCheck(
        local_row_offset >= 0 && local_row_offset <= G.unwrap() &&
            local_row_offset % 4096 == 0 &&
            L.unwrap() <= G.unwrap() - local_row_offset,
        "TP4 epoch local-slice MoE/mHC requires a 4096-row-aligned "
        "in-bounds local interval");
    RuntimeCheck(
        rank >= 0 && rank < 4,
        "TP4 MoE epoch rank must be in [0,4)");
    RuntimeCheck(
        slot == 0 || slot == 1,
        "TP4 MoE epoch slot must be 0 or 1");
    RuntimeCheck(
        epoch > 0 && epoch <= 0xffffffffLL,
        "TP4 MoE ready/done epoch must be in [1,UINT32_MAX]");
    RuntimeCheck(
        !kUsePDL,
        "TP4 cooperative local-slice epoch synchronization does not "
        "support PDL");

    const auto post_params = Tp4MoeLocalSliceMhcPostParams{
        .input0 = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr()),
        .input1 = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
        .input2 = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
        .input3 = reinterpret_cast<const __nv_bfloat16*>(input3.data_ptr()),
        .multicast_input = nullptr,
        .shared_hidden = reinterpret_cast<const __nv_bfloat16*>(
            shared_hidden.data_ptr()),
        .residual =
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .local_num_tokens = static_cast<uint32_t>(L.unwrap()),
        .local_row_offset = static_cast<uint32_t>(local_row_offset),
    };
    const auto params = Tp4MoeLocalSliceEpochParams{
        .post = post_params,
        .flags0 = reinterpret_cast<uint32_t*>(flags0.data_ptr()),
        .flags1 = reinterpret_cast<uint32_t*>(flags1.data_ptr()),
        .flags2 = reinterpret_cast<uint32_t*>(flags2.data_ptr()),
        .flags3 = reinterpret_cast<uint32_t*>(flags3.data_ptr()),
        .rank = static_cast<uint32_t>(rank),
        .slot = static_cast<uint32_t>(slot),
        .epoch = static_cast<uint32_t>(epoch),
    };

    const DLDevice dl_device = device.unwrap();
    int sm_count = 0;
    int cooperative_launch = 0;
    int active_blocks_per_sm = 0;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(
        &sm_count,
        cudaDevAttrMultiProcessorCount,
        dl_device.device_id));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(
        &cooperative_launch,
        cudaDevAttrCooperativeLaunch,
        dl_device.device_id));
    RuntimeDeviceCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        tp4_moe_local_slice_shared_mhc_post_epoch_kernel,
        kTp4MoeMhcThreads,
        0));
    RuntimeCheck(
        cooperative_launch != 0,
        "TP4 local-slice epoch kernel requires cooperative launch");
    RuntimeCheck(
        sm_count > 0 && active_blocks_per_sm > 0 &&
            static_cast<uint64_t>(sm_count) * active_blocks_per_sm >=
                kTp4OwnerPersistentBlocks,
        "TP4 local-slice epoch fixed 592-CTA grid is not fully resident "
        "under this device's SM/occupancy limit");

    auto cooperative_params = params;
    void* cooperative_args[] = {&cooperative_params};
    RuntimeDeviceCheck(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(
            tp4_moe_local_slice_shared_mhc_post_epoch_kernel),
        dim3(kTp4OwnerPersistentBlocks),
        dim3(kTp4MoeMhcThreads),
        cooperative_args,
        0,
        LaunchKernel::resolve_device(dl_device)));
  }

  static void run_local_slice_shared_epoch_split(
      const tvm::ffi::TensorView input0,
      const tvm::ffi::TensorView input1,
      const tvm::ffi::TensorView input2,
      const tvm::ffi::TensorView input3,
      int64_t local_row_offset,
      const tvm::ffi::TensorView shared_hidden,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView flags0,
      const tvm::ffi::TensorView flags1,
      const tvm::ffi::TensorView flags2,
      const tvm::ffi::TensorView flags3,
      int64_t rank,
      int64_t slot,
      int64_t epoch) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto G = SymbolicSize{"global_num_tokens"};
    auto L = SymbolicSize{"local_num_tokens"};
    device.set_options<kDLCUDA>();

    for (const auto tensor : {input0, input1, input2, input3}) {
      TensorMatcher({G, kTp4MoeMhcHidden})
          .with_strides({kTp4MoeMhcHidden, 1})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(tensor);
    }
    TensorMatcher({L, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(shared_hidden);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({L, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    for (const auto flags : {flags0, flags1, flags2, flags3}) {
      TensorMatcher({kTp4MoeEpochFlagCount})
          .with_strides({1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(flags);
    }

    RuntimeCheck(
        G.unwrap() >= 4096 && G.unwrap() <= 131072 &&
            G.unwrap() % 4096 == 0,
        "TP4 split-epoch local-slice MoE/mHC requires global_M in "
        "[4096,131072] and divisible by 4096");
    RuntimeCheck(
        L.unwrap() >= 4096 && L.unwrap() <= 32768 &&
            L.unwrap() % 4096 == 0,
        "TP4 split-epoch local-slice MoE/mHC requires local_M in "
        "[4096,32768] and divisible by 4096");
    RuntimeCheck(
        local_row_offset >= 0 && local_row_offset <= G.unwrap() &&
            local_row_offset % 4096 == 0 &&
            L.unwrap() <= G.unwrap() - local_row_offset,
        "TP4 split-epoch local-slice MoE/mHC requires a 4096-row-aligned "
        "in-bounds local interval");
    RuntimeCheck(rank >= 0 && rank < 4, "TP4 split epoch rank must be in [0,4)");
    RuntimeCheck(
        slot == 0 || slot == 1, "TP4 split epoch slot must be 0 or 1");
    RuntimeCheck(
        epoch > 0 && epoch <= 0xffffffffLL,
        "TP4 split ready/done epoch must be in [1,UINT32_MAX]");
    RuntimeCheck(
        !kUsePDL, "TP4 split epoch synchronization does not support PDL");

    const auto post_params = Tp4MoeLocalSliceMhcPostParams{
        .input0 = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr()),
        .input1 = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
        .input2 = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
        .input3 = reinterpret_cast<const __nv_bfloat16*>(input3.data_ptr()),
        .multicast_input = nullptr,
        .shared_hidden = reinterpret_cast<const __nv_bfloat16*>(
            shared_hidden.data_ptr()),
        .residual =
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .local_num_tokens = static_cast<uint32_t>(L.unwrap()),
        .local_row_offset = static_cast<uint32_t>(local_row_offset),
    };
    const auto control_params = Tp4MoeEpochControlParams{
        .flags0 = reinterpret_cast<uint32_t*>(flags0.data_ptr()),
        .flags1 = reinterpret_cast<uint32_t*>(flags1.data_ptr()),
        .flags2 = reinterpret_cast<uint32_t*>(flags2.data_ptr()),
        .flags3 = reinterpret_cast<uint32_t*>(flags3.data_ptr()),
        .rank = static_cast<uint32_t>(rank),
        .slot = static_cast<uint32_t>(slot),
        .epoch = static_cast<uint32_t>(epoch),
    };
    const DLDevice dl_device = device.unwrap();
    LaunchKernel(1, 32, dl_device)(
        tp4_moe_publish_ready_wait_kernel, control_params);
    LaunchKernel(L.unwrap(), kTp4MoeMhcThreads, dl_device)(
        tp4_moe_local_slice_shared_mhc_post_kernel<false>, post_params);
    LaunchKernel(1, 32, dl_device)(
        tp4_moe_publish_done_kernel, control_params);
  }

  static void run_local_slice_shared_epoch_counter(
      const tvm::ffi::TensorView input0,
      const tvm::ffi::TensorView input1,
      const tvm::ffi::TensorView input2,
      const tvm::ffi::TensorView input3,
      int64_t local_row_offset,
      const tvm::ffi::TensorView shared_hidden,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView flags0,
      const tvm::ffi::TensorView flags1,
      const tvm::ffi::TensorView flags2,
      const tvm::ffi::TensorView flags3,
      const tvm::ffi::TensorView completion_state,
      int64_t rank,
      int64_t slot,
      int64_t epoch) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto G = SymbolicSize{"global_num_tokens"};
    auto L = SymbolicSize{"local_num_tokens"};
    device.set_options<kDLCUDA>();

    for (const auto tensor : {input0, input1, input2, input3}) {
      TensorMatcher({G, kTp4MoeMhcHidden})
          .with_strides({kTp4MoeMhcHidden, 1})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(tensor);
    }
    TensorMatcher({L, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(shared_hidden);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({L, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    for (const auto flags : {flags0, flags1, flags2, flags3}) {
      TensorMatcher({kTp4MoeEpochFlagCount})
          .with_strides({1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(flags);
    }
    TensorMatcher({2})
        .with_strides({1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(completion_state);

    RuntimeCheck(
        G.unwrap() >= 4096 && G.unwrap() <= 131072 &&
            G.unwrap() % 4096 == 0,
        "TP4 counter-epoch local-slice MoE/mHC requires global_M in "
        "[4096,131072] and divisible by 4096");
    RuntimeCheck(
        L.unwrap() >= 4096 && L.unwrap() <= 32768 &&
            L.unwrap() % 4096 == 0,
        "TP4 counter-epoch local-slice MoE/mHC requires local_M in "
        "[4096,32768] and divisible by 4096");
    RuntimeCheck(
        local_row_offset >= 0 && local_row_offset <= G.unwrap() &&
            local_row_offset % 4096 == 0 &&
            L.unwrap() <= G.unwrap() - local_row_offset,
        "TP4 counter-epoch local-slice MoE/mHC requires a 4096-row-aligned "
        "in-bounds local interval");
    RuntimeCheck(
        rank >= 0 && rank < 4, "TP4 counter epoch rank must be in [0,4)");
    RuntimeCheck(
        slot == 0 || slot == 1, "TP4 counter epoch slot must be 0 or 1");
    RuntimeCheck(
        epoch > 0 && epoch <= 0xffffffffLL,
        "TP4 counter ready/done epoch must be in [1,UINT32_MAX]");
    RuntimeCheck(
        !kUsePDL, "TP4 counter epoch synchronization does not support PDL");

    const auto post_params = Tp4MoeLocalSliceMhcPostParams{
        .input0 = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr()),
        .input1 = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
        .input2 = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
        .input3 = reinterpret_cast<const __nv_bfloat16*>(input3.data_ptr()),
        .multicast_input = nullptr,
        .shared_hidden = reinterpret_cast<const __nv_bfloat16*>(
            shared_hidden.data_ptr()),
        .residual =
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .local_num_tokens = static_cast<uint32_t>(L.unwrap()),
        .local_row_offset = static_cast<uint32_t>(local_row_offset),
    };
    const auto params = Tp4MoeLocalSliceEpochCounterParams{
        .post = post_params,
        .flags0 = reinterpret_cast<uint32_t*>(flags0.data_ptr()),
        .flags1 = reinterpret_cast<uint32_t*>(flags1.data_ptr()),
        .flags2 = reinterpret_cast<uint32_t*>(flags2.data_ptr()),
        .flags3 = reinterpret_cast<uint32_t*>(flags3.data_ptr()),
        .completion_state = reinterpret_cast<unsigned long long*>(
            completion_state.data_ptr()),
        .rank = static_cast<uint32_t>(rank),
        .slot = static_cast<uint32_t>(slot),
        .epoch = static_cast<uint32_t>(epoch),
    };
    LaunchKernel(
        kTp4OwnerPersistentBlocks, kTp4MoeMhcThreads, device.unwrap())(
        tp4_moe_local_slice_shared_mhc_post_epoch_counter_kernel<false>,
        params);
  }

  static void run_local_slice_shared_epoch_counter_multimem(
      int64_t multicast_local_ptr,
      const tvm::ffi::TensorView local_partial_anchor,
      int64_t local_row_offset,
      const tvm::ffi::TensorView shared_hidden,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView flags0,
      const tvm::ffi::TensorView flags1,
      const tvm::ffi::TensorView flags2,
      const tvm::ffi::TensorView flags3,
      const tvm::ffi::TensorView completion_state,
      int64_t rank,
      int64_t slot,
      int64_t epoch) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto G = SymbolicSize{"global_num_tokens"};
    auto L = SymbolicSize{"local_num_tokens"};
    device.set_options<kDLCUDA>();

    // The anchor is the selected local unicast slot written by FlashInfer.
    // It is intentionally not read by the kernel: it gives the custom-op ABI
    // a real tensor dependency, holds the symmetric allocation alive, and
    // supplies the global-M/device contract for the raw multicast VA.
    TensorMatcher({G, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(local_partial_anchor);
    TensorMatcher({L, kTp4MoeMhcHidden})
        .with_strides({kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(shared_hidden);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({L, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({L, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    for (const auto flags : {flags0, flags1, flags2, flags3}) {
      TensorMatcher({kTp4MoeEpochFlagCount})
          .with_strides({1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(flags);
    }
    TensorMatcher({2})
        .with_strides({1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(completion_state);

    RuntimeCheck(
        multicast_local_ptr > 0 && (multicast_local_ptr & 0xF) == 0,
        "TP4 NVLS local-slice MoE/mHC requires a positive 16-byte aligned "
        "multicast VA");
    RuntimeCheck(
        G.unwrap() >= 4096 && G.unwrap() <= 131072 &&
            G.unwrap() % 4096 == 0,
        "TP4 NVLS counter-epoch requires global_M in [4096,131072] and "
        "divisible by 4096");
    RuntimeCheck(
        L.unwrap() >= 4096 && L.unwrap() <= 32768 &&
            L.unwrap() % 4096 == 0,
        "TP4 NVLS counter-epoch requires local_M in [4096,32768] and "
        "divisible by 4096");
    RuntimeCheck(
        local_row_offset >= 0 && local_row_offset <= G.unwrap() &&
            local_row_offset % 4096 == 0 &&
            L.unwrap() <= G.unwrap() - local_row_offset,
        "TP4 NVLS counter-epoch requires a 4096-row-aligned in-bounds "
        "local interval");
    RuntimeCheck(
        rank >= 0 && rank < 4,
        "TP4 NVLS counter epoch rank must be in [0,4)");
    RuntimeCheck(
        slot == 0 || slot == 1,
        "TP4 NVLS counter epoch slot must be 0 or 1");
    RuntimeCheck(
        epoch > 0 && epoch <= 0xffffffffLL,
        "TP4 NVLS counter ready/done epoch must be in [1,UINT32_MAX]");
    RuntimeCheck(
        !kUsePDL,
        "TP4 NVLS counter epoch synchronization does not support PDL");

    const auto post_params = Tp4MoeLocalSliceMhcPostParams{
        .input0 = nullptr,
        .input1 = nullptr,
        .input2 = nullptr,
        .input3 = nullptr,
        .multicast_input = reinterpret_cast<const __nv_bfloat16*>(
            static_cast<uintptr_t>(multicast_local_ptr)),
        .shared_hidden = reinterpret_cast<const __nv_bfloat16*>(
            shared_hidden.data_ptr()),
        .residual =
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .local_num_tokens = static_cast<uint32_t>(L.unwrap()),
        .local_row_offset = static_cast<uint32_t>(local_row_offset),
    };
    const auto params = Tp4MoeLocalSliceEpochCounterParams{
        .post = post_params,
        .flags0 = reinterpret_cast<uint32_t*>(flags0.data_ptr()),
        .flags1 = reinterpret_cast<uint32_t*>(flags1.data_ptr()),
        .flags2 = reinterpret_cast<uint32_t*>(flags2.data_ptr()),
        .flags3 = reinterpret_cast<uint32_t*>(flags3.data_ptr()),
        .completion_state = reinterpret_cast<unsigned long long*>(
            completion_state.data_ptr()),
        .rank = static_cast<uint32_t>(rank),
        .slot = static_cast<uint32_t>(slot),
        .epoch = static_cast<uint32_t>(epoch),
    };
    LaunchKernel(
        kTp4OwnerPersistentBlocks, kTp4MoeMhcThreads, device.unwrap())(
        tp4_moe_local_slice_shared_mhc_post_epoch_counter_kernel<true>,
        params);
  }

  static void run_multimem(
      int64_t multicast_ptr,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({M, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    RuntimeCheck(
        M.unwrap() == 65536 || M.unwrap() == 131072,
        "TP4 multimem MoE/mHC post only supports M=65536 or M=131072");
    RuntimeCheck(
        multicast_ptr != 0 && (multicast_ptr & 0xF) == 0,
        "TP4 multimem MoE/mHC post requires a 16-byte aligned multicast VA");
    const auto params = Tp4MoeMhcPostParams{
        .input0 = nullptr,
        .input1 = nullptr,
        .input2 = nullptr,
        .input3 = nullptr,
        .multicast_input = reinterpret_cast<const __nv_bfloat16*>(
            static_cast<uintptr_t>(multicast_ptr)),
        .residual =
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .num_tokens = static_cast<uint32_t>(M.unwrap()),
    };
    LaunchKernel(M.unwrap(), kTp4MoeMhcThreads, device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_moe_mhc_post_kernel<kUsePDL, true>, params);
  }

  static void run_owner_reduce(
      const tvm::ffi::TensorView input0,
      const tvm::ffi::TensorView input1,
      const tvm::ffi::TensorView input2,
      const tvm::ffi::TensorView input3,
      const tvm::ffi::TensorView local_output,
      int64_t owner_rank) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();
    for (const auto tensor : {input0, input1, input2, input3, local_output}) {
      TensorMatcher({M, kTp4MoeMhcHidden})
          .with_strides({kTp4MoeMhcHidden, 1})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(tensor);
    }
    RuntimeCheck(
        M.unwrap() == 65536 || M.unwrap() == 131072,
        "TP4 owner reduction only supports M=65536 or M=131072");
    RuntimeCheck(
        owner_rank >= 0 && owner_rank < 4,
        "TP4 owner reduction requires owner_rank in [0,4)");
    const uint64_t owner_elements =
        static_cast<uint64_t>(M.unwrap() / 4) * kTp4MoeMhcHidden;
    const uint64_t owner_first = owner_rank * owner_elements;
    // A grid-stride loop keeps enough CTAs resident to saturate the NVLink
    // reads without paying scheduler overhead for one CTA per 4 KiB.  These
    // are the same empirically stable CTA counts used by the strict Huge
    // attention owner-gather path for 16K/32K owner-token shards.
    const uint32_t blocks = M.unwrap() == 65536 ? 8192 : 16384;
    LaunchKernel(blocks, kTp4MoeMhcThreads, device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_moe_owner_reduce_kernel<kUsePDL>,
            reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input3.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(local_output.data_ptr()),
            owner_first,
            owner_elements);
  }

  static void run_owner_post(
      const tvm::ffi::TensorView input0,
      const tvm::ffi::TensorView input1,
      const tvm::ffi::TensorView input2,
      const tvm::ffi::TensorView input3,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();
    for (const auto tensor : {input0, input1, input2, input3}) {
      TensorMatcher({M, kTp4MoeMhcHidden})
          .with_strides({kTp4MoeMhcHidden, 1})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(tensor);
    }
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({M, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    RuntimeCheck(
        M.unwrap() == 65536 || M.unwrap() == 131072,
        "TP4 owner mHC post only supports M=65536 or M=131072");
    const auto params = Tp4MoeMhcPostParams{
        .input0 = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr()),
        .input1 = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
        .input2 = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
        .input3 = reinterpret_cast<const __nv_bfloat16*>(input3.data_ptr()),
        .multicast_input = nullptr,
        .residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .num_tokens = static_cast<uint32_t>(M.unwrap()),
    };
    LaunchKernel(
        M.unwrap() / kTp4OwnerPostTokensPerCTA,
        kTp4MoeMhcThreads,
        device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_moe_owner_mhc_post_kernel<kUsePDL>, params);
  }

  static void run_owner_persistent(
      const tvm::ffi::TensorView input0,
      const tvm::ffi::TensorView input1,
      const tvm::ffi::TensorView input2,
      const tvm::ffi::TensorView input3,
      const tvm::ffi::TensorView flags0,
      const tvm::ffi::TensorView flags1,
      const tvm::ffi::TensorView flags2,
      const tvm::ffi::TensorView flags3,
      int64_t owner_rank,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post_mix,
      const tvm::ffi::TensorView comb_mix,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    device.set_options<kDLCUDA>();
    for (const auto tensor : {input0, input1, input2, input3}) {
      TensorMatcher({M, kTp4MoeMhcHidden})
          .with_strides({kTp4MoeMhcHidden, 1})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(tensor);
    }
    for (const auto tensor : {flags0, flags1, flags2, flags3}) {
      TensorMatcher({kTp4OwnerFlagCount})
          .with_strides({1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(tensor);
    }
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({M, kTp4MoeMhcHC})
        .with_strides({kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(post_mix);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHC})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHC, kTp4MoeMhcHC, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(comb_mix);
    TensorMatcher({M, kTp4MoeMhcHC, kTp4MoeMhcHidden})
        .with_strides(
            {kTp4MoeMhcHC * kTp4MoeMhcHidden, kTp4MoeMhcHidden, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    RuntimeCheck(
        M.unwrap() == 65536 || M.unwrap() == 131072,
        "TP4 persistent owner protocol only supports M=65536 or M=131072");
    RuntimeCheck(
        owner_rank >= 0 && owner_rank < 4,
        "TP4 persistent owner protocol requires owner_rank in [0,4)");

    const auto post_params = Tp4MoeMhcPostParams{
        .input0 = reinterpret_cast<const __nv_bfloat16*>(input0.data_ptr()),
        .input1 = reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
        .input2 = reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
        .input3 = reinterpret_cast<const __nv_bfloat16*>(input3.data_ptr()),
        .multicast_input = nullptr,
        .residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post_mix = static_cast<const float*>(post_mix.data_ptr()),
        .comb_mix = static_cast<const float*>(comb_mix.data_ptr()),
        .output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        .num_tokens = static_cast<uint32_t>(M.unwrap()),
    };
    const auto params = Tp4MoeOwnerPersistentParams{
        .post = post_params,
        .flags0 = reinterpret_cast<uint32_t*>(flags0.data_ptr()),
        .flags1 = reinterpret_cast<uint32_t*>(flags1.data_ptr()),
        .flags2 = reinterpret_cast<uint32_t*>(flags2.data_ptr()),
        .flags3 = reinterpret_cast<uint32_t*>(flags3.data_ptr()),
        .owner_rank = static_cast<uint32_t>(owner_rank),
    };
    RuntimeCheck(
        !kUsePDL,
        "TP4 cooperative owner synchronization does not support PDL");
    auto cooperative_params = params;
    void* cooperative_args[] = {&cooperative_params};
    RuntimeDeviceCheck(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(
            tp4_moe_owner_persistent_kernel<kUsePDL>),
        dim3(kTp4OwnerPersistentBlocks),
        dim3(kTp4MoeMhcThreads),
        cooperative_args,
        0,
        LaunchKernel::resolve_device(device.unwrap())));
    LaunchKernel(
        M.unwrap() / kTp4OwnerPostTokensPerCTA,
        kTp4MoeMhcThreads,
        device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_moe_owner_mhc_post_kernel<kUsePDL>, post_params);
  }
};

}  // namespace
