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

  // Every block samples the previous epoch before any block can publish the
  // next one: publication requires all resident blocks to reach the arrival
  // counter below.
  const uint32_t previous_epoch = tp4_moe_mhc_load_acquire_sys(
      local_flags + kTp4OwnerFlagProducedEpoch);
  const uint32_t next_epoch = previous_epoch + 1;

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
  // CTA.  The single system release below transitively publishes the reduced
  // quarter after every producer has crossed the grid barrier.
  auto grid = cooperative_groups::this_grid();
  grid.sync();
  if (blockIdx.x == 0 && tid == 0) {
    __threadfence_system();
    tp4_moe_mhc_store_release_sys(
        local_flags + kTp4OwnerFlagProducedEpoch, next_epoch);
  }
  grid.sync();

  // One CTA polls the four peer epochs.  It releases the local grid with a
  // second system-scope flag, keeping the cross-rank control loop on GPU.
  if (blockIdx.x == 0 && tid == 0) {
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
