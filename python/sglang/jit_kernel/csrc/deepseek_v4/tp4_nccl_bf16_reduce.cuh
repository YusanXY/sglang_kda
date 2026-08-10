// Exact local emulation of the BF16 addition order used by the TP4 NCCL
// ring AllReduce in the strict DSV4 eager-prefill buckets.
//
// The native WO_B path reduces a contiguous [T,4096] BF16 tensor.  Profiling
// and exhaustive value probes on the B300 NCCL build show four 2 MiB channel
// groups per 8 MiB period.  Their sequential rank orders are respectively
//   (1,2,3,0), (2,3,0,1), (0,3,1,2), (0,1,2,3).
// Req16 and each Req128 wave use owner shards whose byte size is an integer
// multiple of that period, so every shard starts at phase zero.  Explicit
// __hadd2 operations preserve the BF16 rounding boundary after every add.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace {

constexpr int64_t kNCCLChannelGroupElements = 1LL << 20;  // 2 MiB BF16
constexpr int64_t kNCCLPeriodElements = 4 * kNCCLChannelGroupElements;
constexpr int kElementsPerThread = 8;
constexpr int kTokensPerReadyGroup = 8;
constexpr int64_t kMaxReadyGroups = 131072 / 2;

SGL_DEVICE uint32_t load_acquire_sys(const uint32_t* pointer) {
  uint32_t value;
  asm volatile(
      "ld.acquire.sys.global.u32 %0, [%1];"
      : "=r"(value)
      : "l"(pointer)
      : "memory");
  return value;
}

SGL_DEVICE void store_release_sys(uint32_t* pointer, uint32_t value) {
  asm volatile(
      "st.release.sys.global.u32 [%0], %1;" ::
          "l"(pointer),
          "r"(value)
      : "memory");
}

template <typename T>
SGL_DEVICE __nv_bfloat162 load_pair(const T* pointer) {
  static_assert(sizeof(T) == sizeof(__nv_bfloat16));
  return *reinterpret_cast<const __nv_bfloat162*>(pointer);
}

SGL_DEVICE __nv_bfloat162 add4_ordered(
    __nv_bfloat162 a,
    __nv_bfloat162 b,
    __nv_bfloat162 c,
    __nv_bfloat162 d,
    int group) {
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

template <typename T, bool kUsePDL>
__global__ void tp4_nccl_ring_bf16_reduce_kernel(
    const T* __restrict__ input0,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const T* __restrict__ input3,
    T* __restrict__ output,
    int64_t num_elements) {
  device::PDLWaitPrimary<kUsePDL>();
  const int64_t block_first =
      static_cast<int64_t>(blockIdx.x) * blockDim.x * kElementsPerThread;
  const int64_t first = block_first + threadIdx.x * 2;
  const int group = static_cast<int>(
      (block_first % kNCCLPeriodElements) / kNCCLChannelGroupElements);
  if (first < num_elements) {
    // Both strict shapes, 16384x4096 and 32768x4096, are multiples of the
    // vector width and channel-group size.  Keep a guarded tail so the host
    // wrapper remains safe if its validation is tightened or extended.
#pragma unroll
    for (int round = 0; round < kElementsPerThread / 2; ++round) {
      // Consecutive warp lanes access consecutive BF16 pairs.  Advancing by
      // one full CTA plane between rounds keeps every 128B sector full; the
      // old thread-contiguous vector mapping issued strided 4B transactions.
      const int64_t index = first + round * blockDim.x * 2;
      if (index + 1 < num_elements) {
        const __nv_bfloat162 a = load_pair(input0 + index);
        const __nv_bfloat162 b = load_pair(input1 + index);
        const __nv_bfloat162 c = load_pair(input2 + index);
        const __nv_bfloat162 d = load_pair(input3 + index);
        *reinterpret_cast<__nv_bfloat162*>(output + index) =
            add4_ordered(a, b, c, d, group);
      }
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// Replicate one rank-owned output shard into the corresponding slice of the
// other three peer-visible output buffers.  The reducer writes the local
// symmetric slice directly, so this kernel deliberately skips the local peer:
// it performs only the three NVLink writes required by an all-gather.
template <typename T, bool kUsePDL>
__global__ void tp4_direct_push_bf16_gather_kernel(
    const T* __restrict__ input,
    T* __restrict__ destination0,
    T* __restrict__ destination1,
    T* __restrict__ destination2,
    int64_t num_elements,
    int64_t output_element_offset) {
  static_assert(sizeof(T) == sizeof(__nv_bfloat16));
  device::PDLWaitPrimary<kUsePDL>();
  using Copy = int4;
  constexpr int kElementsPerCopy = sizeof(Copy) / sizeof(T);
  const int64_t num_copies = num_elements / kElementsPerCopy;
  const int64_t output_copy_offset = output_element_offset / kElementsPerCopy;
  const Copy* source = reinterpret_cast<const Copy*>(input);
  Copy* output0 =
      reinterpret_cast<Copy*>(destination0) + output_copy_offset;
  Copy* output1 =
      reinterpret_cast<Copy*>(destination1) + output_copy_offset;
  Copy* output2 =
      reinterpret_cast<Copy*>(destination2) + output_copy_offset;
  const int64_t thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread; index < num_copies; index += stride) {
    const Copy value = source[index];
    output0[index] = value;
    output1[index] = value;
    output2[index] = value;
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// Fuse the exact four-way BF16 reduction with the symmetric output gather.
// Each rank owns one contiguous token quarter.  Writing that completed quarter
// directly into all four output buffers removes the large reduced intermediate
// read/write and one launch per attention layer while preserving the observed
// NCCL BF16 accumulation order byte-for-byte.
template <typename T, bool kUsePDL>
__global__ void tp4_fused_reduce_push_bf16_gather_kernel(
    const T* __restrict__ input0,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const T* __restrict__ input3,
    T* __restrict__ destination0,
    T* __restrict__ destination1,
    T* __restrict__ destination2,
    T* __restrict__ destination3,
    int64_t num_elements,
    int64_t output_element_offset) {
  static_assert(sizeof(T) == sizeof(__nv_bfloat16));
  device::PDLWaitPrimary<kUsePDL>();
  using Copy = int4;
  constexpr int kElementsPerCopy = sizeof(Copy) / sizeof(T);
  constexpr int kPairsPerCopy = kElementsPerCopy / 2;
  const int64_t num_copies = num_elements / kElementsPerCopy;
  const int64_t output_copy_offset = output_element_offset / kElementsPerCopy;
  const Copy* source0 = reinterpret_cast<const Copy*>(input0);
  const Copy* source1 = reinterpret_cast<const Copy*>(input1);
  const Copy* source2 = reinterpret_cast<const Copy*>(input2);
  const Copy* source3 = reinterpret_cast<const Copy*>(input3);
  Copy* output0 = reinterpret_cast<Copy*>(destination0) + output_copy_offset;
  Copy* output1 = reinterpret_cast<Copy*>(destination1) + output_copy_offset;
  Copy* output2 = reinterpret_cast<Copy*>(destination2) + output_copy_offset;
  Copy* output3 = reinterpret_cast<Copy*>(destination3) + output_copy_offset;
  const int64_t thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t copy_index = thread; copy_index < num_copies;
       copy_index += stride) {
    const Copy a = source0[copy_index];
    const Copy b = source1[copy_index];
    const Copy c = source2[copy_index];
    const Copy d = source3[copy_index];
    Copy reduced;
    const auto* a_pairs = reinterpret_cast<const __nv_bfloat162*>(&a);
    const auto* b_pairs = reinterpret_cast<const __nv_bfloat162*>(&b);
    const auto* c_pairs = reinterpret_cast<const __nv_bfloat162*>(&c);
    const auto* d_pairs = reinterpret_cast<const __nv_bfloat162*>(&d);
    auto* reduced_pairs = reinterpret_cast<__nv_bfloat162*>(&reduced);
    const int group = static_cast<int>(
        ((copy_index * kElementsPerCopy) % kNCCLPeriodElements) /
        kNCCLChannelGroupElements);
#pragma unroll
    for (int pair = 0; pair < kPairsPerCopy; ++pair) {
      reduced_pairs[pair] = add4_ordered(
          a_pairs[pair], b_pairs[pair], c_pairs[pair], d_pairs[pair], group);
    }
    output0[copy_index] = reduced;
    output1[copy_index] = reduced;
    output2[copy_index] = reduced;
    output3[copy_index] = reduced;
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

// Token-group specialization for the no-Graph Req16/Req128 path.  One CTA
// owns a small group of complete tokens, so it can publish a system-scope ready epoch only
// after every BF16 value for that group has reached all four destination
// buffers.  The following mHC-post kernel consumes these epochs directly on
// GPU, replacing the process-wide symmetric-memory barrier with fine-grained
// producer/consumer pipelining and no extra launch.
template <typename T, bool kUsePDL>
__global__ void tp4_fused_reduce_push_bf16_gather_ready_kernel(
    const T* __restrict__ input0,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const T* __restrict__ input3,
    T* __restrict__ destination0,
    T* __restrict__ destination1,
    T* __restrict__ destination2,
    T* __restrict__ destination3,
    uint32_t* __restrict__ ready_owner,
    int64_t owner_tokens,
    int64_t output_element_offset,
    int64_t output_group_offset,
    uint32_t ready_epoch) {
  static_assert(sizeof(T) == sizeof(__nv_bfloat16));
  device::PDLWaitPrimary<kUsePDL>();
  using Copy = int4;
  constexpr int kElementsPerCopy = sizeof(Copy) / sizeof(T);
  constexpr int kPairsPerCopy = kElementsPerCopy / 2;
  constexpr int kCopiesPerToken = 4096 / kElementsPerCopy;
  constexpr int kCopyRounds = kCopiesPerToken / 256;
  static_assert(kCopiesPerToken % 256 == 0);

  const int64_t owner_group = blockIdx.x;
  const int64_t owner_token_first =
      owner_group * kTokensPerReadyGroup;
  const int64_t output_copy_offset =
      output_element_offset / kElementsPerCopy;
  const Copy* source0 = reinterpret_cast<const Copy*>(input0);
  const Copy* source1 = reinterpret_cast<const Copy*>(input1);
  const Copy* source2 = reinterpret_cast<const Copy*>(input2);
  const Copy* source3 = reinterpret_cast<const Copy*>(input3);
  Copy* output0 = reinterpret_cast<Copy*>(destination0) + output_copy_offset;
  Copy* output1 = reinterpret_cast<Copy*>(destination1) + output_copy_offset;
  Copy* output2 = reinterpret_cast<Copy*>(destination2) + output_copy_offset;
  Copy* output3 = reinterpret_cast<Copy*>(destination3) + output_copy_offset;

#pragma unroll
  for (int token_in_group = 0; token_in_group < kTokensPerReadyGroup;
       ++token_in_group) {
    const int64_t owner_token = owner_token_first + token_in_group;
#pragma unroll
    for (int round = 0; round < kCopyRounds; ++round) {
      const int64_t copy_in_token = threadIdx.x + round * blockDim.x;
      const int64_t copy_index =
          owner_token * kCopiesPerToken + copy_in_token;
      const Copy a = source0[copy_index];
      const Copy b = source1[copy_index];
      const Copy c = source2[copy_index];
      const Copy d = source3[copy_index];
      Copy reduced;
      const auto* a_pairs = reinterpret_cast<const __nv_bfloat162*>(&a);
      const auto* b_pairs = reinterpret_cast<const __nv_bfloat162*>(&b);
      const auto* c_pairs = reinterpret_cast<const __nv_bfloat162*>(&c);
      const auto* d_pairs = reinterpret_cast<const __nv_bfloat162*>(&d);
      auto* reduced_pairs = reinterpret_cast<__nv_bfloat162*>(&reduced);
      const int group = static_cast<int>(
          ((copy_index * kElementsPerCopy) % kNCCLPeriodElements) /
          kNCCLChannelGroupElements);
#pragma unroll
      for (int pair = 0; pair < kPairsPerCopy; ++pair) {
        reduced_pairs[pair] = add4_ordered(
            a_pairs[pair], b_pairs[pair], c_pairs[pair], d_pairs[pair], group);
      }
      output0[copy_index] = reduced;
      output1[copy_index] = reduced;
      output2[copy_index] = reduced;
      output3[copy_index] = reduced;
    }
  }

  // Every lane writes a disjoint part of all four peer mappings.  A CTA
  // barrier alone does not guarantee that those peer-memory writes have
  // reached system scope before lane 0 publishes the ready epoch.  Fence each
  // writer first; the following barrier and release store then form the
  // producer side of the consumer's acquire polling protocol.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    // The release store publishes the fully system-visible token group.
    const int64_t global_group = output_group_offset + owner_group;
    store_release_sys(ready_owner + global_group, ready_epoch);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct TP4NcclRingBF16ReduceKernel {
  static void run(
      tvm::ffi::TensorView input0,
      tvm::ffi::TensorView input1,
      tvm::ffi::TensorView input2,
      tvm::ffi::TensorView input3,
      tvm::ffi::TensorView output) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"owner_tokens"};
    constexpr int64_t kHidden = 4096;
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input0);
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input1);
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input2);
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input3);
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(output);
    const int64_t owner_tokens = M.unwrap();
    RuntimeCheck(
        owner_tokens == 16384 || owner_tokens == 32768,
        "TP4 exact BF16 reduction only supports Req16/Req128 owner shards");
    const int64_t num_elements = owner_tokens * kHidden;
    constexpr int kThreads = 256;
    constexpr int64_t kElementsPerBlock = kThreads * kElementsPerThread;
    LaunchKernel(
        dim3((num_elements + kElementsPerBlock - 1) / kElementsPerBlock),
        dim3(kThreads),
        device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_nccl_ring_bf16_reduce_kernel<T, kUsePDL>,
            static_cast<const T*>(input0.data_ptr()),
            static_cast<const T*>(input1.data_ptr()),
            static_cast<const T*>(input2.data_ptr()),
            static_cast<const T*>(input3.data_ptr()),
            static_cast<T*>(output.data_ptr()),
            num_elements);
  }

  static void direct_push_gather(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView peer0,
      tvm::ffi::TensorView peer1,
      tvm::ffi::TensorView peer2,
      tvm::ffi::TensorView peer3,
      int64_t source_rank) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"owner_tokens"};
    constexpr int64_t kHidden = 4096;
    constexpr int64_t kMaxTokens = 131072;
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input);
    TensorMatcher({kMaxTokens, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(peer0);
    TensorMatcher({kMaxTokens, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(peer1);
    TensorMatcher({kMaxTokens, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(peer2);
    TensorMatcher({kMaxTokens, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(peer3);
    const int64_t owner_tokens = M.unwrap();
    RuntimeCheck(
        owner_tokens == 16384 || owner_tokens == 32768,
        "TP4 direct BF16 gather only supports Req16/Req128 owner shards");
    RuntimeCheck(
        source_rank >= 0 && source_rank < 4,
        "TP4 direct BF16 gather requires source_rank in [0, 4)");
    const int64_t num_elements = owner_tokens * kHidden;
    auto* output0 = static_cast<T*>(peer0.data_ptr());
    auto* output1 = static_cast<T*>(peer1.data_ptr());
    auto* output2 = static_cast<T*>(peer2.data_ptr());
    auto* output3 = static_cast<T*>(peer3.data_ptr());
    T* destination0 = nullptr;
    T* destination1 = nullptr;
    T* destination2 = nullptr;
    switch (source_rank) {
      case 0:
        destination0 = output1;
        destination1 = output2;
        destination2 = output3;
        break;
      case 1:
        destination0 = output2;
        destination1 = output3;
        destination2 = output0;
        break;
      case 2:
        destination0 = output3;
        destination1 = output0;
        destination2 = output1;
        break;
      default:
        destination0 = output0;
        destination1 = output1;
        destination2 = output2;
        break;
    }
    constexpr int kThreads = 256;
    // Req16 saturates the peer links at 8192 CTAs; Req128 benefits from twice
    // that parallelism because every rank pushes a 256 MiB owner shard.
    const int blocks = owner_tokens == 16384 ? 8192 : 16384;
    LaunchKernel(dim3(blocks), dim3(kThreads), device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_direct_push_bf16_gather_kernel<T, kUsePDL>,
            static_cast<const T*>(input.data_ptr()),
            destination0,
            destination1,
            destination2,
            num_elements,
            source_rank * num_elements);
  }

  static void fused_reduce_push_gather(
      tvm::ffi::TensorView input0,
      tvm::ffi::TensorView input1,
      tvm::ffi::TensorView input2,
      tvm::ffi::TensorView input3,
      tvm::ffi::TensorView peer0,
      tvm::ffi::TensorView peer1,
      tvm::ffi::TensorView peer2,
      tvm::ffi::TensorView peer3,
      int64_t source_rank) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"owner_tokens"};
    constexpr int64_t kHidden = 4096;
    constexpr int64_t kMaxTokens = 131072;
    TensorMatcher({M, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(input0);
    TensorMatcher({M, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(input1);
    TensorMatcher({M, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(input2);
    TensorMatcher({M, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(input3);
    TensorMatcher({kMaxTokens, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(peer0);
    TensorMatcher({kMaxTokens, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(peer1);
    TensorMatcher({kMaxTokens, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(peer2);
    TensorMatcher({kMaxTokens, kHidden})
        .with_dtype<T>()
        .with_device(device)
        .verify(peer3);
    const int64_t owner_tokens = M.unwrap();
    RuntimeCheck(
        owner_tokens == 16384 || owner_tokens == 32768,
        "TP4 fused BF16 reduce/gather only supports Req16/Req128 owner shards");
    RuntimeCheck(
        source_rank >= 0 && source_rank < 4,
        "TP4 fused BF16 reduce/gather requires source_rank in [0, 4)");
    const int64_t num_elements = owner_tokens * kHidden;
    constexpr int kThreads = 256;
    const int blocks = owner_tokens == 16384 ? 8192 : 16384;
    LaunchKernel(dim3(blocks), dim3(kThreads), device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_fused_reduce_push_bf16_gather_kernel<T, kUsePDL>,
            static_cast<const T*>(input0.data_ptr()),
            static_cast<const T*>(input1.data_ptr()),
            static_cast<const T*>(input2.data_ptr()),
            static_cast<const T*>(input3.data_ptr()),
            static_cast<T*>(peer0.data_ptr()),
            static_cast<T*>(peer1.data_ptr()),
            static_cast<T*>(peer2.data_ptr()),
            static_cast<T*>(peer3.data_ptr()),
            num_elements,
            source_rank * num_elements);
  }

  static void fused_reduce_push_gather_ready(
      tvm::ffi::TensorView input0,
      tvm::ffi::TensorView input1,
      tvm::ffi::TensorView input2,
      tvm::ffi::TensorView input3,
      tvm::ffi::TensorView peer0,
      tvm::ffi::TensorView peer1,
      tvm::ffi::TensorView peer2,
      tvm::ffi::TensorView peer3,
      tvm::ffi::TensorView ready_owner,
      int64_t source_rank,
      int64_t ready_epoch) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"owner_tokens"};
    constexpr int64_t kHidden = 4096;
    constexpr int64_t kMaxTokens = 131072;
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input0);
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input1);
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input2);
    TensorMatcher({M, kHidden}).with_dtype<T>().with_device(device).verify(input3);
    TensorMatcher({kMaxTokens, kHidden}).with_dtype<T>().with_device(device).verify(peer0);
    TensorMatcher({kMaxTokens, kHidden}).with_dtype<T>().with_device(device).verify(peer1);
    TensorMatcher({kMaxTokens, kHidden}).with_dtype<T>().with_device(device).verify(peer2);
    TensorMatcher({kMaxTokens, kHidden}).with_dtype<T>().with_device(device).verify(peer3);
    TensorMatcher({kMaxReadyGroups})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(ready_owner);
    const int64_t owner_tokens = M.unwrap();
    RuntimeCheck(
        owner_tokens == 16384 || owner_tokens == 32768,
        "TP4 token-ready reduce/gather requires Req16/Req128 owner shards");
    RuntimeCheck(source_rank >= 0 && source_rank < 4, "invalid TP4 source rank");
    RuntimeCheck(ready_epoch > 0 && ready_epoch <= UINT32_MAX, "invalid ready epoch");
    const int64_t owner_groups = owner_tokens / kTokensPerReadyGroup;
    const int64_t owner_elements = owner_tokens * kHidden;
    LaunchKernel(dim3(owner_groups), dim3(256), device.unwrap())
        .enable_pdl(kUsePDL)(
            tp4_fused_reduce_push_bf16_gather_ready_kernel<T, kUsePDL>,
            static_cast<const T*>(input0.data_ptr()),
            static_cast<const T*>(input1.data_ptr()),
            static_cast<const T*>(input2.data_ptr()),
            static_cast<const T*>(input3.data_ptr()),
            static_cast<T*>(peer0.data_ptr()),
            static_cast<T*>(peer1.data_ptr()),
            static_cast<T*>(peer2.data_ptr()),
            static_cast<T*>(peer3.data_ptr()),
            static_cast<uint32_t*>(ready_owner.data_ptr()),
            owner_tokens,
            source_rank * owner_elements,
            source_rank * owner_groups,
            static_cast<uint32_t>(ready_epoch));
  }
};

}  // namespace
