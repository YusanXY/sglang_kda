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
};

}  // namespace
