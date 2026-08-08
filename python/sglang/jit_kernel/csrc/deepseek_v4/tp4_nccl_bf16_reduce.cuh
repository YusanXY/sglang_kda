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
  const int64_t first =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kElementsPerThread;
  if (first < num_elements) {
    // Both strict shapes, 16384x4096 and 32768x4096, are multiples of the
    // vector width and channel-group size.  Keep a guarded tail so the host
    // wrapper remains safe if its validation is tightened or extended.
#pragma unroll
    for (int offset = 0; offset < kElementsPerThread; offset += 2) {
      const int64_t index = first + offset;
      if (index + 1 < num_elements) {
        const int group = static_cast<int>(
            (index % kNCCLPeriodElements) / kNCCLChannelGroupElements);
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
};

}  // namespace
