// DeepSeek-V4 sparse-prefill fused gather/dequant for two paged KV sources.
//
// Compressed C4/C128 and SWA caches use the same per-token byte encoding but
// different page sizes and buffers.  The legacy path launches the same Triton
// dequant kernel twice.  This CUDA kernel assigns one warp to one gathered
// token and covers both sources in one launch, writing the exact flat BF16
// workspace consumed by FlashMLA sparse prefill.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace {

constexpr int kDimNope = 448;
constexpr int kDimRope = 64;
constexpr int kOutputDim = kDimNope + kDimRope;
constexpr int kScaleTile = 64;
constexpr int kScaleTiles = kDimNope / kScaleTile;
constexpr int kNopeRopeBytes = kDimNope + kDimRope * 2;
constexpr int kScaleBytesPerToken = kScaleTiles + 1;
constexpr int kWarpsPerBlock = 4;

template <typename T, bool kUsePDL>
__global__ void dual_paged_dequant_kernel(
    const uint8_t* __restrict__ cache_a,
    const int32_t* __restrict__ token_ids_a,
    int64_t num_tokens_a,
    int64_t bytes_per_page_a,
    int32_t page_size_a,
    const uint8_t* __restrict__ cache_b,
    const int32_t* __restrict__ token_ids_b,
    int64_t num_tokens_b,
    int64_t bytes_per_page_b,
    int32_t page_size_b,
    T* __restrict__ output) {
  device::PDLWaitPrimary<kUsePDL>();

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t output_token =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t total_tokens = num_tokens_a + num_tokens_b;
  if (output_token < total_tokens) {
    const bool use_a = output_token < num_tokens_a;
    const int64_t source_token = use_a ? output_token : output_token - num_tokens_a;
    const uint8_t* cache = use_a ? cache_a : cache_b;
    const int32_t* token_ids = use_a ? token_ids_a : token_ids_b;
    const int64_t bytes_per_page = use_a ? bytes_per_page_a : bytes_per_page_b;
    const int32_t page_size = use_a ? page_size_a : page_size_b;

    const int64_t loc = static_cast<int64_t>(token_ids[source_token]);
    const int64_t page = loc / page_size;
    const int64_t in_page = loc - page * page_size;
    const int64_t page_byte_base = page * bytes_per_page;
    const int64_t token_data_base =
        page_byte_base + in_page * kNopeRopeBytes;
    const int64_t token_scale_base =
        page_byte_base + static_cast<int64_t>(page_size) * kNopeRopeBytes +
        in_page * kScaleBytesPerToken;
    T* output_row = output + output_token * kOutputDim;

#pragma unroll
    for (int tile = 0; tile < kScaleTiles; ++tile) {
      uint32_t scale_exp = 0;
      if (lane == 0) {
        scale_exp = cache[token_scale_base + tile];
      }
      scale_exp = __shfl_sync(0xffffffffu, scale_exp, 0);
      // UE8M0 stores the biased FP32 exponent directly.  Exponent zero is
      // below the normal FP32 range and is flushed exactly like the Triton
      // reference implementation.
      const float scale = scale_exp == 0
          ? 0.0f
          : __uint_as_float(scale_exp << 23);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int element = tile * kScaleTile + half * 32 + lane;
        const auto value = reinterpret_cast<const fp8_e4m3_t*>(
            cache + token_data_base)[element];
        output_row[element] =
            device::cast<T>(static_cast<float>(value) * scale);
      }
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int element = half * 32 + lane;
      const auto* rope = reinterpret_cast<const T*>(
          cache + token_data_base + kDimNope);
      output_row[kDimNope + element] = rope[element];
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct DualPagedDequantKernel {
  static void run(
      tvm::ffi::TensorView cache_a,
      tvm::ffi::TensorView token_ids_a,
      int64_t page_size_a,
      tvm::ffi::TensorView cache_b,
      tvm::ffi::TensorView token_ids_b,
      int64_t page_size_b,
      tvm::ffi::TensorView output) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto pages_a = SymbolicSize{"pages_a"};
    auto bytes_a = SymbolicSize{"bytes_a"};
    auto pages_b = SymbolicSize{"pages_b"};
    auto bytes_b = SymbolicSize{"bytes_b"};
    auto tokens_a = SymbolicSize{"tokens_a"};
    auto tokens_b = SymbolicSize{"tokens_b"};
    auto output_tokens = SymbolicSize{"output_tokens"};

    TensorMatcher({pages_a, bytes_a})
        .with_strides({bytes_a, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(cache_a);
    TensorMatcher({tokens_a})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(token_ids_a);
    TensorMatcher({pages_b, bytes_b})
        .with_strides({bytes_b, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(cache_b);
    TensorMatcher({tokens_b})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(token_ids_b);
    TensorMatcher({output_tokens, 1, kOutputDim})
        .with_strides({kOutputDim, kOutputDim, 1})
        .with_dtype<T>()
        .with_device(device)
        .verify(output);

    RuntimeCheck(page_size_a > 0 && page_size_b > 0, "page sizes must be positive");
    RuntimeCheck(
        bytes_a.unwrap() >= page_size_a * (kNopeRopeBytes + kScaleBytesPerToken),
        "cache_a page is too small for the DSV4 KV encoding");
    RuntimeCheck(
        bytes_b.unwrap() >= page_size_b * (kNopeRopeBytes + kScaleBytesPerToken),
        "cache_b page is too small for the DSV4 KV encoding");
    const int64_t total_tokens = tokens_a.unwrap() + tokens_b.unwrap();
    RuntimeCheck(
        output_tokens.unwrap() == total_tokens,
        "output token count must equal tokens_a + tokens_b");
    if (total_tokens == 0) return;

    const dim3 block(kWarpsPerBlock * 32);
    const dim3 grid((total_tokens + kWarpsPerBlock - 1) / kWarpsPerBlock);
    LaunchKernel(grid, block, device.unwrap())
        .enable_pdl(kUsePDL)(
            dual_paged_dequant_kernel<T, kUsePDL>,
            static_cast<const uint8_t*>(cache_a.data_ptr()),
            static_cast<const int32_t*>(token_ids_a.data_ptr()),
            tokens_a.unwrap(),
            bytes_a.unwrap(),
            static_cast<int32_t>(page_size_a),
            static_cast<const uint8_t*>(cache_b.data_ptr()),
            static_cast<const int32_t*>(token_ids_b.data_ptr()),
            tokens_b.unwrap(),
            bytes_b.unwrap(),
            static_cast<int32_t>(page_size_b),
            static_cast<T*>(output.data_ptr()));
  }
};

}  // namespace
