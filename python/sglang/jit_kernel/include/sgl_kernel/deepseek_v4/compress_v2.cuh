#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace device::compress {

/// \brief Per-batch decode plan. Layout: 16 bytes.
struct alignas(16) DecodePlan {
  uint32_t seq_len;
  int32_t write_loc;
  int32_t read_page_0;
  int32_t read_page_1;
};

/// \brief Per-token compress plan (used by c4/c128 prefill). Layout: 16 bytes.
struct alignas(16) CompressPlan {
  // `seq_len` is bounded by the model context (73728) and `buffer_len` by the
  // largest compressor window (128).  Keep both in one 32-bit word so
  // `ragged_id` can address aggregate eager-prefill batches larger than 64K
  // without growing the hot plan from 16 to 32 bytes.
  uint32_t seq_len : 24;
  uint32_t buffer_len : 8;
  uint32_t ragged_id;
  int32_t read_page_0;
  /// \brief Stage 0 (CPU): batch_id (used to look up page table).
  /// \brief Stage 1 (GPU): final state-pool write location.
  int32_t read_page_1;

  static constexpr uint32_t kInvalidSeqLen = (1u << 24) - 1u;

  static SGL_DEVICE __host__ CompressPlan invalid() {
    return CompressPlan{kInvalidSeqLen, 0, 0, -1, -1};
  }

  SGL_DEVICE __host__ bool is_invalid() const {
    return seq_len == kInvalidSeqLen;
  }
};

/// \brief Per-token write plan (used by c4/c128 prefill). Layout: 8 bytes.
struct alignas(8) WritePlan {
  /// \brief Stage 0 (CPU): packed batch/ragged id (20 ragged bits, 12 batch bits).
  /// \brief Stage 1 (GPU): just `ragged_id`.
  uint32_t ragged_id;
  /// \brief Stage 0 (CPU): position + 1 (used to look up state slot).
  /// \brief Stage 1 (GPU): final state-pool write location.
  int32_t write_loc;

  static SGL_DEVICE __host__ WritePlan invalid() {
    return WritePlan{-1u, -1};
  }

  SGL_DEVICE __host__ bool is_invalid() const {
    return ragged_id == -1u;
  }
};

}  // namespace device::compress

namespace host::compress {

using device::compress::CompressPlan;
using device::compress::DecodePlan;
using device::compress::WritePlan;

static_assert(alignof(DecodePlan) == sizeof(DecodePlan));
static_assert(sizeof(DecodePlan) == 16);
static_assert(alignof(CompressPlan) == sizeof(CompressPlan));
static_assert(sizeof(CompressPlan) == 16);
static_assert(alignof(WritePlan) == sizeof(WritePlan));
static_assert(sizeof(WritePlan) == 8);

inline auto verify_plan_d(tvm::ffi::TensorView t, SymbolicSize& N, SymbolicDevice& device) -> const DecodePlan* {
  TensorMatcher({N, sizeof(DecodePlan)})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(t);
  return static_cast<const DecodePlan*>(t.data_ptr());
}

inline auto verify_plan_c(tvm::ffi::TensorView t, SymbolicSize& N, SymbolicDevice& device) -> const CompressPlan* {
  TensorMatcher({N, sizeof(CompressPlan)})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(t);
  return static_cast<const CompressPlan*>(t.data_ptr());
}

inline auto verify_plan_w(tvm::ffi::TensorView t, SymbolicSize& N, SymbolicDevice& device) -> const WritePlan* {
  TensorMatcher({N, sizeof(WritePlan)})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(t);
  return static_cast<const WritePlan*>(t.data_ptr());
}

}  // namespace host::compress
