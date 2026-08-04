/**
 * \file topk_v2.cuh
 * \brief TopK kernel for DeepSeek v4.
 * Adapted from
 * 1:
 *   https://github.com/vllm-project/vllm/blob/a8c6ee9b787d273916206a29b77feebadb80c368/csrc/persistent_topk.cuh
 * 2:
 *   https://github.com/flashinfer-ai/flashinfer/blob/c2b4db2b1a84448d802f0e6ac445243312bd6a4c/include/flashinfer/topk.cuh
 * DarkSharpness never took a detailed look at these 2 implementation, but his claude code did.
 * So we add credit to the reference implementations.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

#include <bit>
#include <cstdint>
#include <iterator>

namespace {

namespace impl = device::topk;
using impl::TopKProblem;

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Register5 = impl::TopKRegister<5>;  // <= 20480, register-resident, 1 read
using Streaming = impl::TopKStreaming;
using Cluster = impl::TopKCluster<8>;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384
constexpr uint32_t kReg5MaxSeqLen = Register5::kMaxSeqLen;  // 20480

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)
#define CLUSTER_TOPK_KERNEL TOPK_KERNEL __cluster_dims__(1, kClusterSize, 1)

constexpr uint32_t kClusterFloor = 65536;
constexpr uint32_t kClusterMaxBatch = 512;
constexpr uint32_t kNumPersistentClusters = 15 * kOccupancy;
constexpr uint32_t kSparsePrefillSWAWindow = 128;
using IndexSort1 = cub::BlockRadixSort<int32_t, kBlockSize, 1>;
using IndexSort2 = cub::BlockRadixSort<int32_t, kBlockSize, 2>;
constexpr uint32_t kHugeIndexSortTopK = 512;
constexpr uint32_t kHugeIndexSortMaxSeqLen = Register5::kMaxSeqLen;
constexpr uint32_t kHugeIndexSortWords =
    (kHugeIndexSortMaxSeqLen + 31) / 32;
using IndexBitsetScan = cub::BlockScan<uint32_t, kBlockSize>;

struct IndexBitsetSortStorage {
  uint32_t words[kHugeIndexSortWords];
  IndexBitsetScan::TempStorage scan;
};

/// Metadata tensor rows (each 8 B / 2 int32). Row 0 is the global plan result;
/// rows 1..N are the (batch_id, seq_len) of items routed to the cluster pool.
struct alignas(8) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;  // N = number of items routed to the cluster pool
};
struct alignas(8) PlanItem {
  uint32_t batch_id;
  uint32_t seq_len;
};
static_assert(sizeof(GlobalMetadata) == 2 * sizeof(int32_t) && sizeof(PlanItem) == sizeof(GlobalMetadata));

struct alignas(16) SparsePrefillEpiloguePlan {
  uint32_t topk_len;
  uint32_t swa_len;
  int32_t compressed_offset;
  int32_t swa_offset;
};

struct TopKLaunchParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;      // optional raw (pre-transform) indices output; nullptr if unused
  const int32_t* __restrict__ positions;  // optional sparse-prefill epilogue
  const int32_t* __restrict__ query_start_loc;
  const int32_t* __restrict__ full_seq_lens;
  const int32_t* __restrict__ swa_gather_lens;
  const int32_t* __restrict__ compressed_base;
  const int32_t* __restrict__ swa_base;
  int32_t* __restrict__ combined_indices;
  int32_t* __restrict__ combined_lens;
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  int64_t score_stride;
  int64_t page_table_stride;
  int64_t combined_indices_stride;
  uint32_t topk;
  uint32_t page_bits;
  uint32_t cluster_floor;  // seq_len > this routes to the cluster path (batch-aware, host-set)
  uint32_t num_reqs;

  SGL_DEVICE const GlobalMetadata& global() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE uint32_t cluster_threshold() const {
    return global().cluster_threshold;
  }
  SGL_DEVICE const PlanItem& item(uint32_t i) const {
    return metadata[1 + i];
  }
  SGL_DEVICE int32_t* get_output_ptr(uint32_t batch_id) const {
    return page_indices + batch_id * static_cast<int64_t>(topk);
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        .in = scores + batch_id * score_stride,
        .out = page_indices + batch_id * k,
        .raw_out = raw_indices != nullptr ? raw_indices + batch_id * k : nullptr,
        .page_table = page_table + batch_id * page_table_stride,
        .topk = topk,
        .seq_len = seq_len,
        .page_bits = page_bits,
    };
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    return this->problem(batch_id, static_cast<uint32_t>(seq_lens[batch_id]));
  }
  SGL_DEVICE uint32_t query_owner(uint32_t batch_id) const {
    const int32_t base = query_start_loc[0];
    uint32_t lo = 0;
    uint32_t hi = num_reqs;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) >> 1;
      if (batch_id <
          static_cast<uint32_t>(query_start_loc[mid + 1] - base)) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo < num_reqs ? lo : num_reqs - 1;
  }
  SGL_DEVICE void prepare_sparse_prefill_epilogue(
      uint32_t batch_id, SparsePrefillEpiloguePlan* plan) const {
    const uint32_t req = query_owner(batch_id);
    const uint32_t seq_len = static_cast<uint32_t>(seq_lens[batch_id]);
    plan->topk_len = seq_len < topk ? seq_len : topk;
    const int32_t pos = positions[batch_id];
    const uint32_t pos_len = static_cast<uint32_t>(pos + 1);
    plan->swa_len =
        pos_len < kSparsePrefillSWAWindow ? pos_len : kSparsePrefillSWAWindow;
    const int32_t gather_start = full_seq_lens[req] - swa_gather_lens[req];
    plan->compressed_offset = compressed_base[req];
    plan->swa_offset = swa_base[req] + pos - static_cast<int32_t>(plan->swa_len) +
        1 - gather_start;
  }
  SGL_DEVICE void sparse_prefill_epilogue(
      uint32_t batch_id,
      const SparsePrefillEpiloguePlan& plan,
      const int32_t* __restrict__ raw_indices_ptr) const {
    if (combined_indices == nullptr) return;
    auto* combined_row = combined_indices + batch_id * combined_indices_stride;
    for (uint32_t t = threadIdx.x; t < plan.topk_len; t += blockDim.x) {
      combined_row[t] = raw_indices_ptr[t] + plan.compressed_offset;
    }
    for (uint32_t t = threadIdx.x; t < plan.swa_len; t += blockDim.x) {
      combined_row[plan.topk_len + t] = plan.swa_offset + static_cast<int32_t>(t);
    }
    if (threadIdx.x == 0) {
      combined_lens[batch_id] = static_cast<int32_t>(plan.topk_len + plan.swa_len);
    }
  }
};

/**
 * \brief Persistent cluster kernel for the long items. It will handle long inputs.
 * The short items are handled by the separate topk_kernel.
 */
template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_persistent_cluster_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  __shared__ impl::MaxSmem<Cluster::Smem> smem;
  const uint32_t num_cluster_items = params.global().num_cluster_items;
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
#pragma unroll 1
  for (uint32_t w = blockIdx.x; w < num_cluster_items; w += kNumPersistentClusters) {
    const auto it = params.item(w);
    const auto problem = params.problem(it.batch_id, it.seq_len);
    Cluster::forward<false>(problem, &smem);
    __syncthreads();
  }
}

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      __builtin_assume(tx < kMaxTopK);
      f(tx, i);
    }
  }
}

template <bool kPDL>
SGL_DEVICE void trivial_transform(const TopKProblem& problem) {
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t) {
    problem.transform_output(tx, tx < problem.seq_len ? static_cast<int32_t>(tx) : -1);
  });
}

SGL_DEVICE void problem_transform_bitset512(
    TopKProblem& problem,
    const int32_t* source_ptr,
    IndexBitsetSortStorage* storage) {
  const uint32_t tx = threadIdx.x;
  for (uint32_t word = tx; word < kHugeIndexSortWords;
       word += kBlockSize) {
    storage->words[word] = 0;
  }
  __syncthreads();

  if (tx < kHugeIndexSortTopK) {
    const int32_t raw = source_ptr[tx];
    if (raw >= 0 && static_cast<uint32_t>(raw) < problem.seq_len) {
      atomicOr(
          &storage->words[static_cast<uint32_t>(raw) >> 5],
          1u << (static_cast<uint32_t>(raw) & 31));
    }
  }
  __syncthreads();

  const uint32_t bits =
      tx < kHugeIndexSortWords ? storage->words[tx] : 0;
  uint32_t write_base = 0;
  IndexBitsetScan(storage->scan).ExclusiveSum(__popc(bits), write_base);

  if (tx < kHugeIndexSortWords) {
    uint32_t remaining = bits;
    while (remaining != 0) {
      const uint32_t bit = static_cast<uint32_t>(__ffs(remaining) - 1);
      problem.transform_output(write_base++, static_cast<int32_t>(tx * 32 + bit));
      remaining &= remaining - 1;
    }
  }
}

SGL_DEVICE void problem_transform(
    TopKProblem& problem, int32_t* output_ptr, void* sort_smem) {
  static_assert(kMaxTopK % kBlockSize == 0);
  // The radix selector emits all elements above the threshold through an
  // atomic counter. The selected set is exact, but its order depends on CTA
  // scheduling. Sparse attention accumulates in that order, so the otherwise
  // harmless permutation becomes model-level run-to-run drift. Sort the raw
  // token indices in this existing transform CTA: this is deterministic and
  // also turns paged-KV reads into increasing-address accesses.
  const auto end_bit = 32 - __clz(problem.seq_len - 1);
  constexpr int32_t kInvalid = std::numeric_limits<int32_t>::max();
  const auto source_ptr = problem.out;
  problem.out = output_ptr;
  if (problem.topk == kHugeIndexSortTopK &&
      problem.seq_len <= kHugeIndexSortMaxSeqLen) {
    problem_transform_bitset512(
        problem,
        source_ptr,
        static_cast<IndexBitsetSortStorage*>(sort_smem));
    return;
  }
  if (problem.topk <= kBlockSize) {
    int32_t source_index[1] = {
        threadIdx.x < problem.topk ? source_ptr[threadIdx.x] : kInvalid};
    IndexSort1(*static_cast<IndexSort1::TempStorage*>(sort_smem))
        .Sort(source_index, 0, end_bit);
    if (threadIdx.x < problem.topk) {
      problem.transform_output(threadIdx.x, source_index[0]);
    }
  } else {
    int32_t source_index[2];
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
      const auto pos = threadIdx.x * 2 + i;
      source_index[i] = pos < problem.topk ? source_ptr[pos] : kInvalid;
    }
    IndexSort2(*static_cast<IndexSort2::TempStorage*>(sort_smem))
        .Sort(source_index, 0, end_bit);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
      const auto pos = threadIdx.x * 2 + i;
      if (pos < problem.topk) problem.transform_output(pos, source_index[i]);
    }
  }
}

/**
 * \brief Main kernel for the short items and epilogue of long items.
 * \tparam kPDL whether to use PDL to synchronize with the cluster kernel (if any)
 * \tparam kLevel:
 * - Level 0: max_seq_len <= 8192           -> trivial + register<2>
 * - Level 1: max_seq_len <= 16384          -> trivial + register<4>
 * - Level 2: max_seq_len <= cluster_floor  -> trivial + register<4/5> + streaming
 * - Level 3: max_seq_len > cluster_floor   -> + epilogue process of cluster path
 */
template <bool kPDL, int kLevel>
TOPK_KERNEL void topk_main_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();
  __shared__ impl::MaxSmem<
      Register2::Smem,
      Register4::Smem,
      Register5::Smem,
      Streaming::Smem,
      IndexBitsetSortStorage,
      IndexSort1::TempStorage,
      IndexSort2::TempStorage>
      smem;
  __shared__ SparsePrefillEpiloguePlan sparse_epilogue_plan;
  if (problem.seq_len <= problem.topk) {
    trivial_transform<kPDL>(problem);
    if (params.combined_indices != nullptr && threadIdx.x == 0) {
      params.prepare_sparse_prefill_epilogue(blockIdx.x, &sparse_epilogue_plan);
    }
    __syncthreads();
    if (params.combined_indices != nullptr) {
      params.sparse_prefill_epilogue(
          blockIdx.x,
          sparse_epilogue_plan,
          params.raw_indices + blockIdx.x * static_cast<int64_t>(params.topk));
    }
    return;
  }
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;
  const int32_t* epilogue_indices_ptr =
      params.raw_indices != nullptr
      ? params.raw_indices + blockIdx.x * static_cast<int64_t>(params.topk)
      : topk_indices;

  constexpr bool kHandleCluster = (kLevel == 3);
  // non-trivial path: dispatch based on level and seq_len
  const auto cluster_threshold = kHandleCluster ? params.cluster_threshold() : kU32Max;
  if constexpr (kLevel == 0) {
    __builtin_assume(problem.seq_len <= kReg2MaxSeqLen);
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    __builtin_assume(problem.seq_len <= kReg4MaxSeqLen);
    Register4::forward<kPDL>(problem, &smem);  // max_seq_len <= 16384 guarantees seq <= 16384
  } else {
    static_assert(kLevel == 2 || kLevel == 3, "we only support level = 0,1,2,3 now");
    // if using cluster, we can delay the PDL wait
    constexpr bool kPDLEarly = kPDL && !kHandleCluster;
    constexpr bool kPDLFinal = kPDL && kHandleCluster;
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= kReg5MaxSeqLen) {
      // DSV4 Flash's strict C4 context tops out at 18432.  Keeping the fifth
      // float4 vector resident avoids rereading the complete logits row in
      // the streaming collect pass for the important 64K-prefix workload.
      Register5::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= cluster_threshold) {
      Streaming::forward<kPDLEarly>(problem, &smem);
    } else {  // cluster path do nothing here
      problem.out = params.get_output_ptr(blockIdx.x);
      // The persistent cluster kernel publishes its raw result in global
      // output, so this CTA has no shared raw-index copy to reuse.
      epilogue_indices_ptr =
          params.raw_indices + blockIdx.x * static_cast<int64_t>(params.topk);
    }
    device::PDLWaitPrimary<kPDLFinal>();
  }

  // page-table transform pass (gathers kept out of the hot scatter loop),
  // then trigger the dependent kernel only after the full output is written.
  device::PDLTriggerSecondary<kPDL>();
  __syncthreads();
  problem_transform(problem, params.get_output_ptr(blockIdx.x), &smem);
  if (params.combined_indices != nullptr && threadIdx.x == 0) {
    params.prepare_sparse_prefill_epilogue(blockIdx.x, &sparse_epilogue_plan);
  }
  __syncthreads();
  if (params.combined_indices != nullptr) {
    params.sparse_prefill_epilogue(
        blockIdx.x, sparse_epilogue_plan, epilogue_indices_ptr);
  }
}

template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  __shared__ impl::MaxSmem<
      Streaming::Smem,
      Cluster::Smem,
      IndexBitsetSortStorage,
      IndexSort1::TempStorage,
      IndexSort2::TempStorage>
      smem;
  __shared__ SparsePrefillEpiloguePlan sparse_epilogue_plan;
  // One rank owns final output for both the trivial and clustered paths.
  const auto worker_rank = blockIdx.x % kClusterSize;
  if (problem.seq_len <= problem.topk) {
    if (blockIdx.y == worker_rank) {
      trivial_transform<kPDL>(problem);
      if (params.combined_indices != nullptr && threadIdx.x == 0) {
        params.prepare_sparse_prefill_epilogue(blockIdx.x, &sparse_epilogue_plan);
      }
      __syncthreads();
      if (params.combined_indices != nullptr) {
        params.sparse_prefill_epilogue(
            blockIdx.x,
            sparse_epilogue_plan,
            params.raw_indices + blockIdx.x * static_cast<int64_t>(params.topk));
      }
    }
    return;
  }
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  // randomly elect one worker rank to avoid workload imbalance

  // for small batch, we will fuse in the cluster case
  if (problem.seq_len <= kReg4MaxSeqLen) {
    if (blockIdx.y == worker_rank) Register4::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= kReg5MaxSeqLen) {
    if (blockIdx.y == worker_rank) Register5::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= params.cluster_floor) {
    if (blockIdx.y == worker_rank) Streaming::forward<kPDL>(problem, &smem);
  } else {
    auto cluster = cooperative_groups::this_cluster();
    problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
    Cluster::forward<kPDL>(problem, &smem);  // write to peer's output shared memory
    cluster.sync();
  }

  device::PDLWaitPrimary<kPDL>();
  __syncthreads();
  if (blockIdx.y == worker_rank) {
    problem_transform(problem, params.get_output_ptr(blockIdx.x), &smem);
    if (params.combined_indices != nullptr && threadIdx.x == 0) {
      params.prepare_sparse_prefill_epilogue(blockIdx.x, &sparse_epilogue_plan);
    }
    __syncthreads();
    if (params.combined_indices != nullptr) {
      params.sparse_prefill_epilogue(
          blockIdx.x,
          sparse_epilogue_plan,
          params.raw_indices + blockIdx.x * static_cast<int64_t>(params.topk));
    }
  }
}

// --- Plan: choose cluster_threshold from the seq_len distribution -----------
__global__ __launch_bounds__(kBlockSize, 1) void topk_plan(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,  // [0]=GlobalMetadata, [1+i]=PlanItem
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
  // Candidate (threshold T_j, cap_j) pairs, T strictly increasing. The plan lowers
  // cluster_threshold to T_j while #(items with seq_len > T_j) <= cap_j, so cap_j
  // bounds how many long items go to the persistent pool. The pool runs N items in
  // ceil(N / kNumPersistentClusters) waves; the longer the seq the more waves pay
  // off (streaming a single block over a long item is very slow), so cap_j is the
  // measured cluster-vs-streaming crossover (B200, occ2) and GROWS with T -- a flat
  // cap = pool size only fits the shortest (~98K, one-wave) bucket. (Plan is tunable.)
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
      {65536, 30},    // (65536,98304]:    ~1 pool wave, streams beyond 30
      {98304, 48},    // (98304,131072]
      {131072, 60},   // (131072,196608]
      {196608, 80},   // (196608,262144]
      {262144, 112},  // (262144,393216]
      {393216, 128},  // (393216,inf):     longest -- worth many pool waves; a top
                      // threshold here lets overloaded ~280-393K batches still stream
  };
  constexpr uint32_t kNumCandidates = std::size(kCandidates);
  static_assert(kCandidates[0].threshold == kClusterFloor);

  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;
  __shared__ uint32_t s_count;

  const auto tx = threadIdx.x;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  if (tx == 0) s_count = 0;
  __syncthreads();

  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) atomicAdd(&s_counts[count - 1], 1);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kCandidates[kNumCandidates - 1].threshold;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];  // # items with seq_len > kCandidates[j].threshold
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  const auto cluster_threshold = max(s_threshold, kClusterFloor);

  // Compact items with seq_len > threshold into metadata[1..N]: their batch ids
  // are the work list the persistent cluster pool fetches.
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl};
    }
  }
  __syncthreads();
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {.cluster_threshold = cluster_threshold, .num_cluster_items = s_count};
  }
}

#ifndef SGLANG_DSV4_TOPK_DEVICE_ONLY

struct TopKKernel {
  static void plan(  //
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView metadata,
      const uint32_t static_cluster_threshold) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({Bp1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto device = device_.unwrap();
    LaunchKernel(1, kBlockSize, device)(  //
        topk_plan,
        static_cast<const uint32_t*>(seq_lens.data_ptr()),
        static_cast<PlanItem*>(metadata.data_ptr()),
        batch_size,
        static_cluster_threshold);
  }

  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::TensorView metadata,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices,
      const tvm::ffi::Optional<tvm::ffi::TensorView> positions,
      const tvm::ffi::Optional<tvm::ffi::TensorView> query_start_loc,
      const tvm::ffi::Optional<tvm::ffi::TensorView> full_seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> swa_gather_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> compressed_base,
      const tvm::ffi::Optional<tvm::ffi::TensorView> swa_base,
      const tvm::ffi::Optional<tvm::ffi::TensorView> combined_indices,
      const tvm::ffi::Optional<tvm::ffi::TensorView> combined_lens) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, -1})  // page_table
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_table);
    TensorMatcher({B, K})  // page_indices
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);
    TensorMatcher({Bp1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device_).verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    const bool sparse_epilogue = combined_indices.has_value();
    RuntimeCheck(
        sparse_epilogue == positions.has_value() &&
            sparse_epilogue == query_start_loc.has_value() &&
            sparse_epilogue == full_seq_lens.has_value() &&
            sparse_epilogue == swa_gather_lens.has_value() &&
            sparse_epilogue == compressed_base.has_value() &&
            sparse_epilogue == swa_base.has_value() &&
            sparse_epilogue == combined_lens.has_value(),
        "sparse-prefill topk epilogue tensors must be provided together");
    RuntimeCheck(!sparse_epilogue || raw_indices_ptr != nullptr,
                 "sparse-prefill topk epilogue requires raw_indices output");

    const int32_t* positions_ptr = nullptr;
    const int32_t* query_start_loc_ptr = nullptr;
    const int32_t* full_seq_lens_ptr = nullptr;
    const int32_t* swa_gather_lens_ptr = nullptr;
    const int32_t* compressed_base_ptr = nullptr;
    const int32_t* swa_base_ptr = nullptr;
    int32_t* combined_indices_ptr = nullptr;
    int32_t* combined_lens_ptr = nullptr;
    int64_t combined_indices_stride = 0;
    uint32_t num_reqs = 0;
    if (sparse_epilogue) {
      auto R = SymbolicSize{"num_reqs"};
      auto C = SymbolicSize{"combined_stride"};
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(positions.value());
      TensorMatcher({-1}).with_dtype<int32_t>().with_device(device_).verify(query_start_loc.value());
      TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(full_seq_lens.value());
      TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(swa_gather_lens.value());
      TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(compressed_base.value());
      TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(swa_base.value());
      TensorMatcher({B, -1})
          .with_strides({C, 1})
          .with_dtype<int32_t>()
          .with_device(device_)
          .verify(combined_indices.value());
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(combined_lens.value());
      RuntimeCheck(query_start_loc.value().shape()[0] == R.unwrap() + 1,
                   "query_start_loc must have num_reqs + 1 entries");
      RuntimeCheck(R.unwrap() > 0 && R.unwrap() <= 128,
                   "sparse-prefill topk supports 1..128 requests");
      RuntimeCheck(combined_indices.value().shape()[1] >= K.unwrap() + kSparsePrefillSWAWindow,
                   "combined_indices row is too narrow");
      positions_ptr = static_cast<const int32_t*>(positions.value().data_ptr());
      query_start_loc_ptr = static_cast<const int32_t*>(query_start_loc.value().data_ptr());
      full_seq_lens_ptr = static_cast<const int32_t*>(full_seq_lens.value().data_ptr());
      swa_gather_lens_ptr = static_cast<const int32_t*>(swa_gather_lens.value().data_ptr());
      compressed_base_ptr = static_cast<const int32_t*>(compressed_base.value().data_ptr());
      swa_base_ptr = static_cast<const int32_t*>(swa_base.value().data_ptr());
      combined_indices_ptr = static_cast<int32_t*>(combined_indices.value().data_ptr());
      combined_lens_ptr = static_cast<int32_t*>(combined_lens.value().data_ptr());
      combined_indices_stride = C.unwrap();
      num_reqs = static_cast<uint32_t>(R.unwrap());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto max_seq_len = static_cast<uint32_t>(L.unwrap());
    const auto device = device_.unwrap();

    // The fused kernel runs one 8-block cluster per batch element, and B200 fits one
    // wave of exactly 15 such clusters (occ2). For batch <= 15 it stays latency-bound,
    // so the 8-way split beats streaming from a much lower seq (measured crossover
    // ~36-40K); batch 16 spills into a 2nd wave (+25%) and keeps the 64K floor.
    // The floor is chosen on the host per launch.
    constexpr uint32_t kClusterFloorSmall = 32768;
    constexpr uint32_t kSmallBatchLowFloor = 15;
    const auto params = TopKLaunchParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .positions = positions_ptr,
        .query_start_loc = query_start_loc_ptr,
        .full_seq_lens = full_seq_lens_ptr,
        .swa_gather_lens = swa_gather_lens_ptr,
        .compressed_base = compressed_base_ptr,
        .swa_base = swa_base_ptr,
        .combined_indices = combined_indices_ptr,
        .combined_lens = combined_lens_ptr,
        .metadata = static_cast<const PlanItem*>(metadata.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .combined_indices_stride = combined_indices_stride,
        .topk = topk,
        .page_bits = page_bits,
        .cluster_floor = (batch_size <= kSmallBatchLowFloor) ? kClusterFloorSmall : kClusterFloor,
        .num_reqs = num_reqs,
    };

    const bool use_cluster = (max_seq_len > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
    constexpr bool kUsePDL = true;
    if (use_cluster) {
      if (batch_size <= kNumPersistentClusters) {
        LaunchKernel({batch_size, kClusterSize}, kBlockSize, device)
            .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
            .launch(topk_small_batch_kernel<kUsePDL>, params);
      } else {
        const uint32_t num_clusters = std::min(batch_size, kNumPersistentClusters);
        LaunchKernel({num_clusters, kClusterSize}, kBlockSize, device)
            .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
            .launch(topk_persistent_cluster_kernel<kUsePDL>, params);
        LaunchKernel(batch_size, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_main_kernel<kUsePDL, /*kLevel=*/3>, params);
      }
    } else if (max_seq_len <= kReg2MaxSeqLen) {
      LaunchKernel(batch_size, kBlockSize, device)
          .config({.use_pdl = kUsePDL})
          .launch(topk_main_kernel<kUsePDL, /*kLevel=*/0>, params);
    } else if (max_seq_len <= kReg4MaxSeqLen) {
      LaunchKernel(batch_size, kBlockSize, device)
          .config({.use_pdl = kUsePDL})
          .launch(topk_main_kernel<kUsePDL, /*kLevel=*/1>, params);
    } else {
      LaunchKernel(batch_size, kBlockSize, device)
          .config({.use_pdl = kUsePDL})
          .launch(topk_main_kernel<kUsePDL, /*kLevel=*/2>, params);
    }
  }
};

#endif  // SGLANG_DSV4_TOPK_DEVICE_ONLY

}  // namespace
