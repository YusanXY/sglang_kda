#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "v2_sm100_mqa_logits.cuh"
#ifndef SGLANG_DSV4_TOPK_WARP_BITSET_SORT
#error "clustered MQA combined-only TopK requires warp-bitset sort"
#endif
#define SGLANG_DSV4_TOPK_DEVICE_ONLY
#include "../../deepseek_v4/topk_v2.cuh"
#undef SGLANG_DSV4_TOPK_DEVICE_ONLY

// JIT build epoch v70b: force the extension content key to include the strict
// Req16/Req32 11-bit register Top-k specialization from the transitive header.

void launch_blockq8_tmem2(
    int grid_size,
    int num_q_tokens_total,
    int logits_stride,
    int block_table_stride,
    const int* context_lens,
    float* logits,
    const int* block_table,
    const int* schedule_meta,
    CUtensorMap tensor_map_q,
    CUtensorMap tensor_map_sf_q,
    CUtensorMap tensor_map_kv,
    CUtensorMap tensor_map_sf_kv,
    CUtensorMap tensor_map_weights,
    cudaStream_t stream) {
    using Storage = deep_gemm::layout::MQALogitsSharedStorage<
        false, 64, 128, 8, 256, 1, 4, 2, float>;
    constexpr int kSmemBytes = static_cast<int>(sizeof(Storage));
    constexpr int kThreads = 128 + 256;

    auto kernel = &deep_gemm::sm100_paged_mqa_logits<
        false,
        16, 64,
        128, 64,
        true, false,
        1, 4,
        256, 64,
        128, 256,
        float, float,
        2>;

    auto status = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemBytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaFuncSetAttribute failed: ") +
            cudaGetErrorString(status));
    }

    cudaLaunchAttribute cluster_attr{};
    cluster_attr.id = cudaLaunchAttributeClusterDimension;
    cluster_attr.val.clusterDim = {2, 1, 1};

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(grid_size, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = stream;
    config.attrs = &cluster_attr;
    config.numAttrs = 1;

    status = cudaLaunchKernelEx(
        &config,
        kernel,
        static_cast<uint32_t>(num_q_tokens_total),
        static_cast<uint32_t>(logits_stride),
        static_cast<uint32_t>(block_table_stride),
        reinterpret_cast<const uint32_t*>(context_lens),
        logits,
        reinterpret_cast<const uint32_t*>(block_table),
        nullptr,
        reinterpret_cast<const uint32_t*>(schedule_meta),
        tensor_map_q,
        tensor_map_sf_q,
        tensor_map_kv,
        tensor_map_sf_kv,
        tensor_map_weights);

    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cluster2 Q16 cta_group::2 KV kernel launch failed: ") +
            cudaGetErrorString(status));
    }
}

void launch_topk512_sparse_prefill(
    int batch_size,
    int num_reqs,
    int max_context_len,
    int logits_stride,
    int page_table_stride,
    int combined_indices_stride,
    const float* logits,
    const int* seq_lens,
    const int* page_table,
    int* page_indices,
    int* raw_indices,
    const int* positions,
    const int* query_start_loc,
    const int* full_seq_lens,
    const int* swa_gather_lens,
    const int* compressed_base,
    const int* swa_base,
    int* combined_indices,
    int* combined_lens,
    cudaStream_t stream) {
    // The strict Huge runtime caps the raw context at 73728 tokens.  C4
    // therefore never exceeds 18432 positions and cannot enter v2's >64K
    // cluster route.  Keeping the launch inside this extension removes the
    // second Python/C++ submission and makes routing a static three-way CUDA
    // specialization without requiring the generic plan tensor.
    constexpr uint32_t kTopK = 512;
    constexpr uint32_t kPageBits = 6;
    constexpr uint32_t kHugeMaxC4Context = 18432;
    if (max_context_len <= 0 ||
        max_context_len > static_cast<int>(kHugeMaxC4Context) ||
        logits_stride < max_context_len || logits_stride % 256 != 0) {
        throw std::runtime_error(
            "Huge C4 top-k v2 requires 0 < context <= 18432 and a 256-aligned logits stride");
    }
    const auto params = TopKLaunchParams{
        .scores = logits,
        .seq_lens = seq_lens,
        .page_table = page_table,
        .page_indices = page_indices,
        .raw_indices = raw_indices,
        .positions = positions,
        .query_start_loc = query_start_loc,
        .full_seq_lens = full_seq_lens,
        .swa_gather_lens = swa_gather_lens,
        .compressed_base = compressed_base,
        .swa_base = swa_base,
        .combined_indices = combined_indices,
        .combined_lens = combined_lens,
        .metadata = nullptr,
        .score_stride = logits_stride,
        .page_table_stride = page_table_stride,
        .combined_indices_stride = combined_indices_stride,
        .topk = kTopK,
        .page_bits = kPageBits,
        .cluster_floor = kClusterFloor,
        .num_reqs = static_cast<uint32_t>(num_reqs),
    };

    cudaLaunchAttribute pdl_attr{};
    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr.val.programmaticStreamSerializationAllowed = true;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(batch_size, 1, 1);
    config.blockDim = dim3(kBlockSize, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &pdl_attr;
    config.numAttrs = 1;

    // Strict no-Graph high-load Huge consumes only combined_indices in the
    // following sparse-attention stage.  Select the combined-only template at
    // the real clustered MQA+TopK launch boundary: the standalone TVM-FFI
    // transform wrapper is not used by this path.  Keep lower-load Huge
    // configurations on their existing CUDA specialization.
    // TP4 query sharding turns the strict global Req16/Req32 scheduler waves
    // into request-aligned Req4/Req8 local launches.  Keep those launches on
    // the same combined-only epilogue as their unsharded counterparts: sparse
    // attention consumes only combined_indices, so publishing page_indices
    // and raw_indices would add two full [M,512] global stores plus a reload.
    // Retain the pre-sharding shapes for standalone callers and frozen paths.
    const bool combined_only =
        (num_reqs == 4 && batch_size == 4 * 4096) ||
        (num_reqs == 8 && batch_size == 8 * 4096) ||
        (num_reqs == 16 && batch_size == 16 * 4096) ||
        (num_reqs == 32 && batch_size == 32 * 4096) ||
        (num_reqs == 128 && batch_size == 128 * 4096);
    cudaError_t status;
    if (combined_only) {
        if (max_context_len <= static_cast<int>(kReg2MaxSeqLen)) {
            status = cudaLaunchKernelEx(
                &config, topk_main_kernel<true, 0, true, true>, params);
        } else if (max_context_len <= static_cast<int>(kReg4MaxSeqLen)) {
            status = cudaLaunchKernelEx(
                &config, topk_main_kernel<true, 1, true>, params);
        } else {
            status = cudaLaunchKernelEx(
                &config, topk_main_kernel<true, 2, true>, params);
        }
    } else {
        if (max_context_len <= static_cast<int>(kReg2MaxSeqLen)) {
            status = cudaLaunchKernelEx(
                &config, topk_main_kernel<true, 0>, params);
        } else if (max_context_len <= static_cast<int>(kReg4MaxSeqLen)) {
            status = cudaLaunchKernelEx(
                &config, topk_main_kernel<true, 1>, params);
        } else {
            status = cudaLaunchKernelEx(
                &config, topk_main_kernel<true, 2>, params);
        }
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("clustered MQA top-k v2 launch failed: ") +
            cudaGetErrorString(status));
    }
}
