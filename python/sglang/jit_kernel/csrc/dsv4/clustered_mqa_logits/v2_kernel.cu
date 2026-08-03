#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "v2_sm100_mqa_logits.cuh"
#define SGLANG_DSV4_TOPK_DEVICE_ONLY
#include "../../deepseek_v4/topk_v1.cuh"
#undef SGLANG_DSV4_TOPK_DEVICE_ONLY

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
    static_assert(kTopK == 512 && kTopKBlockSize == 512);
    constexpr auto kernel = topk_transform_kernel<false>;
    constexpr int smem_bytes = kSMEM + sizeof(int32_t);
    static const auto setup_status = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);
    if (setup_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("top-k cudaFuncSetAttribute failed: ") +
            cudaGetErrorString(setup_status));
    }

    const auto params = TopKParams{
        logits,
        seq_lens,
        page_table,
        page_indices,
        raw_indices,
        positions,
        query_start_loc,
        full_seq_lens,
        swa_gather_lens,
        compressed_base,
        swa_base,
        combined_indices,
        combined_lens,
        logits_stride,
        page_table_stride,
        combined_indices_stride,
        static_cast<uint32_t>(num_reqs),
        6,
    };
    topk_transform_kernel<false>
        <<<batch_size, kTopKBlockSize, smem_bytes, stream>>>(params);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("clustered MQA top-k launch failed: ") +
            cudaGetErrorString(status));
    }
}
