#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "v2_sm100_mqa_logits.cuh"

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
