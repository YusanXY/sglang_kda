#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace {

void check_cu(CUresult result, const char* what) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::ostringstream oss;
    oss << what << " failed: " << (name ? name : "unknown")
        << " (" << (text ? text : "no detail") << ")";
    throw std::runtime_error(oss.str());
}

CUtensorMap make_tma_2d(
        void* base,
        CUtensorMapDataType dtype,
        uint64_t dim0,
        uint64_t dim1,
        uint64_t stride1_bytes,
        uint32_t box0,
        uint32_t box1,
        CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    CUtensorMap map{};
    const cuuint64_t global_dims[2] = {dim0, dim1};
    const cuuint64_t global_strides[1] = {stride1_bytes};
    const cuuint32_t box_dims[2] = {box0, box1};
    const cuuint32_t element_strides[2] = {1, 1};
    check_cu(
        cuTensorMapEncodeTiled(
            &map,
            dtype,
            2,
            base,
            global_dims,
            global_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled(2D)");
    return map;
}

CUtensorMap make_tma_3d(
        void* base,
        CUtensorMapDataType dtype,
        uint64_t dim0,
        uint64_t dim1,
        uint64_t dim2,
        uint64_t stride1_bytes,
        uint64_t stride2_bytes,
        uint32_t box0,
        uint32_t box1,
        uint32_t box2,
        CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    CUtensorMap map{};
    const cuuint64_t global_dims[3] = {dim0, dim1, dim2};
    const cuuint64_t global_strides[2] = {stride1_bytes, stride2_bytes};
    const cuuint32_t box_dims[3] = {box0, box1, box2};
    const cuuint32_t element_strides[3] = {1, 1, 1};
    check_cu(
        cuTensorMapEncodeTiled(
            &map,
            dtype,
            3,
            base,
            global_dims,
            global_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled(3D)");
    return map;
}

}  // namespace

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
    cudaStream_t stream);

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
    cudaStream_t stream);

torch::Tensor v2_forward_out(
        const torch::Tensor& q,
        const torch::Tensor& fused_kv,
        const torch::Tensor& weights,
        const torch::Tensor& context_lens,
        const torch::Tensor& block_table,
        const torch::Tensor& schedule_meta,
        const torch::Tensor& logits_full,
        int64_t max_context_len) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(fused_kv.device() == q.device(), "fused_kv device mismatch");
    TORCH_CHECK(weights.device() == q.device(), "weights device mismatch");
    TORCH_CHECK(context_lens.device() == q.device(), "context_lens device mismatch");
    TORCH_CHECK(block_table.device() == q.device(), "block_table device mismatch");
    TORCH_CHECK(schedule_meta.device() == q.device(), "schedule_meta device mismatch");
    TORCH_CHECK(logits_full.device() == q.device(), "logits workspace device mismatch");

    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "q must be float8_e4m3fn");
    TORCH_CHECK(fused_kv.scalar_type() == at::kByte, "fused_kv must be uint8");
    TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be float32");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    TORCH_CHECK(schedule_meta.scalar_type() == at::kInt, "schedule_meta must be int32");
    TORCH_CHECK(logits_full.scalar_type() == at::kFloat, "logits workspace must be float32");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(fused_kv.is_contiguous(), "fused_kv must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    TORCH_CHECK(block_table.stride(1) == 1, "block_table inner dimension must be contiguous");
    TORCH_CHECK(schedule_meta.is_contiguous(), "schedule_meta must be contiguous");
    TORCH_CHECK(logits_full.is_contiguous(), "logits workspace must be contiguous");

    TORCH_CHECK(q.dim() == 3, "q must have [M,H,D] layout");
    TORCH_CHECK(q.size(0) > 0 && q.size(0) % 16 == 0 &&
                q.size(1) == 64 && q.size(2) == 128,
                "clustered MQA requires M divisible by 16, H=64, D=128");
    TORCH_CHECK(fused_kv.dim() == 4 &&
                fused_kv.size(1) == 64 &&
                fused_kv.size(2) == 1 &&
                fused_kv.size(3) == 132,
                "fused_kv must have [num_pages,64,1,132] layout");
    TORCH_CHECK(fused_kv.stride(0) == 8448 &&
                fused_kv.stride(1) == 132 &&
                fused_kv.stride(3) == 1,
                "fused_kv must use the 8448-byte DeepGEMM page ABI");
    TORCH_CHECK(weights.dim() == 2 &&
                weights.size(0) == q.size(0) &&
                weights.size(1) == 64,
                "weights must have [M,64] layout");
    TORCH_CHECK(context_lens.dim() == 2 &&
                context_lens.size(0) == q.size(0) / 16 &&
                context_lens.size(1) == 16,
                "context_lens must have [M/16,16] layout");
    TORCH_CHECK(block_table.dim() == 2 &&
                block_table.size(0) == q.size(0) / 16,
                "block_table must have [M/16,max_pages] layout");
    TORCH_CHECK(schedule_meta.dim() == 2 &&
                schedule_meta.size(1) == 2 &&
                schedule_meta.size(0) >= 2,
                "schedule_meta must have [num_sms+1,2] layout");
    TORCH_CHECK(max_context_len > 0 &&
                max_context_len <= block_table.size(1) * 64,
                "max_context_len is outside block-table capacity");
    TORCH_CHECK(logits_full.dim() == 2 &&
                logits_full.size(0) == q.size(0) &&
                logits_full.size(1) >= max_context_len &&
                logits_full.size(1) % 256 == 0,
                "logits workspace must be [M, aligned_context>=max_context]");

    constexpr int kHeadDim = 128;
    constexpr int kHeads = 64;
    constexpr int kPage = 64;
    constexpr int kPageBytes = 8448;
    constexpr int kScaleOffsetBytes = kPage * kHeadDim;
    constexpr int kBlockQ = 8;
    constexpr int kQueriesPerQTma = 4;
    constexpr int kQueriesPerCluster = 16;

    const auto total_q = static_cast<int>(q.size(0));
    const auto num_pages = static_cast<int>(fused_kv.size(0));
    const auto padded_context = static_cast<int>(logits_full.size(1));

    auto* fused_base = static_cast<uint8_t*>(fused_kv.data_ptr());
    auto tensor_map_q = make_tma_2d(
        q.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        kHeadDim,
        static_cast<uint64_t>(total_q) * kHeads,
        static_cast<uint64_t>(q.stride(1)),
        kHeadDim,
        kQueriesPerQTma * kHeads,
        CU_TENSOR_MAP_SWIZZLE_128B);
    auto tensor_map_kv = make_tma_3d(
        fused_base,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        kHeadDim,
        kPage,
        num_pages,
        kHeadDim,
        kPageBytes,
        kHeadDim,
        kPage,
        1,
        CU_TENSOR_MAP_SWIZZLE_128B);
    auto tensor_map_sf_kv = make_tma_2d(
        fused_base + kScaleOffsetBytes,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        kPage,
        num_pages,
        kPageBytes,
        kPage,
        1);
    auto tensor_map_weights = make_tma_2d(
        weights.data_ptr(),
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        kHeads,
        total_q,
        static_cast<uint64_t>(weights.stride(0)) * sizeof(float),
        kHeads,
        kQueriesPerCluster);

    constexpr int kClusterSize = 2;
    const int num_clusters = static_cast<int>(schedule_meta.size(0) - 1);
    const int grid_size = num_clusters * kClusterSize;
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    launch_blockq8_tmem2(
        grid_size,
        total_q,
        padded_context,
        static_cast<int>(block_table.stride(0)),
        context_lens.data_ptr<int>(),
        logits_full.data_ptr<float>(),
        block_table.data_ptr<int>(),
        schedule_meta.data_ptr<int>(),
        tensor_map_q,
        tensor_map_sf_kv,
        tensor_map_kv,
        tensor_map_sf_kv,
        tensor_map_weights,
        stream);

    return logits_full.narrow(1, 0, max_context_len);
}

torch::Tensor v2_forward(
        const torch::Tensor& q,
        const torch::Tensor& fused_kv,
        const torch::Tensor& weights,
        const torch::Tensor& context_lens,
        const torch::Tensor& block_table,
        const torch::Tensor& schedule_meta,
        int64_t max_context_len) {
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0, "q must have nonempty [M,H,D] layout");
    const auto padded_context = static_cast<int>((max_context_len + 255) / 256 * 256);
    auto logits_full = torch::empty(
        {q.size(0), padded_context},
        q.options().dtype(torch::kFloat32));
    return v2_forward_out(
        q,
        fused_kv,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        logits_full,
        max_context_len);
}

void v2_forward_topk(
        const torch::Tensor& q,
        const torch::Tensor& fused_kv,
        const torch::Tensor& weights,
        const torch::Tensor& context_lens,
        const torch::Tensor& grouped_block_table,
        const torch::Tensor& schedule_meta,
        const torch::Tensor& logits_workspace,
        const torch::Tensor& page_table,
        const torch::Tensor& page_indices,
        const torch::Tensor& raw_indices,
        const torch::Tensor& positions,
        const torch::Tensor& query_start_loc,
        const torch::Tensor& full_seq_lens,
        const torch::Tensor& swa_gather_lens,
        const torch::Tensor& compressed_base,
        const torch::Tensor& swa_base,
        const torch::Tensor& combined_indices,
        const torch::Tensor& combined_lens,
        int64_t max_context_len) {
    auto logits = v2_forward_out(
        q,
        fused_kv,
        weights,
        context_lens,
        grouped_block_table,
        schedule_meta,
        logits_workspace,
        max_context_len);

    const auto same_device = [&q](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == q.device(), name, " device mismatch");
    };
    const auto int32_tensor = [&same_device](const torch::Tensor& tensor, const char* name) {
        same_device(tensor, name);
        TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    };
    int32_tensor(page_table, "page_table");
    int32_tensor(page_indices, "page_indices");
    int32_tensor(raw_indices, "raw_indices");
    int32_tensor(positions, "positions");
    int32_tensor(query_start_loc, "query_start_loc");
    int32_tensor(full_seq_lens, "full_seq_lens");
    int32_tensor(swa_gather_lens, "swa_gather_lens");
    int32_tensor(compressed_base, "compressed_base");
    int32_tensor(swa_base, "swa_base");
    int32_tensor(combined_indices, "combined_indices");
    int32_tensor(combined_lens, "combined_lens");

    const auto total_q = q.size(0);
    TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == total_q &&
                page_table.stride(1) == 1,
                "page_table must be strided [M,max_pages]");
    TORCH_CHECK(page_indices.dim() == 2 && page_indices.size(0) == total_q &&
                page_indices.size(1) == 512 && page_indices.is_contiguous(),
                "page_indices must be contiguous [M,512]");
    TORCH_CHECK(raw_indices.sizes() == page_indices.sizes() && raw_indices.is_contiguous(),
                "raw_indices must be contiguous [M,512]");
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) == total_q && positions.is_contiguous(),
                "positions must be contiguous [M]");
    TORCH_CHECK(full_seq_lens.dim() == 1 && full_seq_lens.is_contiguous(),
                "full_seq_lens must be contiguous [R]");
    const auto num_reqs = full_seq_lens.size(0);
    TORCH_CHECK(num_reqs > 0 && num_reqs <= 16,
                "clustered sparse-prefill supports 1..16 requests");
    TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == num_reqs + 1 &&
                query_start_loc.is_contiguous(),
                "query_start_loc must be contiguous [R+1]");
    for (const auto* tensor : {&swa_gather_lens, &compressed_base, &swa_base}) {
        TORCH_CHECK(tensor->dim() == 1 && tensor->size(0) == num_reqs && tensor->is_contiguous(),
                    "request epilogue tensors must be contiguous [R]");
    }
    TORCH_CHECK(combined_indices.dim() == 2 && combined_indices.size(0) == total_q &&
                combined_indices.size(1) >= 640 && combined_indices.stride(1) == 1,
                "combined_indices must be strided [M,>=640]");
    TORCH_CHECK(combined_lens.dim() == 1 && combined_lens.size(0) == total_q &&
                combined_lens.is_contiguous(),
                "combined_lens must be contiguous [M]");

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    launch_topk512_sparse_prefill(
        static_cast<int>(total_q),
        static_cast<int>(num_reqs),
        static_cast<int>(logits.stride(0)),
        static_cast<int>(page_table.stride(0)),
        static_cast<int>(combined_indices.stride(0)),
        logits.data_ptr<float>(),
        context_lens.data_ptr<int>(),
        page_table.data_ptr<int>(),
        page_indices.data_ptr<int>(),
        raw_indices.data_ptr<int>(),
        positions.data_ptr<int>(),
        query_start_loc.data_ptr<int>(),
        full_seq_lens.data_ptr<int>(),
        swa_gather_lens.data_ptr<int>(),
        compressed_base.data_ptr<int>(),
        swa_base.data_ptr<int>(),
        combined_indices.data_ptr<int>(),
        combined_lens.data_ptr<int>(),
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &v2_forward, "cluster2 Q16 cta_group::2 KV SM100 paged MQA logits");
    m.def(
        "forward_out",
        &v2_forward_out,
        "cluster2 Q16 paged MQA logits into a persistent device workspace");
    m.def(
        "forward_topk",
        &v2_forward_topk,
        "cluster2 Q16 paged MQA logits plus exact sparse-prefill top-k");
}
