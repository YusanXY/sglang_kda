from __future__ import annotations

import hashlib
import os
from pathlib import Path


_DSV4_JIT_SPEC = None
_DSV4_RAW_MODULE = None


_EXPECTED_LAUNCHER_SHA256 = (
    "bc8ed7c95c18265f4e57607d263263de86eeb754a12d741cdf74fcc23489961b"
)

_RUN_PROLOGUE = """  Array<Tensor> run(int64_t moe_tactic, bool enable_pdl = true,
                    bool use_routing_scales_on_input = false,
                    bool use_deep_seek_fp8 = false) override {
    check_routing();
    prepare_routing();

    // Execute routing
"""

_OVERLAPPED_RUN_PROLOGUE = """  Array<Tensor> run(int64_t moe_tactic, bool enable_pdl = true,
                    bool use_routing_scales_on_input = false,
                    bool use_deep_seek_fp8 = false) override {
    check_routing();
    prepare_routing();

    // The MoE runner, workspace and TMA descriptors only depend on launch
    // geometry and tensor pointers. Prepare them while the preceding GPU work
    // is still running, before submitting routing, so routing can flow into
    // the GEMMs without a CPU submission bubble.
    check_moe();
    prepare_moe(moe_tactic);

    // Execute routing
"""

_POST_ROUTING_PREPARE = """                       mRoutingLogitsDtype, norm_topk_prob, replay_ptr);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
"""

_POST_ROUTING_LAUNCH = """                       mRoutingLogitsDtype, norm_topk_prob, replay_ptr);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
"""

_MOE_RUNNER_MEMBER = (
    "  int64_t moe_tactic{-1};\n"
    "  std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner> moe_runner;"
)

_MOE_RUNNER_CONSTRUCTION = """    if (this->mDtypeAct == btg::Dtype::E4m3 && this->mDtypeWeights == btg::Dtype::E4m3 &&
        args->mUseDeepSeekFp8) {
      moe_runner = std::make_unique<RunnerType>(this->mDtypeWeights, args->mUseDeepSeekFp8,
                                                (int32_t)tile_tokens_dim, this->use_shuffled_weight,
                                                this->weight_layout, usePerTokenScalingGemm1,
                                                usePerTokenScalingGemm2, false, false);
    } else {
      moe_runner = std::make_unique<RunnerType>(
          this->mDtypeAct, this->mDtypeWeights, args->mUseDeepSeekFp8, (int32_t)tile_tokens_dim,
          this->activation_type, this->use_shuffled_weight, this->weight_layout,
          usePerTokenScalingGemm1, usePerTokenScalingGemm2);
    }
"""

_DSV4_MOE_RUNNER_CONSTRUCTION = """    auto make_runner = [&]() -> std::shared_ptr<RunnerType> {
      if (this->mDtypeAct == btg::Dtype::E4m3 &&
          this->mDtypeWeights == btg::Dtype::E4m3 && args->mUseDeepSeekFp8) {
        return std::make_shared<RunnerType>(
            this->mDtypeWeights, args->mUseDeepSeekFp8,
            static_cast<int32_t>(tile_tokens_dim), this->use_shuffled_weight,
            this->weight_layout, usePerTokenScalingGemm1,
            usePerTokenScalingGemm2, false, false);
      }
      return std::make_shared<RunnerType>(
          this->mDtypeAct, this->mDtypeWeights, args->mUseDeepSeekFp8,
          static_cast<int32_t>(tile_tokens_dim), this->activation_type,
          this->use_shuffled_weight, this->weight_layout,
          usePerTokenScalingGemm1, usePerTokenScalingGemm2);
    };
    if (dsv4_shared_finalize_state.active) {
      auto& persistent = dsv4_persistent_moe_state;
      int const act_dtype = static_cast<int>(this->mDtypeAct);
      int const weight_dtype = static_cast<int>(this->mDtypeWeights);
      int const activation = static_cast<int>(this->activation_type);
      int const layout = static_cast<int>(this->weight_layout);
      bool const runner_matches =
          persistent.runner &&
          persistent.runner_tile == tile_tokens_dim &&
          persistent.runner_act_dtype == act_dtype &&
          persistent.runner_weight_dtype == weight_dtype &&
          persistent.runner_activation == activation &&
          persistent.runner_layout == layout &&
          persistent.runner_deepseek_fp8 == args->mUseDeepSeekFp8 &&
          persistent.runner_scale_gemm1 == usePerTokenScalingGemm1 &&
          persistent.runner_scale_gemm2 == usePerTokenScalingGemm2;
      if (!runner_matches) {
        persistent.runner = make_runner();
        persistent.runner_tile = tile_tokens_dim;
        persistent.runner_act_dtype = act_dtype;
        persistent.runner_weight_dtype = weight_dtype;
        persistent.runner_activation = activation;
        persistent.runner_layout = layout;
        persistent.runner_deepseek_fp8 = args->mUseDeepSeekFp8;
        persistent.runner_scale_gemm1 = usePerTokenScalingGemm1;
        persistent.runner_scale_gemm2 = usePerTokenScalingGemm2;
        // A tactic and its byte workspaces are valid only for the selected
        // runner geometry.  Device tensor variants remain cached separately.
        persistent.num_tokens = -1;
        persistent.tactic_ready = false;
        persistent.tactic = -1;
        persistent.workspace_sizes_ready = false;
        persistent.workspace_fc1_bytes = 0;
        persistent.workspace_fc2_bytes = 0;
      }
      moe_runner = persistent.runner;
    } else {
      moe_runner = make_runner();
    }
"""

_MOE_TACTIC_AND_WORKSPACE = """    if (moe_tactic == -1) {
      moe_tactic = moe_runner->getDefaultValidConfigIndex(
          args->top_k, args->hidden_size, args->intermediate_size, args->local_num_experts,
          args->num_tokens);
    }
    auto valid_cfgs =
        moe_runner->getValidConfigIndices(args->top_k, args->hidden_size, args->intermediate_size,
                                          args->local_num_experts, args->num_tokens);
    auto valid_it = std::find(valid_cfgs.begin(), valid_cfgs.end(), moe_tactic);
    FLASHINFER_CHECK(valid_it != valid_cfgs.end(), "Invalid MoE tactic ", moe_tactic,
                     " for tile_N=", tile_tokens_dim, ". Number of valid tactics for this tile is ",
                     valid_cfgs.size(),
                     ". This often indicates a stale or mismatched autotuner cache entry.");
    this->moe_tactic = moe_tactic;

    auto workspace_sizes = moe_runner->getWorkspaceSizeInBytes(*args, moe_tactic);
    workspace_fc1 = alloc_tensor({std::get<0>(workspace_sizes)}, dl_int8, hidden_states.device());
    workspace_fc2 = alloc_tensor({std::get<1>(workspace_sizes)}, dl_int8, hidden_states.device());
"""

_DSV4_MOE_TACTIC_AND_WORKSPACE = """    auto& persistent = dsv4_persistent_moe_state;
    bool const use_persistent = dsv4_shared_finalize_state.active;
    // Tactic validity and workspace byte counts depend on aggregate M.  Prefix
    // construction and the true req=16 batch deliberately alternate between
    // M=4096 and M=65536, so retain the runner but refresh the shape-dependent
    // planning state whenever M changes.
    if (use_persistent && persistent.num_tokens != args->num_tokens) {
      persistent.num_tokens = args->num_tokens;
      persistent.tactic_ready = false;
      persistent.tactic = -1;
      persistent.workspace_sizes_ready = false;
      persistent.workspace_fc1_bytes = 0;
      persistent.workspace_fc2_bytes = 0;
    }
    if (use_persistent && persistent.tactic_ready) {
      if (moe_tactic != -1) {
        TVM_FFI_ICHECK_EQ(moe_tactic, persistent.tactic);
      }
      moe_tactic = persistent.tactic;
    } else {
      if (moe_tactic == -1) {
        moe_tactic = moe_runner->getDefaultValidConfigIndex(
            args->top_k, args->hidden_size, args->intermediate_size,
            args->local_num_experts, args->num_tokens);
      }
      auto valid_cfgs = moe_runner->getValidConfigIndices(
          args->top_k, args->hidden_size, args->intermediate_size,
          args->local_num_experts, args->num_tokens);
      auto valid_it = std::find(valid_cfgs.begin(), valid_cfgs.end(), moe_tactic);
      FLASHINFER_CHECK(
          valid_it != valid_cfgs.end(), "Invalid MoE tactic ", moe_tactic,
          " for tile_N=", tile_tokens_dim,
          ". Number of valid tactics for this tile is ", valid_cfgs.size(),
          ". This often indicates a stale or mismatched autotuner cache entry.");
      if (use_persistent) {
        persistent.tactic = moe_tactic;
        persistent.tactic_ready = true;
      }
    }
    this->moe_tactic = moe_tactic;

    int64_t workspace_fc1_bytes;
    int64_t workspace_fc2_bytes;
    if (use_persistent && persistent.workspace_sizes_ready) {
      workspace_fc1_bytes = persistent.workspace_fc1_bytes;
      workspace_fc2_bytes = persistent.workspace_fc2_bytes;
    } else {
      auto workspace_sizes = moe_runner->getWorkspaceSizeInBytes(*args, moe_tactic);
      workspace_fc1_bytes = static_cast<int64_t>(std::get<0>(workspace_sizes));
      workspace_fc2_bytes = static_cast<int64_t>(std::get<1>(workspace_sizes));
      if (use_persistent) {
        persistent.workspace_fc1_bytes = workspace_fc1_bytes;
        persistent.workspace_fc2_bytes = workspace_fc2_bytes;
        persistent.workspace_sizes_ready = true;
      }
    }
    workspace_fc1 = alloc_tensor({workspace_fc1_bytes}, dl_int8, hidden_states.device());
    workspace_fc2 = alloc_tensor({workspace_fc2_bytes}, dl_int8, hidden_states.device());
"""

_INCLUDE_ANCHOR = "#include <cuda_runtime.h>\n"
_NAMESPACE_ANCHOR = "namespace flashinfer {\n"
_EXPORT_ANCHOR = (
    "TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_moe, "
    "trtllm_fp4_block_scale_moe);\n"
)

_DSV4_FUSED_FINALIZE_SUPPORT = r"""
struct Dsv4SharedFinalizeState {
  __nv_bfloat16 const* shared_output = nullptr;
  int64_t num_tokens = 0;
  int64_t hidden_dim = 0;
  float routed_scale = 1.0f;
  bool active = false;
};

// Each TP rank owns one Python execution thread and one CUDA device.  Keeping
// this state thread-local lets the model provide the shared-expert tensor
// before entering the existing TRTLLM launcher without adding it to the
// upstream FlashInfer public ABI.
thread_local Dsv4SharedFinalizeState dsv4_shared_finalize_state;

struct Dsv4PersistentMoeState {
  size_t tensor_cursor = 0;
  // A logical allocator slot can request different routing-dependent shapes
  // across decoder layers.  Cache each shape variant on device instead of
  // reallocating every call or incorrectly forcing the first layer's shape on
  // all following layers.
  std::vector<std::vector<Tensor>> tensor_variants;
  std::shared_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner> runner;
  int64_t runner_tile = -1;
  int runner_act_dtype = -1;
  int runner_weight_dtype = -1;
  int runner_activation = -1;
  int runner_layout = -1;
  bool runner_deepseek_fp8 = false;
  bool runner_scale_gemm1 = false;
  bool runner_scale_gemm2 = false;
  int64_t num_tokens = -1;
  bool tactic_ready = false;
  int64_t tactic = -1;
  bool workspace_sizes_ready = false;
  int64_t workspace_fc1_bytes = 0;
  int64_t workspace_fc2_bytes = 0;
};

thread_local Dsv4PersistentMoeState dsv4_persistent_moe_state;

void dsv4_begin_persistent_moe_call() {
  dsv4_persistent_moe_state.tensor_cursor = 0;
}

Tensor dsv4_alloc_tensor(tvm::ffi::Shape shape, DLDataType dtype, DLDevice device) {
  if (!dsv4_shared_finalize_state.active) {
    return alloc_tensor(shape, dtype, device);
  }
  auto& persistent = dsv4_persistent_moe_state;
  size_t const slot = persistent.tensor_cursor++;
  if (slot == persistent.tensor_variants.size()) {
    persistent.tensor_variants.emplace_back();
  }
  auto& variants = persistent.tensor_variants[slot];
  for (Tensor const& tensor : variants) {
    bool matches = tensor.ndim() == shape.size();
    for (int i = 0; matches && i < tensor.ndim(); ++i) {
      matches = tensor.size(i) == shape[i];
    }
    matches = matches && tensor.dtype() == dtype;
    matches = matches && tensor.device().device_type == device.device_type;
    matches = matches && tensor.device().device_id == device.device_id;
    if (matches) {
      return tensor;
    }
  }
  variants.push_back(alloc_tensor(shape, dtype, device));
  return variants.back();
}

void dsv4_set_shared_finalize(TensorView shared_output, double routed_scale) {
  TVM_FFI_ICHECK(shared_output.device().device_type == kDLCUDA);
  TVM_FFI_ICHECK((shared_output.dtype() == DLDataType{kDLBfloat, 16, 1}));
  TVM_FFI_ICHECK_EQ(shared_output.ndim(), 2);
  TVM_FFI_ICHECK(shared_output.IsContiguous());
  dsv4_shared_finalize_state.shared_output =
      static_cast<__nv_bfloat16 const*>(shared_output.data_ptr());
  dsv4_shared_finalize_state.num_tokens = shared_output.size(0);
  dsv4_shared_finalize_state.hidden_dim = shared_output.size(1);
  dsv4_shared_finalize_state.routed_scale = static_cast<float>(routed_scale);
  dsv4_shared_finalize_state.active = true;
}

struct alignas(16) Dsv4Bf16x8 {
  __nv_bfloat16 value[8];
};

__device__ __forceinline__ float dsv4_weight_to_float(float value) {
  return value;
}

__device__ __forceinline__ float dsv4_weight_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename WeightType>
__global__ void dsv4MoeFinalizeSharedKernel(
    int num_tokens,
    int hidden_dim,
    int hidden_dim_padded,
    int top_k,
    __nv_bfloat16 const* __restrict__ gemm2_output,
    int const* __restrict__ expanded_to_permuted,
    WeightType const* __restrict__ expert_weights,
    __nv_bfloat16 const* __restrict__ shared_output,
    float routed_scale,
    __nv_bfloat16* __restrict__ output) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  constexpr int kMaxTopK = 64;
  int const token = static_cast<int>(blockIdx.x);
  __shared__ int permuted[kMaxTopK];
  __shared__ float weights[kMaxTopK];
  for (int k = threadIdx.x; k < top_k; k += blockDim.x) {
    int const expanded = token * top_k + k;
    permuted[k] = expanded_to_permuted[expanded];
    weights[k] = dsv4_weight_to_float(expert_weights[expanded]);
  }
  __syncthreads();

  int const hidden_vecs = hidden_dim / 8;
  int const padded_vecs = hidden_dim_padded / 8;
  auto const* gemm_vec = reinterpret_cast<Dsv4Bf16x8 const*>(gemm2_output);
  auto const* shared_vec = reinterpret_cast<Dsv4Bf16x8 const*>(shared_output);
  auto* output_vec = reinterpret_cast<Dsv4Bf16x8*>(output);
  for (int vec = threadIdx.x; vec < hidden_vecs; vec += blockDim.x) {
    float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < top_k; ++k) {
      int const source = permuted[k];
      if (source < 0) continue;
      float const weight = weights[k];
      Dsv4Bf16x8 const values = gemm_vec[source * padded_vecs + vec];
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        accum[element] += weight * __bfloat162float(values.value[element]);
      }
    }
    Dsv4Bf16x8 const shared = shared_vec[token * hidden_vecs + vec];
    Dsv4Bf16x8 result;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      result.value[element] = __float2bfloat16_rn(
          accum[element] * routed_scale + __bfloat162float(shared.value[element]));
    }
    output_vec[token * hidden_vecs + vec] = result;
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void dsv4_launch_shared_finalize(
    Tensor const& gemm2_output,
    TensorView expert_weights,
    Tensor const& expanded_to_permuted,
    TensorView output,
    int top_k,
    bool enable_pdl) {
  auto const& state = dsv4_shared_finalize_state;
  TVM_FFI_ICHECK(state.active);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);
  TVM_FFI_ICHECK_EQ(output.size(0), state.num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), state.hidden_dim);
  TVM_FFI_ICHECK_EQ(output.size(1) % 8, 0);
  bool const weights_are_bf16 =
      expert_weights.dtype() == DLDataType{kDLBfloat, 16, 1};
  bool const weights_are_fp32 =
      expert_weights.dtype() == DLDataType{kDLFloat, 32, 1};
  TVM_FFI_ICHECK(weights_are_bf16 || weights_are_fp32)
      << "DSV4 fused finalize expects BF16 or FP32 routing weights";
  TVM_FFI_ICHECK((expanded_to_permuted.dtype() == DLDataType{kDLInt, 32, 1}));
  TVM_FFI_ICHECK_LE(top_k, 64);

  cudaStream_t const stream = get_stream(output.device());
  if (weights_are_bf16) {
    dsv4MoeFinalizeSharedKernel<__nv_bfloat16>
        <<<state.num_tokens, 256, 0, stream>>>(
            static_cast<int>(state.num_tokens),
            static_cast<int>(state.hidden_dim),
            static_cast<int>(gemm2_output.size(1)), top_k,
            static_cast<__nv_bfloat16 const*>(gemm2_output.data_ptr()),
            static_cast<int const*>(expanded_to_permuted.data_ptr()),
            static_cast<__nv_bfloat16 const*>(expert_weights.data_ptr()),
            state.shared_output, state.routed_scale,
            static_cast<__nv_bfloat16*>(output.data_ptr()));
  } else {
    dsv4MoeFinalizeSharedKernel<float>
        <<<state.num_tokens, 256, 0, stream>>>(
            static_cast<int>(state.num_tokens),
            static_cast<int>(state.hidden_dim),
            static_cast<int>(gemm2_output.size(1)), top_k,
            static_cast<__nv_bfloat16 const*>(gemm2_output.data_ptr()),
            static_cast<int const*>(expanded_to_permuted.data_ptr()),
            static_cast<float const*>(expert_weights.data_ptr()),
            state.shared_output, state.routed_scale,
            static_cast<__nv_bfloat16*>(output.data_ptr()));
  }
  cudaError_t const error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess)
      << "dsv4 fused MoE finalize launch failed: " << cudaGetErrorString(error);
}
"""

_FP4_MULTI_TILE_START = """  // Determine supported tile sizes
  std::vector<int32_t> mSupportedTileN = FP4BlockScaleLauncher::getSupportedTileNums(mDtypeAct);
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
"""

_FP4_MULTI_TILE_END = """  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl);
"""

_FP4_SINGLE_TILE_BLOCK = """  // Resolve the autotuner tactic first. Huge mode is strict and only
  // needs the selected launcher, so do not construct and destroy launchers for every
  // supported tile on every decoder layer.
  std::vector<int32_t> mSupportedTileN = FP4BlockScaleLauncher::getSupportedTileNums(mDtypeAct);
  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
  args->num_tokens = num_tokens;
  args->num_experts = num_experts;
  args->hidden_size = hidden_size;
  args->hidden_size_output = output.size(1);
  args->top_k = top_k;
  args->n_group = n_group.value_or(0);
  args->topk_group = topk_group.value_or(0);
  args->local_expert_offset = local_expert_offset;
  args->local_num_experts = local_num_experts;
  args->intermediate_size = intermediate_size;
  args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
  bool const dsv4_fuse_shared = dsv4_shared_finalize_state.active;
  TVM_FFI_ICHECK(!dsv4_fuse_shared || do_finalize)
      << "DSV4 shared finalize requires do_finalize=true";
  args->do_finalize = do_finalize && !dsv4_fuse_shared;
  args->output = output.data_ptr();
  args->output_scale = nullptr;

  if (dsv4_fuse_shared) {
    dsv4_begin_persistent_moe_call();
  }

  auto launcher = std::make_unique<FP4BlockScaleLauncher>(
      static_cast<RoutingInputMode>(routing_input_mode), routing_logits, routing_bias,
      hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm1_bias,
      gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
      output1_scales_scalar, output1_scales_gate_scalar, output2_scales_scalar, per_token_scales,
      topk_ids, topk_weights);
  launcher->init(std::move(args), static_cast<int32_t>(tile_N), routing_method_type,
                 /*use_shuffled_weight=*/true, /*weight_layout=*/0,
                 static_cast<ActivationType>(act_type), mDtypeAct, mDtypeWeights, norm_topk_prob);
  launcher->set_routing_replay_out(routing_replay_out);
  Array<Tensor> result = launcher->run(config, enable_pdl);
  if (dsv4_fuse_shared) {
    TVM_FFI_ICHECK_EQ(result.size(), 3);
    dsv4_launch_shared_finalize(
        result[0], topk_weights, result[2], output, top_k, enable_pdl);
    // The Python wrapper owns `output` and returns it whenever the public
    // do_finalize argument is true; the intermediate array is ignored there.
    return result;
  }
  return result;
"""


def _patch_launcher(source: str) -> str:
    if source.count(_INCLUDE_ANCHOR) != 1:
        raise RuntimeError("DSV4 MoE patch expected one cuda_runtime include")
    source = source.replace(
        _INCLUDE_ANCHOR,
        _INCLUDE_ANCHOR + "#include <cuda_bf16.h>\n#include <memory>\n",
    )
    if source.count("alloc_tensor(") != 48:
        raise RuntimeError("DSV4 MoE workspace patch expected 48 tensor allocations")
    # Redirect launcher-local temporary allocations into a model-thread scoped
    # arena while Huge is active. The helper itself is injected afterwards, so
    # its fallback call remains the original FlashInfer allocator.
    source = source.replace("alloc_tensor(", "dsv4_alloc_tensor(")
    if source.count(_NAMESPACE_ANCHOR) != 1:
        raise RuntimeError("DSV4 MoE patch expected one flashinfer namespace")
    source = source.replace(
        _NAMESPACE_ANCHOR,
        _NAMESPACE_ANCHOR + _DSV4_FUSED_FINALIZE_SUPPORT,
    )
    if source.count(_MOE_RUNNER_MEMBER) != 1:
        raise RuntimeError("DSV4 MoE runner patch expected one runner member")
    source = source.replace(
        _MOE_RUNNER_MEMBER,
        "  int64_t moe_tactic{-1};\n"
        "  std::shared_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner> moe_runner;",
    )
    if source.count(_MOE_RUNNER_CONSTRUCTION) != 1:
        raise RuntimeError("DSV4 MoE runner patch found unexpected construction code")
    source = source.replace(_MOE_RUNNER_CONSTRUCTION, _DSV4_MOE_RUNNER_CONSTRUCTION)
    # The allocation calls in this anchor were renamed above as part of the
    # complete 48-call arena redirection.
    tactic_workspace = _MOE_TACTIC_AND_WORKSPACE.replace(
        "alloc_tensor(", "dsv4_alloc_tensor("
    )
    replacement = _DSV4_MOE_TACTIC_AND_WORKSPACE.replace(
        "alloc_tensor(", "dsv4_alloc_tensor("
    )
    if source.count(tactic_workspace) != 1:
        raise RuntimeError("DSV4 MoE tactic/workspace patch found unexpected code")
    source = source.replace(tactic_workspace, replacement)
    if source.count(_RUN_PROLOGUE) != 1:
        raise RuntimeError(
            "DSV4 MoE overlap patch expected exactly one FP4 run prologue"
        )
    if source.count(_POST_ROUTING_PREPARE) != 1:
        raise RuntimeError(
            "DSV4 MoE overlap patch expected exactly one FP4 post-routing prepare"
        )
    source = source.replace(_RUN_PROLOGUE, _OVERLAPPED_RUN_PROLOGUE).replace(
        _POST_ROUTING_PREPARE, _POST_ROUTING_LAUNCH
    )

    if source.count(_FP4_MULTI_TILE_START) != 1:
        raise RuntimeError(
            "DSV4 MoE overlap patch expected exactly one FP4 multi-tile prologue"
        )
    start = source.index(_FP4_MULTI_TILE_START)
    end = source.find(_FP4_MULTI_TILE_END, start)
    if end < 0:
        raise RuntimeError("DSV4 MoE overlap patch could not find FP4 multi-tile epilogue")
    end += len(_FP4_MULTI_TILE_END)
    original_block = source[start:end]
    required_fragments = (
        "for (int32_t curr_tile_N : mSupportedTileN)",
        "std::make_unique<FP4BlockScaleLauncher>",
        "resolveMoeTileAndConfig(config_index",
        "launchers_map.find",
    )
    if any(original_block.count(fragment) != 1 for fragment in required_fragments):
        raise RuntimeError("DSV4 MoE overlap patch found an unexpected FP4 launcher block")
    source = source[:start] + _FP4_SINGLE_TILE_BLOCK + source[end:]
    if source.count(_EXPORT_ANCHOR) != 1:
        raise RuntimeError("DSV4 MoE patch expected one FP4 export")
    return source.replace(
        _EXPORT_ANCHOR,
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv4_set_shared_finalize, "
        "dsv4_set_shared_finalize);\n" + _EXPORT_ANCHOR,
    )


def _get_patched_launcher(flashinfer_csrc_dir: Path) -> Path:
    source_path = flashinfer_csrc_dir / "trtllm_fused_moe_kernel_launcher.cu"
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != _EXPECTED_LAUNCHER_SHA256:
        raise RuntimeError(
            "Unsupported FlashInfer MoE launcher for DSV4 Huge overlap: "
            f"expected sha256={_EXPECTED_LAUNCHER_SHA256}, got {source_sha256} "
            f"from {source_path}"
        )

    patched = _patch_launcher(source_bytes.decode("utf-8")).encode("utf-8")
    patched_sha256 = hashlib.sha256(patched).hexdigest()
    output_dir = (
        Path.home()
        / ".cache"
        / "sglang"
        / "dsv4_moe_overlap"
        / patched_sha256[:16]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    if output_path.exists() and output_path.read_bytes() == patched:
        return output_path

    temporary_path = output_dir / f"{source_path.name}.{os.getpid()}.tmp"
    temporary_path.write_bytes(patched)
    os.replace(temporary_path, output_path)
    return output_path


def gen_dsv4_trtllm_gen_fused_moe_sm100_module():
    global _DSV4_JIT_SPEC
    import flashinfer
    from flashinfer.artifacts import ArtifactPath, CheckSumHash
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec
    from flashinfer.jit.cubin_loader import (
        ensure_symlink,
        get_artifact,
        get_meta_hash,
        verify_symlinked_headers,
    )
    from flashinfer.jit.fused_moe import BMM_EXPORT_HEADERS

    flashinfer_data_dir = Path(flashinfer.__file__).resolve().parent / "data"
    flashinfer_csrc_dir = flashinfer_data_dir / "csrc"
    patched_launcher = _get_patched_launcher(flashinfer_csrc_dir)

    include_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/include"
    checksum_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/checksums.txt"
    checksum = get_artifact(checksum_path, CheckSumHash.TRTLLM_GEN_BMM)
    assert checksum, f"Failed to get checksums.txt from {checksum_path}"
    meta_hash = get_meta_hash(checksum)

    header_name = "flashinferMetaInfo"
    metainfo = get_artifact(f"{include_path}/{header_name}.h", meta_hash)
    assert metainfo, f"{header_name}.h not found"

    bmm_export_path = f"{include_path}/trtllmGen_bmm_export"
    for header in BMM_EXPORT_HEADERS:
        artifact = get_artifact(
            f"{bmm_export_path}/{header}", get_meta_hash(checksum, header)
        )
        assert artifact, f"{header} not found"

    # flashinfer-cubin is commonly installed in a root-owned, read-only wheel.
    # Keep the packaged artifacts immutable and stage only the include symlink
    # in this process's writable JIT workspace.
    cubin_include_root = (
        jit_env.FLASHINFER_WORKSPACE_DIR / "dsv4_cubin_include"
    )
    symlink_path = (
        cubin_include_root
        / "flashinfer"
        / "trtllm"
        / "batched_gemm"
        / "trtllmGen_bmm_export"
    )
    ensure_symlink(symlink_path, jit_env.FLASHINFER_CUBIN_DIR / bmm_export_path)
    verify_symlinked_headers(symlink_path, BMM_EXPORT_HEADERS, checksum)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 12]
    )

    _DSV4_JIT_SPEC = gen_jit_spec(
        "sgl_dsv4_fused_moe_trtllm_sm100_overlap",
        [
            flashinfer_csrc_dir / "nv_internal/cpp/kernels/quantization.cu",
            flashinfer_csrc_dir / "nv_internal/cpp/common/envUtils.cpp",
            flashinfer_csrc_dir / "nv_internal/cpp/common/logger.cpp",
            flashinfer_csrc_dir / "nv_internal/cpp/common/stringUtils.cpp",
            flashinfer_csrc_dir / "nv_internal/cpp/common/tllmException.cpp",
            flashinfer_csrc_dir / "nv_internal/cpp/common/memoryUtils.cu",
            patched_launcher,
            flashinfer_csrc_dir / "trtllm_fused_moe_runner.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_llama4.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu",
            flashinfer_csrc_dir
            / "fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu",
            flashinfer_csrc_dir / "trtllm_batched_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_GEN_EXPORT_FLASHINFER",
            "-DTLLM_ENABLE_CUDA",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_BMM}\\"',
        ]
        + nvcc_flags,
        extra_include_paths=[
            cubin_include_root,
            jit_env.FLASHINFER_CUBIN_DIR / include_path,
            flashinfer_csrc_dir / "nv_internal",
            flashinfer_csrc_dir / "nv_internal/include",
        ],
    )
    return _DSV4_JIT_SPEC


def set_dsv4_shared_finalize(shared_output, routed_scaling_factor: float) -> None:
    global _DSV4_RAW_MODULE
    if _DSV4_RAW_MODULE is None:
        # The FlashInfer wrapper owns cubin-loader initialization and invokes
        # our monkey-patched generator. Reusing that exact JIT spec exposes the
        # additional strict-Huge setter without loading a second CUDA module.
        from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

        get_trtllm_moe_sm100_module()
        if _DSV4_JIT_SPEC is None:
            raise RuntimeError("DSV4 Huge MoE JIT module was not initialized")
        _DSV4_RAW_MODULE = _DSV4_JIT_SPEC.build_and_load()
    _DSV4_RAW_MODULE.dsv4_set_shared_finalize(
        shared_output, float(routed_scaling_factor)
    )
