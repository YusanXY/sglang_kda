#pragma once

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>

#include "v2_mqa_logits_layout.cuh"
#include "v2_sm100_paged_mqa_logits.cuh"

// Shared SM100 MQA logits core plus contiguous-KV and paged entries
// Both entries use the same q / sf_q / kv / sf_kv / weights TMA signature

namespace deep_gemm {

// Ring-buffer counter avoiding `% kNumStages`, which ptxas can lower poorly for TMEM paths
template <uint32_t kNumStages>
struct RingPipeline {
    uint32_t stage_idx = 0, phase = 0;

    CUTLASS_DEVICE cute::tuple<uint32_t, uint32_t> advance(const uint32_t& step = 1) {
        const uint32_t current_stage_idx = stage_idx, current_phase = phase;
        stage_idx += step;
        if (stage_idx >= kNumStages) {
            stage_idx -= kNumStages;
            phase ^= 1;
        }
        return {current_stage_idx, current_phase};
    }
};

// Convert runtime valid-token count to `cute::Int` so token loops stay compile-time constant
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens, Fn&& fn) {
    if constexpr (kCandidate <= 1) {
        fn(cute::Int<1>{});
    } else if (num_block_tokens >= kCandidate) {
        fn(cute::Int<kCandidate>{});
    } else {
        dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens, static_cast<Fn&&>(fn));
    }
}

// Shared device core parameterized by dtype and scheduler geometry/addressing
template <bool kIsFP4, uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t, typename reduce_dtype_t, typename MakeScheduler,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          uint32_t kNumKVPeers = 1>
CUTLASS_DEVICE void sm100_mqa_logits_core_impl(const uint32_t logits_stride,
                                               logits_dtype_t* logits,
                                               const cute::TmaDescriptor& tensor_map_q,
                                               const cute::TmaDescriptor& tensor_map_sf_q,
                                               const cute::TmaDescriptor& tensor_map_kv,
                                               const cute::TmaDescriptor& tensor_map_sf_kv,
                                               const cute::TmaDescriptor& tensor_map_weights,
                                               const MakeScheduler& make_scheduler) {
    const auto sm_idx = blockIdx.x;
    const auto cluster_rank = cute::block_rank_in_cluster();
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    // Every cta_group::2 pass consumes two query rows from each rank, so N256
    // uses 256 TMEM columns.  Two stages fill the architectural 512-column
    // allocation and alternate across the four Q-pair passes.
    static constexpr uint32_t kNumTmemStages = 2;
    static constexpr uint32_t kQueriesPerMMA = 4;
    static constexpr uint32_t kQueriesPerRankPerMMA = kQueriesPerMMA / 2;
    static constexpr uint32_t kNumQPairs = BLOCK_Q / kQueriesPerRankPerMMA;
    static constexpr uint32_t kQueriesPerQTma = 4;
    static constexpr uint32_t kNumQTmas = BLOCK_Q / kQueriesPerQTma;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    // One cta_group::2 instruction gathers two local M128 halves into M256.
    // Four N256 instructions reuse that same KV tile to cover the cluster's
    // sixteen queries (two local query rows from each rank per instruction).
    static constexpr uint32_t UMMA_M = 256;
    static constexpr uint32_t UMMA_N = kQueriesPerMMA * kNumHeads;
    static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    static constexpr uint32_t kNumSFQ  = kIsFP4 ? math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kNumSFKV = kIsFP4 ? math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t kLocalSplitKV = SPLIT_KV / 2;
    static constexpr uint32_t kQueriesPerCluster = BLOCK_Q * 2;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = kLocalSplitKV * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kIsFP4 ? (kRealNumSFQ * sizeof(int)) : 0;
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kIsFP4 ? (kNumSFKV * sizeof(int)) : (kLocalSplitKV * sizeof(float));
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kQueriesPerCluster * kNumHeads * sizeof(reduce_dtype_t);

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(kNumKVPeers == 2, "V25 requires a two-CTA cluster");
    DG_STATIC_ASSERT(kNumMathWarpGroups == 2 and kNumMathThreads == 256,
                     "V25 alternates four Q pairs across two consumer groups");
    DG_STATIC_ASSERT(SPLIT_KV == UMMA_M and kLocalSplitKV == 128,
                     "V25 requires a logical M256 split with M128 per CTA");
    DG_STATIC_ASSERT(BLOCK_Q == 8 and kNumQPairs == 4,
                     "V25 requires Q8 per CTA and four N256 passes");

    using SharedStorage = layout::MQALogitsSharedStorage<kIsFP4, kNumHeads, kHeadDim, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, reduce_dtype_t>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            smem.full_q_barriers[i].init(1);
            smem.empty_q_barriers[i].init(kNumMathThreads + 32);
            smem.q_peer_full_barriers[i].init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv_barriers[i].init(1);
            smem.empty_kv_barriers[i].init(kIsFP4 ? 1 : kNumMathThreads);
            smem.kv_peer_full_barriers[i].init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            smem.full_tmem_barriers[i].init(1);
            smem.empty_tmem_barriers[i].init(128);
            smem.tmem_peer_empty_barriers[i].init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    // Allocator2Sm is a paired consumer-warp operation.  Both CTAs must
    // execute it from the same warp id; use warp 0 exactly as the validated
    // CUTLASS cta_group::2 pipeline does.
    if (warp_idx == 0)
        cute::TMEM::Allocator2Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
    __syncthreads();
    cute::cluster_sync();

    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    constexpr uint32_t kNumSpecializedRegisters = 56;
    // Each math warpgroup owns one TMEM stage and consumes alternating Q
    // pairs.  Only one epilogue's accumulator state is live at a time.
    constexpr uint32_t kNumMathRegisters = 168;

    // The paged scheduling metadata is produced by the dependency-launched
    // metadata kernel.  Every CTA must wait before constructing a scheduler;
    // otherwise the two ranks can observe different loop bounds and deadlock
    // at the cluster/TMEM handshakes.
    cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
            // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

                const uint32_t q_token_base = scheduler.get_q_tma_token_base(q_block_idx);
                #pragma unroll
                for (uint32_t q_tma = 0; q_tma < kNumQTmas; ++q_tma) {
                    tma::copy<kNumQKBytesPerToken,
                              kQueriesPerQTma * kNumHeads, 0>(
                        &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                        smem.smem_q[q_stage_idx] +
                            q_tma * kQueriesPerQTma * kNumHeads *
                                kNumQKBytesPerToken,
                        0,
                        (q_token_base + q_tma * kQueriesPerQTma) *
                            kNumHeads);
                }
                if constexpr (kIsFP4)
                    tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx], smem.smem_sf_q[q_stage_idx], 0, q_token_base);
                const uint32_t cluster_q_token_base = scheduler.get_cluster_q_tma_token_base(q_block_idx);
                tma::copy<kNumHeads, kQueriesPerCluster, 0>(
                    &tensor_map_weights, &smem.full_q_barriers[q_stage_idx],
                    smem.smem_weights[q_stage_idx], 0, cluster_q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t cached_kv_page_base = 0;
        uint32_t cached_kv_page_coord = 0;
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                if constexpr (decltype(scheduler)::kIsPaged) {
                    constexpr uint32_t kPageKV = decltype(scheduler)::kPageKV;
                    constexpr uint32_t kNumPagesPerSplit = decltype(scheduler)::kNumPagesPerSplit;
                    DG_STATIC_ASSERT(kNumPagesPerSplit <= 32, "Split spans more pages than a warp can cache");

                    const uint32_t kv_page_base = (kv_base + kv_split_idx) * kNumPagesPerSplit;
                    if (kv_page_base < cached_kv_page_base or kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
                        cached_kv_page_base = (kv_page_base / 32) * 32;
                        cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(cached_kv_page_base + lane_idx);
                    }

                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    if (cute::elect_one_sync())
                        smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                    __syncwarp();

                    constexpr uint32_t kLocalPages = kNumPagesPerSplit / 2;
                    DG_STATIC_ASSERT(kNumPagesPerSplit % 2 == 0,
                                     "cta_group::2 requires an even page split");
                    int page_coords[kLocalPages];
                    #pragma unroll
                    for (uint32_t page_idx = 0; page_idx < kLocalPages; ++ page_idx) {
                        const uint32_t split_page = cluster_rank * kLocalPages + page_idx;
                        const auto src_lane = static_cast<int>(
                            kv_page_base - cached_kv_page_base + split_page);
                        page_coords[page_idx] = __shfl_sync(
                            0xffffffff, cached_kv_page_coord, src_lane);
                    }

                    if (cute::elect_one_sync()) {
                        auto* kv_barrier = &smem.full_kv_barriers[kv_stage_idx];
                        kv_barrier->arrive_and_expect_tx(
                            SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                        // Each CTA loads one disjoint M128 half.  The two
                        // halves use identical local SMEM offsets and are
                        // gathered directly by cta_group::2 UMMA.
                        #pragma unroll
                        for (uint32_t page_idx = 0; page_idx < kLocalPages; ++ page_idx) {
                            tma::copy<kNumQKBytesPerToken, kPageKV, 0,
                                      typename SharedStorage::qk_dtype_t, true>(
                                &tensor_map_kv, kv_barrier,
                                smem.smem_kv[kv_stage_idx] +
                                    page_idx * kPageKV * kNumQKBytesPerToken,
                                0, 0, 1, page_coords[page_idx]);
                            tma::copy<kPageKV, 1, 0>(
                                &tensor_map_sf_kv, kv_barrier,
                                smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                                0, page_coords[page_idx]);
                        }
                    }
                    __syncwarp();
                } else if (cute::elect_one_sync()) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);

                    const uint32_t kv_tma_offset =
                        scheduler.get_kv_tma_offset(kv_base, kv_split_idx) +
                        cluster_rank * kLocalSplitKV;
                    tma::copy<kNumQKBytesPerToken, kLocalSplitKV, 0>(
                        &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                        smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
                    tma::copy<kLocalSplitKV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
                    smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_STATIC_ASSERT(not kIsFP4, "V25 currently targets the formal FP8 path");
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            if (cluster_rank == 1) {
                if (cute::elect_one_sync())
                    smem.q_peer_full_barriers[q_stage_idx].arrive(0u, 1u);
            } else {
                smem.q_peer_full_barriers[q_stage_idx].wait(q_phase);
            }
            __syncwarp();

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

                if (cluster_rank == 1) {
                    if (cute::elect_one_sync())
                        smem.kv_peer_full_barriers[kv_stage_idx].arrive(0u, 1u);
                } else {
                    smem.kv_peer_full_barriers[kv_stage_idx].wait(kv_phase);
                    // All leader-warp lanes enter the CUTLASS wrappers;
                    // their internal elect_one_sync chooses the issuing
                    // lane for both MMA and commit.
                    {
                        auto instr_desc = cute::UMMA::make_instr_desc<
                            cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                            UMMA_M, UMMA_N, cute::UMMA::Major::K,
                            cute::UMMA::Major::K>();
                        auto runtime_instr_desc =
                            cute::UMMA::make_runtime_instr_desc(instr_desc);

                        #pragma unroll
                        for (uint32_t q_pair = 0; q_pair < kNumQPairs; ++q_pair) {
                            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                            const uint32_t tmem_addr = tmem_stage_idx * UMMA_N;
                            smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                            smem.tmem_peer_empty_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                            ptx::tcgen05_after_thread_sync();

                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
                                auto a_desc = mma::sm100::make_umma_desc<
                                    cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                        smem.smem_kv[kv_stage_idx], 0, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<
                                    cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                        smem.smem_q[q_stage_idx],
                                        q_pair * 2 * kNumHeads, k * UMMA_K);
                                // CUTLASS' cta_group::2 wrapper emits the
                                // required eight mask registers.  The
                                // DeepGEMM-local five-operand wrapper is a
                                // cta_group::1-shaped encoding and cannot be
                                // used for this two-SM instruction.
                                cute::SM100_MMA_F8F6F4_2x1SM_SS::fma(
                                    a_desc, b_desc, tmem_addr, k,
                                    runtime_instr_desc);
                            }
                            cutlass::arch::umma_arrive_multicast_2x1SM(
                                reinterpret_cast<uint64_t*>(
                                    &smem.full_tmem_barriers[tmem_stage_idx]),
                                0x3);
                        }
                    }
                }
                __syncwarp();
            }
            smem.empty_q_barriers[q_stage_idx].arrive();
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        // Rank 1 forwards its local TMEM-consumer completion to rank 0.
        // The leader waits this phase in addition to its own 128-thread
        // empty barrier before reusing either accumulator stage.
        if (cluster_rank == 1) {
            auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                    #pragma unroll
                    for (uint32_t q_pair = 0; q_pair < kNumQPairs; ++q_pair) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                        if (cute::elect_one_sync()) {
                            smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                            smem.tmem_peer_empty_barriers[tmem_stage_idx].arrive(0u, 1u);
                        }
                        __syncwarp();
                    }
                }
            }
        }
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = (warp_idx % 4) * 32 + lane_idx;
        DG_STATIC_ASSERT(kNumMathWarpGroups == 2,
                         "one consumer warpgroup per Q-pair TMEM stage");
        tmem_pipeline.advance(math_warpgroup_idx);

        constexpr bool kIsReduceBF16 = not cute::is_same_v<reduce_dtype_t, float>;
        DG_STATIC_ASSERT(not kIsReduceBF16 or kNumHeads % 2 == 0, "bf16 weights need even kNumHeads");
        DG_STATIC_ASSERT(kNumHeads == 4 or kNumHeads == 8 or kNumHeads == 16 or kNumHeads == 32 or kNumHeads == 64,
                         "Unsupported TMEM load size");
        float accum[kNumHeads];

        auto tmem_load_no_fence = [](auto num_elems_t, const uint32_t& addr, float* load_dst) {
            constexpr uint32_t N = decltype(num_elems_t)::value;
            using Loader = cute::conditional_t<N == 2,  cute::SM100_TMEM_LOAD_32dp32b2x,
                           cute::conditional_t<N == 4,  cute::SM100_TMEM_LOAD_32dp32b4x,
                           cute::conditional_t<N == 8,  cute::SM100_TMEM_LOAD_32dp32b8x,
                           cute::conditional_t<N == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                           cute::conditional_t<N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                                                        cute::SM100_TMEM_LOAD_32dp32b64x>>>>>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(addr, reinterpret_cast<uint32_t*>(load_dst)[Is]...);
            }(cute::make_index_sequence<N>{});
        };

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                const auto kv_offset = scheduler.get_logits_col(
                    kv_base, kv_split_idx,
                    cluster_rank * kLocalSplitKV + math_thread_idx);

                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                reduce_dtype_t scale_kv = 0;

                // WG0 consumes q-pairs 0/2 through stage 0; WG1 consumes
                // q-pairs 1/3 through stage 1.  This lets the producer issue
                // the next pair while the other WG performs ReLU/reduction.
                #pragma unroll
                for (uint32_t q_iter = 0;
                     q_iter < kNumQPairs / kNumMathWarpGroups;
                     ++q_iter) {
                    const uint32_t q_pair =
                        math_warpgroup_idx + q_iter * kNumMathWarpGroups;
                    constexpr uint32_t kTmemStridePerPair = kNumMathWarpGroups;
                    CUTE_TIE_DECL(tmem_pipeline.advance(kTmemStridePerPair),
                                  tmem_stage_idx, tmem_phase);
                    smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                    ptx::tcgen05_after_thread_sync();
                    // UMMA is issued only after both CTAs' KV TMA barriers are
                    // ready, so its completion also makes the local scale
                    // visible.  Load it once, after the first TMEM completion.
                    if (q_iter == 0) {
                        scale_kv = static_cast<reduce_dtype_t>(ptx::ld_shared(
                            smem.smem_sf_kv[kv_stage_idx] + math_thread_idx));
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < kQueriesPerMMA; ++i) {
                        // The N256 B tile concatenates two rows from rank 0
                        // and the same local row offsets from rank 1.
                        const uint32_t local_q =
                            q_pair * kQueriesPerRankPerMMA +
                            (i % kQueriesPerRankPerMMA);
                        const uint32_t q_in_cluster =
                            (i < kQueriesPerRankPerMMA)
                                ? local_q
                                : BLOCK_Q + local_q;
                        const auto smem_weights_row =
                            smem.smem_weights[q_stage_idx] + q_in_cluster * kNumHeads;
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                        if constexpr (kNumHeads == 8) {
                            tmem_load_no_fence(cute::Int<kNumHeads>{}, tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else if constexpr (kNumHeads == 16) {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);
                            cutlass::arch::fence_view_async_tmem_load();
                        }

                        if (i == kQueriesPerMMA - 1) {
                            ptx::tcgen05_before_thread_sync();
                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                        }

                        reduce_dtype_t reduced;
                        if constexpr (kIsReduceBF16) {
                            auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                            auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                            const auto transform = [&](const uint32_t& j, const nv_bfloat162& sum) {
                                const auto a = ptx::cvt_relu_bf16x2_f32(make_float2(accum[j], accum[j + 1]));
                                const auto packed_row =
                                    reinterpret_cast<const uint32_t*>(smem_weights_row);
                                const auto packed = ptx::ld_shared(packed_row + j / 2);
                                const auto b = ptx::exchange(
                                    *reinterpret_cast<const nv_bfloat162*>(&packed), 0);
                                return __hfma2(a, b, sum);
                            };

                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }

                            auto sum = __hadd2_rn(sum_0, sum_1);
                            reduced = __hadd_rn(sum.x, sum.y);
                        } else {
                            auto sum_0 = make_float2(0, 0);
                            auto sum_1 = make_float2(0, 0);
                            const auto transform = [&](const uint32_t& j, const float2& sum) {
                                auto a_0 = make_float2(accum[j], accum[j + 1]);
                                auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
                                auto b = make_float2(
                                    ptx::ld_shared(smem_weights_row + j),
                                    ptx::ld_shared(smem_weights_row + j + 1));
                                return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
                            };

                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }

                            auto sum = __fadd2_rn(sum_0, sum_1);
                            reduced = (sum.x + sum.y) / 2;
                        }
                        auto result = static_cast<logits_dtype_t>(reduced * scale_kv);
                        const auto q_offset = scheduler.get_cluster_logits_row(q_in_cluster) *
                                              static_cast<uint64_t>(logits_stride);
                        if constexpr (kIsCompressedLogits) {
                            const uint32_t rel_kv = kv_offset - seq_k_start[i];
                            const uint32_t len = seq_k_end[i] - seq_k_start[i];
                            if (rel_kv < len)
                                logits[q_offset + rel_kv] = result;
                        } else {
                            logits[q_offset + kv_offset] = result;
                        }
                    }
                }
                // Both math warpgroups arrive before this local M128 KV half
                // can be reused by the TMA producer.
                smem.empty_kv_barriers[kv_stage_idx].arrive();
            }

            smem.empty_q_barriers[q_stage_idx].arrive();
        }
    }

    __syncthreads();
    cute::cluster_sync();
    if (warp_idx == 0) {
        cute::TMEM::Allocator2Sm allocator;
        allocator.release_allocation_lock();
        allocator.free(ptx::ld_shared(&smem.tmem_ptr_in_smem), kNumTmemCols);
    }
}

// Unified contiguous-KV entry for both FP4 and FP8, selected by `kIsFP4`
template <bool kIsFP4,
          uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_mqa_logits(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                      const uint32_t logits_stride,
                      const uint32_t* cu_seq_len_k_start,
                      const uint32_t* cu_seq_len_k_end,
                      logits_dtype_t* logits,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, seq_k_start, seq_k_end);
    };

    sm100_mqa_logits_core_impl<kIsFP4, kNumHeads, kHeadDim, kIsCompressedLogits, BLOCK_Q, SPLIT_KV,
                               kNumQStages, kNumKVStages, kNumSMs,
                               kNumSpecializedThreads, kNumMathThreads, logits_dtype_t,
                               reduce_dtype_t, decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

// Unified paged entry for both FP4 and FP8, selected by `kIsFP4`
// V25 pairs two Q8 CTAs.  The CTAs walk identical KV chunks and contribute
// disjoint Q halves to four cta_group::2 N256 passes.
template <bool kIsFP4, uint32_t kTokensPerRequest, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t PAGE_KV,
          bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV, uint32_t kSplitsPerChunk,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_paged_mqa_logits(const uint32_t num_q_tokens_total,
                            const uint32_t logits_stride, const uint32_t block_table_stride,
                            const uint32_t* context_lens, logits_dtype_t* logits,
                            const uint32_t* block_table, const uint32_t* indices,
                            const uint32_t* schedule_meta,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    static constexpr uint32_t BLOCK_Q = 8;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    DG_STATIC_ASSERT(SPLIT_KV == PAGE_KV * kNumPagesPerSplit, "Invalid split/page size");

    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* /*seq_k_start*/, uint32_t* /*seq_k_end*/) {
        const uint32_t cluster_rank = cute::block_rank_in_cluster();
        const uint32_t cluster_idx = sm_idx / 2;
        return sched::SM100PagedMQALogitsScheduler<kTokensPerRequest, kIsContextLens2D, kIsVarlen,
                                                   kNumHeads, SPLIT_KV, PAGE_KV, kSplitsPerChunk>(
            cluster_idx, cluster_rank, context_lens, schedule_meta, indices,
            block_table, block_table_stride, num_q_tokens_total);
    };

    // Paged uses `kNumSMs = 0`; schedule meta drives the grid stride
    sm100_mqa_logits_core_impl<kIsFP4, kNumHeads, kHeadDim, false, BLOCK_Q, SPLIT_KV,
                               kNumQStages, kNumKVStages, 0,
                               kNumSpecializedThreads, kNumMathThreads, logits_dtype_t,
                               reduce_dtype_t, decltype(make_scheduler), kNumMathWarpGroups, 2>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

} // namespace deep_gemm
