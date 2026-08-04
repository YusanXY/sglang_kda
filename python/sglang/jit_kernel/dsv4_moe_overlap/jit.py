from __future__ import annotations

import hashlib
import os
from pathlib import Path


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


def _patch_launcher(source: str) -> str:
    if source.count(_RUN_PROLOGUE) != 1:
        raise RuntimeError(
            "DSV4 MoE overlap patch expected exactly one FP4 run prologue"
        )
    if source.count(_POST_ROUTING_PREPARE) != 1:
        raise RuntimeError(
            "DSV4 MoE overlap patch expected exactly one FP4 post-routing prepare"
        )
    return source.replace(_RUN_PROLOGUE, _OVERLAPPED_RUN_PROLOGUE).replace(
        _POST_ROUTING_PREPARE, _POST_ROUTING_LAUNCH
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

    symlink_path = (
        jit_env.FLASHINFER_CUBIN_DIR
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

    return gen_jit_spec(
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
            jit_env.FLASHINFER_CUBIN_DIR,
            jit_env.FLASHINFER_CUBIN_DIR / include_path,
            flashinfer_csrc_dir / "nv_internal",
            flashinfer_csrc_dir / "nv_internal/include",
        ],
    )
