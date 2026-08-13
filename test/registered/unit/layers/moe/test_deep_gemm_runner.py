import sys

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def test_deep_gemm_runner_skips_empty_masked_batch():
    # Standard EP pads masked inputs even when a DP-attention rank has no local
    # token.  expected_m, rather than the backing tensor shape, carries the
    # semantic row count in that case.
    runner = object.__new__(DeepGemmRunnerCore)
    runner_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((4, 256, 16), dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.empty((4, 256, 1), dtype=torch.float32),
        use_masked_gemm=True,
        masked_m=torch.zeros((4,), dtype=torch.int32),
        expected_m=0,
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((4, 32, 16), dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty((4, 16, 16), dtype=torch.float8_e4m3fn),
        use_fp8=True,
    )

    output = runner.run(runner_input, quant_info, running_state={})

    assert output.hidden_states.shape == (4, 256, 16)
    assert output.hidden_states.dtype == torch.bfloat16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
