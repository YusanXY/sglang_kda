import pytest

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    strict_batch1_swa_token_count,
)


@pytest.mark.parametrize(
    ("num_qo_tokens", "max_seq_len", "expected"),
    [
        (4096, 4096, 4096),
        (4096, 65536, 4223),
        (4096, 69632, 4223),
        (1, 65536, 128),
    ],
)
def test_strict_batch1_swa_token_count(
    num_qo_tokens, max_seq_len, expected
):
    assert strict_batch1_swa_token_count(num_qo_tokens, max_seq_len, 128) == expected


@pytest.mark.parametrize("args", [(-1, 1, 128), (1, -1, 128), (1, 1, 0)])
def test_strict_batch1_swa_token_count_rejects_invalid_geometry(args):
    with pytest.raises(ValueError, match="invalid strict batch-1 SWA geometry"):
        strict_batch1_swa_token_count(*args)
