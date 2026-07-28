from types import SimpleNamespace

import pytest

from sglang.srt.utils import common


def test_get_amdgpu_memory_capacity_parses_rocm_smi_json(monkeypatch):
    monkeypatch.setattr(common.shutil, "which", lambda _: "/mock/rocm-smi")
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '{"card0":{"VRAM Total Memory (B)":"206141652992"},'
                '"card1":{"VRAM Total Memory (B)":"205520896000"}}'
            ),
            stderr="",
        ),
    )

    assert common.get_amdgpu_memory_capacity() == pytest.approx(
        205520896000 / (1 << 20)
    )
