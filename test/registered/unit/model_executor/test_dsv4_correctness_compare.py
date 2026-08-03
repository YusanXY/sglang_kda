"""Hermetic correctness artifact comparison tests."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).parents[4]
SCRIPT = REPO_ROOT / "scripts/dsv4_e2e/compare_correctness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_correctness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare_correctness = _load_module()


def _save(path, token, logits):
    torch.save(
        {
            "next_token_ids": torch.tensor([token], dtype=torch.int64),
            "next_token_logits": torch.tensor([logits], dtype=torch.float32),
        },
        path,
    )


def test_compare_reports_metrics_and_atomically_writes_json(tmp_path):
    native = tmp_path / "native.pt"
    huge = tmp_path / "huge.pt"
    output = tmp_path / "nested/report.json"
    _save(native, 17, [1.0, 2.0, 3.0])
    _save(huge, 17, [1.0, 2.001, 3.0])

    report = compare_correctness.compare(native, huge)
    assert report["status"] == "PASS"
    assert report["token_equal"] is True
    assert report["cosine"] >= 0.999
    assert report["max_abs"] == pytest.approx(0.001, abs=1e-6)
    assert report["mean_abs"] == pytest.approx(0.001 / 3, abs=1e-6)

    compare_correctness.write_json_atomic(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_compare_fails_gate_for_token_or_cosine_mismatch(tmp_path, monkeypatch):
    native = tmp_path / "native.pt"
    huge = tmp_path / "huge.pt"
    output = tmp_path / "failed.json"
    _save(native, 17, [1.0, 2.0, 3.0])
    _save(huge, 18, [-1.0, -2.0, -3.0])

    report = compare_correctness.compare(native, huge)
    assert report["status"] == "FAILED"
    assert report["token_equal"] is False
    assert report["cosine"] == pytest.approx(-1.0)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_correctness.py",
            "--native",
            str(native),
            "--huge",
            str(huge),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        compare_correctness.main()
    assert exit_info.value.code == 3
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "FAILED"


def test_compare_rejects_shape_mismatch(tmp_path):
    native = tmp_path / "native.pt"
    huge = tmp_path / "huge.pt"
    _save(native, 17, [1.0, 2.0, 3.0])
    _save(huge, 17, [1.0, 2.0])

    with pytest.raises(ValueError, match="logits shape mismatch"):
        compare_correctness.compare(native, huge)


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"next_token_ids": torch.tensor([1])}, "expected exactly keys"),
        (
            {
                "next_token_ids": torch.tensor([1]),
                "next_token_logits": torch.tensor([[float("nan")]]),
            },
            "non-finite",
        ),
    ],
)
def test_compare_rejects_invalid_artifacts(tmp_path, payload, match):
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match=match):
        compare_correctness._load_artifact(invalid)
