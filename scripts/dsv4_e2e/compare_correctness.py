#!/usr/bin/env python3
"""Compare native and huge-kernel one_batch correctness artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import torch


REQUIRED_KEYS = {"next_token_ids", "next_token_logits"}
MIN_COSINE = 0.999


def _load_artifact(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != REQUIRED_KEYS:
        actual = (
            sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        )
        raise ValueError(
            f"{path}: expected exactly keys {sorted(REQUIRED_KEYS)}, got {actual}"
        )

    token_ids = payload["next_token_ids"]
    logits = payload["next_token_logits"]
    if not isinstance(token_ids, torch.Tensor) or not isinstance(logits, torch.Tensor):
        raise ValueError(f"{path}: correctness values must be torch tensors")
    if token_ids.numel() == 0 or logits.numel() == 0:
        raise ValueError(f"{path}: correctness tensors must be non-empty")
    invalid_token_dtype = (
        token_ids.is_floating_point()
        or token_ids.is_complex()
        or token_ids.dtype == torch.bool
    )
    if invalid_token_dtype:
        raise ValueError(f"{path}: next_token_ids must have an integer dtype")
    if not logits.is_floating_point():
        raise ValueError(f"{path}: next_token_logits must have a floating dtype")
    logits = logits.to(dtype=torch.float32)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{path}: next_token_logits contains non-finite values")
    return token_ids, logits


def compare(native_path: Path, huge_path: Path) -> dict:
    native_ids, native_logits = _load_artifact(native_path)
    huge_ids, huge_logits = _load_artifact(huge_path)
    if native_ids.shape != huge_ids.shape:
        raise ValueError(
            f"token shape mismatch: native={tuple(native_ids.shape)}, "
            f"huge={tuple(huge_ids.shape)}"
        )
    if native_logits.shape != huge_logits.shape:
        raise ValueError(
            f"logits shape mismatch: native={tuple(native_logits.shape)}, "
            f"huge={tuple(huge_logits.shape)}"
        )

    native_flat = native_logits.reshape(-1).to(dtype=torch.float64)
    huge_flat = huge_logits.reshape(-1).to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(native_flat) * torch.linalg.vector_norm(
        huge_flat
    )
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value == 0.0:
        raise ValueError("cosine similarity requires finite, nonzero logits norms")

    cosine = float((torch.dot(native_flat, huge_flat) / denominator).item())
    absolute_error = (native_flat - huge_flat).abs()
    token_equal = bool(torch.equal(native_ids, huge_ids))
    passed = token_equal and cosine >= MIN_COSINE
    return {
        "status": "PASS" if passed else "FAILED",
        "native": str(native_path),
        "huge": str(huge_path),
        "token_shape": list(native_ids.shape),
        "logits_shape": list(native_logits.shape),
        "token_equal": token_equal,
        "cosine": cosine,
        "cosine_threshold": MIN_COSINE,
        "max_abs": float(absolute_error.max().item()),
        "mean_abs": float(absolute_error.mean().item()),
    }


def write_json_atomic(path: Path, report: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--huge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = compare(args.native, args.huge)
    write_json_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if report["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
