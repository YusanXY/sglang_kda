"""ModelRunner selection shared by Scheduler workers and standalone benches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs


def get_model_runner_class(server_args: ServerArgs) -> type[ModelRunner]:
    backend = server_args.dsv4_worker_backend
    if backend == "native":
        from sglang.srt.model_executor.model_runner import ModelRunner

        return ModelRunner
    if backend == "huge_kernel":
        from sglang.srt.model_executor.dsv4_huge_kernel_model_runner import (
            Dsv4HugeKernelModelRunner,
        )

        return Dsv4HugeKernelModelRunner
    raise ValueError(f"Unknown dsv4 worker backend: {backend!r}")


def create_model_runner(**runner_kwargs: Any) -> ModelRunner:
    """Construct the selected runner; never silently changes backend."""

    server_args = runner_kwargs["server_args"]
    runner_cls = get_model_runner_class(server_args)
    return runner_cls(**runner_kwargs)
