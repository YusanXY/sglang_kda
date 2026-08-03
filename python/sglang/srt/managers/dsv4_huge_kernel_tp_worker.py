"""Dedicated TP worker for the strict DSV4 whole-layer development path."""

from sglang.srt.managers.tp_worker import TpModelWorker


class Dsv4HugeKernelTpModelWorker(TpModelWorker):
    """A separate worker type so future fusion state does not leak into native."""

    def __init__(self, *args, **kwargs):
        server_args = kwargs["server_args"]
        if server_args.dsv4_worker_backend != "huge_kernel":
            raise ValueError(
                "Dsv4HugeKernelTpModelWorker requires "
                "--dsv4-worker-backend=huge_kernel"
            )
        super().__init__(*args, **kwargs)
