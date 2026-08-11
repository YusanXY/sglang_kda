"""CPU-only unit tests for GroupCoordinator.all_gatherv argument handling."""

from contextlib import contextmanager, nullcontext

import pytest
import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakePyNcclCommunicator:
    disabled = False

    def __init__(self):
        self.calls = []

    @contextmanager
    def change_state(self, *, enable):
        assert enable
        yield

    def group_start(self):
        self.calls.append(("group_start",))

    def all_gather(self, output, input_, *, sizes):
        self.calls.append(("all_gather", output, input_, sizes))

    def group_end(self):
        self.calls.append(("group_end",))


def _make_group(*, world_size=2, rank=0):
    group = object.__new__(GroupCoordinator)
    group.world_size = world_size
    group.rank_in_group = rank
    group.pynccl_comm = _FakePyNcclCommunicator()
    group.use_symmetric_memory = lambda *_args, **_kwargs: nullcontext()
    return group


def test_all_gatherv_groups_multiple_inputs_with_preallocated_outputs():
    group = _make_group()
    sizes = [2, 3]
    hidden = torch.empty((2, 4), dtype=torch.bfloat16)
    packed_route = torch.empty((2, 8), dtype=torch.int32)
    global_hidden = torch.empty((5, 4), dtype=torch.bfloat16)
    global_packed_route = torch.empty((5, 8), dtype=torch.int32)

    result = group.all_gatherv(
        [hidden, packed_route],
        sizes=sizes,
        output=[global_hidden, global_packed_route],
    )

    assert result[0] is global_hidden
    assert result[1] is global_packed_route
    calls = group.pynccl_comm.calls
    assert [call[0] for call in calls] == [
        "group_start",
        "all_gather",
        "all_gather",
        "group_end",
    ]
    assert calls[1][1] is global_hidden
    assert calls[1][2] is hidden
    assert calls[1][3] is sizes
    assert calls[2][1] is global_packed_route
    assert calls[2][2] is packed_route
    assert calls[2][3] is sizes


def test_all_gatherv_preserves_single_tensor_output_compatibility():
    group = _make_group()
    local = torch.empty((2, 4), dtype=torch.bfloat16)
    output = torch.empty((5, 4), dtype=torch.bfloat16)

    result = group.all_gatherv(local, sizes=[2, 3], output=output)

    assert result[0] is output
    assert [call[0] for call in group.pynccl_comm.calls] == [
        "group_start",
        "all_gather",
        "group_end",
    ]


def test_all_gatherv_preserves_allocating_list_input_compatibility():
    group = _make_group()
    hidden = torch.empty((2, 4), dtype=torch.bfloat16)
    packed_route = torch.empty((2, 8), dtype=torch.int32)

    result = group.all_gatherv([hidden, packed_route], sizes=[2, 3])

    assert [tuple(out.shape) for out in result] == [(5, 4), (5, 8)]
    assert [out.dtype for out in result] == [torch.bfloat16, torch.int32]
    assert [call[0] for call in group.pynccl_comm.calls] == [
        "group_start",
        "all_gather",
        "all_gather",
        "group_end",
    ]


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([torch.empty((5, 4), dtype=torch.bfloat16)], "output list length"),
        (
            [
                torch.empty((5, 5), dtype=torch.bfloat16),
                torch.empty((5, 8), dtype=torch.int32),
            ],
            r"output\[0\] shape",
        ),
        (
            [
                torch.empty((5, 4), dtype=torch.float32),
                torch.empty((5, 8), dtype=torch.int32),
            ],
            r"output\[0\] dtype",
        ),
        (
            [
                torch.empty((5, 4), dtype=torch.bfloat16, device="meta"),
                torch.empty((5, 8), dtype=torch.int32),
            ],
            r"output\[0\] device",
        ),
    ],
)
def test_all_gatherv_rejects_invalid_output_lists(outputs, message):
    group = _make_group()
    inputs = [
        torch.empty((2, 4), dtype=torch.bfloat16),
        torch.empty((2, 8), dtype=torch.int32),
    ]

    with pytest.raises(ValueError, match=message):
        group.all_gatherv(inputs, sizes=[2, 3], output=outputs)

    assert group.pynccl_comm.calls == []


def test_all_gatherv_rejects_mismatched_input_devices():
    group = _make_group()
    inputs = [
        torch.empty((2, 4), dtype=torch.bfloat16),
        torch.empty((2, 8), dtype=torch.int32, device="meta"),
    ]

    with pytest.raises(ValueError, match=r"input_\[1\] device"):
        group.all_gatherv(inputs, sizes=[2, 3])

    assert group.pynccl_comm.calls == []


@pytest.mark.parametrize(
    ("sizes", "message"),
    [
        ([2], "sizes.*length"),
        ([2, -1], "non-negative integers"),
        ([1, 4], r"input_\[0\] leading dimension"),
    ],
)
def test_all_gatherv_rejects_invalid_sizes(sizes, message):
    group = _make_group()
    inputs = [
        torch.empty((2, 4), dtype=torch.bfloat16),
        torch.empty((2, 8), dtype=torch.int32),
    ]

    with pytest.raises(ValueError, match=message):
        group.all_gatherv(inputs, sizes=sizes)

    assert group.pynccl_comm.calls == []
