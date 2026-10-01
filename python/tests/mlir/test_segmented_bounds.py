# python/tests/mlir/test_segmented_bounds.py
"""Device-side segment bounds and block-size admission for segmented kernels.

Host validation sees one snapshot of the offsets. The kernels reload them
from device memory at every launch, so the bound that keeps every read inside
``[0, value_count)`` has to live in the kernel. The CUDA tests here launch
through the driver, below the Python validation, with ranges that validation
would reject.
"""

import re
from itertools import accumulate, pairwise

import pytest
import torch
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage
from swage import _runtime
from swage._segmented_qualification import (
    _semantic_module,
    _validate_offsets,
    launch_gpu,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)

_KERNEL = "segmented_sum"
_VALUE_COUNT = 1000
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1

# Each sequence fails host validation. Apart from the last, every unclamped
# read stays inside the NaN guard that `_guarded_values` allocates.
INVALID_OFFSETS = [
    pytest.param((0, 400, 1500), id="end-past-value-count"),
    pytest.param((-300, 250, 1000), id="negative-start"),
    pytest.param((0, 700, 300, 1000), id="decreasing-pair"),
    pytest.param((1200, 1700, 1900), id="start-past-value-count"),
    pytest.param((_INT32_MIN, 600, _INT32_MAX), id="i32-extremes"),
]

# One sequence holding every violation above, for the ABIs that add task
# indirection: a negative start, an end and a start past the value count, and
# a decreasing pair.
_MIXED_OFFSETS = (-300, 250, 1500, 700, 300, 1900)

# The value count is the first i32 of each ABI, directly after the pointers.
# The last column is the number of ranges the kernel loads to index values.
CLAMPED_KERNELS = [
    pytest.param(
        "_compile_segmented_reduction_ptx", {"block_size": 32}, 3, 1,
        id="direct-32",
    ),
    pytest.param(
        "_compile_segmented_reduction_ptx", {"block_size": 128}, 3, 1,
        id="direct-128",
    ),
    pytest.param(
        "_compile_segmented_reduction_ptx",
        {"block_size": 32, "use_task_ids": True}, 4, 1,
        id="task-ids-warp",
    ),
    pytest.param(
        "_compile_segmented_reduction_ptx",
        {"block_size": 128, "use_task_ids": True}, 4, 1,
        id="task-ids-cta",
    ),
    pytest.param(
        "_compile_fused_segmented_reduction_ptx", {}, 4, 2, id="fused"
    ),
    pytest.param(
        "_compile_persistent_segmented_reduction_ptx", {}, 10, 3,
        id="persistent",
    ),
    pytest.param(
        "_compile_split_partial_reduction_ptx", {}, 3, 1, id="split-partial"
    ),
]


def _compile(compiler, target, **arguments):
    """Compile the canonical identity sum through one native entry point."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(_semantic_module("sum"))
        return getattr(native_swage, compiler)(
            module, kernel_name=_KERNEL, target=target, **arguments
        )


def _load(compiler, entry=_KERNEL, **arguments):
    """Compile for the current device and return the loaded function."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    _, ptx = _compile(compiler, f"sm_{major}{minor}", **arguments)
    _, function = _runtime._get_driver().load(ptx, entry)
    return function


def _guarded_values():
    """Place the kernel's values between two NaN guards.

    The kernel receives a pointer to ``_VALUE_COUNT`` valid elements. The
    ``_VALUE_COUNT`` elements after them are NaN, so a read past the value
    count poisons its sum. An equal guard precedes them, so a read before
    element zero does too, and neither stray read leaves the allocation.

    Returns:
        The device view the kernel reads, and a host copy of its values.
        The values are position-dependent small integers, so every sum is
        exact in f32 and a shifted window changes it.
    """
    host_values = ((torch.arange(_VALUE_COUNT) * 7) % 13 + 1).to(torch.float32)
    buffer = torch.full((3 * _VALUE_COUNT,), float("nan"), device="cuda")
    values = buffer[_VALUE_COUNT : 2 * _VALUE_COUNT]
    values.copy_(host_values)
    return values, host_values


def _clamped_sums(host_values, ranges):
    """Sum each range after the clamp the kernels apply."""
    count = host_values.numel()
    sums = []
    for start, end in ranges:
        start = min(max(start, 0), count)
        end = min(max(end, start), count)
        sums.append(host_values[start:end].sum())
    return torch.stack(sums)


def _device_i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _device_ranges(ranges):
    """Flatten half-open ranges into the plan's begin, end record layout."""
    return _device_i32([bound for bounds in ranges for bound in bounds])


def _nan_output(count):
    return torch.full((count,), float("nan"), device="cuda")


def _assert_clamped(output, host_values, ranges):
    """Require finite outputs equal to the sums over the clamped ranges."""
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert torch.equal(output.cpu(), _clamped_sums(host_values, ranges))


@pytest.mark.parametrize(("block_size", "warps"), [(96, 3), (160, 5)])
def test_native_compiler_rejects_non_power_of_two_warp_counts(
    block_size, warps
):
    """Reject block sizes whose all-reduce stores from incomplete lanes."""
    with pytest.raises(
        ValueError,
        match=(
            "block-size must give a power-of-two warp count, "
            rf"got {block_size} \({warps} warps\)"
        ),
    ):
        _compile(
            "_compile_segmented_reduction_ptx", "sm_86", block_size=block_size
        )


@pytest.mark.parametrize(
    ("compiler", "arguments", "value_count_index", "ranges"), CLAMPED_KERNELS
)
def test_values_ranges_are_clamped_against_the_abi_value_count(
    compiler, arguments, value_count_index, ranges
):
    """Bound every loaded values range by the count its own ABI carries."""
    lowered, _ = _compile(compiler, "sm_86", **arguments)

    signature = re.search(r"llvm\.func @\w+\(([^)]*)\)", lowered)
    assert signature is not None
    parameters = [
        parameter.split(":")[0].strip()
        for parameter in signature.group(1).split(",")
    ]
    value_count = re.escape(parameters[value_count_index])
    # A clamp is two signed maxima and two signed minima; both minima take
    # the value count as their upper bound.
    upper_bounds = re.findall(
        rf"llvm\.intr\.smin\(%\w+, {value_count}\) : \(i32, i32\)", lowered
    )
    assert len(upper_bounds) == 2 * ranges
    assert lowered.count("llvm.intr.smin") == 2 * ranges
    assert lowered.count("llvm.intr.smax") == 2 * ranges


def test_split_merge_ranges_stay_unclamped():
    """Leave scratch ranges as loaded: the merge ABI has no value count."""
    lowered, _ = _compile("_compile_split_merge_reduction_ptx", "sm_86")

    assert "llvm.intr.smin" not in lowered
    assert "llvm.intr.smax" not in lowered


@requires_cuda
@pytest.mark.parametrize("block_size", [32, 128])
@pytest.mark.parametrize("offsets", INVALID_OFFSETS)
def test_direct_kernel_clamps_offsets_that_validation_rejects(
    offsets, block_size
):
    """Keep one-CTA reads inside the value count without host validation."""
    segment_count = len(offsets) - 1
    with pytest.raises(ValueError):
        _validate_offsets(list(offsets), _VALUE_COUNT, segment_count)
    values, host_values = _guarded_values()
    device_offsets = _device_i32(offsets)
    output = _nan_output(segment_count)
    function = _load("_compile_segmented_reduction_ptx", block_size=block_size)

    _runtime._get_driver().launch_segmented(
        function,
        (segment_count,),
        block_size,
        torch.cuda.current_stream().cuda_stream,
        (
            values.data_ptr(),
            device_offsets.data_ptr(),
            output.data_ptr(),
            _VALUE_COUNT,
            segment_count,
        ),
    )

    _assert_clamped(output, host_values, pairwise(offsets))


@requires_cuda
@pytest.mark.parametrize("block_size", [32, 128], ids=["warp", "cta"])
def test_task_id_kernels_clamp_offsets(block_size):
    """Clamp behind task indirection, in the warp and the CTA reduction."""
    segment_count = len(_MIXED_OFFSETS) - 1
    values, host_values = _guarded_values()
    device_offsets = _device_i32(_MIXED_OFFSETS)
    output = _nan_output(segment_count)
    task_ids = _device_i32(list(reversed(range(segment_count))))
    function = _load(
        "_compile_segmented_reduction_ptx",
        block_size=block_size,
        use_task_ids=True,
    )

    _runtime._get_driver().launch_segmented_tasks(
        function,
        (segment_count,),
        block_size,
        torch.cuda.current_stream().cuda_stream,
        (
            values.data_ptr(),
            device_offsets.data_ptr(),
            output.data_ptr(),
            task_ids.data_ptr(),
            _VALUE_COUNT,
            segment_count,
        ),
    )

    _assert_clamped(output, host_values, pairwise(_MIXED_OFFSETS))


@requires_cuda
def test_fused_kernel_clamps_offsets_in_both_branches():
    """Clamp in the four-per-block warp slots and in the CTA tasks."""
    segment_count = len(_MIXED_OFFSETS) - 1
    values, host_values = _guarded_values()
    device_offsets = _device_i32(_MIXED_OFFSETS)
    output = _nan_output(segment_count)
    warp_ids = [0, 2, 3]
    cta_ids = [1, 4]
    task_ids = _device_i32([*warp_ids, *cta_ids])
    function = _load("_compile_fused_segmented_reduction_ptx")

    _runtime._get_driver().launch_segmented_mixed(
        function,
        ((len(warp_ids) + 3) // 4 + len(cta_ids),),
        128,
        torch.cuda.current_stream().cuda_stream,
        (
            values.data_ptr(),
            device_offsets.data_ptr(),
            output.data_ptr(),
            task_ids.data_ptr(),
            _VALUE_COUNT,
            len(warp_ids),
            len(cta_ids),
        ),
    )

    _assert_clamped(output, host_values, pairwise(_MIXED_OFFSETS))


@requires_cuda
def test_persistent_kernel_clamps_offsets_and_partial_ranges():
    """Clamp the warp queue, the CTA queue, and the split partial queue."""
    direct_count = len(_MIXED_OFFSETS) - 1
    values, host_values = _guarded_values()
    device_offsets = _device_i32(_MIXED_OFFSETS)
    # One extra output slot receives the merge of two partials whose ranges
    # start before element zero and end past the value count.
    partial_ranges = [(-200, 450), (450, 1800)]
    output = _nan_output(direct_count + 1)
    warp_tasks = _device_i32([0, 2, 3])
    cta_tasks = _device_i32([1, 4])
    device_partials = _device_ranges(partial_ranges)
    partial_merges = _device_i32([0, 0])
    merge_records = _device_i32([direct_count, 0, len(partial_ranges)])
    scratch = _nan_output(len(partial_ranges))
    counters = torch.zeros(3 + 1, dtype=torch.int32, device="cuda")
    function = _load("_compile_persistent_segmented_reduction_ptx")

    _runtime._get_driver().launch_persistent(
        function,
        (2,),
        512,
        torch.cuda.current_stream().cuda_stream,
        (
            values.data_ptr(),
            device_offsets.data_ptr(),
            output.data_ptr(),
            warp_tasks.data_ptr(),
            cta_tasks.data_ptr(),
            device_partials.data_ptr(),
            partial_merges.data_ptr(),
            merge_records.data_ptr(),
            scratch.data_ptr(),
            counters.data_ptr(),
            _VALUE_COUNT,
            warp_tasks.numel(),
            cta_tasks.numel(),
            len(partial_ranges),
            1,
        ),
    )

    _assert_clamped(
        output,
        host_values,
        [*pairwise(_MIXED_OFFSETS), (0, _VALUE_COUNT)],
    )
    assert torch.equal(
        scratch.cpu(), _clamped_sums(host_values, partial_ranges)
    )


@requires_cuda
def test_split_partial_kernel_clamps_plan_ranges():
    """Clamp plan-owned partial ranges, which also index the values."""
    ranges = [(-200, 450), (450, 1800), (1200, 1300), (900, 200)]
    values, host_values = _guarded_values()
    device_ranges = _device_ranges(ranges)
    scratch = _nan_output(len(ranges))
    function = _load(
        "_compile_split_partial_reduction_ptx", entry=f"{_KERNEL}__partial"
    )

    _runtime._get_driver().launch_segmented(
        function,
        (len(ranges),),
        512,
        torch.cuda.current_stream().cuda_stream,
        (
            values.data_ptr(),
            device_ranges.data_ptr(),
            scratch.data_ptr(),
            _VALUE_COUNT,
            len(ranges),
        ),
    )

    _assert_clamped(scratch, host_values, ranges)


@requires_cuda
@pytest.mark.parametrize("block_size", [40, 100])
def test_partly_filled_last_warp_gives_exact_sums(block_size):
    """Admit a power-of-two warp count whose last warp is not full."""
    lengths = [1, 31, 33, 97, 300, 4097]
    offsets = list(accumulate(lengths, initial=0))
    host_values = ((torch.arange(offsets[-1]) * 7) % 13 + 1).to(torch.float32)
    expected = _clamped_sums(host_values, pairwise(offsets))
    values = host_values.cuda()
    device_offsets = _device_i32(offsets)

    # Repeat the launch: a fault that depends on which lane's store survives
    # would not show every time.
    for _ in range(4):
        output = _nan_output(len(lengths))
        launch_gpu(
            values, device_offsets, output, "sum", block_size=block_size
        )
        torch.cuda.synchronize()

        assert torch.equal(output.cpu(), expected)
