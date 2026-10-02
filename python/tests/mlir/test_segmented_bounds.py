# python/tests/mlir/test_segmented_bounds.py
"""Device-side segment bounds and block-size admission for segmented kernels.

Host validation sees one snapshot of the offsets. The kernels reload them
from device memory at every launch, so the bound that keeps every read inside
``[0, value_count)`` has to live in the kernel. The CUDA tests here launch
through the driver, below the Python validation, with ranges that validation
would reject. Plan-owned ranges take the same bound against the buffer they
index: partial ranges against the value count, merge ranges against the
partial count.

Three loaded indices take a bound of their own: a segment ID read from a task
buffer and the output segment of a merge record against the segment count,
and the merge ID of a persistent partial against the merge count. An index
outside its bound is skipped. The tests for them place every buffer such an
index could reach between guards and require the guards to be unchanged.
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
    _SOFTMAX_MODULE,
    _launch_segmented_sum_tasks,
    _semantic_module,
    _validate_offsets,
    _validate_softmax_tensors,
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

# Elements in each guard that `_canaried` allocates around a buffer.
_GUARD = 64
# No reduction of the test values produces this, so a store shows.
_CANARY = -1.0

# Indices that name nothing in a batch of `count` segments or merges. The
# near ones reach a few elements outside a buffer, well inside the guards:
# without a device bound they would change a guard element. The extremes
# would leave the allocation. The last two of each set also miss the three
# claim counters when they are used as a persistent merge ID.
STRAY_IDS = [
    pytest.param(lambda count: (count, -1, count + 2, -5), id="near"),
    pytest.param(lambda _: (_INT32_MAX, _INT32_MIN), id="i32-extremes"),
]

# Each kernel maps the parameter index of a count to the number of loaded
# indices it compares with that count. The segment count is the last i32 of
# every ABI that loads a segment ID or an output segment. The merge count is
# the fifth i32 of the persistent ABI.
BOUNDED_ID_KERNELS = [
    pytest.param(
        "_compile_segmented_reduction_ptx",
        {"block_size": 32, "use_task_ids": True}, {6: 1},
        id="task-ids-warp",
    ),
    pytest.param(
        "_compile_segmented_reduction_ptx",
        {"block_size": 128, "use_task_ids": True}, {6: 1},
        id="task-ids-cta",
    ),
    pytest.param(
        "_compile_fused_segmented_reduction_ptx", {}, {7: 2}, id="fused"
    ),
    pytest.param(
        "_compile_persistent_segmented_reduction_ptx", {}, {14: 1, 15: 3},
        id="persistent",
    ),
    pytest.param(
        "_compile_split_merge_reduction_ptx", {}, {5: 1}, id="split-merge"
    ),
]

# Each kernel maps the parameter index of a buffer length to the number of
# ranges it clamps against that length. The value count is the first i32 of
# every ABI that reads values, directly after the pointers, and bounds the
# ranges into values. The partial count bounds the merge ranges into scratch:
# it is the fourth i32 of the persistent ABI and the first of the merge ABI.
CLAMPED_KERNELS = [
    pytest.param(
        "_compile_segmented_reduction_ptx", {"block_size": 32}, {3: 1},
        id="direct-32",
    ),
    pytest.param(
        "_compile_segmented_reduction_ptx", {"block_size": 128}, {3: 1},
        id="direct-128",
    ),
    pytest.param(
        "_compile_segmented_reduction_ptx",
        {"block_size": 32, "use_task_ids": True}, {4: 1},
        id="task-ids-warp",
    ),
    pytest.param(
        "_compile_segmented_reduction_ptx",
        {"block_size": 128, "use_task_ids": True}, {4: 1},
        id="task-ids-cta",
    ),
    pytest.param(
        "_compile_fused_segmented_reduction_ptx", {}, {4: 2}, id="fused"
    ),
    pytest.param(
        "_compile_persistent_segmented_reduction_ptx", {}, {10: 3, 13: 1},
        id="persistent",
    ),
    pytest.param(
        "_compile_split_partial_reduction_ptx", {}, {3: 1}, id="split-partial"
    ),
    pytest.param(
        "_compile_split_merge_reduction_ptx", {}, {3: 1}, id="split-merge"
    ),
]


def _compile(
    compiler, target, *, module_text=None, kernel_name=_KERNEL, **arguments
):
    """Compile one program through one native entry point.

    The program is the canonical identity sum unless ``module_text`` names
    another one.
    """
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(module_text or _semantic_module("sum"))
        return getattr(native_swage, compiler)(
            module, kernel_name=kernel_name, target=target, **arguments
        )


def _load(compiler, entry=_KERNEL, **arguments):
    """Compile for the current device and return the loaded function."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    _, ptx = _compile(compiler, f"sm_{major}{minor}", **arguments)
    _, function = _runtime._get_driver().load(ptx, entry)
    return function


def _guarded(host_values):
    """Place a buffer the kernel reads between two NaN guards.

    The kernel receives a pointer to the valid elements. Each guard is at
    least as long as the buffer, so a read past its end or before element
    zero poisons the result and neither stray read leaves the allocation.
    """
    count = host_values.numel()
    guard = max(count, 32)
    buffer = torch.full((count + 2 * guard,), float("nan"), device="cuda")
    guarded = buffer[guard : guard + count]
    guarded.copy_(host_values)
    return guarded


def _guarded_values():
    """Build the guarded values every out-of-range launch reads.

    Returns:
        The device view the kernel reads, and a host copy of its values.
        The values are position-dependent small integers, so every sum is
        exact in f32 and a shifted window changes it.
    """
    host_values = ((torch.arange(_VALUE_COUNT) * 7) % 13 + 1).to(torch.float32)
    return _guarded(host_values), host_values


def _canaried(host, fill):
    """Place a buffer between two guards that hold ``fill``.

    Returns:
        The device view a kernel receives, and the whole allocation. An
        index up to ``_GUARD`` elements outside the view stays inside the
        allocation and addresses a guard element.
    """
    count = host.numel()
    buffer = torch.full(
        (count + 2 * _GUARD,), fill, dtype=host.dtype, device="cuda"
    )
    view = buffer[_GUARD : _GUARD + count]
    view.copy_(host)
    return view, buffer


def _canaried_output(count):
    """Build an output whose slots and guards all hold the canary."""
    return _canaried(torch.full((count,), _CANARY), _CANARY)


def _canaried_i32(values):
    """Build an i32 buffer between zero guards.

    A zero guard read as an offset or as a merge record names an empty
    range, so a stray read that reaches it still ends in a store.
    """
    return _canaried(torch.tensor(values, dtype=torch.int32), 0)


def _assert_only_stored(buffer, stored):
    """Require the canary everywhere except the slots named in ``stored``."""
    torch.cuda.synchronize()
    expected = torch.full((buffer.numel(),), _CANARY)
    for slot, value in stored.items():
        expected[_GUARD + slot] = value
    assert torch.equal(buffer.cpu(), expected)


def _assert_guards_hold(buffer, fill):
    """Require both guards of a canaried buffer to hold their fill."""
    torch.cuda.synchronize()
    host = buffer.cpu()
    guard = torch.full((_GUARD,), fill, dtype=host.dtype)
    assert torch.equal(host[:_GUARD], guard)
    assert torch.equal(host[-_GUARD:], guard)


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


@pytest.mark.parametrize(("compiler", "arguments", "bounds"), CLAMPED_KERNELS)
def test_loaded_ranges_are_clamped_against_the_abi_buffer_length(
    compiler, arguments, bounds
):
    """Bound every loaded range by the buffer length its own ABI carries."""
    lowered, _ = _compile(compiler, "sm_86", **arguments)

    signature = re.search(r"llvm\.func @\w+\(([^)]*)\)", lowered)
    assert signature is not None
    parameters = [
        parameter.split(":")[0].strip()
        for parameter in signature.group(1).split(",")
    ]
    # A clamp is two signed maxima and two signed minima; both minima take
    # the buffer length as their upper bound.
    for length_index, ranges in bounds.items():
        length = re.escape(parameters[length_index])
        upper_bounds = re.findall(
            rf"llvm\.intr\.smin\(%\w+, {length}\) : \(i32, i32\)", lowered
        )
        assert len(upper_bounds) == 2 * ranges
    total = 2 * sum(bounds.values())
    assert lowered.count("llvm.intr.smin") == total
    assert lowered.count("llvm.intr.smax") == total


@pytest.mark.parametrize(
    ("compiler", "arguments", "bounds"), BOUNDED_ID_KERNELS
)
def test_loaded_ids_are_compared_with_the_abi_count(
    compiler, arguments, bounds
):
    """Bound every loaded index by the count its own ABI carries."""
    lowered, _ = _compile(compiler, "sm_86", **arguments)

    signature = re.search(r"llvm\.func @\w+\(([^)]*)\)", lowered)
    assert signature is not None
    parameters = [
        parameter.split(":")[0].strip()
        for parameter in signature.group(1).split(",")
    ]
    assert len(parameters) == max(bounds) + 1
    # One unsigned comparison of the loaded i32 word bounds it on both
    # sides: a negative word is a large unsigned one.
    for count_index, sites in bounds.items():
        count = re.escape(parameters[count_index])
        comparisons = re.findall(
            rf'llvm\.icmp "ult" %\w+, {count} : i32', lowered
        )
        assert len(comparisons) == sites


@pytest.mark.parametrize(
    ("value_count", "output_count", "bound"),
    [(10, 6, 6), (10, 10, 10), (10, 16, 10)],
    ids=["shorter-output", "equal", "longer-output"],
)
def test_softmax_validation_returns_the_shorter_buffer_as_the_kernel_bound(
    value_count, output_count, bound
):
    """Give the map_store kernel a count that fits both element buffers."""
    values = torch.ones(value_count)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32)
    output = torch.zeros(output_count)

    assert _validate_softmax_tensors(
        values, offsets, output, require_cuda=False
    ) == (bound, 2)


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
            segment_count,
        ),
    )

    _assert_clamped(output, host_values, pairwise(_MIXED_OFFSETS))


@requires_cuda
def test_persistent_kernel_clamps_offsets_partial_and_merge_ranges():
    """Clamp the warp, CTA, and split partial queues, and the merge."""
    direct_count = len(_MIXED_OFFSETS) - 1
    values, host_values = _guarded_values()
    device_offsets = _device_i32(_MIXED_OFFSETS)
    # Two extra output slots receive one merge each. The first group's
    # partials start before element zero and end past the value count. The
    # second group's partials are valid, but its merge record names scratch
    # slots 3 and 4 of four: the length still matches its two partials, so
    # the merge runs, and it must read slot 3 alone.
    partial_ranges = [(-200, 450), (450, 1800), (100, 300), (300, 500)]
    merge_ranges = [(0, 2), (3, 5)]
    merge_count = len(merge_ranges)
    output = _nan_output(direct_count + merge_count)
    warp_tasks = _device_i32([0, 2, 3])
    cta_tasks = _device_i32([1, 4])
    device_partials = _device_ranges(partial_ranges)
    partial_merges = _device_i32([0, 0, 1, 1])
    merge_records = _device_i32(
        [
            field
            for merge_id, (begin, end) in enumerate(merge_ranges)
            for field in (direct_count + merge_id, begin, end)
        ]
    )
    scratch = _guarded(torch.full((len(partial_ranges),), float("nan")))
    counters = torch.zeros(3 + merge_count, dtype=torch.int32, device="cuda")
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
            merge_count,
            output.numel(),
        ),
    )

    partial_sums = _clamped_sums(host_values, partial_ranges)
    _assert_clamped(
        output[:direct_count], host_values, pairwise(_MIXED_OFFSETS)
    )
    assert torch.equal(scratch.cpu(), partial_sums)
    assert torch.equal(
        output[direct_count:].cpu(), _clamped_sums(partial_sums, merge_ranges)
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
@pytest.mark.parametrize(
    "block_size", [1, 31, 33, 40, 64, 97, 100, 256, 512, 1024]
)
@pytest.mark.parametrize("kind", ["sum", "max", "min", "softmax"])
def test_admitted_block_sizes_reduce_position_dependent_data(block_size, kind):
    """Run admitted block sizes other than 32 and 128 with data.

    The sizes cover a single lane, one partly filled warp, a partly filled
    last warp behind full ones (33, 40, 97, 100), and whole warps up to
    1024 threads. Segment lengths fall below and above each block size.

    Sum, max, and min use ``(2 * (index % 67) - 65) / 4``, nonzero multiples of
    0.25 whose sums are exact in f32 under any order at these lengths, and
    are compared with no tolerance. A window moved by 1 to 64 elements, or
    a dropped or repeated element, changes a sum.

    Softmax uses quarter-step logits with spread 4 and the tolerance of
    test_softmax_store_stays_inside_the_validated_output against float64
    PyTorch. It is a measured tolerance: the largest deviation is 2.8e-06,
    at block size 1, where one lane chains all 4097 terms.
    """
    from swage._segmented_qualification import launch_softmax_gpu

    lengths = [0, 1, 31, 33, 97, 300, 4097]
    offsets = list(accumulate(lengths, initial=0))
    index = torch.arange(offsets[-1])
    if kind == "softmax":
        host_values = ((index % 17) - 8).to(torch.float32) / 4
        expected = torch.cat(
            [
                torch.softmax(host_values[begin:end].double(), 0)
                for begin, end in pairwise(offsets)
            ]
        ).float()
        tolerance = {"rtol": 1e-5, "atol": 0}
    else:
        host_values = (2 * (index % 67) - 65).to(torch.float32) / 4
        identity, reduce = {
            "sum": (0.0, torch.sum),
            "max": (float("-inf"), torch.amax),
            "min": (float("inf"), torch.amin),
        }[kind]
        expected = torch.stack(
            [
                reduce(host_values[begin:end].double())
                if end > begin
                else torch.tensor(identity, dtype=torch.float64)
                for begin, end in pairwise(offsets)
            ]
        ).float()
        tolerance = {"rtol": 0, "atol": 0}
    values = host_values.cuda()
    device_offsets = _device_i32(offsets)

    # Repeat the launch: a fault that depends on which lane's store survives
    # would not show every time.
    for _ in range(4):
        output = _nan_output(expected.numel())
        if kind == "softmax":
            launch_softmax_gpu(
                values, device_offsets, output, block_size=block_size
            )
        else:
            launch_gpu(
                values, device_offsets, output, kind, block_size=block_size
            )
        torch.cuda.synchronize()

        torch.testing.assert_close(output.cpu(), expected, **tolerance)


@requires_cuda
def test_split_merge_kernel_clamps_plan_ranges():
    """Clamp plan-owned merge ranges by the partial count they index."""
    host_scratch = torch.arange(1, 7, dtype=torch.float32)
    partial_count = host_scratch.numel()
    ranges = [
        (-2, 3),
        (3, partial_count + 5),
        (partial_count + 2, partial_count + 9),
        (4, 1),
    ]
    scratch = _guarded(host_scratch)
    records = _device_i32(
        [
            field
            for segment, (begin, end) in enumerate(ranges)
            for field in (segment, begin, end)
        ]
    )
    output = _nan_output(len(ranges))
    function = _load(
        "_compile_split_merge_reduction_ptx", entry=f"{_KERNEL}__merge"
    )

    _runtime._get_driver().launch_segmented(
        function,
        (len(ranges),),
        512,
        torch.cuda.current_stream().cuda_stream,
        (
            scratch.data_ptr(),
            output.data_ptr(),
            records.data_ptr(),
            partial_count,
            len(ranges),
            output.numel(),
        ),
    )

    _assert_clamped(output, host_scratch, ranges)


@requires_cuda
@pytest.mark.parametrize("block_size", [32, 128])
def test_softmax_store_stays_inside_the_validated_output(block_size):
    """Bound the map_store write by the output the host validated.

    The softmax kernel stores one element per input element, so its range
    bounds a write as well as a read. The output may be shorter than the
    values. The offsets then change through a path the host cannot see, and
    the last segment claims every value.
    """
    covered = 600
    values, host_values = _guarded_values()
    values.div_(4)
    host_values = host_values / 4
    output_buffer = torch.full((covered + _VALUE_COUNT,), -1.0, device="cuda")
    output = output_buffer[:covered]
    offsets = _device_i32((0, 400, covered))
    bound, segment_count = _validate_softmax_tensors(values, offsets, output)
    stale = (0, 400, _VALUE_COUNT)
    with pytest.raises(ValueError, match="output has 600 elements"):
        _validate_softmax_tensors(values, _device_i32(stale), output)
    offsets.copy_(_device_i32(stale))
    function = _load(
        "_compile_segmented_reduction_ptx",
        entry="ragged_softmax",
        module_text=_SOFTMAX_MODULE,
        kernel_name="ragged_softmax",
        block_size=block_size,
    )

    _runtime._get_driver().launch_segmented(
        function,
        (segment_count,),
        block_size,
        torch.cuda.current_stream().cuda_stream,
        (
            values.data_ptr(),
            offsets.data_ptr(),
            output.data_ptr(),
            bound,
            segment_count,
        ),
    )
    torch.cuda.synchronize()

    assert torch.equal(
        output_buffer[covered:].cpu(), torch.full((_VALUE_COUNT,), -1.0)
    )
    expected = torch.cat(
        [
            torch.softmax(host_values[begin:end].double(), 0)
            for begin, end in ((0, 400), (400, covered))
        ]
    ).float()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=0)


@requires_cuda
@pytest.mark.parametrize("block_size", [32, 128], ids=["warp", "cta"])
@pytest.mark.parametrize("stray_ids", STRAY_IDS)
def test_task_id_kernels_skip_ids_outside_the_segment_count(
    stray_ids, block_size
):
    """Store nothing for a task whose segment ID names no segment.

    The task buffer is the caller's, so it can change after the host
    validated it. Segment 1 has no task: a stray ID must not be redirected
    to a valid segment either.
    """
    offsets = (0, 250, 600, 1000)
    segment_count = len(offsets) - 1
    values, host_values = _guarded_values()
    device_offsets, _ = _canaried_i32(offsets)
    output, output_buffer = _canaried_output(segment_count)
    task_ids = _device_i32([0, *stray_ids(segment_count), 2])
    with pytest.raises(ValueError, match="valid segment IDs"):
        _launch_segmented_sum_tasks(
            values, device_offsets, output, task_ids, block_size=block_size
        )
    function = _load(
        "_compile_segmented_reduction_ptx",
        block_size=block_size,
        use_task_ids=True,
    )

    _runtime._get_driver().launch_segmented_tasks(
        function,
        (task_ids.numel(),),
        block_size,
        torch.cuda.current_stream().cuda_stream,
        (
            values.data_ptr(),
            device_offsets.data_ptr(),
            output.data_ptr(),
            task_ids.data_ptr(),
            _VALUE_COUNT,
            task_ids.numel(),
            segment_count,
        ),
    )

    sums = _clamped_sums(host_values, pairwise(offsets))
    _assert_only_stored(output_buffer, {0: sums[0], 2: sums[2]})


@requires_cuda
@pytest.mark.parametrize("stray_ids", STRAY_IDS)
def test_fused_kernel_skips_ids_outside_the_segment_count(stray_ids):
    """Skip stray IDs in the four-per-block warp slots and the CTA tasks."""
    offsets = (0, 20, 300, 330, 1000)
    segment_count = len(offsets) - 1
    values, host_values = _guarded_values()
    device_offsets, _ = _canaried_i32(offsets)
    output, output_buffer = _canaried_output(segment_count)
    stray = stray_ids(segment_count)
    # Both schedules see every stray ID, and in the warp schedule a stray
    # ID shares a block with a valid one.
    warp_ids = [0, *stray, 2]
    cta_ids = [*stray, 3]
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
            segment_count,
        ),
    )

    sums = _clamped_sums(host_values, pairwise(offsets))
    _assert_only_stored(
        output_buffer, {0: sums[0], 2: sums[2], 3: sums[3]}
    )


@requires_cuda
@pytest.mark.parametrize("stray_ids", STRAY_IDS)
def test_persistent_kernel_skips_ids_outside_their_counts(stray_ids):
    """Skip stray segment IDs, merge IDs, and merge output segments.

    Segments 0 to 3 are direct work and segments 4 and 5 are split in two
    partials each. Four more partials carry the stray indices: two belong to
    merges whose record names a stray output segment, and two name a stray
    merge ID. Every partial still writes its own scratch slot. A stray merge
    ID must leave the completion counters alone, and a stray output segment
    must leave the output alone.
    """
    offsets = (0, 20, 300, 330, 400, 700, 1000)
    segment_count = len(offsets) - 1
    values, host_values = _guarded_values()
    device_offsets, _ = _canaried_i32(offsets)
    output, output_buffer = _canaried_output(segment_count)
    stray = stray_ids(segment_count)
    warp_tasks = _device_i32([0, *stray, 2])
    cta_tasks = _device_i32([1, *stray, 3])
    partial_ranges = [
        (400, 550), (550, 700),  # merge 0, segment 4
        (700, 850), (850, 1000),  # merge 1, segment 5
        (100, 200),  # merge 2, stray output segment
        (200, 300),  # merge 3, stray output segment
        (0, 50), (50, 100),  # stray merge IDs
    ]
    merge_records = [(4, 0, 2), (5, 2, 4), (stray[0], 4, 5), (stray[1], 5, 6)]
    merge_count = len(merge_records)
    # Completion counter 3 + ID follows three claim counters. The last two
    # stray IDs miss those, so a counter update without a bound would land
    # in a guard and not corrupt the queues.
    stray_merges = stray_ids(merge_count)[-2:]
    partial_merges = _device_i32([0, 0, 1, 1, 2, 3, *stray_merges])
    device_partials = _device_ranges(partial_ranges)
    device_merges, _ = _canaried_i32(
        [field for record in merge_records for field in record]
    )
    scratch, scratch_buffer = _canaried_output(len(partial_ranges))
    counters, counters_buffer = _canaried_i32([0] * (3 + merge_count))
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
            device_merges.data_ptr(),
            scratch.data_ptr(),
            counters.data_ptr(),
            _VALUE_COUNT,
            warp_tasks.numel(),
            cta_tasks.numel(),
            len(partial_ranges),
            merge_count,
            segment_count,
        ),
    )

    sums = _clamped_sums(host_values, pairwise(offsets))
    _assert_only_stored(output_buffer, dict(enumerate(sums)))
    partial_sums = _clamped_sums(host_values, partial_ranges)
    _assert_only_stored(scratch_buffer, dict(enumerate(partial_sums)))
    _assert_guards_hold(counters_buffer, 0)
    # Each merge counted its own partials and nothing else.
    assert counters[3:].tolist() == [2, 2, 1, 1]


@requires_cuda
@pytest.mark.parametrize("stray_ids", STRAY_IDS)
def test_split_merge_kernel_skips_output_segments_outside_the_segment_count(
    stray_ids,
):
    """Store nothing for a merge record whose segment names no segment."""
    host_scratch = torch.arange(1, 7, dtype=torch.float32)
    partial_count = host_scratch.numel()
    segment_count = 3
    scratch = _guarded(host_scratch)
    output, output_buffer = _canaried_output(segment_count)
    ranges = [(0, 3), (3, 5), (0, 6), (5, 6)]
    stray = stray_ids(segment_count)
    segments = [0, stray[0], stray[1], 2]
    records = _device_i32(
        [
            field
            for segment, (begin, end) in zip(segments, ranges, strict=True)
            for field in (segment, begin, end)
        ]
    )
    function = _load(
        "_compile_split_merge_reduction_ptx", entry=f"{_KERNEL}__merge"
    )

    _runtime._get_driver().launch_segmented(
        function,
        (len(ranges),),
        512,
        torch.cuda.current_stream().cuda_stream,
        (
            scratch.data_ptr(),
            output.data_ptr(),
            records.data_ptr(),
            partial_count,
            len(ranges),
            segment_count,
        ),
    )

    sums = _clamped_sums(host_scratch, ranges)
    _assert_only_stored(output_buffer, {0: sums[0], 2: sums[3]})
