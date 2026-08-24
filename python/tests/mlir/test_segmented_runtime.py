# python/tests/mlir/test_segmented_runtime.py
"""Differential qualification for native segmented reductions."""

import threading

import pytest
import torch
from swage._segmented_qualification import (
    _launch_segmented_sum_tasks,
    _prepare_planned_sum,
    _validate_counts,
    _validate_offsets,
    _validate_tensors,
    cpu_oracle,
    cpu_softmax_oracle,
    launch_gpu,
    launch_softmax_gpu,
)


def _case(lengths):
    """Build deterministic values and offsets for segment lengths."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.tensor(
        [((index % 17) - 8) / 4 for index in range(offsets[-1])],
        dtype=torch.float32,
    )
    return values, torch.tensor(offsets, dtype=torch.int32)


CASES = [
    pytest.param([0], id="empty"),
    pytest.param([1], id="singleton"),
    pytest.param([128], id="block-boundary"),
    pytest.param([129], id="non-multiple"),
    pytest.param([4097], id="large"),
    pytest.param([33, 33, 33, 33], id="uniform"),
    pytest.param([1, 257, 2, 3837], id="skewed"),
    pytest.param([0, 2, 0, 0, 3], id="repeated-empty"),
]


TASK_CASES = [
    pytest.param([], id="no-segments"),
    pytest.param([0], id="empty"),
    pytest.param([32, 33], id="policy-boundary"),
    pytest.param([0, 2, 0, 0, 3], id="repeated-empty"),
    pytest.param([1, 257, 2, 3837], id="skewed"),
    pytest.param([4097], id="large"),
]


def _pytorch_reference(values, offsets, kind):
    """Compute the PyTorch reference while preserving empty identities."""
    results = []
    for index in range(len(offsets) - 1):
        segment = values[offsets[index] : offsets[index + 1]]
        if kind == "sum":
            results.append(segment.sum())
        elif segment.numel():
            results.append(segment.max())
        else:
            results.append(torch.tensor(float("-inf"), dtype=torch.float32))
    return torch.stack(results)


@pytest.mark.parametrize(
    ("offsets", "value_count", "output_count", "message"),
    [
        ([1, 1], 1, 1, "start at zero"),
        ([0, -1], 1, 1, "must not be negative"),
        ([0, 2, 1], 2, 2, "nondecreasing"),
        ([0, 2], 1, 1, "exceeds value count"),
        ([0, 1, 1], 1, 1, "output has 1 elements"),
    ],
)
def test_rejects_malformed_offsets(offsets, value_count, output_count, message):
    """Reject malformed metadata before a CUDA pointer is launched."""
    with pytest.raises(ValueError, match=message):
        _validate_offsets(offsets, value_count, output_count)


@pytest.mark.parametrize("count", [-1, 1 << 31])
def test_rejects_counts_outside_i32(count):
    """Keep explicit value and segment counts inside the CUDA ABI."""
    with pytest.raises(ValueError, match="nonnegative i32"):
        _validate_counts(count, 0)
    with pytest.raises(ValueError, match="nonnegative i32"):
        _validate_counts(0, count)


def test_rejects_wrong_offset_dtype_rank_and_undersized_output():
    """Validate tensor metadata before device or pointer access."""
    values = torch.empty(2)
    output = torch.empty(1)
    with pytest.raises(TypeError, match="offsets must have dtype torch.int32"):
        _validate_tensors(values, torch.tensor([0, 2]), output)
    with pytest.raises(TypeError, match="offsets must have rank one"):
        _validate_tensors(
            values,
            torch.tensor([[0, 2]], dtype=torch.int32),
            output,
        )
    with pytest.raises(ValueError, match="output has 1 elements"):
        _validate_tensors(
            values,
            torch.tensor([0, 1, 2], dtype=torch.int32),
            output,
        )


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("lengths", CASES)
def test_cpu_reduction_matches_pytorch(lengths, kind):
    """Execute sequential reductions through upstream mlir-runner."""
    values, offsets = _case(lengths)

    actual = cpu_oracle(values, offsets, kind)
    expected = _pytorch_reference(values, offsets, kind)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("lengths", CASES)
def test_gpu_reduction_matches_pytorch_and_cpu_oracle(lengths, kind):
    """Qualify one-CTA reductions against both independent references."""
    host_values, host_offsets = _case(lengths)
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(len(lengths), device="cuda")

    launch_gpu(values, offsets, output, kind)

    torch.testing.assert_close(
        output.cpu(),
        _pytorch_reference(host_values, host_offsets, kind),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        output.cpu(),
        cpu_oracle(host_values, host_offsets, kind),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("block_size", [32, 128], ids=["warp", "cta"])
@pytest.mark.parametrize("lengths", TASK_CASES)
def test_gpu_task_sum_matches_exact_segment_lengths(lengths, block_size):
    """Execute every segment through each private M7 task policy."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.full(
        (len(lengths),), float("nan"), device="cuda", dtype=torch.float32
    )
    task_ids = torch.arange(len(lengths), device="cuda", dtype=torch.int32)

    _launch_segmented_sum_tasks(
        values, device_offsets, output, task_ids, block_size=block_size
    )

    torch.testing.assert_close(
        output.cpu(), torch.tensor(lengths, dtype=torch.float32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("lengths", TASK_CASES)
def test_prepared_mixed_sum_matches_exact_segment_lengths(lengths):
    """Execute one fused launch with stable warp IDs before stable CTA IDs."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.full(
        (len(lengths),), float("nan"), device="cuda", dtype=torch.float32
    )

    prepared = _prepare_planned_sum(values, device_offsets, output)
    prepared.mixed()

    torch.testing.assert_close(
        output.cpu(), torch.tensor(lengths, dtype=torch.float32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_sum_rejects_output_aliases_before_driver_work(monkeypatch):
    """Reject either input alias before compilation or driver access."""
    from swage import _runtime

    monkeypatch.setattr(
        _runtime,
        "_get_driver",
        lambda: pytest.fail("driver must not be accessed"),
    )
    values = torch.ones(2, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="values buffer"):
        _prepare_planned_sum(values, offsets, values)

    offset_output = offsets[:2].view(torch.float32)
    with pytest.raises(ValueError, match="offsets buffer"):
        _prepare_planned_sum(values, offsets, offset_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed"])
def test_prepared_sum_uses_current_non_default_stream(policy):
    """Launch every prepared policy on PyTorch's current stream."""
    lengths = [0, 32, 33, 4097]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.full((len(lengths),), float("nan"), device="cuda")
    prepared = _prepare_planned_sum(values, device_offsets, output)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        getattr(prepared, policy)()
    stream.synchronize()

    torch.testing.assert_close(
        output.cpu(), torch.tensor(lengths, dtype=torch.float32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("policy", ["warp", "cta", "mixed"])
def test_prepared_sum_waits_for_task_initialization_across_streams(
    monkeypatch, policy
):
    """Order asynchronous task initialization before another stream launches."""
    from swage import _runtime

    preparation_stream = torch.cuda.Stream()
    launch_stream = torch.cuda.Stream()
    launch_observed = torch.cuda.Event()
    launch_completed = threading.Event()

    class _Driver:
        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_segmented_tasks(self, *_arguments):
            launch_observed.record(launch_stream)

        def launch_segmented_mixed(self, *_arguments):
            launch_observed.record(launch_stream)

    values = torch.ones(34, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1, 34], device="cuda", dtype=torch.int32)
    output = torch.empty(2, device="cuda", dtype=torch.float32)
    all_task_storage = torch.empty(2, device="cuda", dtype=torch.int32)
    mixed_task_storage = torch.empty(2, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    monkeypatch.setattr(_runtime, "_get_driver", _Driver)

    def asynchronous_arange(*_args, **_kwargs):
        torch.cuda._sleep(2_000_000_000)
        return all_task_storage

    def asynchronous_tensor(_data, *, dtype, device):
        assert dtype == torch.int32
        assert device == offsets.device
        return mixed_task_storage

    monkeypatch.setattr(torch, "arange", asynchronous_arange)
    monkeypatch.setattr(torch, "tensor", asynchronous_tensor)
    with torch.cuda.stream(preparation_stream):
        prepared = _prepare_planned_sum(values, offsets, output)
    with torch.cuda.stream(launch_stream):
        getattr(prepared, policy)()

    def observe_launch():
        launch_observed.synchronize()
        launch_completed.set()

    observer = threading.Thread(target=observe_launch)
    observer.start()
    bypassed_initialization = launch_completed.wait(timeout=0.5)
    preparation_stream.synchronize()
    launch_stream.synchronize()
    observer.join()

    assert not bypassed_initialization


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("lengths", "expected_ids", "expected_counts", "expected_grid"),
    [
        ([33, 1], [1, 0], (1, 1), (2,)),
        ([1, 2, 3, 4, 33], [0, 1, 2, 3, 4], (4, 1), (2,)),
        ([1, 2, 3, 4, 5, 33], [0, 1, 2, 3, 4, 5], (5, 1), (3,)),
        ([0, 32], [0, 1], (2, 0), (1,)),
        ([33, 34], [0, 1], (0, 2), (2,)),
    ],
)
def test_prepared_mixed_uses_one_ordered_fused_launch(
    monkeypatch, lengths, expected_ids, expected_counts, expected_grid
):
    """Submit stable warp IDs before CTA IDs through one fused kernel."""
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.loads = []
            self.launches = []

        def load(self, ptx, kernel_name):
            self.loads.append((ptx, kernel_name))
            function = len(self.loads)
            return function + 100, function

        def launch_segmented_tasks(
            self, function, grid, block, stream, arguments
        ):
            self.launches.append(
                ("pure", function, grid, block, stream, arguments)
            )

        def launch_segmented_mixed(
            self, function, grid, block, stream, arguments
        ):
            self.launches.append(
                ("mixed", function, grid, block, stream, arguments)
            )

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.empty(len(lengths), device="cuda")
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    original_tensor = torch.tensor
    task_tensors = []

    def capture_tensor(data, *args, **kwargs):
        tensor = original_tensor(data, *args, **kwargs)
        if kwargs.get("dtype") == torch.int32 and tensor.device.type == "cuda":
            task_tensors.append((list(data), tensor))
        return tensor

    monkeypatch.setattr(torch, "tensor", capture_tensor)

    prepared = _prepare_planned_sum(values, device_offsets, output)
    prepared.mixed()

    assert len(driver.launches) == 1
    assert len(task_tensors) == 1
    kind, _, grid, block, _, arguments = driver.launches[0]
    mixed_ids, mixed_tasks = task_tensors[-1]
    assert kind == "mixed"
    assert grid == expected_grid
    assert block == 128
    assert mixed_ids == expected_ids
    assert arguments[3] == mixed_tasks.data_ptr()
    assert arguments[5:] == expected_counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_empty_prepared_sum_does_not_compile_allocate_or_launch(monkeypatch):
    """Return three no-ops after classification when there are no segments."""
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )
    from swage import _runtime

    def fail(*_args, **_kwargs):
        pytest.fail("empty work must not compile, allocate, or access driver")

    monkeypatch.setattr(
        native_swage, "_compile_segmented_reduction_ptx", fail
    )
    monkeypatch.setattr(
        native_swage, "_compile_fused_segmented_reduction_ptx", fail
    )
    monkeypatch.setattr(torch, "arange", fail)
    monkeypatch.setattr(torch, "tensor", fail)
    monkeypatch.setattr(_runtime, "_get_driver", fail)
    values = torch.empty(0, device="cuda", dtype=torch.float32)
    offsets = torch.empty(1, device="cuda", dtype=torch.int32)
    offsets.zero_()
    output = torch.empty(0, device="cuda", dtype=torch.float32)

    prepared = _prepare_planned_sum(values, offsets, output)

    assert prepared.warp() is None
    assert prepared.cta() is None
    assert prepared.mixed() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_sum_rejects_a_different_current_device(monkeypatch):
    """Do not dispatch a prepared policy after the current device changes."""
    from swage import _runtime

    class _Driver:
        def __init__(self):
            self.launches = []

        def load(self, _ptx, _kernel_name):
            return 1, 1

        def launch_segmented_tasks(self, *arguments):
            self.launches.append(arguments)

    values = torch.ones(34, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1, 34], device="cuda", dtype=torch.int32)
    output = torch.empty(2, device="cuda")
    driver = _Driver()
    monkeypatch.setattr(_runtime, "_get_driver", lambda: driver)
    prepared = _prepare_planned_sum(values, offsets, output)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 999)

    with pytest.raises(ValueError, match="prepared device"):
        prepared.mixed()
    assert not driver.launches


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_mixed_sum_is_repeatable():
    """Run the same fused launch twice without stale warp or CTA state."""
    lengths = [0, 32, 33, 0, 1, 4097]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    values = torch.ones(offsets[-1], device="cuda", dtype=torch.float32)
    device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    output = torch.empty(len(lengths), device="cuda", dtype=torch.float32)
    prepared = _prepare_planned_sum(values, device_offsets, output)

    prepared.mixed()
    first = output.clone()
    output.fill_(float("nan"))
    prepared.mixed()

    torch.testing.assert_close(output, first, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["sum", "max"])
def test_gpu_reduction_is_repeatable(kind):
    """Run the same loaded shape twice without stale CTA state."""
    host_values, host_offsets = _case([0, 2, 0, 129])
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(4, device="cuda")

    launch_gpu(values, offsets, output, kind)
    first = output.clone()
    output.fill_(float("nan"))
    launch_gpu(values, offsets, output, kind)

    torch.testing.assert_close(output, first)


def test_cpu_max_propagates_nan_and_uses_negative_infinity_identity():
    """Make max NaN and empty semantics explicit in the CPU oracle."""
    values = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0])
    offsets = torch.tensor([0, 3, 3, 5], dtype=torch.int32)

    actual = cpu_oracle(values, offsets, "max")

    torch.testing.assert_close(
        actual,
        torch.tensor([float("nan"), float("-inf"), 5.0]),
        equal_nan=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_max_propagates_nan_and_uses_negative_infinity_identity():
    """Match the specified max semantics through CTA reduction."""
    host_values = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0])
    host_offsets = torch.tensor([0, 3, 3, 5], dtype=torch.int32)
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(3, device="cuda")

    launch_gpu(values, offsets, output, "max")

    expected = torch.tensor([float("nan"), float("-inf"), 5.0])
    torch.testing.assert_close(output.cpu(), expected, equal_nan=True)
    torch.testing.assert_close(
        output.cpu(),
        cpu_oracle(host_values, host_offsets, "max"),
        equal_nan=True,
    )


# Softmax tolerances, measured on an RTX A6000 at sm_86.
#
# The dominant term is the f32 rounding of `x * 1.44269502`, whose relative
# effect on exp2 is about 6e-08 per unit of intra-segment spread. It is
# identical on both backends, so it does not cancel against PyTorch. Every
# distribution below except one-outlier caps its spread at 4, and _GPU_RTOL
# is sized for a cap of 8. Widening a segment past that needs a larger
# constant, so the fix for a failure just above rtol is the distribution.
_GPU_RTOL, _GPU_ATOL = 2e-6, 1e-7

# Anything compared against cpu_softmax_oracle carries a hard 5e-06 relative
# floor, because the oracle parses printMemrefF32's six-significant-digit
# text. That floor is the transport, not the arithmetic.
_ORACLE_RTOL, _ORACLE_ATOL = 1e-5, 1e-6


def _softmax_case(lengths, outlier=None):
    """Build a softmax case, optionally planting one dominant value."""
    values, offsets = _case(lengths)
    if outlier is not None:
        values = values.clone()
        values[int(offsets[1])] = outlier
    return values, offsets


SOFTMAX_CASES = [
    pytest.param([0] * 8, None, id="all-empty"),
    pytest.param([1] * 64, None, id="all-ones"),
    pytest.param([1, 2, 3, 4] * 24, None, id="many-tiny"),
    pytest.param([4096, 3, 2731], None, id="few-huge"),
    pytest.param([1, 127, 640], 100.0, id="one-outlier"),
    pytest.param([0, 5, 0, 7, 0, 3, 0, 1, 0], None, id="alternating-empty"),
]


def _pytorch_softmax_reference(values, offsets):
    """Compute segmented softmax in float32, with the shift written out.

    The maximum shift is the thing under test, so it is explicit rather than
    delegated to torch.softmax. Empty segments contribute nothing, so the
    result is a concatenation of length offsets[-1].
    """
    pieces = []
    for index in range(len(offsets) - 1):
        segment = values[offsets[index] : offsets[index + 1]]
        if not segment.numel():
            continue
        shifted = segment - segment.max()
        exponentials = shifted.exp()
        pieces.append(exponentials / exponentials.sum())
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces)


@pytest.mark.parametrize(("lengths", "outlier"), SOFTMAX_CASES)
def test_cpu_softmax_matches_pytorch(lengths, outlier):
    """Execute the sequential softmax through upstream mlir-runner."""
    values, offsets = _softmax_case(lengths, outlier)

    actual = cpu_softmax_oracle(values, offsets)
    expected = _pytorch_softmax_reference(values, offsets)

    torch.testing.assert_close(
        actual, expected, rtol=_ORACLE_RTOL, atol=_ORACLE_ATOL
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(("lengths", "outlier"), SOFTMAX_CASES)
def test_gpu_softmax_matches_pytorch_and_cpu_oracle(lengths, outlier):
    """Qualify the three-phase kernel against both independent references."""
    host_values, host_offsets = _softmax_case(lengths, outlier)
    covered = int(host_offsets[-1])
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(covered, device="cuda")

    launch_softmax_gpu(values, offsets, output)
    # A zero-length output enqueues no copy on .cpu(), so without this the
    # all-empty row would compare two empty tensors and observe nothing
    # about whether the launch even succeeded.
    torch.cuda.synchronize()

    expected = _pytorch_softmax_reference(host_values, host_offsets)
    torch.testing.assert_close(
        output.cpu(), expected, rtol=_GPU_RTOL, atol=_GPU_ATOL
    )
    torch.testing.assert_close(
        output.cpu(),
        cpu_softmax_oracle(host_values, host_offsets),
        rtol=_ORACLE_RTOL,
        atol=_ORACLE_ATOL,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_softmax_is_repeatable():
    """Run the same loaded shape twice without stale CTA state."""
    host_values, host_offsets = _softmax_case([0, 2, 0, 129])
    values = host_values.cuda()
    offsets = host_offsets.cuda()
    output = torch.empty(int(host_offsets[-1]), device="cuda")

    launch_softmax_gpu(values, offsets, output)
    first = output.clone()
    output.fill_(float("nan"))
    launch_softmax_gpu(values, offsets, output)

    torch.testing.assert_close(output, first)


def test_cpu_softmax_of_singleton_is_exactly_one():
    """A one-element segment normalizes to 1.0 within the text transport.

    The assertion uses no tolerance, but the value reaches it through
    printMemrefF32's six significant digits, so its real strictness is the
    5e-06 floor documented above and not bit equality. That is still about
    twice as tight as _ORACLE_RTOL. The bit-exactness claim belongs to the
    GPU twin, which compares device memory directly.
    """
    values, offsets = _softmax_case([1, 1, 1])

    actual = cpu_softmax_oracle(values, offsets)

    torch.testing.assert_close(actual, torch.ones(3), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_softmax_of_singleton_is_exactly_one():
    """The recompute schedule makes the singleton quotient exact.

    map_store repeats the identical subtract, multiply, and exp2 sequence
    the reduce used, so the sum equals the exponential bit for bit and the
    quotient is exactly 1.0 regardless of ex2.approx's accuracy. This is the
    test that trips if the two region clones ever lower differently, which
    is the only way the recompute can stop agreeing with itself.
    """
    host_values, host_offsets = _softmax_case([1, 1, 1])
    output = torch.empty(3, device="cuda")

    launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)

    torch.testing.assert_close(output.cpu(), torch.ones(3), rtol=0, atol=0)


def _semantic_edge_case():
    """Three segments: a NaN, all negative infinity, and a finite maximum."""
    values = torch.tensor(
        [float("nan"), 1.0, float("-inf"), float("-inf"), float("-inf"), 0.0],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    # A NaN propagates through maximumf and ex2. A segment that is entirely
    # negative infinity has a negative-infinity maximum, so every difference
    # is NaN. A segment where negative infinity sits under a finite maximum
    # is well defined, because exp2 of negative infinity is exactly zero.
    expected = torch.tensor(
        [float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 1.0],
        dtype=torch.float32,
    )
    return values, offsets, expected


def test_cpu_softmax_propagates_nan_and_is_nan_for_all_negative_infinity():
    """Pin the three float edge cases on the sequential lowering."""
    values, offsets, expected = _semantic_edge_case()

    actual = cpu_softmax_oracle(values, offsets)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_softmax_propagates_nan_and_is_nan_for_all_negative_infinity():
    """Pin the same three edge cases on the one-CTA kernel."""
    host_values, host_offsets, expected = _semantic_edge_case()
    output = torch.empty(6, device="cuda")

    launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)

    torch.testing.assert_close(
        output.cpu(), expected, rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_softmax_leaves_output_beyond_final_offset_untouched():
    """An empty CTA that stored would write a neighbour's first slot."""
    host_values, host_offsets = _softmax_case([0, 5, 0, 7, 0])
    covered = int(host_offsets[-1])
    output = torch.full((covered + 8,), -1.0, device="cuda")

    launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)

    torch.testing.assert_close(
        output[covered:].cpu(), torch.full((8,), -1.0), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rejects_softmax_output_shorter_than_final_offset():
    """The softmax ABI needs one output element per covered value."""
    host_values, host_offsets = _softmax_case([3, 4])
    output = torch.empty(6, device="cuda")

    with pytest.raises(ValueError, match="output has 6 elements for 7 values"):
        launch_softmax_gpu(host_values.cuda(), host_offsets.cuda(), output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rejects_softmax_output_aliasing_values():
    """map_store's no-alias obligation is enforced before any launch."""
    host_values, host_offsets = _softmax_case([4, 4])
    values = host_values.cuda()

    with pytest.raises(ValueError, match="must not overlap the values buffer"):
        launch_softmax_gpu(values, host_offsets.cuda(), values.narrow(0, 0, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rejects_softmax_output_aliasing_offsets():
    """The kernel re-reads offsets, so an aliased output voids validation."""
    host_values, host_offsets = _softmax_case([4, 4])
    count = host_offsets.numel()
    shared = torch.empty(16, dtype=torch.int32, device="cuda")
    shared[:count] = host_offsets.cuda()
    offsets = shared.narrow(0, 0, count)
    output = shared.view(torch.float32).narrow(0, 0, 8)

    with pytest.raises(ValueError, match="must not overlap the offsets buffer"):
        launch_softmax_gpu(host_values.cuda(), offsets, output)
