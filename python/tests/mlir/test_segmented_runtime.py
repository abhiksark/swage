# python/tests/mlir/test_segmented_runtime.py
"""Differential qualification for native segmented reductions."""

import pytest
import torch
from swage._segmented_qualification import (
    _validate_counts,
    _validate_offsets,
    _validate_tensors,
    cpu_oracle,
    launch_gpu,
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
def test_rejects_malformed_offsets(
    offsets, value_count, output_count, message
):
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
        output.cpu(), _pytorch_reference(host_values, host_offsets, kind),
        rtol=1e-5, atol=1e-5,
    )
    torch.testing.assert_close(
        output.cpu(), cpu_oracle(host_values, host_offsets, kind),
        rtol=1e-5, atol=1e-5,
    )


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
        output.cpu(), cpu_oracle(host_values, host_offsets, "max"),
        equal_nan=True,
    )
