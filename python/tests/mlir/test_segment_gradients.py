# python/tests/mlir/test_segment_gradients.py
"""Gradients of the public segmented calls.

ADR-0024 decides how `swage.segment_reduce` and `swage.segment_softmax`
record a gradient. This file pins, on the GPU:

- That a call with values that require grad records a node, and that
  `gradcheck` and `gradgradcheck` pass in float64 at ranks one and two,
  with int32 and int64 offsets, empty segments, and rows past the final
  offset.
- The gradient of each kind against a formula evaluated on the call's own
  forward output and against the gradient of `torch.segment_reduce`. No
  test pins the forward bits of rank-two values, which another schedule
  may change.
- The interactions: `out`, `torch.no_grad()`, `torch.inference_mode()`,
  in-place writes, an expanded upstream gradient, determinism, streams,
  `torch.compile`, `SWAGE_NO_COMPILE=1`, and resources over many calls.
"""

import gc
import os
import subprocess
import sys
import textwrap

import pytest
import swage
import torch
from swage import _autograd, _runtime, _segments
from swage import _segmented_qualification as qualification
from test_public_segments import (  # noqa: F401 (fixtures)
    _public_depth,
    _selection_lengths,
    driver_calls,
    empty_kernel_memo,
    event_counts,
)
from test_segmented_numerics import _softmax_bound
from test_segmented_runtime import _EPS32, _bits, _offsets

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
pytestmark = _needs_cuda

KINDS = ["sum", "mean", "max", "min"]
EXTREMES = ["max", "min"]
DTYPES = [torch.float32, torch.float64]
OFFSET_DTYPES = [torch.int32, torch.int64]
# None stands for rank-one values; a number is the column count of [N, D].
COLUMNS = [None, 1, 3, 64]
# Segments of a gradcheck batch: two empty ones, and two rows past the final
# offset that belong to no segment.
_CHECK_LENGTHS = [2, 0, 5, 1, 0, 3]
_PAST_FINAL = 2


def _values(rows, columns, dtype, seed):
    """Return values on the device that require grad.

    The values are distinct multiples of 1/64 in a shuffled order, so no
    two are equal and a maximum or a minimum has no tie, also after a
    perturbation of `gradcheck`. Every value is exact in float32.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (rows,) if columns is None else (rows, columns)
    count = rows * (1 if columns is None else columns)
    order = torch.randperm(count, generator=generator).to(torch.float64)
    values = ((order - count / 2) / 64).reshape(shape)
    return values.to(dtype).cuda().requires_grad_()


def _batch(
    lengths,
    *,
    columns=None,
    dtype=torch.float64,
    offsets_dtype=torch.int32,
    past_final=0,
    seed=0,
):
    """Return values that require grad and their offsets, on the device."""
    offsets = torch.tensor(_offsets(lengths), dtype=offsets_dtype).cuda()
    return _values(sum(lengths) + past_final, columns, dtype, seed), offsets


def _empty(offsets, like):
    """Return a mask of the empty segments that broadcasts against `like`."""
    return (offsets[1:] == offsets[:-1]).view(-1, *[1] * (like.dim() - 1))


def _masked_reduce(kind, offsets):
    """Return the call with the results of empty segments set to zero.

    An empty segment's result is a constant NaN or infinity, whose numerical
    Jacobian column is NaN, so `gradcheck` compares nothing there.
    """

    def reduce(values):
        result = swage.segment_reduce(values, offsets, kind)
        return result.masked_fill(_empty(offsets, result), 0.0)

    return reduce


def _gradient(kind, values, offsets, upstream):
    """Return the gradient of one call for one upstream gradient."""
    result = swage.segment_reduce(values, offsets, kind)
    (gradient,) = torch.autograd.grad(result, values, upstream)
    return result, gradient


def _torch_gradient(kind, values, offsets, upstream):
    """Return the gradient of `torch.segment_reduce` for the same input."""
    leaf = values.detach().clone().requires_grad_()
    result = torch.segment_reduce(leaf, kind, offsets=offsets.long(), axis=0)
    (gradient,) = torch.autograd.grad(result, leaf, upstream)
    return gradient


def _lengths(offsets, like):
    """Return the segment lengths in float64, broadcast against `like`."""
    lengths = (offsets[1:] - offsets[:-1]).to(torch.float64)
    return lengths.view(-1, *[1] * (like.dim() - 1))


def _broadcast(per_segment, offsets, rows):
    """Copy a per-segment tensor to its rows on the host, 0 past the end."""
    host_offsets = offsets.cpu().tolist()
    shape = (rows, *per_segment.shape[1:])
    result = torch.zeros(shape, dtype=per_segment.dtype)
    for segment, (begin, end) in enumerate(zip(host_offsets, host_offsets[1:])):
        result[begin:end] = per_segment[segment].cpu()
    return result


@pytest.mark.parametrize("columns", [None, 3])
@pytest.mark.parametrize("kind", KINDS)
def test_a_call_with_values_that_require_grad_records_a_node(kind, columns):
    """Return a result with a grad_fn, and send a gradient to the values."""
    values, offsets = _batch([2, 3], columns=columns)

    result = swage.segment_reduce(values, offsets, kind)
    result.sum().backward()

    assert result.requires_grad
    assert type(result.grad_fn).__name__ == "SegmentReduceBackward"
    assert values.grad is not None
    assert values.grad.shape == values.shape
    assert offsets.grad is None


@pytest.mark.parametrize("offsets_dtype", OFFSET_DTYPES, ids=["i32", "i64"])
@pytest.mark.parametrize("columns", COLUMNS, ids=["r1", "c1", "c3", "c64"])
@pytest.mark.parametrize("kind", KINDS)
def test_gradcheck_and_gradgradcheck_in_float64(kind, columns, offsets_dtype):
    """Check the first and second derivatives numerically in float64.

    The batch has empty segments and rows past the final offset. Segments
    stay a few rows long, so the rounding of the forward stays far below the
    tolerance of the check. With 64 columns the check runs in fast mode,
    which compares one random projection of the Jacobian.
    """
    values, offsets = _batch(
        _CHECK_LENGTHS,
        columns=columns,
        offsets_dtype=offsets_dtype,
        past_final=_PAST_FINAL,
    )
    reduce = _masked_reduce(kind, offsets)
    fast = columns == 64

    assert torch.autograd.gradcheck(reduce, (values,), fast_mode=fast)
    assert torch.autograd.gradgradcheck(reduce, (values,), fast_mode=fast)


@pytest.mark.parametrize("kind", KINDS)
def test_gradcheck_across_the_schedules_of_rank_one(kind):
    """Check segments on both sides of the warp and the chunk limits.

    Lengths 32 and 33 change from the warp to the CTA tree, and 4096 and
    4097 from the CTA tree to split chunks. The check runs in fast mode.
    """
    values, offsets = _batch([32, 33, 4096, 4097], seed=1)
    reduce = _masked_reduce(kind, offsets)

    assert torch.autograd.gradcheck(reduce, (values,), fast_mode=True)
    assert torch.autograd.gradgradcheck(reduce, (values,), fast_mode=True)


@pytest.mark.parametrize("offsets_dtype", OFFSET_DTYPES, ids=["i32", "i64"])
@pytest.mark.parametrize("columns", COLUMNS, ids=["r1", "c1", "c3", "c64"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_the_gradient_of_a_sum_is_an_exact_copy(dtype, columns, offsets_dtype):
    """Copy the gradient of a segment to its rows, as PyTorch does.

    Every row of a segment receives the upstream gradient of the segment
    unchanged, signed zeros and NaN included. The rows past the final
    offset receive zero. The gradient equals that of `torch.segment_reduce`
    bit for bit.
    """
    values, offsets = _batch(
        [3, 0, 40, 1],
        columns=columns,
        dtype=dtype,
        offsets_dtype=offsets_dtype,
        past_final=2,
    )
    generator = torch.Generator().manual_seed(4)
    shape = (4,) if columns is None else (4, columns)
    upstream = torch.randn(shape, generator=generator, dtype=dtype)
    upstream.view(-1)[:3] = torch.tensor([-0.0, float("nan"), float("inf")])
    upstream = upstream.cuda()

    _, gradient = _gradient("sum", values, offsets, upstream)

    expected = _broadcast(upstream, offsets, values.shape[0])
    assert _bits(gradient.cpu()) == _bits(expected)
    assert _bits(gradient.cpu()) == _bits(
        _torch_gradient("sum", values, offsets, upstream).cpu()
    )


@pytest.mark.parametrize("offsets_dtype", OFFSET_DTYPES, ids=["i32", "i64"])
@pytest.mark.parametrize("columns", COLUMNS, ids=["r1", "c1", "c3", "c64"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_the_gradient_of_a_mean_is_one_division(dtype, columns, offsets_dtype):
    """Divide the gradient of a segment once by its length.

    The expected value is the division in float64 rounded to the dtype,
    which equals the correctly rounded division in the dtype. Against
    `torch.segment_reduce` the gradient agrees to within one rounding of
    the dtype.
    """
    values, offsets = _batch(
        [3, 0, 7, 1, 4097],
        columns=columns,
        dtype=dtype,
        offsets_dtype=offsets_dtype,
        past_final=2,
    )
    generator = torch.Generator().manual_seed(5)
    shape = (5,) if columns is None else (5, columns)
    upstream = torch.randn(shape, generator=generator, dtype=dtype).cuda()

    _, gradient = _gradient("mean", values, offsets, upstream)

    lengths = _lengths(offsets, upstream).clamp(min=1)
    share = (upstream.double() / lengths).to(dtype)
    expected = _broadcast(share, offsets, values.shape[0])
    assert _bits(gradient.cpu()) == _bits(expected)
    theirs = _torch_gradient("mean", values, offsets, upstream)
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(gradient, theirs, rtol=eps, atol=0)


@pytest.mark.parametrize("offsets_dtype", OFFSET_DTYPES, ids=["i32", "i64"])
@pytest.mark.parametrize("columns", COLUMNS, ids=["r1", "c1", "c3", "c64"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("kind", EXTREMES)
def test_an_extreme_without_a_tie_sends_the_whole_gradient_to_it(
    kind, dtype, columns, offsets_dtype
):
    """Give the gradient of a segment to its one extreme element.

    Without a tie the gradient equals that of `torch.segment_reduce` bit
    for bit, for an upstream gradient of either sign.
    """
    values, offsets = _batch(
        [3, 0, 40, 1, 4097],
        columns=columns,
        dtype=dtype,
        offsets_dtype=offsets_dtype,
        past_final=2,
        seed=6,
    )
    generator = torch.Generator().manual_seed(6)
    shape = (5,) if columns is None else (5, columns)
    upstream = torch.randn(shape, generator=generator, dtype=dtype).cuda()

    result, gradient = _gradient(kind, values, offsets, upstream)

    extreme = _broadcast(result.detach(), offsets, values.shape[0])
    tied = values.detach().cpu() == extreme
    tied[sum([3, 0, 40, 1, 4097]):] = False
    expected = torch.where(
        tied, _broadcast(upstream, offsets, values.shape[0]), 0.0
    )
    assert _bits(gradient.cpu()) == _bits(expected)
    assert _bits(gradient.cpu()) == _bits(
        _torch_gradient(kind, values, offsets, upstream).cpu()
    )


# One segment per row: values, the result of a maximum, and which elements
# tie with it.
_TIES = [
    ([3.0, 3.0, 1.0], [True, True, False]),
    ([2.0, 2.0, 2.0], [True, True, True]),
    ([4.0, float("nan"), float("nan")], [False, True, True]),
    ([float("-inf")] * 3, [True, True, True]),
    ([-0.0, 0.0, -1.0], [True, True, False]),
    ([float("inf"), 1.0, float("inf")], [True, False, True]),
]


def _tie_batch(kind):
    """Return the tie segments, each of three rows, for one kind.

    For a minimum every value is negated, which keeps the ties of a
    maximum. One row past the final offset holds zero, which equals the
    zero row that the gather of the backward pads with.
    """
    sign = 1.0 if kind == "max" else -1.0
    rows = [sign * value for values, _ in _TIES for value in values]
    values = torch.tensor([*rows, 0.0], dtype=torch.float64)
    offsets = torch.tensor(_offsets([3] * len(_TIES)), dtype=torch.int32)
    tied = torch.tensor([flag for _, flags in _TIES for flag in flags])
    return values.cuda().requires_grad_(), offsets.cuda(), tied


@pytest.mark.parametrize(
    "upstream", [2.0, -3.0, float("nan"), float("inf")], ids=str
)
@pytest.mark.parametrize("kind", EXTREMES)
def test_tied_elements_share_the_gradient_equally(kind, upstream):
    """Divide the gradient once among the tied elements, for any sign.

    The elements that equal the result tie, `-0.0` and `0.0` included, and
    the NaN elements tie when the result is NaN. Every other element and
    the row past the final offset receive exactly `0.0`, also when the
    gradient is infinite or NaN.
    """
    values, offsets, tied = _tie_batch(kind)
    count = len(_TIES)

    _, gradient = _gradient(
        kind, values, offsets, torch.full((count,), upstream).double().cuda()
    )

    shares = tied.view(count, 3).sum(1, keepdim=True).double()
    expected = torch.where(tied.view(count, 3), upstream / shares, 0.0)
    expected = torch.cat((expected.view(-1), torch.zeros(1)))
    assert _bits(gradient.cpu()) == _bits(expected.double())


@pytest.mark.parametrize("kind", EXTREMES)
def test_ties_hold_per_column(kind):
    """Count the ties of every column of `[N, D]` values on its own.

    Column 0 holds the tie segments, and column 1 holds distinct values,
    whose extreme is one element per segment.
    """
    values, offsets, tied = _tie_batch(kind)
    rows = values.shape[0]
    distinct = torch.arange(rows, dtype=torch.float64).cuda()
    columns = torch.stack((values.detach(), distinct), 1)
    columns = columns.contiguous().requires_grad_()
    count = len(_TIES)
    upstream = torch.full((count, 2), -2.0, dtype=torch.float64).cuda()

    _, gradient = _gradient(kind, columns, offsets, upstream)

    gradient = gradient.cpu()
    shares = tied.view(count, 3).sum(1, keepdim=True).double()
    first = torch.where(tied.view(count, 3), -2.0 / shares, 0.0)
    assert _bits(gradient[: count * 3, 0]) == _bits(first.view(-1))
    second = torch.zeros(count, 3, dtype=torch.float64)
    second[:, 2 if kind == "max" else 0] = -2.0
    assert _bits(gradient[: count * 3, 1]) == _bits(second.view(-1))
    assert _bits(gradient[-1]) == _bits(torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize(
    "upstream", [2.0, -2.0, float("nan")], ids=["positive", "negative", "nan"]
)
@pytest.mark.parametrize("kind", EXTREMES)
def test_pytorch_gives_ties_the_same_gradient_on_cuda_and_on_the_cpu(
    kind, upstream
):
    """Record how `torch.segment_reduce` treats ties on CUDA.

    The documented difference between the two tie rules was found on the
    CPU. This compares the gradient of `torch.segment_reduce` on CUDA with
    the one on the CPU for the same tied segments, and the Swage gradient
    with both where the rules agree: for a positive gradient, PyTorch also
    divides it equally.
    """
    values, offsets, _ = _tie_batch(kind)
    count = len(_TIES)
    upstream_cuda = torch.full((count,), upstream).double().cuda()

    cuda = _torch_gradient(kind, values, offsets, upstream_cuda)
    cpu = _torch_gradient(
        kind, values.cpu(), offsets.cpu(), upstream_cuda.cpu()
    )
    _, ours = _gradient(kind, values, offsets, upstream_cuda)

    assert _bits(cuda.cpu()) == _bits(cpu)
    # A NaN divided by the tie count is NaN, so only a negative gradient
    # shows the difference of the two rules.
    if upstream < 0:
        assert _bits(ours.cpu()) != _bits(cpu)
    else:
        assert _bits(ours.cpu()) == _bits(cpu)


@pytest.mark.parametrize("columns", [None, 3])
@pytest.mark.parametrize("kind", KINDS)
def test_empty_segments_and_rows_past_the_final_offset_get_nothing(
    kind, columns
):
    """Drop the gradient of an empty segment and give zero past the end.

    The upstream gradient of every empty segment is NaN, which must reach
    no element.
    """
    values, offsets = _batch([0, 3, 0, 0, 2, 0], columns=columns, past_final=3)
    shape = (6,) if columns is None else (6, columns)
    upstream = torch.ones(shape, dtype=torch.float64)
    upstream[[0, 2, 3, 5]] = float("nan")

    _, gradient = _gradient(kind, values, offsets, upstream.cuda())

    gradient = gradient.cpu()
    assert gradient.isfinite().all()
    assert _bits(gradient[5:]) == _bits(torch.zeros_like(gradient[5:]))


@pytest.mark.parametrize("kind", KINDS)
def test_a_linear_layer_trains_through_the_pooling(kind):
    """Train a layer whose output is pooled per segment.

    The gradient at the layer output is the gradient of the pooling. It
    equals the gradient that `torch.segment_reduce` gives, to within one
    float32 rounding for a mean, and the parameter gradients follow.
    """
    torch.manual_seed(7)
    layer = torch.nn.Linear(8, 4).cuda()
    reference = torch.nn.Linear(8, 4).cuda()
    reference.load_state_dict(layer.state_dict())
    features = torch.randn(30, 8).cuda()
    offsets = torch.tensor([0, 4, 4, 19, 30], dtype=torch.int64).cuda()
    weight = torch.randn(4, 4).cuda()

    hidden = layer(features)
    hidden.retain_grad()
    (swage.segment_reduce(hidden, offsets, kind) * weight).sum().backward()
    theirs = reference(features)
    theirs.retain_grad()
    pooled = torch.segment_reduce(theirs, kind, offsets=offsets, axis=0)
    (pooled * weight).sum().backward()

    eps = torch.finfo(torch.float32).eps
    torch.testing.assert_close(hidden.grad, theirs.grad, rtol=eps, atol=0)
    for ours, theirs in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(ours.grad, theirs.grad)


@pytest.mark.parametrize("kind", KINDS)
def test_no_grad_and_inference_mode_record_nothing(kind):
    """Record no gradient where PyTorch records none, and take `out` there."""
    values, offsets = _batch([2, 3], columns=3)
    expected = swage.segment_reduce(values.detach(), offsets, kind)

    with torch.no_grad():
        plain = swage.segment_reduce(values, offsets, kind)
        out = torch.empty(2, 3, dtype=values.dtype, device="cuda")
        written = swage.segment_reduce(values, offsets, kind, out=out)
    with torch.inference_mode():
        inferred = swage.segment_reduce(values, offsets, kind)

    for result in (plain, written, inferred):
        assert not result.requires_grad
        assert result.grad_fn is None
        assert _bits(result.cpu()) == _bits(expected.cpu())
    assert written is out
    assert inferred.is_inference()


@pytest.mark.parametrize("kind", KINDS)
def test_out_is_refused_while_recording(kind):
    """Refuse `out` with a ValueError and enqueue nothing."""
    values, offsets = _batch([2, 3])
    out = torch.full((2,), -5.0, dtype=values.dtype, device="cuda")

    with pytest.raises(ValueError, match="^out must be None while values"):
        swage.segment_reduce(values, offsets, kind, out=out)

    torch.cuda.synchronize()
    assert (out == -5.0).all()


@pytest.mark.parametrize("kind", KINDS)
def test_offsets_made_under_inference_mode_are_kept_as_a_copy(kind):
    """Record a gradient with inference offsets made inside the context."""
    values, offsets = _batch([2, 0, 3], columns=3)
    with torch.inference_mode():
        inferred = offsets.clone()

    _, expected = _gradient(kind, values, offsets, torch.ones(3, 3).cuda())
    _, gradient = _gradient(kind, values, inferred, torch.ones(3, 3).cuda())

    assert _bits(gradient.cpu()) == _bits(expected.cpu())


@pytest.mark.parametrize("kind", KINDS)
def test_an_in_place_change_to_the_offsets_fails_the_backward(kind):
    """Raise the standard error when the kept offsets changed in place."""
    values, offsets = _batch([2, 3])
    result = swage.segment_reduce(values, offsets, kind)

    offsets.add_(0)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


@pytest.mark.parametrize("kind", EXTREMES)
def test_an_in_place_change_to_values_or_result_fails_an_extreme(kind):
    """Raise when the kept values or the kept result changed in place."""
    for change in ("values", "result"):
        values, offsets = _batch([2, 3])
        result = swage.segment_reduce(values, offsets, kind)
        with torch.no_grad():
            (values if change == "values" else result).add_(0.0)

        with pytest.raises(
            RuntimeError, match="modified by an inplace operation"
        ):
            result.sum().backward()


@pytest.mark.parametrize("kind", ["sum", "mean"])
def test_an_in_place_change_to_the_result_keeps_the_backward(kind):
    """A sum or a mean keeps no result, so writing to it is recorded."""
    values, offsets = _batch([2, 3])
    result = swage.segment_reduce(values, offsets, kind)

    result.mul_(2.0)
    result.sum().backward()

    divisor = torch.tensor([2.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    expected = 2.0 / divisor if kind == "mean" else torch.full((5,), 2.0)
    assert _bits(values.grad.cpu()) == _bits(expected.double())


def test_a_write_to_the_offsets_that_pytorch_does_not_count():
    """Stay inside the buffers when the kept offsets change unseen.

    A write through `.data` does not advance the version counter, so the
    backward runs on offsets that were not validated. Its result is not a
    validated gradient, and every index stays clamped, so the process and
    its CUDA context stay usable. It runs in a process of its own.
    """
    script = textwrap.dedent(
        """
        import swage, torch
        values = torch.randn(12, 3, dtype=torch.float64).cuda()
        values.requires_grad_()
        offsets = torch.tensor([0, 2, 7, 12], dtype=torch.int32).cuda()
        for kind in ("sum", "mean"):
            result = swage.segment_reduce(values, offsets, kind)
            offsets.data.copy_(torch.tensor([0, 9, 1, 40], dtype=torch.int32))
            result.sum().backward()
            torch.cuda.synchronize()
            offsets.data.copy_(torch.tensor([0, 2, 7, 12], dtype=torch.int32))
        again = swage.segment_reduce(values.detach(), offsets, "sum")
        print(again.shape[0], bool(values.grad.isfinite().all()))
        """
    )
    environment = dict(os.environ)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["3", "True"]


@pytest.mark.parametrize("columns", [None, 3])
@pytest.mark.parametrize("kind", KINDS)
def test_an_expanded_upstream_gradient_reaches_the_sum_kernel(kind, columns):
    """Take the stride-0 gradient that `.sum()` sends, in both orders.

    The second derivative with respect to the upstream gradient is a
    segment sum, which the sum kernel computes from a contiguous copy of
    the expanded gradient that `.sum()` sends back.
    """
    values, offsets = _batch([2, 0, 3, 1], columns=columns, past_final=1)
    result = swage.segment_reduce(values, offsets, kind)
    upstream = torch.ones_like(result, requires_grad=True)

    (gradient,) = torch.autograd.grad(
        result, values, upstream, create_graph=True
    )
    (second,) = torch.autograd.grad(gradient.sum(), upstream)

    lengths = _lengths(offsets, second)
    expected = lengths if kind == "sum" else torch.ones_like(lengths)
    expected = expected.expand_as(second).masked_fill(
        _empty(offsets, second).cpu(), 0.0
    )
    assert _bits(second.cpu()) == _bits(expected.contiguous())


@pytest.mark.parametrize("kind", KINDS)
def test_derivatives_are_deterministic(kind):
    """Give the same bits twice, also under deterministic algorithms."""
    values, offsets = _batch([2, 0, 300, 4097], columns=None, seed=8)
    upstream = torch.randn(4, dtype=torch.float64).cuda()

    def derivatives():
        result = swage.segment_reduce(values, offsets, kind)
        weight = upstream.clone().requires_grad_()
        (first,) = torch.autograd.grad(
            result, values, weight, create_graph=True
        )
        (second,) = torch.autograd.grad((first * first).sum(), weight)
        return _bits(first.detach().cpu()), _bits(second.cpu())

    before = derivatives()
    torch.use_deterministic_algorithms(True)
    try:
        during = derivatives()
    finally:
        torch.use_deterministic_algorithms(False)

    assert before == during == derivatives()


@pytest.mark.parametrize("kind", KINDS)
def test_reduction_gradients_do_not_depend_on_the_batch(kind):
    """Keep the gradient of a segment when the selection rule changes.

    A batch of as many 4097 to 8192-element segments as the device has SMs
    is reduced by another tree than the same batch with one segment fewer.
    The gradient of every shared segment keeps its bits.
    """
    count = torch.cuda.get_device_properties(0).multi_processor_count
    lengths = _selection_lengths(count)
    values, offsets = _batch(lengths, dtype=torch.float32, seed=9)
    fewer = offsets[:count].clone()
    upstream = torch.randn(count, dtype=torch.float32).cuda()

    _, full = _gradient(kind, values, offsets, upstream)
    _, short = _gradient(kind, values, fewer, upstream[: count - 1])

    shared = int(offsets[count - 1])
    assert _bits(full[:shared].cpu()) == _bits(short[:shared].cpu())


def test_a_second_derivative_runs_on_the_stream_of_its_forward(monkeypatch):
    """Enqueue the sum kernel of a second derivative on the forward stream.

    The autograd engine runs a node on the stream that was current when
    the node was made, so a second derivative of a call made on a side
    stream sums on that stream.
    """
    values, offsets = _batch([2, 0, 5, 40], columns=3)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        result = swage.segment_reduce(values, offsets, "sum")
        upstream = torch.ones_like(result, requires_grad=True)
        (gradient,) = torch.autograd.grad(
            result, values, upstream, create_graph=True
        )
    side.synchronize()
    driver = _runtime._get_driver()
    streams = []

    def recording(original):
        def launch(function, grid, block, stream, arguments):
            streams.append(stream)
            return original(function, grid, block, stream, arguments)

        return launch

    for name in ("launch_segmented", "launch_segmented_tasks"):
        monkeypatch.setattr(driver, name, recording(getattr(driver, name)))

    (second,) = torch.autograd.grad((gradient * 2.0).sum(), upstream)
    torch.cuda.synchronize()

    assert streams
    assert set(streams) == {side.cuda_stream}
    assert _bits(second.cpu()[:, 0]) == _bits(
        torch.tensor([4.0, 0.0, 10.0, 80.0], dtype=torch.float64)
    )


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("kind", KINDS)
def test_gradients_flow_through_a_compiled_function(kind, backend, monkeypatch):
    """Record the gradient of a call that a compiled function makes.

    The call is a graph break; the Functions are made inside the trace, at
    the first call of the process. The gradient at the input equals that of
    the same function run eagerly, bit for bit.
    """
    torch._dynamo.reset()
    monkeypatch.setattr(_segments, "_UNTRACED", {})
    monkeypatch.setattr(_autograd, "_FUNCTIONS", {})
    values, offsets = _batch([3, 0, 500, 497], columns=8, dtype=torch.float32)

    def model(values):
        return swage.segment_reduce(values * 2.0, offsets, kind).tanh()

    compiled = torch.compile(model, backend=backend)
    (ours,) = torch.autograd.grad(compiled(values).sum(), values)
    (eager,) = torch.autograd.grad(model(values).sum(), values)

    assert _autograd._FUNCTIONS
    torch.testing.assert_close(ours, eager, rtol=1e-6, atol=0)
    if backend == "eager":
        assert _bits(ours.cpu()) == _bits(eager.cpu())


@pytest.mark.parametrize("kind", KINDS)
def test_the_first_derivative_of_a_reduction_needs_no_kernel(
    kind, empty_kernel_memo, monkeypatch  # noqa: F811 (fixture)
):
    """Run the backward with compiling switched off and no kernel held.

    The backward of a reduction is PyTorch operations only. A second
    derivative needs the sum kernel of its batch, and refuses to compile
    it.
    """
    monkeypatch.setenv("SWAGE_NO_COMPILE", "0")
    values, offsets = _batch([2, 0, 3])
    result = swage.segment_reduce(values, offsets, kind)
    upstream = torch.ones_like(result, requires_grad=True)
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")

    (gradient,) = torch.autograd.grad(
        result, values, upstream, create_graph=True
    )
    with pytest.raises(RuntimeError, match="^SWAGE_NO_COMPILE=1 refuses"):
        torch.autograd.grad(gradient.sum(), upstream)

    assert gradient.isfinite().all()


def test_calls_with_backward_passes_leave_nothing_behind(
    driver_calls, event_counts  # noqa: F811 (fixtures)
):
    """Run many calls with backward passes and keep resources flat.

    Every iteration takes offsets it has not seen, records a gradient,
    runs a first derivative, and every fifth one a second derivative.
    After two warming iterations, 100 more load no module, synchronize no
    context, create no CUDA event, and end with the device memory of the
    start, and the cycle collector finds nothing.
    """
    values = _values(4000, 3, torch.float32, 10)
    calls = 100

    def one_iteration(seed):
        generator = torch.Generator().manual_seed(seed)
        lengths = torch.randint(0, 60, (50,), generator=generator).tolist()
        offsets = torch.tensor(_offsets(lengths), dtype=torch.int32).cuda()
        kind = KINDS[seed % len(KINDS)]
        result = swage.segment_reduce(values, offsets, kind)
        upstream = torch.ones_like(result, requires_grad=seed % 5 == 0)
        (gradient,) = torch.autograd.grad(
            result, values, upstream, create_graph=seed % 5 == 0
        )
        if seed % 5 == 0:
            torch.autograd.grad(gradient.sum(), upstream)

    for seed in (0, 1):
        one_iteration(seed)
    torch.cuda.synchronize()
    loads = driver_calls["cuModuleLoadData"]
    allocated = torch.cuda.memory_allocated()
    event_counts.clear()
    gc.collect()

    gc.disable()
    try:
        for seed in range(2, calls + 2):
            one_iteration(seed)
        torch.cuda.synchronize()
        unreachable = gc.collect()
    finally:
        gc.enable()

    assert unreachable == 0
    assert torch.cuda.memory_allocated() == allocated
    assert driver_calls["cuModuleLoadData"] == loads
    assert driver_calls["cuModuleUnload"] == 0
    assert driver_calls["cuCtxSynchronize"] == 0
    assert event_counts == {}


def _softmax_case(lengths, spread, columns=None, seed=0):
    """Return float32 logits that require grad, their offsets, and `g`.

    The logits of a segment lie in `[top - spread, top]`, with both ends
    planted in every column, and the upstream gradient is standard normal.
    """
    generator = torch.Generator().manual_seed(seed)
    width = 1 if columns is None else columns
    segments = []
    for index, length in enumerate(lengths):
        top = float(index % 5) - 2.0
        block = top - spread * torch.rand(length, width, generator=generator)
        if length >= 2:
            block[0], block[-1] = top, top - spread
        segments.append(block)
    logits = torch.cat(segments)
    if columns is None:
        logits = logits.view(-1)
    upstream = torch.randn(logits.shape, generator=generator)
    offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    return (
        logits.cuda().requires_grad_(),
        offsets.cuda(),
        upstream.cuda(),
    )


def _softmax_gradient(logits, offsets, upstream):
    """Return the softmax and its gradient for one upstream gradient."""
    result = swage.segment_softmax(logits, offsets)
    (gradient,) = torch.autograd.grad(result, logits, upstream)
    return result.detach(), gradient


def _segment_slices(offsets):
    """Return the row slice of every non-empty segment."""
    host = offsets.cpu().tolist()
    return [slice(a, b) for a, b in zip(host, host[1:]) if b > a]


def _backward_bound(result, upstream, depth):
    """Bound the backward of one segment against the formula on its `y`.

    For the returned `y`, the exact gradient is `y * (g - s)` with
    `s = sum(g * y)`. The backward forms `g * y` with one rounding, sums it
    with `k` rounding additions, subtracts once, and multiplies once. To
    first order the error of an element is at most
    `(k + 3) * eps32 * max|g| * y * max(1, sum(y))`, per column.
    """
    scale = upstream.abs().amax(0) * result.sum(0).clamp(min=1.0)
    return (depth + 3) * _EPS32 * scale * result


@pytest.mark.parametrize("columns", COLUMNS, ids=["r1", "c1", "c3", "c64"])
@pytest.mark.parametrize("spread", [8, 20, 50, 80])
def test_the_softmax_backward_is_the_formula_on_its_own_result(spread, columns):
    """Bound the backward against `y * (g - s)` evaluated on the returned y.

    The sum of the backward is the sum kernel of `segment_reduce`, whose
    rounding depends on its schedule, so the bound uses `k` of that
    schedule: the rank-one bound of "Sum rounding" for scalars and
    `[N, 1]` values, and `n - 1`, which bounds every tree, for rows.
    """
    lengths = [2, 129, 1024, 4096, 1, 300]
    logits, offsets, upstream = _softmax_case(lengths, spread, columns)

    result, gradient = _softmax_gradient(logits, offsets, upstream)

    for rows in _segment_slices(offsets):
        y = result[rows].double().cpu()
        g = upstream[rows].double().cpu()
        if y.dim() == 1:
            y, g = y.view(-1, 1), g.view(-1, 1)
        exact = y * (g - (g * y).sum(0))
        length = rows.stop - rows.start
        depth = length - 1 if columns not in (None, 1) else _public_depth(
            length
        )
        actual = gradient[rows].double().cpu().view_as(exact)
        error = (actual - exact).abs()
        assert (error <= _backward_bound(y, g, depth)).all(), (
            f"rows {rows}: largest error {error.max().item():.3e}"
        )


@pytest.mark.parametrize("columns", [None, 3])
def test_the_softmax_backward_of_one_long_segment(columns):
    """Bound the backward of a segment of 100,003 rows the same way."""
    logits, offsets, upstream = _softmax_case([100_003], 20, columns, seed=3)

    result, gradient = _softmax_gradient(logits, offsets, upstream)

    y = result.double().cpu()
    g = upstream.double().cpu()
    if y.dim() == 1:
        y, g = y.view(-1, 1), g.view(-1, 1)
    exact = y * (g - (g * y).sum(0))
    depth = 100_002 if columns else _public_depth(100_003)
    error = (gradient.double().cpu().view_as(exact) - exact).abs()
    assert (error <= _backward_bound(y, g, depth)).all()


@pytest.mark.parametrize("columns", [None, 3])
@pytest.mark.parametrize("spread", [8, 50])
def test_the_softmax_gradient_agrees_with_float64_pytorch(spread, columns):
    """Compare end to end with the float64 gradient of `torch.softmax`.

    The tolerance is the error the forward may carry into the formula,
    three times the forward bound of "Ragged Softmax" times `max|g| * y`,
    plus the bound of the backward on its own result.
    """
    lengths = [2, 129, 1024, 300]
    logits, offsets, upstream = _softmax_case(lengths, spread, columns, seed=1)

    result, gradient = _softmax_gradient(logits, offsets, upstream)

    for rows in _segment_slices(offsets):
        x = logits[rows].detach().double().cpu()
        g = upstream[rows].double().cpu()
        if x.dim() == 1:
            x, g = x.view(-1, 1), g.view(-1, 1)
        leaf = x.clone().requires_grad_()
        reference = torch.softmax(leaf, 0)
        (expected,) = torch.autograd.grad(reference, leaf, g)
        length = rows.stop - rows.start
        rank_two = columns is not None
        forward = _softmax_bound(
            x, reference.detach(), length - 1 if rank_two else None
        ).amax(0)
        depth = length - 1 if rank_two else _public_depth(length)
        y = reference.detach()
        tolerance = 3 * forward * g.abs().amax(0) * y + _backward_bound(
            y, g, depth
        )
        actual = gradient[rows].double().cpu().view_as(expected)
        assert ((actual - expected).abs() <= tolerance).all()


@pytest.mark.parametrize("columns", [None, 3])
def test_the_softmax_gradient_follows_pytorch_on_special_values(columns):
    """Give NaN to a NaN segment only, and zero to a logit of probability 0.

    A segment that holds a NaN or a positive infinity, or only negative
    infinities, has NaN results, and its gradient is NaN, as the float64
    gradient of `torch.softmax` is. A negative infinity beside a finite
    maximum has the result `0.0` and a zero gradient. The other segments
    keep finite gradients.
    """
    length = 33
    ramp = torch.linspace(-2.0, 2.0, length)
    segments = [ramp.clone() for _ in range(5)]
    segments[1][length // 2] = float("nan")
    segments[2][-1] = float("inf")
    segments[3] = torch.full((length,), float("-inf"))
    segments[4][0] = float("-inf")
    logits = torch.cat(segments)
    if columns is not None:
        logits = torch.stack([logits] * columns, 1)
    offsets = torch.tensor(_offsets([length] * 5), dtype=torch.int32)
    generator = torch.Generator().manual_seed(2)
    upstream = torch.randn(logits.shape, generator=generator)
    values = logits.cuda().requires_grad_()

    _, gradient = _softmax_gradient(values, offsets.cuda(), upstream.cuda())

    leaf = logits.double().view(5, length, -1).clone().requires_grad_()
    reference = torch.softmax(leaf, 1)
    (expected,) = torch.autograd.grad(
        reference, leaf, upstream.double().view(5, length, -1)
    )
    actual = gradient.cpu().view(5, length, -1)
    assert torch.equal(actual.isnan(), expected.isnan())
    assert actual[1:4].isnan().all()
    assert actual[[0, 4]].isfinite().all()
    assert (actual[4, 0] == 0.0).all()


@pytest.mark.parametrize("columns", [None, 3])
def test_the_softmax_second_derivative_agrees_with_float64_pytorch(columns):
    """Compare a Hessian-vector product with float64 `torch.softmax`.

    There is no float64 softmax, so `gradgradcheck` cannot run. The product
    of the second derivative with a vector is compared with the float64
    product instead, within `1e-5` of the largest magnitude of the
    expected product of each segment.
    """
    lengths = [2, 40, 129, 300]
    logits, offsets, upstream = _softmax_case(lengths, 8, columns, seed=4)
    generator = torch.Generator().manual_seed(5)
    direction = torch.randn(logits.shape, generator=generator).cuda()

    result = swage.segment_softmax(logits, offsets)
    (gradient,) = torch.autograd.grad(
        result, logits, upstream, create_graph=True
    )
    (product,) = torch.autograd.grad(gradient, logits, direction)

    for rows in _segment_slices(offsets):
        x = logits[rows].detach().double().cpu()
        g = upstream[rows].double().cpu()
        v = direction[rows].double().cpu()
        leaf = x.clone().requires_grad_()
        (first,) = torch.autograd.grad(
            torch.softmax(leaf, 0), leaf, g, create_graph=True
        )
        (expected,) = torch.autograd.grad(first, leaf, v)
        scale = expected.abs().amax(0).clamp(min=1e-30)
        actual = product[rows].double().cpu()
        assert ((actual - expected).abs() <= 1e-5 * scale).all()


def test_the_softmax_backward_is_refused_under_capture_by_its_name():
    """Refuse the sum of a softmax backward on a capturing stream.

    The autograd engine runs the backward on the stream of its forward, so
    a forward on a side stream and a capture on that stream put the
    backward under capture. It raises before its host copy, in a message
    that names the backward, and the capture stays usable.
    """
    logits, offsets, upstream = _softmax_case([3, 40], 8, seed=6)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        result = swage.segment_softmax(logits, offsets)
        doubled = torch.zeros_like(upstream)
    side.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, stream=side):
        with pytest.raises(
            RuntimeError,
            match=(
                "^the backward of segment_softmax cannot run while the "
                "current stream captures a CUDA graph"
            ),
        ):
            torch.autograd.grad(result, logits, upstream)
        torch.mul(upstream, 2.0, out=doubled)

    doubled.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(doubled, 2 * upstream)


def test_the_softmax_backward_runs_on_the_stream_of_its_forward(monkeypatch):
    """Enqueue the sum of the softmax backward on the forward stream."""
    logits, offsets, upstream = _softmax_case([3, 40, 300], 8, columns=3)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        result = swage.segment_softmax(logits, offsets)
    side.synchronize()
    driver = _runtime._get_driver()
    streams = []

    def recording(original):
        def launch(function, grid, block, stream, arguments):
            streams.append(stream)
            return original(function, grid, block, stream, arguments)

        return launch

    for name in ("launch_segmented", "launch_segmented_tasks"):
        monkeypatch.setattr(driver, name, recording(getattr(driver, name)))

    torch.autograd.grad(result, logits, upstream)
    torch.cuda.synchronize()

    assert streams
    assert set(streams) == {side.cuda_stream}


def test_the_softmax_backward_refuses_a_sum_kernel_not_held(
    empty_kernel_memo, monkeypatch  # noqa: F811 (fixture)
):
    """Raise under `SWAGE_NO_COMPILE=1` when the sum kernel is not held."""
    monkeypatch.setenv("SWAGE_NO_COMPILE", "0")
    logits, offsets, upstream = _softmax_case([3, 40], 8)
    result = swage.segment_softmax(logits, offsets)
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )
    monkeypatch.setenv("SWAGE_NO_COMPILE", "1")

    with pytest.raises(RuntimeError, match="^SWAGE_NO_COMPILE=1 refuses"):
        torch.autograd.grad(result, logits, upstream)


def test_a_softmax_with_values_that_require_grad_records_a_node():
    """Return a result with a grad_fn, and send a gradient to the values."""
    logits, offsets, upstream = _softmax_case([2, 0, 5], 8, columns=3)

    result = swage.segment_softmax(logits, offsets)
    result.backward(upstream)

    assert type(result.grad_fn).__name__ == "SegmentSoftmaxBackward"
    assert logits.grad.shape == logits.shape


def test_the_softmax_records_nothing_without_grad_mode_and_takes_out():
    """Record no gradient under `no_grad` and `inference_mode`."""
    logits, offsets, _ = _softmax_case([2, 5], 8)
    expected = swage.segment_softmax(logits.detach(), offsets)

    with torch.no_grad():
        out = torch.empty_like(expected)
        written = swage.segment_softmax(logits, offsets, out=out)
    with torch.inference_mode():
        inferred = swage.segment_softmax(logits, offsets)

    assert written is out
    for result in (written, inferred):
        assert not result.requires_grad
        assert _bits(result.cpu()) == _bits(expected.cpu())
    with pytest.raises(ValueError, match="^out must be None while values"):
        swage.segment_softmax(logits, offsets, out=torch.empty_like(out))


def test_the_softmax_keeps_its_result_and_its_offsets():
    """Raise when the kept result or the kept offsets changed in place.

    Offsets made inside `torch.inference_mode()` are kept as a copy, and
    give the gradient of ordinary offsets.
    """
    for change in ("result", "offsets"):
        logits, offsets, upstream = _softmax_case([2, 5], 8)
        result = swage.segment_softmax(logits, offsets)
        with torch.no_grad():
            (result if change == "result" else offsets).add_(0)
        with pytest.raises(
            RuntimeError, match="modified by an inplace operation"
        ):
            torch.autograd.grad(result, logits, upstream)

    logits, offsets, upstream = _softmax_case([2, 5], 8)
    with torch.inference_mode():
        inferred = offsets.clone()
    _, expected = _softmax_gradient(logits, offsets, upstream)
    _, gradient = _softmax_gradient(logits, inferred, upstream)
    assert _bits(gradient.cpu()) == _bits(expected.cpu())


@pytest.mark.parametrize("columns", [None, 3])
def test_the_softmax_takes_an_expanded_upstream_gradient(columns):
    """Take the stride-0 gradient that `.sum()` sends.

    With `g = 1` the gradient is `y * (1 - sum(y))` for the returned `y`,
    which is near zero, and it stays within the backward bound of that
    formula.
    """
    lengths = [2, 40, 300]
    logits, offsets, _ = _softmax_case(lengths, 8, columns)

    result = swage.segment_softmax(logits, offsets)
    result.sum().backward()

    for rows in _segment_slices(offsets):
        y = result[rows].detach().double().cpu()
        y = y.view(-1, 1) if y.dim() == 1 else y
        g = torch.ones_like(y)
        exact = y * (g - y.sum(0))
        length = rows.stop - rows.start
        depth = length - 1 if columns else _public_depth(length)
        actual = logits.grad[rows].double().cpu().view_as(exact)
        assert ((actual - exact).abs() <= _backward_bound(y, g, depth)).all()


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_softmax_gradients_flow_through_a_compiled_function(
    backend, monkeypatch
):
    """Record the gradient of a softmax that a compiled function makes."""
    torch._dynamo.reset()
    monkeypatch.setattr(_segments, "_UNTRACED", {})
    monkeypatch.setattr(_autograd, "_FUNCTIONS", {})
    logits, offsets, upstream = _softmax_case([3, 0, 500, 497], 8, columns=8)

    def model(values):
        return swage.segment_softmax(values * 2.0, offsets) * 3.0

    compiled = torch.compile(model, backend=backend)
    (ours,) = torch.autograd.grad(compiled(logits), logits, upstream)
    (eager,) = torch.autograd.grad(model(logits), logits, upstream)

    assert _autograd._FUNCTIONS
    torch.testing.assert_close(ours, eager, rtol=1e-6, atol=0)
    if backend == "eager":
        assert _bits(ours.cpu()) == _bits(eager.cpu())


def test_softmax_backward_passes_leave_nothing_behind(
    driver_calls, event_counts  # noqa: F811 (fixtures)
):
    """Run many softmax calls with backward passes and keep resources flat.

    Every iteration takes offsets it has not seen, records a gradient, and
    runs a backward, whose sum kernel classifies and launches like a call.
    """
    calls = 100

    def one_iteration(seed):
        generator = torch.Generator().manual_seed(seed)
        lengths = torch.randint(0, 60, (50,), generator=generator).tolist()
        logits, offsets, upstream = _softmax_case(lengths, 8, 3, seed)
        result = swage.segment_softmax(logits, offsets)
        torch.autograd.grad(result, logits, upstream)

    for seed in (0, 1):
        one_iteration(seed)
    torch.cuda.synchronize()
    loads = driver_calls["cuModuleLoadData"]
    allocated = torch.cuda.memory_allocated()
    event_counts.clear()
    gc.collect()

    gc.disable()
    try:
        for seed in range(2, calls + 2):
            one_iteration(seed)
        torch.cuda.synchronize()
        unreachable = gc.collect()
    finally:
        gc.enable()

    assert unreachable == 0
    assert torch.cuda.memory_allocated() == allocated
    assert driver_calls["cuModuleLoadData"] == loads
    assert driver_calls["cuModuleUnload"] == 0
    assert driver_calls["cuCtxSynchronize"] == 0
    assert event_counts == {}
