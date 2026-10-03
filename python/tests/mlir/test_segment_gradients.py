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
    _selection_lengths,
    driver_calls,
    empty_kernel_memo,
    event_counts,
)
from test_segmented_runtime import _bits, _offsets

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
pytestmark = _needs_cuda

KINDS = ["sum", "mean"]
DTYPES = [torch.float32, torch.float64]
OFFSET_DTYPES = [torch.int32, torch.int64]
# None stands for rank-one values; a number is the column count of [N, D].
COLUMNS = [None, 1, 3, 64]
# Segments of a gradcheck batch: two empty ones, and two rows past the final
# offset that belong to no segment.
_CHECK_LENGTHS = [2, 0, 5, 1, 0, 3]
_PAST_FINAL = 2


def _values(rows, columns, dtype, seed):
    """Return values on the device that require grad."""
    generator = torch.Generator().manual_seed(seed)
    shape = (rows,) if columns is None else (rows, columns)
    values = torch.randn(shape, generator=generator, dtype=torch.float64)
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


@pytest.mark.parametrize("kind", KINDS)
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
    expected = torch.ones_like(lengths) if kind == "mean" else lengths
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


@pytest.mark.parametrize("kind", ["max", "min"])
def test_max_and_min_have_no_backward_yet(kind):
    """Refuse a gradient for the extremes until their tie rule exists."""
    values, offsets = _batch([2, 3])

    with pytest.raises(
        ValueError,
        match=rf"^kind '{kind}' has no backward yet; pass values.detach\(\)$",
    ):
        swage.segment_reduce(values, offsets, kind)

    result = swage.segment_reduce(values.detach(), offsets, kind)
    assert not result.requires_grad


def test_segment_softmax_has_no_backward_yet():
    """Refuse a gradient for the softmax until its backward exists."""
    values, offsets = _batch([2, 3], dtype=torch.float32)

    with pytest.raises(
        ValueError,
        match=r"^segment_softmax has no backward yet; pass values.detach\(\)$",
    ):
        swage.segment_softmax(values, offsets)
