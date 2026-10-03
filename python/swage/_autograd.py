# python/swage/_autograd.py
"""Gradients of the public segmented calls.

A call records a gradient through the `torch.autograd.Function` subclasses
of this module. Their backward is fixed, and nothing chooses between two
implementations at run time:

- PyTorch operations find the segment of every row from the offsets on the
  device, copy the gradient of a segment to its rows, and divide.
- The sum kernel of `segment_reduce` does every floating-point segment sum
  that a backward needs. Second derivatives need one.

The module loads no PyTorch when it is imported. The Functions are made at
the first call that records a gradient, after `_runtime._import_torch` has
accepted PyTorch, and are kept per PyTorch module. ADR-0024 records the
decision.
"""

import types

from . import _segmented_qualification as _qualification

# The Functions of each PyTorch module, made at the first call that records
# a gradient. Two threads that make them at once each keep a correct set,
# and the dictionary keeps one of them.
_FUNCTIONS = {}

def functions(torch):
    """Return the autograd Functions of the segmented calls for `torch`.

    Args:
        torch: The PyTorch module that `_runtime._import_torch` returned.

    Returns:
        A namespace with `SegmentReduce`, the Function of `segment_reduce`,
        and `_SegmentSum` and `_SegmentBroadcast`, the two Functions each
        backward is composed of.
    """
    built = _FUNCTIONS.get(torch)
    if built is None:
        built = _FUNCTIONS[torch] = _build(torch)
    return built


def _segment_ids(torch, offsets, rows):
    """Return the segment of every row as an int64 tensor on the device.

    A row at or past the final offset gets the segment count, which indexes
    the zero row that `_gather` appends. The clamp keeps offsets that were
    changed without PyTorch counting the write from indexing below zero:
    every result lies between zero and the segment count.
    """
    index = torch.arange(rows, dtype=offsets.dtype, device=offsets.device)
    return (torch.searchsorted(offsets, index, right=True) - 1).clamp_(min=0)


def _gather(torch, per_segment, offsets, rows):
    """Copy the row of each segment to the rows of that segment.

    Args:
        torch: The PyTorch module.
        per_segment: `[S]` or `[S, D]`, one row per segment.
        offsets: The offsets of the call, on the device.
        rows: The number of rows of the values of the call.

    Returns:
        `[rows]` or `[rows, D]`. A row past the final offset holds zero.
        The copy is exact, signed zeros and NaN included.
    """
    pad = per_segment.new_zeros((1, *per_segment.shape[1:]))
    return torch.cat((per_segment, pad)).index_select(
        0, _segment_ids(torch, offsets, rows)
    )


def _per_segment(column, like):
    """View a per-segment column so it broadcasts against `like`."""
    return column.view(-1, *[1] * (like.dim() - 1))


def _tie_shares(torch, grad, offsets, values, result, rows):
    """Return where the elements tie and the share of the gradient each gets.

    The tied elements of a segment are the elements that equal its maximum
    or minimum, or the NaN elements when that is NaN. `-0.0` and `0.0` are
    equal. Each tied element of segment `j` gets `grad[j]` divided once by
    their number. The count is an integer prefix sum, which is exact and
    deterministic.

    Args:
        torch: The PyTorch module.
        grad: The upstream gradient, one row per segment.
        offsets: The offsets of the call.
        values: The values of the call, detached.
        result: The result of the call, detached.
        rows: The number of rows of the values.

    Returns:
        A boolean tensor of the shape of the values that marks the tied
        elements, and the share of each segment, one row per segment. A row
        past the final offset is never tied.
    """
    ids = _segment_ids(torch, offsets, rows)
    pad = result.new_zeros((1, *result.shape[1:]))
    extreme = torch.cat((result, pad)).index_select(0, ids)
    covered = _per_segment(ids < offsets.numel() - 1, values)
    tied = ((values == extreme) | (values.isnan() & extreme.isnan())) & covered
    running = torch.cat(
        (
            tied.new_zeros((1, *tied.shape[1:]), dtype=torch.int32),
            tied.to(torch.int32).cumsum(0, dtype=torch.int32),
        )
    )
    # Clamped so that offsets changed without PyTorch counting the write
    # cannot index outside the prefix sums.
    bounds = offsets.clamp(0, rows)
    count = running.index_select(0, bounds[1:]) - running.index_select(
        0, bounds[:-1]
    )
    # A segment with an element always has a tied one; the clamp keeps an
    # empty segment from dividing by zero.
    return tied, grad / count.clamp(min=1).to(grad.dtype)


def _kept_offsets(offsets):
    """Return the offsets a backward keeps.

    The caller's tensor is kept, and its version counter makes a backward
    raise after an in-place change. An inference tensor has no counter and
    cannot be saved for a backward, so it is kept as a copy.
    """
    return offsets.clone() if offsets.is_inference() else offsets


def _build(torch):
    """Make the four Functions for one PyTorch module."""
    # Imported here: the public module imports this one.
    from . import _segments

    class _SegmentSum(torch.autograd.Function):
        """Sum per segment with the sum kernel of `segment_reduce`.

        Its backward is `_SegmentBroadcast`, and its forward is the
        backward of `_SegmentBroadcast`, so a backward composed of the two
        has a second derivative. The forward refuses a capturing stream
        before its host copy, in the name of the call whose backward it
        serves.
        """

        @staticmethod
        def forward(values, offsets, call):
            # An upstream gradient may be expanded or a lazy negation view;
            # the kernel reads contiguous storage.
            values = values.detach().resolve_neg().contiguous()
            _segments._refuse_capture(torch, call, values, offsets)
            output = torch.empty(
                (offsets.numel() - 1, *values.shape[1:]),
                dtype=values.dtype,
                device=values.device,
            )
            _segments._launch_reduction(
                values,
                offsets,
                "sum",
                output,
                _qualification._element_of(torch, values),
            )
            return output

        @staticmethod
        def setup_context(ctx, inputs, output):
            values, offsets, call = inputs
            ctx.save_for_backward(offsets)
            ctx.rows = values.shape[0]
            ctx.call = call

        @staticmethod
        def backward(ctx, grad):
            (offsets,) = ctx.saved_tensors
            return (
                _SegmentBroadcast.apply(grad, offsets, ctx.rows, ctx.call),
                None,
                None,
            )

    class _SegmentBroadcast(torch.autograd.Function):
        """Copy a per-segment tensor to the rows of its segments.

        `call` names the call whose second derivative the backward of this
        Function, a segment sum, serves.
        """

        @staticmethod
        def forward(per_segment, offsets, rows, call):
            return _gather(torch, per_segment.detach(), offsets, rows)

        @staticmethod
        def setup_context(ctx, inputs, output):
            _, offsets, _, call = inputs
            ctx.save_for_backward(offsets)
            ctx.call = call

        @staticmethod
        def backward(ctx, grad):
            (offsets,) = ctx.saved_tensors
            return _SegmentSum.apply(grad, offsets, ctx.call), None, None, None

    class SegmentReduce(torch.autograd.Function):
        """The Function of `segment_reduce`.

        Its node in a graph is `SegmentReduceBackward`.
        """

        @staticmethod
        def forward(values, offsets, kind):
            output = torch.empty(
                (max(offsets.numel() - 1, 0), *values.shape[1:]),
                dtype=values.dtype,
                device=values.device,
            )
            detached = values.detach()
            _segments._launch_reduction(
                detached,
                offsets,
                kind,
                output,
                _qualification._element_of(torch, detached),
            )
            return output

        @staticmethod
        def setup_context(ctx, inputs, output):
            values, offsets, kind = inputs
            kept = _kept_offsets(offsets)
            if kind in ("max", "min"):
                # The tied elements are found again from the values and the
                # result, which the forward computed exactly.
                ctx.save_for_backward(kept, values, output)
            else:
                ctx.save_for_backward(kept)
            ctx.kind = kind
            ctx.rows = values.shape[0]

        @staticmethod
        def backward(ctx, grad):
            if not ctx.needs_input_grad[0]:
                return None, None, None
            call = "a second derivative of segment_reduce"
            if ctx.kind in ("max", "min"):
                offsets, values, result = ctx.saved_tensors
                tied, shares = _tie_shares(
                    torch,
                    grad,
                    offsets,
                    values.detach(),
                    result.detach(),
                    ctx.rows,
                )
                spread = _SegmentBroadcast.apply(
                    shares, offsets, ctx.rows, call
                )
                # Selected, never multiplied, so an element that does not
                # tie gets exactly zero also from an infinite or a NaN share.
                return torch.where(tied, spread, 0.0), None, None
            (offsets,) = ctx.saved_tensors
            if ctx.kind == "mean":
                lengths = (offsets[1:] - offsets[:-1]).clamp(min=1)
                # One correctly rounded division per segment, by the length
                # converted to the dtype as the forward converts it.
                grad = grad / _per_segment(lengths.to(grad.dtype), grad)
            return (
                _SegmentBroadcast.apply(grad, offsets, ctx.rows, call),
                None,
                None,
            )

    return types.SimpleNamespace(
        SegmentReduce=SegmentReduce,
        _SegmentSum=_SegmentSum,
        _SegmentBroadcast=_SegmentBroadcast,
    )
