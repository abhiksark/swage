# python/swage/_segments.py
"""Public segmented calls over one values buffer and its offsets.

Each call validates its tensors, classifies the offsets on the host, and
enqueues the kernels on the current PyTorch CUDA stream. The scheduling, the
kernels, and their caches belong to the private runner; this module adds the
public argument contract and nothing else. The kernels are compiled in the
process, or taken from the artifact that `SWAGE_ARTIFACT_DIR` selects.
"""

from . import _artifact, _runtime
from . import _segmented_programs as _programs
from . import _segmented_qualification as _qualification
from . import _segmented_runtime as _execution
from . import _segmented_validation as _validation
from ._frontend import _INSTALLATION

_KINDS = ("sum", "max", "min", "mean")


def segment_reduce(values, offsets, kind, *, out=None):
    """Reduce every segment of `values` to one result on the GPU.

    Segment `i` is `values[offsets[i]:offsets[i + 1]]`. The call validates
    the tensors, copies the offsets to the host to validate and classify
    them, and enqueues the kernels on the current PyTorch CUDA stream. It
    returns without waiting for the result.

    Rank-two values are `[N, D]`: `N` rows of `D` features. The offsets
    delimit rows, and every column of a segment is reduced on its own, as
    `torch.segment_reduce` does along axis 0. Such a call runs one kernel
    with one block per segment, in which a thread reduces a column in row
    order. It classifies nothing and splits no segment, so a long segment
    occupies one block for its whole length, and with few columns few
    threads reduce it. `[N, 1]` values are reduced by the schedules of
    rank-one values.

    Every call repeats the host work, also when the offsets are the ones of
    the call before, so a call costs more than its kernels. A call keeps no
    plan and compares no version counter, so the tensors may be inference
    tensors. Compiled kernels are kept in the process and reused by later
    calls, and a call compiles only the kernels its batch launches. When
    `SWAGE_ARTIFACT_DIR` names an artifact that `python -m swage.compile`
    wrote, the kernels come from it and nothing is compiled.

    Args:
        values: Contiguous `torch.float32` or `torch.float64` CUDA tensor
            on the current device, of rank one or of rank two, `[N, D]`.
            It must not require grad: the call records no gradient.
            Nothing is cast: float64 values are reduced in float64.
        offsets: Contiguous rank-one `torch.int32` or `torch.int64` tensor
            on the same device, with one entry more than there are
            segments. It starts at zero, never decreases, and ends at or
            below the number of values, or of rows for rank-two values. Two
            equal neighbors describe an empty segment. int64 offsets are
            checked and narrowed on the host, and the kernels read a
            private int32 copy that the call uploads.
        kind: `"sum"`, `"max"`, `"min"`, or `"mean"`. The sum of an empty
            segment is `0.0`, its maximum is negative infinity, its minimum
            is positive infinity, and its mean is NaN. A maximum or a
            minimum over a NaN is NaN, and a sum follows IEEE-754 addition.
            A maximum and a minimum are exact. The rounding of a sum
            depends on the schedule the call selects from the segment
            lengths, the batch, and the device, and no argument pins it. A
            mean is that sum divided once by the length of the segment.
        out: Optional result tensor: contiguous, of the dtype of `values`,
            on the device of `values`, with exactly one element per
            segment, or of shape `[S, D]` for `S` segments of `[N, D]`
            values, sharing no memory with `values` or `offsets`, and not
            requiring grad. It is never resized.

    Returns:
        `out`, or a new tensor of the dtype and on the device of `values`
        when `out` is None, with one element per segment, or one row of
        `D` elements per segment for `[N, D]` values. The kernels that
        write it are
        enqueued and may not have finished. The version counter of the
        tensor is advanced when a kernel is enqueued.

    Raises:
        TypeError: An argument is not a tensor, or a tensor has the wrong
            dtype, rank, or device type.
        ValueError: `kind` is not a supported kind; a tensor is not
            contiguous, is a lazy view, requires grad, or is on another
            device; `out` has the wrong size or overlaps an input; or the
            offsets break their contract.
        RuntimeError: PyTorch is missing or older than the supported
            release; the native bindings are missing and no artifact is
            selected; the artifact that `SWAGE_ARTIFACT_DIR` selects cannot
            be used or does not hold the kernels of the call; numpy is
            missing; CUDA is unavailable; the current stream is capturing
            a CUDA graph; or a kernel would have to be compiled while
            `SWAGE_NO_COMPILE=1` is set.
    """
    torch = _runtime._import_torch()
    if type(kind) is not str or kind not in _KINDS:
        raise ValueError(
            f"kind must be 'sum', 'max', 'min', or 'mean', got {kind!r}"
        )
    _require_inputs(torch, values, offsets)
    rank = values.dim()
    if rank not in (1, 2):
        raise TypeError("values must have rank one or two")
    segment_count = max(offsets.numel() - 1, 0)
    # One result per segment, and per column for `[N, D]` values.
    shape = (segment_count, *values.shape[1:])
    # The element type of the program, or None for values no program takes.
    element = _programs._element_of(torch, values)
    _require_out(torch, out, shape, "segment", values, offsets, element)
    _require_bindings("segment_reduce")
    _require_numpy("segment_reduce")
    _refuse_capture(torch, "segment_reduce", values, offsets)
    if element is None:
        raise TypeError("values must have dtype torch.float32 or torch.float64")
    output = _result(torch, out, shape, values, values.dtype)
    if rank == 2 and values.shape[1] != 1:
        _qualification._launch_columns(
            values,
            offsets,
            output,
            module_text=_programs._semantic_module(kind, element, 2),
            kernel_name=_programs._reduction_kernel(kind, element, 2),
            validate_offsets=_validation._validate_offsets,
            int64_offsets=True,
        )
        return output
    kernel_values, kernel_output = values, output
    if rank == 2:
        kernel_values, kernel_output = _one_column_as_scalars(values, output)
    _qualification._launch_planned_reduction(
        kernel_values,
        offsets,
        kernel_output,
        module_text=_programs._semantic_module(kind, element),
        kernel_name=_programs._reduction_kernel(kind, element),
    )
    return output


def segment_softmax(values, offsets, *, out=None):
    """Apply a softmax within every segment of `values` on the GPU.

    Segment `i` is `values[offsets[i]:offsets[i + 1]]`, and the result holds
    the softmax of each segment at the positions of its values. The call
    validates the tensors, copies the offsets to the host to validate them,
    and enqueues one kernel on the current PyTorch CUDA stream. It returns
    without waiting for the result.

    Rank-two values are `[N, D]`: `N` rows of `D` features. The offsets
    delimit rows, and every column of a segment is normalized on its own
    over the rows of that segment, as `torch.softmax(values[a:b], dim=0)`
    does. Such a call runs one kernel with one block per segment, in which
    a thread normalizes a column in row order. `[N, 1]` values are
    normalized by the kernel of rank-one values.

    Args:
        values: Contiguous `torch.float32` CUDA tensor on the current
            device, of rank one or of rank two, `[N, D]`. It must not
            require grad: the call records no gradient. float64 values are
            refused: the device has no 64-bit exp2, so there is no float64
            softmax kernel.
        offsets: Contiguous rank-one `torch.int32` or `torch.int64` tensor
            on the same device, with one entry more than there are
            segments. It starts at zero, never decreases, and ends at the
            number of values, or of rows for rank-two values, so every
            value belongs to a segment. Two equal neighbors describe an
            empty segment, which has no result element. int64 offsets are
            checked and narrowed on the host, and the kernel reads a
            private int32 copy that the call uploads.
        out: Optional result tensor: contiguous, `torch.float32`, on the
            device of `values`, with the shape of `values`, sharing no
            memory with `values` or `offsets`, and not requiring grad. It
            is never resized.

    Returns:
        `out`, or a new tensor on the device of `values` when `out` is
        None, with the shape of `values`. The kernel that writes it is
        enqueued and may not have finished. The version counter of the
        tensor is advanced when a kernel is enqueued.

    Raises:
        TypeError: An argument is not a tensor, or a tensor has the wrong
            dtype, rank, or device type.
        ValueError: A tensor is not contiguous, is a lazy view, requires
            grad, or is on another device; `out` has the wrong size or
            overlaps an input; or the offsets break their contract.
        RuntimeError: PyTorch is missing or older than the supported
            release; the native bindings are missing and no artifact is
            selected; the artifact that `SWAGE_ARTIFACT_DIR` selects cannot
            be used or does not hold the kernel; numpy is missing; CUDA is
            unavailable; the current stream is capturing a CUDA graph; or
            the kernel would have to be compiled while `SWAGE_NO_COMPILE=1`
            is set.
    """
    torch = _runtime._import_torch()
    _require_inputs(torch, values, offsets)
    rank = values.dim()
    if rank not in (1, 2):
        raise TypeError("values must have rank one or two")
    # One result per value: the result has the shape of the values.
    shape = tuple(values.shape)
    element = "f32" if values.dtype == torch.float32 else None
    _require_out(torch, out, shape, "value", values, offsets, element)
    _require_bindings("segment_softmax")
    _require_numpy("segment_softmax")
    _refuse_capture(torch, "segment_softmax", values, offsets)
    if values.dtype == torch.float64:
        raise TypeError(
            "values must have dtype torch.float32; segment_softmax has no "
            "float64 kernel because the device has no 64-bit exp2"
        )
    output = _result(torch, out, shape, values, torch.float32)
    if rank == 2 and values.shape[1] != 1:
        _qualification._launch_columns(
            values,
            offsets,
            output,
            module_text=_programs._softmax_text(2),
            kernel_name="ragged_softmax_r2",
            validate_offsets=_validate_covering_offsets,
            int64_offsets=True,
            clamp_rows_to_output=True,
        )
        return output
    kernel_values, kernel_output = values, output
    if rank == 2:
        kernel_values, kernel_output = _one_column_as_scalars(values, output)
    value_count, segment_count, host_offsets = _validation._validate_shapes(
        kernel_values,
        offsets,
        kernel_output,
        _validate_covering_offsets,
        int64_offsets=True,
    )
    # A batch without segments enqueues nothing and uploads nothing.
    kernel_offsets = offsets
    if segment_count:
        kernel_offsets = _validation._kernel_offsets(
            torch, offsets, host_offsets
        )
    _qualification._enqueue_softmax(
        torch,
        kernel_values,
        kernel_offsets,
        kernel_output,
        value_count,
        segment_count,
        _execution._target_description().cta_block_threads,
    )
    return output


def _one_column_as_scalars(values, output):
    """View `[N, 1]` values and their result as tensors of rank one.

    One column of rows is a run of scalars, which the kernels of rank-one
    values take. A view needs contiguous storage, which the kernels need as
    well, so a tensor that is not contiguous is refused here under its
    public name. The views share the storage and the version counter of
    their tensors.
    """
    for name, tensor in (("values", values), ("out", output)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return values.view(-1), output.view(-1)


def _require_inputs(torch, values, offsets):
    """Check what the public contract adds for the two input tensors.

    The private runner validates dtype, rank, contiguity, lazy views, the
    device, and the offsets themselves. This check comes first because the
    size of the result is read from the tensors, and because a caller of a
    public function is told how to proceed without gradients.
    """
    for name, tensor in (("values", values), ("offsets", offsets)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if values.requires_grad:
        raise ValueError(
            "values must not require grad; a segmented call records no "
            "gradient, so pass values.detach()"
        )


def _require_out(torch, out, shape, unit, values, offsets, element):
    """Validate a caller-supplied result tensor under its public name.

    Args:
        torch: The PyTorch module.
        out: The `out` argument, or None when the call allocates the result.
        shape: The shape of the result: one element per segment or value,
            or, for rank-two values, one row per segment and one column per
            feature, or the shape of the values.
        unit: What one element belongs to, `"segment"` or `"value"`.
        values: The values tensor, already known to be a tensor.
        offsets: The offsets tensor, already known to be a tensor.
        element: The element type of the program the call runs, or None
            when the call takes no values of this dtype. The dtype of `out`
            is then not judged here: the shared validation refuses the
            values.
    """
    if out is None:
        return
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor or None")
    if element is not None and out.dtype != values.dtype:
        raise TypeError(f"out must have the dtype of values, {values.dtype}")
    if len(shape) == 2:
        if tuple(out.shape) != tuple(shape):
            meaning = (
                "the shape of values"
                if unit == "value"
                else f"one row per {unit} and one column per feature"
            )
            raise ValueError(
                f"out must have shape {tuple(shape)}, {meaning}; found "
                f"{tuple(out.shape)}"
            )
    else:
        (count,) = shape
        if out.dim() != 1:
            raise TypeError("out must have rank one")
        if out.numel() != count:
            raise ValueError(
                f"out must have exactly {count} elements, one per {unit}; "
                f"found {out.numel()}"
            )
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    _validation._validate_storage("out", out)
    if out.requires_grad:
        raise ValueError(
            "out must not require grad; a segmented call records no gradient"
        )
    if out.device != values.device:
        raise ValueError(
            f"out must be on the device of values: found {out.device}, "
            f"values are on {values.device}"
        )
    for name, buffer in (("values", values), ("offsets", offsets)):
        if _share_memory(buffer, out):
            raise ValueError(f"out must not overlap {name}")


def _share_memory(buffer, out):
    """Return whether two tensors have a byte of device memory in common.

    The extents are exact for the contiguous rank-one tensors the calls
    admit. The comparison cannot see two virtual mappings of one physical
    allocation.
    """
    buffer_start = buffer.data_ptr()
    buffer_end = buffer_start + buffer.numel() * buffer.element_size()
    out_start = out.data_ptr()
    out_end = out_start + out.numel() * out.element_size()
    return buffer_start < out_end and out_start < buffer_end


def _require_bindings(call):
    """Require what compiles or holds the kernels of a call.

    A selected artifact is read and verified here, once per process, and
    the native bindings are then not needed. Without one, a wheel-only
    install raises the missing-bindings error.

    Raises:
        RuntimeError: `SWAGE_ARTIFACT_DIR` names an artifact that cannot be
            used, or no artifact is selected and the bindings are missing.
    """
    if _artifact.selected() is not None:
        return
    try:
        from mlir_swage._mlir_libs._swageDialectsNanobind import (  # noqa: F401
            swage,
        )
    except Exception as error:
        raise RuntimeError(
            f"Swage {call}() requires the build-tree mlir_swage bindings, "
            "which the swage-compiler wheel does not include; nothing was "
            f"launched. See {_INSTALLATION} for the native build"
        ) from error


def _require_numpy(call):
    """Require numpy, which holds the host copy of the offsets.

    The offsets are validated and classified as a numpy array, with the
    bindings and with an artifact. Without this check a missing numpy
    surfaces inside PyTorch, in words that name neither the call nor what
    to install.
    """
    try:
        import numpy  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            f"Swage {call}() requires numpy, which cannot be imported; "
            "nothing was launched. Install 'swage-compiler[pytorch]', which "
            f"declares it. See {_INSTALLATION} for the requirements"
        ) from error


def _refuse_capture(torch, call, values, offsets):
    """Refuse a call while the current stream captures a CUDA graph.

    A call copies the offsets to the host, which a capturing stream cannot
    do, and builds its task records from that copy, which a replay would
    not repeat. The check comes before the copy, so a refused call leaves
    the capture usable.
    """
    on_cuda = any(tensor.device.type == "cuda" for tensor in (values, offsets))
    if on_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{call} cannot run while the current stream captures a CUDA "
            "graph: every call copies the offsets to the host and schedules "
            "from that copy, which a replay would not repeat"
        )


def _result(torch, out, shape, values, dtype):
    """Return the tensor a call writes: `out`, or a new one of `shape`."""
    if out is not None:
        return out
    return torch.empty(shape, dtype=dtype, device=values.device)


def _validate_covering_offsets(offsets, value_count, output_count):
    """Validate softmax offsets that must cover every value.

    The private softmax admits offsets that end below the value count and
    leaves the values past them unwritten. A public result has one element
    per value, so an element that belongs to no segment would be returned
    uninitialized.
    """
    segment_count = _validation._validate_softmax_offsets(
        offsets, value_count, output_count
    )
    final = int(offsets[-1])
    if final != value_count:
        raise ValueError(
            f"offsets must end at the value count for a softmax: final "
            f"offset {final}, value count {value_count}"
        )
    return segment_count
