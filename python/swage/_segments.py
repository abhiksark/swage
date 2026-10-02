# python/swage/_segments.py
"""Public segmented calls over one values buffer and its offsets.

Each call validates its tensors, classifies the offsets on the host, and
enqueues the kernels on the current PyTorch CUDA stream. The scheduling, the
kernels, and their caches belong to the private runner; this module adds the
public argument contract and nothing else. The kernels are compiled in the
process, or taken from the artifact that `SWAGE_ARTIFACT_DIR` selects.
"""

from . import _artifact, _runtime
from . import _segmented_qualification as _qualification
from ._frontend import _INSTALLATION

_KINDS = ("sum", "max")


def segment_reduce(values, offsets, kind, *, out=None):
    """Reduce every segment of `values` to one f32 result on the GPU.

    Segment `i` is `values[offsets[i]:offsets[i + 1]]`. The call validates
    the tensors, copies the offsets to the host to validate and classify
    them, and enqueues the kernels on the current PyTorch CUDA stream. It
    returns without waiting for the result.

    Every call repeats the host work, also when the offsets are the ones of
    the call before, so a call costs more than its kernels. Compiled kernels
    are kept in the process and reused by later calls. When
    `SWAGE_ARTIFACT_DIR` names an artifact that `python -m swage.compile`
    wrote, the kernels come from it and nothing is compiled.

    Args:
        values: Contiguous rank-one `torch.float32` CUDA tensor on the
            current device. It must not require grad: the call records no
            gradient.
        offsets: Contiguous rank-one `torch.int32` tensor on the same
            device, with one entry more than there are segments. It starts
            at zero, never decreases, and ends at or below the number of
            values. Two equal neighbors describe an empty segment. It must
            not be an inference tensor.
        kind: `"sum"` or `"max"`. The sum of an empty segment is `0.0` and
            its maximum is negative infinity. A maximum over a NaN is NaN,
            and a sum follows IEEE-754 addition. The rounding of a sum
            depends on the schedule the call selects from the segment
            lengths, the batch, and the device, and no argument pins it.
        out: Optional result tensor: contiguous, rank one, `torch.float32`,
            on the device of `values`, with exactly one element per segment,
            sharing no memory with `values` or `offsets`, and not requiring
            grad. It is never resized.

    Returns:
        `out`, or a new tensor on the device of `values` when `out` is
        None, with one element per segment. The kernels that write it are
        enqueued and may not have finished. The version counter of the
        tensor is advanced.

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
            be used or does not hold the kernels of the call; CUDA is
            unavailable; the current stream is capturing a CUDA graph; or
            a kernel would have to be compiled while `SWAGE_NO_COMPILE=1`
            is set.
    """
    torch = _runtime._import_torch()
    if type(kind) is not str or kind not in _KINDS:
        raise ValueError(f"kind must be 'sum' or 'max', got {kind!r}")
    _require_inputs(torch, values, offsets)
    segment_count = max(offsets.numel() - 1, 0)
    _require_out(torch, out, segment_count, "segment", values, offsets)
    _require_bindings("segment_reduce")
    _require_numpy("segment_reduce")
    _refuse_capture(torch, "segment_reduce", values, offsets)
    output = _result(torch, out, segment_count, values)
    _qualification._launch_planned_reduction(
        values,
        offsets,
        output,
        module_text=_qualification._semantic_module(kind),
        kernel_name=f"segmented_{kind}",
    )
    return output


def segment_softmax(values, offsets, *, out=None):
    """Apply a softmax within every segment of `values` on the GPU.

    Segment `i` is `values[offsets[i]:offsets[i + 1]]`, and the result holds
    the softmax of each segment at the positions of its values. The call
    validates the tensors, copies the offsets to the host to validate them,
    and enqueues one kernel on the current PyTorch CUDA stream. It returns
    without waiting for the result.

    Args:
        values: Contiguous rank-one `torch.float32` CUDA tensor on the
            current device. It must not require grad: the call records no
            gradient.
        offsets: Contiguous rank-one `torch.int32` tensor on the same
            device, with one entry more than there are segments. It starts
            at zero, never decreases, and ends at the number of values, so
            every value belongs to a segment. Two equal neighbors describe
            an empty segment, which has no result element.
        out: Optional result tensor: contiguous, rank one, `torch.float32`,
            on the device of `values`, with exactly one element per value,
            sharing no memory with `values` or `offsets`, and not requiring
            grad. It is never resized.

    Returns:
        `out`, or a new tensor on the device of `values` when `out` is
        None, with one element per value. The kernel that writes it is
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
            be used or does not hold the kernel; CUDA is unavailable; the
            current stream is capturing a CUDA graph; or the kernel would
            have to be compiled while `SWAGE_NO_COMPILE=1` is set.
    """
    torch = _runtime._import_torch()
    _require_inputs(torch, values, offsets)
    value_count = values.numel()
    _require_out(torch, out, value_count, "value", values, offsets)
    _require_bindings("segment_softmax")
    _require_numpy("segment_softmax")
    _refuse_capture(torch, "segment_softmax", values, offsets)
    output = _result(torch, out, value_count, values)
    value_count, segment_count, _ = _qualification._validate_shapes(
        values, offsets, output, _validate_covering_offsets
    )
    _qualification._enqueue_softmax(
        torch,
        values,
        offsets,
        output,
        value_count,
        segment_count,
        _qualification._target_description().cta_block_threads,
    )
    return output


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


def _require_out(torch, out, count, unit, values, offsets):
    """Validate a caller-supplied result tensor under its public name.

    Args:
        torch: The PyTorch module.
        out: The `out` argument, or None when the call allocates the result.
        count: Number of elements the result has.
        unit: What one element belongs to, `"segment"` or `"value"`.
        values: The values tensor, already known to be a tensor.
        offsets: The offsets tensor, already known to be a tensor.
    """
    if out is None:
        return
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor or None")
    if out.dtype != torch.float32:
        raise TypeError("out must have dtype torch.float32")
    if out.dim() != 1:
        raise TypeError("out must have rank one")
    if out.numel() != count:
        raise ValueError(
            f"out must have exactly {count} elements, one per {unit}; "
            f"found {out.numel()}"
        )
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    _qualification._validate_storage("out", out)
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


def _result(torch, out, count, values):
    """Return the tensor a call writes: `out`, or a new one of `count`."""
    if out is not None:
        return out
    return torch.empty(count, dtype=torch.float32, device=values.device)


def _validate_covering_offsets(offsets, value_count, output_count):
    """Validate softmax offsets that must cover every value.

    The private softmax admits offsets that end below the value count and
    leaves the values past them unwritten. A public result has one element
    per value, so an element that belongs to no segment would be returned
    uninitialized.
    """
    segment_count = _qualification._validate_softmax_offsets(
        offsets, value_count, output_count
    )
    final = int(offsets[-1])
    if final != value_count:
        raise ValueError(
            f"offsets must end at the value count for a softmax: final "
            f"offset {final}, value count {value_count}"
        )
    return segment_count
