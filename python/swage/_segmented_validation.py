# python/swage/_segmented_validation.py
"""Tensor, offset, count, and alias contracts for segmented execution."""

from . import _runtime

_I32_LIMIT = 1 << 31


def _validate_counts(value_count, segment_count):
    """Validate the two explicit signed-i32 CUDA ABI counts."""
    for name, count in (
        ("value count", value_count),
        ("segment count", segment_count),
    ):
        if type(count) is not int or not 0 <= count < _I32_LIMIT:
            raise ValueError(f"{name} must be a nonnegative i32")


def _validate_offset_sequence(offsets, value_count):
    """Validate the offset array itself and return the segment count."""
    if not offsets:
        raise ValueError("offsets must contain at least the initial zero")
    segment_count = len(offsets) - 1
    _validate_counts(value_count, segment_count)
    if offsets[0] != 0:
        raise ValueError("offsets must start at zero")
    previous = 0
    for offset in offsets:
        if type(offset) is not int or not -(1 << 31) <= offset < _I32_LIMIT:
            raise ValueError("offsets must contain signed i32 values")
        if offset < 0:
            raise ValueError("offsets must not be negative")
        if offset < previous:
            raise ValueError("offsets must be nondecreasing")
        previous = offset
    if offsets[-1] > value_count:
        raise ValueError(
            f"final offset {offsets[-1]} exceeds value count {value_count}"
        )
    return segment_count


def _validate_offsets(offsets, value_count, output_count):
    """Require one output element per segment, the reduction ABI."""
    segment_count = _validate_offset_sequence(offsets, value_count)
    if type(output_count) is not int or output_count < segment_count:
        raise ValueError(
            f"output has {output_count} elements for {segment_count} segments"
        )
    return segment_count


def _validate_softmax_offsets(offsets, value_count, output_count):
    """Require one output element per covered value, the map_store ABI."""
    segment_count = _validate_offset_sequence(offsets, value_count)
    required = offsets[-1]
    if type(output_count) is not int or output_count < required:
        raise ValueError(
            f"output has {output_count} elements for {required} values"
        )
    return segment_count


def _validate_disjoint(name, buffer, output):
    """Reject an output that overlaps a buffer the kernel reads."""
    buffer_start = buffer.data_ptr()
    buffer_end = buffer_start + buffer.numel() * buffer.element_size()
    output_start = output.data_ptr()
    output_end = output_start + output.numel() * output.element_size()
    if buffer_start < output_end and output_start < buffer_end:
        raise ValueError(f"output must not overlap the {name} buffer")


def _validate_shapes(
    values, offsets, output, validate_offsets, *, require_cuda=True
):
    """Validate tensor shapes against one of the two output ABIs."""
    torch = _runtime._import_torch()
    for name, tensor in (
        ("values", values),
        ("offsets", offsets),
        ("output", output),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    for name, tensor in (("values", values), ("output", output)):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
    if offsets.dtype != torch.int32:
        raise TypeError("offsets must have dtype torch.int32")
    for name, tensor in (
        ("values", values),
        ("offsets", offsets),
        ("output", output),
    ):
        if tensor.dim() != 1:
            raise TypeError(f"{name} must have rank one")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    value_count = values.numel()
    host_offsets = offsets.detach().cpu().tolist()
    segment_count = validate_offsets(host_offsets, value_count, output.numel())
    _validate_disjoint("values", values, output)
    _validate_disjoint("offsets", offsets, output)
    if not require_cuda:
        return value_count, segment_count, host_offsets
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in PyTorch")
    current_device = torch.cuda.current_device()
    for name, tensor in (
        ("values", values),
        ("offsets", offsets),
        ("output", output),
    ):
        if tensor.device.type != "cuda":
            raise TypeError(f"{name} must be a CUDA tensor")
        if tensor.device.index != current_device:
            raise ValueError(f"{name} must be on the current CUDA device")
    return value_count, segment_count, host_offsets


def _validate_tensors(values, offsets, output, *, require_cuda=True):
    """Validate the reduction tensors and return their explicit counts."""
    value_count, segment_count, _ = _validate_shapes(
        values, offsets, output, _validate_offsets, require_cuda=require_cuda
    )
    return value_count, segment_count


def _validate_softmax_tensors(values, offsets, output, *, require_cuda=True):
    """Validate the softmax tensors, including the aliasing obligation."""
    value_count, segment_count, _ = _validate_shapes(
        values,
        offsets,
        output,
        _validate_softmax_offsets,
        require_cuda=require_cuda,
    )
    return value_count, segment_count
