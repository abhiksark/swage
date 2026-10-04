# python/swage/_segmented_validation.py
"""Tensor, offset, count, storage, and alias checks of segmented launches.

A launch reads its offsets on the host once, validates them, and passes the
counts they give to its kernels. A prepared launch also records the storage
and the offsets version it was prepared with and compares both before each
enqueue.
"""

from . import _runtime
from . import _segmented_programs as _programs
from . import _segmented_runtime as _execution

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
    """Validate the offset array itself and return the segment count.

    Args:
        offsets: Host offsets. An int32 or int64 array, the host copy of a
            tensor, is checked with array operations. Any other sequence is
            checked one element at a time, which also rejects a value that
            is not a signed i32 integer. An array that passes holds no value
            outside the signed i32 range either: every offset lies between
            the initial zero and the final offset, which is at most the
            value count.
        value_count: Number of values the offsets index.

    Returns:
        The segment count, one less than the number of offsets.

    Raises:
        ValueError: If the offsets are empty, do not start at zero, hold a
            value that is not a signed i32 integer, a negative value, or a
            decrease, or end past the value count. The first invalid offset
            decides the message, in both forms.
    """
    import numpy

    if not len(offsets):
        raise ValueError("offsets must contain at least the initial zero")
    segment_count = len(offsets) - 1
    _validate_counts(value_count, segment_count)
    if offsets[0] != 0:
        raise ValueError("offsets must start at zero")
    if isinstance(offsets, numpy.ndarray) and offsets.dtype in (
        numpy.int32,
        numpy.int64,
    ):
        # The first decrease is the first invalid offset: every offset
        # before it is at least the initial zero, so a negative offset is
        # always a decrease.
        decreasing = offsets[1:] < offsets[:-1]
        if decreasing.any():
            if offsets[1:][decreasing.argmax()] < 0:
                raise ValueError("offsets must not be negative")
            raise ValueError("offsets must be nondecreasing")
    else:
        previous = 0
        for offset in offsets:
            if type(offset) is not int or not -(1 << 31) <= offset < _I32_LIMIT:
                raise ValueError("offsets must contain signed i32 values")
            if offset < 0:
                raise ValueError("offsets must not be negative")
            if offset < previous:
                raise ValueError("offsets must be nondecreasing")
            previous = offset
    final = int(offsets[-1])
    if final > value_count:
        raise ValueError(
            f"final offset {final} exceeds value count {value_count}"
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
    """Require one output element per covered value, the map_store ABI.

    The bound is the final offset rather than the value count, because
    offsets may cover fewer values than the buffer holds and binding to the
    value count would reject a correctly sized output.
    """
    segment_count = _validate_offset_sequence(offsets, value_count)
    required = int(offsets[-1])
    if type(output_count) is not int or output_count < required:
        raise ValueError(
            f"output has {output_count} elements for {required} values"
        )
    return segment_count


def _validate_disjoint(name, buffer, output):
    """Reject an output that overlaps a buffer the kernel reads.

    ADR-0008 names only the values buffer, but the offsets buffer needs the
    same treatment: the kernel re-reads offsets from device memory after
    other CTAs have already stored through an aliased output, which voids
    the host-side offset walk entirely.

    Both tensors are known contiguous and rank one by this point, so the
    byte extent is exact and a half-open intersection is exact. It cannot
    see two virtual mappings of one physical allocation, nor aliasing
    created after this returns.
    """
    buffer_start = buffer.data_ptr()
    buffer_end = buffer_start + buffer.numel() * buffer.element_size()
    output_start = output.data_ptr()
    output_end = output_start + output.numel() * output.element_size()
    if buffer_start < output_end and output_start < buffer_end:
        raise ValueError(f"output must not overlap the {name} buffer")


def _validate_storage(name, tensor):
    """Reject a lazy view, whose storage does not hold the values it shows.

    The kernels read and write tensor storage through raw pointers, so for
    a negation or conjugate view they would compute with the base. The
    public launch applies the same rule.
    """
    if tensor.is_neg():
        raise ValueError(
            f"{name} must not be a lazy negation view; pass "
            "tensor.resolve_neg()"
        )
    if tensor.is_conj():
        raise ValueError(
            f"{name} must not be a lazy conjugate view; pass "
            "tensor.resolve_conj()"
        )


def _validate_shapes(
    values,
    offsets,
    output,
    validate_offsets,
    *,
    require_cuda=True,
    int64_offsets=False,
    element="f32",
    rank=1,
):
    """Validate tensor shapes against one of the two output ABIs.

    Args:
        values: The values tensor.
        offsets: The offsets tensor.
        output: The result tensor.
        validate_offsets: The validator of the output ABI, called with the
            host offsets as an int32 array, the value count, and the number
            of output elements.
        require_cuda: Whether the tensors must be on the current CUDA
            device.
        int64_offsets: Whether `torch.int64` offsets are admitted beside
            `torch.int32` ones. A caller that sets it must give its kernels
            `_kernel_offsets`: a kernel reads int32 words.
        element: The element type of the program the caller runs, `"f32"`
            or `"f64"`. The values and the output must have its dtype: a
            kernel reads and writes elements of one width.
        rank: The rank of the values and of the output of the program, one
            or two. Rank-two values are `[rows, columns]`, the offsets
            delimit rows, and the output has the columns of the values.
            The validator then receives the row counts.

    Returns:
        The value count, which is the number of rows for rank two, the
        segment count, and the offsets on the host as
        one int32 array. Copying a CUDA tensor to the host waits for the
        work queued on it; that copy is the only device synchronization
        here, and no Python integer is created per offset. int64 offsets
        are validated as they are and narrowed afterwards, so a value that
        would wrap into a valid offset is refused.
    """
    torch = _runtime._import_torch()
    for name, tensor in (
        ("values", values),
        ("offsets", offsets),
        ("output", output),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    dtype = _programs._element_dtype(torch, element)
    for name, tensor in (("values", values), ("output", output)):
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
    if offsets.dtype != torch.int32 and not (
        int64_offsets and offsets.dtype == torch.int64
    ):
        raise TypeError(
            "offsets must have dtype torch.int32"
            + (" or torch.int64" if int64_offsets else "")
        )
    for name, tensor, required in (
        ("values", values, rank),
        ("offsets", offsets, 1),
        ("output", output, rank),
    ):
        if tensor.dim() != required:
            raise TypeError(
                f"{name} must have rank {'one' if required == 1 else 'two'}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        _validate_storage(name, tensor)
    for name, tensor in (("values", values), ("output", output)):
        if tensor.requires_grad:
            raise ValueError(
                f"{name} must not require grad; segmented kernels write "
                "through raw pointers"
            )
    if rank == 2:
        if output.shape[1] != values.shape[1]:
            raise ValueError(
                f"output has {output.shape[1]} columns for "
                f"{values.shape[1]} columns of values"
            )
        if values.shape[1] >= _I32_LIMIT:
            raise ValueError("feature count must be a nonnegative i32")

    # The offsets delimit the first axis: elements of rank-one values and
    # rows of rank-two values.
    value_count = values.shape[0]
    host_offsets = offsets.detach().cpu().numpy()
    if offsets.dtype == torch.int64:
        import numpy

        # Valid offsets lie between zero and the value count, which is
        # below 2**31, so the narrowing below is exact.
        _validate_offset_sequence(host_offsets, value_count)
        host_offsets = host_offsets.astype(numpy.int32)
    segment_count = validate_offsets(host_offsets, value_count, output.shape[0])
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


def _kernel_offsets(torch, offsets, host_offsets):
    """Return the int32 offsets tensor a kernel reads.

    A kernel loads each offset as an i32 word. int32 offsets are read where
    the caller keeps them. Validated int64 offsets are uploaded as a
    private int32 tensor on the current stream, and the caller's tensor is
    never a kernel argument. The caller of this function retains the
    result on the launch stream.

    Args:
        torch: The PyTorch module.
        offsets: The validated offsets tensor of the caller.
        host_offsets: Its int32 host copy, as `_validate_shapes` returns.
    """
    if offsets.dtype == torch.int32:
        return offsets
    return torch.tensor(host_offsets, dtype=torch.int32, device=offsets.device)


def _validate_tensors(values, offsets, output, *, require_cuda=True):
    """Validate the reduction tensors and return their explicit counts."""
    value_count, segment_count, _ = _validate_shapes(
        values, offsets, output, _validate_offsets, require_cuda=require_cuda
    )
    return value_count, segment_count


def _validate_softmax_tensors(values, offsets, output, *, require_cuda=True):
    """Validate the softmax tensors, including the aliasing obligation.

    Returns:
        The element bound the kernel receives in its value-count slot, and
        the segment count. The map_store kernel reads values and writes
        output at the same element index, and clamps every range it loads
        from the device to that bound, so the bound is the length of the
        shorter buffer. Validated offsets end at or below both lengths, so
        the bound never shortens a validated segment.
    """
    value_count, segment_count, _ = _validate_shapes(
        values,
        offsets,
        output,
        _validate_softmax_offsets,
        require_cuda=require_cuda,
    )
    return min(value_count, output.numel()), segment_count


def _validate_warp_count(block_size):
    """Require a block size whose warp count is a power of two.

    The block-wide reduction combines one partial result per warp, and that
    combination is complete in every participating lane only for a
    power-of-two number of warps. A partly filled last warp is admitted.
    """
    width = _execution._target_description().subgroup_width
    warp_count = (block_size + width - 1) // width
    if warp_count & (warp_count - 1):
        raise ValueError(
            f"block size must give a power-of-two warp count, got {block_size}"
        )


def _offsets_version(offsets):
    """Return the version counter a prepared launch compares at launch.

    PyTorch advances the counter on every in-place write through the tensor
    or one of its views, so the comparison is one host attribute read and
    never waits for the device. It cannot see a write through `.data`, a raw
    pointer, memory shared with another library, or another kernel, and a
    replayed CUDA graph does not run it.
    """
    if offsets.is_inference():
        raise ValueError(
            "offsets must not be an inference tensor; a prepared launch "
            "compares its version counter"
        )
    return offsets._version


def _require_unchanged_offsets(offsets, version):
    """Refuse to launch a plan whose offsets changed in place."""
    if offsets._version != version:
        raise RuntimeError("offsets changed after preparation; prepare again")


def _output_version_step(torch, offsets, output):
    """Return how far one advance of the output version moves offsets.

    Every launch advances the version counter of its output. Views of one
    tensor share one counter, also views of another dtype that share no
    byte, so that advance can move the offsets version as well. A prepared
    launch would then read its own output write as changed offsets.

    Returns:
        1 when offsets and output share a version counter and 0 otherwise,
        found by advancing the output version once. A prepared launch adds
        it to the offsets version it expects each time it launches.
    """
    before = offsets._version
    _runtime._advance_version(torch, output)
    return offsets._version - before


def _storage_binding(values, offsets, output):
    """Return what ties a prepared launch to the storage of its tensors.

    A prepared launch passes the counts taken at preparation with the data
    pointers read at launch. PyTorch can give a tensor object other storage
    without advancing its version counter, for example through
    `tensor.data = other`, so the counts would then describe memory the
    tensor no longer has. The data pointer, the element count, and the
    dtype of each tensor together fix the bytes a kernel may touch, and a
    launch compares them with this record. Reading them is host work only.

    It cannot see a tensor that was rebound and then bound back to the
    prepared address, count, and dtype over storage that was freed and
    allocated again in between.

    Returns:
        The data pointer, element count, and dtype of values, then of
        offsets, then of output, so every third entry is a data pointer.
    """
    return (
        values.data_ptr(),
        values.numel(),
        values.dtype,
        offsets.data_ptr(),
        offsets.numel(),
        offsets.dtype,
        output.data_ptr(),
        output.numel(),
        output.dtype,
    )


def _refuse_rebound_storage(binding, prepared):
    """Raise for the first tensor whose storage is not the prepared one.

    Args:
        binding: The `_storage_binding` of the tensors now.
        prepared: The one recorded at preparation, which differs.
    """
    for index, name in enumerate(("values", "offsets", "output")):
        pointer, count, dtype = binding[3 * index : 3 * index + 3]
        was = prepared[3 * index : 3 * index + 3]
        if (pointer, count, dtype) != was:
            raise RuntimeError(
                f"{name} is bound to other storage than at preparation: "
                f"found {count} {dtype} elements at {pointer:#x}, prepared "
                f"with {was[1]} {was[2]} elements at {was[0]:#x}; prepare "
                "again"
            )
