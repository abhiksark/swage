# python/swage/_segmented_qualification.py
"""Private launches of segment programs: one-shot, prepared, and persistent.

A launch validates its tensors (`_segmented_validation`), plans its batch
when it is planned (`_segmented_plan`), compiles and leases its kernels and
binds their arguments by the launch contracts (`_segmented_runtime`), and
enqueues them on the current stream. The public calls of `_segments` run on
these entry points; `_segmented_oracle` holds the CPU references.
"""

import threading
from collections.abc import Callable
from typing import NamedTuple

from . import _cuda_backend, _runtime
from . import _segmented_plan as _plan
from . import _segmented_programs as _programs
from . import _segmented_runtime as _execution
from . import _segmented_validation as _validation


class _PreparedReduction(NamedTuple):
    """Prepared pure and classified launch policies.

    Each callable owns what it launches: it holds a lease on each of its
    kernels and keeps its task storage allocated for as long as it is
    referenced, also after the kernel memo has forgotten them. A CUDA graph
    that captured a launch keeps the module of the kernel loaded, and
    borrows the task storage, so the callable must outlive the graph.

    A callable raises RuntimeError instead of launching when the current
    CUDA context is not the one it was prepared in.
    """

    warp: Callable[[], None]
    cta: Callable[[], None]
    mixed: Callable[[], None]


class _PreparedPersistentSum(NamedTuple):
    """One prepared persistent launch and its fixed task metadata.

    `launch` owns its kernel lease, its task storage, and one queue: it
    keeps them for as long as it is referenced, and a CUDA graph that
    captured a launch borrows the storage, so `launch` must outlive the
    graph.

    One queue admits one launch at a time. `launch` raises RuntimeError
    instead of launching when another thread is inside it, when an earlier
    launch is still in flight on another stream, and when the current CUDA
    context is not the one it was prepared in. A launch queued behind an
    earlier one on the same stream is ordered by the stream and admitted.
    Two things are not checked: launches recorded during a graph capture,
    and replays of a graph.
    """

    launch: Callable[[], None]
    resident_blocks: int
    warp_tasks: int
    cta_tasks: int
    partial_tasks: int
    merge_tasks: int


def _native_bindings():
    """Return the native bindings, which the private-only paths compile with.

    No artifact holds the kernels of these paths, so they never ask one.
    """
    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    return native_swage


def _block_size(torch, block_size):
    """Validate the block size of a direct kernel and return it.

    Args:
        torch: The PyTorch module.
        block_size: Threads per block, or None for the CTA block of the
            target description.

    Raises:
        ValueError: The size is not a positive integer, its warp count is
            not a power of two, or it exceeds the device limit.
    """
    if block_size is None:
        block_size = _execution._target_description().cta_block_threads
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block size must be a positive integer")
    _validation._validate_warp_count(block_size)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    return block_size


def _launch_one(torch, kernel, module_text, tensors, counts, blocks, named):
    """Bind, lease, and enqueue one kernel on the current stream.

    Args:
        torch: The PyTorch module.
        kernel: The compiled `_Kernel`.
        module_text: Its program, whose parameter roles order the user
            values.
        tensors: The values, the offsets the kernel reads, and the output,
            then any task tensor the launch reads. The stream keeps each
            alive until the kernel has run.
        counts: The counts of the program by role.
        blocks: Blocks of the one-dimensional grid.
        named: Derived counts and plan pointers by contract key.
    """
    values, offsets, output = tensors[:3]
    user = _execution._user_arguments(
        module_text,
        values=values.data_ptr(),
        offsets=offsets.data_ptr(),
        output=output.data_ptr(),
        **counts,
    )
    arguments = _execution._bind(kernel, user, named)
    context = _cuda_backend.ensure_context(torch, torch.cuda.current_device())
    lease = _execution._lease(torch, kernel)
    stream = torch.cuda.current_stream()
    try:
        _execution._enqueue(
            torch, lease, kernel, arguments, blocks, stream, context=context
        )
    finally:
        lease.release()
    for tensor in tensors:
        tensor.record_stream(stream)
    _runtime._advance_version(torch, output)


def _launch_columns(
    values,
    offsets,
    output,
    *,
    module_text,
    kernel_name,
    validate_offsets,
    int64_offsets=False,
    block_size=None,
    clamp_rows_to_output=False,
):
    """Validate and enqueue the column kernel of one rank-two program.

    The kernel is the direct schedule of a program over `[rows, columns]`
    values: one block per segment, in which thread `t` runs the program
    for the columns `t`, `t + block_size`, and so on, each alone. A launch
    classifies nothing and uploads no task record. Its host work is the
    validation of the offsets.

    Args:
        values: Contiguous `[rows, columns]` CUDA tensor of the element
            type the program declares.
        offsets: Contiguous CUDA segment offsets, which delimit rows.
        output: Disjoint contiguous CUDA output with the columns of the
            values and the rows the validator requires.
        module_text: MLIR text of the program.
        kernel_name: Name of its segment function.
        validate_offsets: The validator of the output ABI, as for
            `_validate_shapes`.
        int64_offsets: Whether `torch.int64` offsets are admitted. The
            kernel then reads a private int32 copy.
        block_size: Threads per block, or None for the CTA block of the
            target description.
        clamp_rows_to_output: Whether the kernel writes the rows it reads,
            as a map store does. The row count it receives is then the
            number of rows of the shorter of the values and the output, as
            `_validate_softmax_tensors` returns it for rank one.
    """
    torch = _runtime._import_torch()
    value_count, segment_count, host_offsets = _validation._validate_shapes(
        values,
        offsets,
        output,
        validate_offsets,
        int64_offsets=int64_offsets,
        element=_programs._program_element(module_text),
        rank=2,
    )
    if clamp_rows_to_output:
        value_count = min(value_count, output.shape[0])
    feature_count = values.shape[1]
    block_size = _block_size(torch, block_size)
    # Without a segment or without a column there is nothing to write.
    if segment_count == 0 or feature_count == 0:
        return None

    kernel = _execution._compile_once(
        _execution._native_swage()._compile_segmented_reduction_ptx,
        module_text,
        kernel_name=kernel_name,
        block_size=block_size,
        target=_execution._target(torch, torch.cuda.current_device()),
    )
    kernel_offsets = _validation._kernel_offsets(torch, offsets, host_offsets)
    _launch_one(
        torch,
        kernel,
        module_text,
        (values, kernel_offsets, output),
        {
            "value_count": value_count,
            "segment_count": segment_count,
            "feature_count": feature_count,
        },
        segment_count,
        {},
    )
    return None


def launch_gpu(values, offsets, output, kind, block_size=None):
    """Launch one internally qualified segmented reduction.

    The kernel is compiled once per kind, element type, rank, block size,
    and target, and loaded once per CUDA context. An omitted block size is
    the CTA block of the target description. float64 values run the f64
    program of the kind, and the output then is float64 as well. Rank-two
    values run the column kernel of the kind, one block per segment.
    """
    torch = _runtime._import_torch()
    element = _programs._element_of(torch, values) or "f32"
    if getattr(values, "ndim", 1) == 2:
        return _launch_columns(
            values,
            offsets,
            output,
            module_text=_programs._semantic_module(kind, element, 2),
            kernel_name=_programs._reduction_kernel(kind, element, 2),
            validate_offsets=_validation._validate_offsets,
            block_size=block_size,
        )
    # An unknown kind is refused also for a batch without segments.
    module_text = _programs._semantic_module(kind, element)
    value_count, segment_count, _ = _validation._validate_shapes(
        values, offsets, output, _validation._validate_offsets, element=element
    )
    block_size = _block_size(torch, block_size)
    if segment_count == 0:
        return None

    kernel = _execution._compile_once(
        _native_bindings()._compile_segmented_reduction_ptx,
        module_text,
        kernel_name=_programs._reduction_kernel(kind, element),
        block_size=block_size,
        target=_execution._target(torch, torch.cuda.current_device()),
    )
    _launch_one(
        torch,
        kernel,
        module_text,
        (values, offsets, output),
        {"value_count": value_count, "segment_count": segment_count},
        segment_count,
        {},
    )
    return None


def _launch_segmented_sum_tasks(
    values, offsets, output, task_ids, *, block_size
):
    """Launch one internal identity-sum task list with a warp or CTA block."""
    torch = _runtime._import_torch()
    value_count, segment_count = _validation._validate_tensors(
        values, offsets, output
    )
    if not isinstance(task_ids, torch.Tensor):
        raise TypeError("task_ids must be a torch.Tensor")
    if task_ids.dtype != torch.int32:
        raise TypeError("task_ids must have dtype torch.int32")
    if task_ids.dim() != 1:
        raise TypeError("task_ids must have rank one")
    if not task_ids.is_contiguous():
        raise ValueError("task_ids must be contiguous")
    _validation._validate_storage("task_ids", task_ids)
    # A kernel loads a task ID after other CTAs have stored through the
    # output, so IDs that share its memory are read after they were
    # overwritten.
    _validation._validate_disjoint("task_ids", task_ids, output)
    if task_ids.device.type != "cuda":
        raise TypeError("task_ids must be a CUDA tensor")
    if task_ids.device.index != torch.cuda.current_device():
        raise ValueError("task_ids must be on the current CUDA device")
    host_task_ids = task_ids.detach().cpu().numpy()
    task_count = len(host_task_ids)
    _validation._validate_counts(value_count, task_count)
    if task_count and not (
        0 <= host_task_ids.min() and host_task_ids.max() < segment_count
    ):
        raise ValueError("task_ids must contain valid segment IDs")
    blocks = _execution._target_description()
    if block_size not in {blocks.subgroup_width, blocks.cta_block_threads}:
        raise ValueError(
            f"task block size must be {blocks.subgroup_width} or "
            f"{blocks.cta_block_threads}"
        )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if task_count == 0:
        return None

    module_text = _programs._semantic_module("sum")
    kernel = _execution._compile_once(
        _native_bindings()._compile_segmented_reduction_ptx,
        module_text,
        kernel_name="segmented_sum",
        block_size=block_size,
        target=_execution._target(torch, torch.cuda.current_device()),
        use_task_ids=True,
    )
    _launch_one(
        torch,
        kernel,
        module_text,
        (values, offsets, output, task_ids),
        {"value_count": value_count, "segment_count": segment_count},
        task_count,
        {"task_ids": task_ids.data_ptr(), "task_count": task_count},
    )
    return None


def _prepare_planned_sum(
    values,
    offsets,
    output,
    *,
    warp_max_elements=None,
    cta_chunk_elements=None,
):
    """Prepare the canonical identity sum used by existing qualification."""
    return _prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=_programs._semantic_module("sum"),
        kernel_name="segmented_sum",
        warp_max_elements=warp_max_elements,
        cta_chunk_elements=cta_chunk_elements,
        select_schedule=False,
    )


def _prepared_stream(torch, device_index, prepared_context, label):
    """Return the current stream of a prepared launch, or refuse to launch.

    The kernels are loaded in one CUDA context and are not valid in
    another, also on the same device.

    Raises:
        ValueError: The current device is not the prepared one.
        RuntimeError: The current CUDA context is not the prepared one.
    """
    if torch.cuda.current_device() != device_index:
        raise ValueError(f"prepared {label} must launch on its prepared device")
    if _cuda_backend.ensure_context(torch, device_index) != prepared_context:
        raise RuntimeError(
            f"prepared {label} must launch in its prepared CUDA context"
        )
    return torch.cuda.current_stream()


def _prepare_planned_reduction(
    values,
    offsets,
    output,
    *,
    module_text,
    kernel_name,
    warp_max_elements=None,
    cta_chunk_elements=None,
    select_schedule=True,
):
    """Prepare static policies for one private capture-free reduction.

    Args:
        values: Contiguous rank-one CUDA input tensor, float32 or float64
            as the program declares its values.
        offsets: Contiguous rank-one CUDA i32 segment offsets. They must not
            change in place after preparation; values may.
        output: Disjoint contiguous CUDA output of the dtype of values, one
            value per segment.
        module_text: Native qualification MLIR with the semantic program.
        kernel_name: Name of the segment function in the module.
        warp_max_elements: Largest segment assigned to direct warp work.
        cta_chunk_elements: Largest input range assigned to one CTA task.
        select_schedule: Allow a conservative direct-CTA choice for batches
            of moderately long segments with the default chunk size and at
            most 32 relative units of element arithmetic work.

    Returns:
        Prepared warp, CTA, and classified mixed launch callables. Mixed
        execution includes ordered partial and merge launches for split work,
        or aliases CTA when the preparation-time selection avoids splitting.
        Each callable raises RuntimeError instead of launching when offsets
        changed in place after preparation. Each callable is bound to the
        storage of values, offsets, and output at preparation: in-place
        writes to values and output are fine, and a tensor that was given
        another data pointer, element count, or dtype since, for example
        through `tensor.data = other`, raises RuntimeError before anything
        is enqueued. Each launch advances the version counter of output,
        and preparation advances it once to learn whether offsets share
        that counter. Kernels are compiled once per kernel and target and
        loaded once per CUDA context, so a preparation compiles and loads
        only the kernels the process has not already compiled and loaded.
    """
    if type(select_schedule) is not bool:
        raise TypeError("select_schedule must be a bool")
    torch = _runtime._import_torch()
    warp_max_elements, cta_chunk_elements = _plan._planning_limits(
        warp_max_elements, cta_chunk_elements
    )
    blocks = _execution._target_description()
    # A fused block serves one warp task per subgroup.
    warp_slots = blocks.cta_block_threads // blocks.subgroup_width
    validate, found = _plan._classifying_validator(
        warp_max_elements, cta_chunk_elements
    )
    value_count, segment_count, host_offsets = _validation._validate_shapes(
        values,
        offsets,
        output,
        validate,
        element=_programs._program_element(module_text),
    )
    offsets_version = _validation._offsets_version(offsets)
    # A launch advances the output version, which the offsets may share.
    version_step = _validation._output_version_step(torch, offsets, output)
    offsets_version += version_step
    # Each launch builds this record again and compares the two. The record
    # also carries the data pointers that the kernels were bound to.
    prepared_storage = _validation._storage_binding(values, offsets, output)

    native_swage = _execution._native_swage()
    target = _execution._target(torch, torch.cuda.current_device())
    small_element_program = _plan._admit_program(
        module_text, kernel_name, warp_max_elements, cta_chunk_elements
    )
    # One buffer holds the warp ids, the CTA ids, the partial ranges, the
    # merge records, and the merge of every partial task, in that order.
    (
        records,
        direct_warp_count,
        direct_cta_count,
        partial_count,
        merge_count,
    ) = _plan._classification(
        found,
        host_offsets,
        value_count=value_count,
        segment_count=segment_count,
        warp_max_elements=warp_max_elements,
        cta_chunk_elements=cta_chunk_elements,
    )
    direct_count = direct_warp_count + direct_cta_count
    use_direct_cta = select_schedule and _plan._selects_direct_cta(
        torch,
        values.device,
        host_offsets,
        segment_count,
        merge_count,
        cta_chunk_elements,
        small_element_program,
    )

    def unchanged():
        _validation._require_unchanged_offsets(offsets, offsets_version)
        storage = _validation._storage_binding(values, offsets, output)
        if storage != prepared_storage:
            _validation._refuse_rebound_storage(storage, prepared_storage)

    if segment_count == 0:

        def no_launch():
            unchanged()
            return None

        return _PreparedReduction(no_launch, no_launch, no_launch)

    def compile_kernel(compiler, **options):
        return _execution._compile_once(
            getattr(native_swage, compiler),
            module_text,
            kernel_name=kernel_name,
            target=target,
            **options,
        )

    warp_kernel = compile_kernel(
        "_compile_segmented_reduction_ptx",
        block_size=blocks.subgroup_width,
        use_task_ids=True,
    )
    cta_kernel = compile_kernel(
        "_compile_segmented_reduction_ptx",
        block_size=blocks.cta_block_threads,
        use_task_ids=True,
    )
    mixed_kernel = None
    if direct_count:
        mixed_kernel = compile_kernel("_compile_fused_segmented_reduction_ptx")
    split = partial_count and not use_direct_cta
    if split:
        partial_kernel = compile_kernel("_compile_split_partial_reduction_ptx")
        merge_kernel = compile_kernel("_compile_split_merge_reduction_ptx")

    device = offsets.device
    device_index = device.index
    prepared_context = _cuda_backend.ensure_context(torch, device_index)
    warp_lease = _execution._lease(torch, warp_kernel)
    cta_lease = _execution._lease(torch, cta_kernel)
    if mixed_kernel is not None:
        mixed_lease = _execution._lease(torch, mixed_kernel)
    if split:
        partial_lease = _execution._lease(torch, partial_kernel)
        merge_lease = _execution._lease(torch, merge_kernel)
    all_tasks = _plan._identity_ids(torch, device, segment_count)
    # One upload carries every record the mixed policy launches. The fused
    # kernel reads the warp ids and then the CTA ids at the start of the
    # buffer; the partial ranges and the merge records follow them. The
    # direct-CTA selection launches none of them: every segment is split
    # then, so there is no direct id either.
    task_records = scratch = None
    named = {}
    if len(records) and not use_direct_cta:
        task_records = torch.tensor(records, dtype=torch.int32, device=device)
        mixed_pointer = task_records.data_ptr()
        partial_pointer = mixed_pointer + 4 * direct_count
        named = {
            "task_ids": mixed_pointer,
            "warp_task_count": direct_warp_count,
            "cta_task_count": direct_cta_count,
            "partial_ranges": partial_pointer,
            "merge_records": partial_pointer + 8 * partial_count,
            "partial_count": partial_count,
            "merge_count": merge_count,
        }
        if partial_count:
            # One partial result per chunk, of the element type.
            scratch = torch.empty(
                partial_count, dtype=values.dtype, device=device
            )
            named["scratch"] = scratch.data_ptr()
    tasks_ready = torch.cuda.Event()
    tasks_ready.record(torch.cuda.current_stream(device_index))

    values_pointer, offsets_pointer, output_pointer = prepared_storage[::3]
    user = _execution._user_arguments(
        module_text,
        values=values_pointer,
        offsets=offsets_pointer,
        output=output_pointer,
        value_count=value_count,
        segment_count=segment_count,
    )
    # One task per segment, in segment order: the task list is the first
    # `segment_count` ids of `all_tasks`.
    every_segment = {
        "task_ids": all_tasks.data_ptr(),
        "task_count": segment_count,
    }
    warp_arguments = _execution._bind(warp_kernel, user, every_segment)
    cta_arguments = _execution._bind(cta_kernel, user, every_segment)
    if mixed_kernel is not None:
        mixed_arguments = _execution._bind(mixed_kernel, user, named)
    if split:
        partial_arguments = _execution._bind(partial_kernel, user, named)
        merge_arguments = _execution._bind(merge_kernel, user, named)

    tasks_ready_complete = False

    def wrote_output():
        nonlocal offsets_version
        _runtime._advance_version(torch, output)
        offsets_version += version_step

    def wait_for_tasks(stream):
        nonlocal tasks_ready_complete
        if tasks_ready_complete:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepared reduction must launch once after task "
                "initialization before CUDA graph capture; a launch that "
                "only queued the wait does not count, so launch, "
                "synchronize, and launch again"
            )
        if tasks_ready.query():
            tasks_ready_complete = True
        else:
            stream.wait_event(tasks_ready)

    def submit(lease, kernel, arguments):
        unchanged()
        stream = _prepared_stream(
            torch, device_index, prepared_context, "reduction"
        )
        wait_for_tasks(stream)
        _execution._enqueue(
            torch,
            lease,
            kernel,
            arguments,
            segment_count,
            stream,
            capturing=_cuda_backend.is_current_stream_capturing(torch),
            context=prepared_context,
        )
        for tensor in (values, offsets, output, all_tasks):
            tensor.record_stream(stream)
        wrote_output()
        return None

    def warp():
        return submit(warp_lease, warp_kernel, warp_arguments)

    def cta():
        return submit(cta_lease, cta_kernel, cta_arguments)

    def mixed():
        unchanged()
        stream = _prepared_stream(
            torch, device_index, prepared_context, "reduction"
        )
        wait_for_tasks(stream)
        # Asked once for the call; the launches below share the answer.
        known = {
            "capturing": _cuda_backend.is_current_stream_capturing(torch),
            "context": prepared_context,
        }
        if direct_count:
            _execution._enqueue(
                torch,
                mixed_lease,
                mixed_kernel,
                mixed_arguments,
                (direct_warp_count + warp_slots - 1) // warp_slots
                + direct_cta_count,
                stream,
                **known,
            )
            for tensor in (values, offsets, output, task_records):
                tensor.record_stream(stream)
        if partial_count:
            _execution._enqueue(
                torch,
                partial_lease,
                partial_kernel,
                partial_arguments,
                partial_count,
                stream,
                **known,
            )
            for tensor in (values, offsets, task_records, scratch):
                tensor.record_stream(stream)
            _execution._enqueue(
                torch,
                merge_lease,
                merge_kernel,
                merge_arguments,
                merge_count,
                stream,
                **known,
            )
            for tensor in (offsets, output, task_records, scratch):
                tensor.record_stream(stream)
        wrote_output()
        return None

    return _PreparedReduction(warp, cta, cta if use_direct_cta else mixed)


def _launch_planned_reduction(
    values, offsets, output, *, module_text, kernel_name
):
    """Validate, classify, and enqueue the mixed schedule of one batch.

    This is `_prepare_planned_reduction(...).mixed()` with the default
    limits and automatic selection, for a caller that launches once: the
    same validation, classification, selection rule, kernels, and launch
    arguments, and therefore the same result bits. It prepares nothing it
    does not launch. No pure warp kernel is compiled or loaded, the pure
    CTA kernel and the shared segment ids are used only for a batch the
    selection rule sends there, and no CUDA event is created.

    Everything `_validate_shapes` checks is checked, and the kernels keep
    their device-side bounds. Four guards of a prepared launch are left
    out, because they protect the time between a preparation and a later
    launch, which does not exist here:

    - The offsets version counter is not compared. The offsets are copied
      to the host and their records are enqueued within this call, and the
      caller runs nothing in between. An inference tensor, which has no
      counter, is therefore admitted.
    - The storage of the tensors is not compared. The data pointers are
      read after validation and passed to the driver in the same call.
    - The CUDA context is not compared. The kernels are loaded, or found
      loaded, in the context that is current at this call.
    - No event orders the task records before the kernels. They are
      uploaded and read on the one stream that is current at this call.

    A write to the offsets by another thread or by a kernel on another
    stream, between the host copy and the enqueue, is not detected. The
    result is then not a validated one. The kernels clamp every range they
    load, so such a write cannot move an access outside the buffers.

    int64 offsets are admitted. The kernels read a private int32 copy of
    them, which is uploaded behind the task records in the same tensor, or
    as a tensor of its own for a batch that uploads no records.

    Args:
        values: Contiguous rank-one CUDA input tensor, float32 or float64
            as the program declares its values.
        offsets: Contiguous rank-one CUDA i32 or i64 segment offsets.
        output: Disjoint contiguous CUDA output of the dtype of values, one
            value per segment.
        module_text: Native qualification MLIR with the semantic program.
        kernel_name: Name of the segment function in the module.

    Returns:
        None. The version counter of the output is advanced once, after the
        enqueue; a batch without segments enqueues nothing and leaves it.
    """
    torch = _runtime._import_torch()
    warp_max_elements, cta_chunk_elements = _plan._planning_limits(None, None)
    blocks = _execution._target_description()
    validate, found = _plan._classifying_validator(
        warp_max_elements, cta_chunk_elements
    )
    value_count, segment_count, host_offsets = _validation._validate_shapes(
        values,
        offsets,
        output,
        validate,
        int64_offsets=True,
        element=_programs._program_element(module_text),
    )
    native_swage = _execution._native_swage()
    device = offsets.device
    target = _execution._target(torch, device.index)
    small_element_program = _plan._admit_program(
        module_text, kernel_name, warp_max_elements, cta_chunk_elements
    )
    (
        records,
        direct_warp_count,
        direct_cta_count,
        partial_count,
        merge_count,
    ) = _plan._classification(
        found,
        host_offsets,
        value_count=value_count,
        segment_count=segment_count,
        warp_max_elements=warp_max_elements,
        cta_chunk_elements=cta_chunk_elements,
    )
    if segment_count == 0:
        return None

    def compile_kernel(compiler, **options):
        return _execution._compile_once(
            getattr(native_swage, compiler),
            module_text,
            kernel_name=kernel_name,
            target=target,
            **options,
        )

    narrowed = offsets.dtype != torch.int32
    if _plan._selects_direct_cta(
        torch,
        device,
        host_offsets,
        segment_count,
        merge_count,
        cta_chunk_elements,
        small_element_program,
    ):
        # One task per segment, in segment order, on the pure CTA kernel.
        cta_kernel = compile_kernel(
            "_compile_segmented_reduction_ptx",
            block_size=blocks.cta_block_threads,
            use_task_ids=True,
        )
        all_tasks = _plan._identity_ids(torch, device, segment_count)
        _launch_one(
            torch,
            cta_kernel,
            module_text,
            (
                values,
                _validation._kernel_offsets(torch, offsets, host_offsets),
                output,
                all_tasks,
            ),
            {"value_count": value_count, "segment_count": segment_count},
            segment_count,
            {"task_ids": all_tasks.data_ptr(), "task_count": segment_count},
        )
        return None

    # Every kernel is held before anything is uploaded or enqueued, so a
    # refused compile leaves the device untouched.
    direct_count = direct_warp_count + direct_cta_count
    launched = []
    if direct_count:
        # A fused block serves one warp task per subgroup.
        warp_slots = blocks.cta_block_threads // blocks.subgroup_width
        launched.append(
            (
                compile_kernel("_compile_fused_segmented_reduction_ptx"),
                (direct_warp_count + warp_slots - 1) // warp_slots
                + direct_cta_count,
            )
        )
    if partial_count:
        launched.append(
            (
                compile_kernel("_compile_split_partial_reduction_ptx"),
                partial_count,
            )
        )
        launched.append(
            (
                compile_kernel("_compile_split_merge_reduction_ptx"),
                merge_count,
            )
        )
    context = _cuda_backend.ensure_context(torch, device.index)
    leases = [_execution._lease(torch, kernel) for kernel, _ in launched]
    try:
        # One upload carries every record: the warp ids, the CTA ids, the
        # partial ranges, and the merge records, in that order. The private
        # copy of int64 offsets follows them in the same tensor.
        if narrowed:
            import numpy

            record_words = len(records)
            records = numpy.concatenate(
                (numpy.asarray(records, dtype=numpy.int32), host_offsets)
            )
        task_records = torch.tensor(records, dtype=torch.int32, device=device)
        mixed_pointer = task_records.data_ptr()
        partial_pointer = mixed_pointer + 4 * direct_count
        if narrowed:
            offsets_pointer = mixed_pointer + 4 * record_words
            retained = (values, output, task_records)
        else:
            offsets_pointer = offsets.data_ptr()
            retained = (values, offsets, output, task_records)
        named = {
            "task_ids": mixed_pointer,
            "warp_task_count": direct_warp_count,
            "cta_task_count": direct_cta_count,
            "partial_ranges": partial_pointer,
            "merge_records": partial_pointer + 8 * partial_count,
            "partial_count": partial_count,
            "merge_count": merge_count,
        }
        if partial_count:
            scratch = torch.empty(
                partial_count, dtype=values.dtype, device=device
            )
            retained += (scratch,)
            named["scratch"] = scratch.data_ptr()
        user = _execution._user_arguments(
            module_text,
            values=values.data_ptr(),
            offsets=offsets_pointer,
            output=output.data_ptr(),
            value_count=value_count,
            segment_count=segment_count,
        )
        arguments = [
            _execution._bind(kernel, user, named) for kernel, _ in launched
        ]
        stream = torch.cuda.current_stream()
        for lease, (kernel, grid), bound in zip(leases, launched, arguments):
            _execution._enqueue(
                torch, lease, kernel, bound, grid, stream, context=context
            )
    finally:
        for lease in leases:
            lease.release()
    for tensor in retained:
        tensor.record_stream(stream)
    _runtime._advance_version(torch, output)
    return None


def _prepare_persistent_sum(
    values,
    offsets,
    output,
    *,
    warp_max_elements=None,
    cta_chunk_elements=None,
    resident_blocks=None,
):
    """Prepare a private resident kernel with split completion handling.

    The offsets must not change in place after preparation; the prepared
    launch raises RuntimeError instead of launching when they did. The
    launch is bound to the storage of values, offsets, and output at
    preparation: in-place writes to values and output are fine, and a
    tensor that was given another data pointer, element count, or dtype
    since, for example through `tensor.data = other`, raises RuntimeError
    before anything is enqueued. Each launch advances the version counter
    of output, and preparation advances it once to learn whether offsets
    share that counter. The kernel is compiled once per target and loaded
    once per CUDA context.
    """
    if resident_blocks is not None and (
        type(resident_blocks) is not int
        or resident_blocks <= 0
        or resident_blocks > (1 << 32) - 1
    ):
        raise ValueError("resident_blocks must be a positive u32")
    torch = _runtime._import_torch()
    warp_max_elements, cta_chunk_elements = _plan._planning_limits(
        warp_max_elements, cta_chunk_elements
    )
    validate, found = _plan._classifying_validator(
        warp_max_elements, cta_chunk_elements
    )
    value_count, segment_count, host_offsets = _validation._validate_shapes(
        values, offsets, output, validate
    )
    offsets_version = _validation._offsets_version(offsets)
    # A launch advances the output version, which the offsets may share.
    version_step = _validation._output_version_step(torch, offsets, output)
    offsets_version += version_step
    # Each launch builds this record again and compares the two. The record
    # also carries the data pointers that the kernel was bound to.
    prepared_storage = _validation._storage_binding(values, offsets, output)

    target = _execution._target(torch, torch.cuda.current_device())
    kernel_name = "segmented_sum"
    module_text = _programs._semantic_module("sum")
    _plan._admit_program(
        module_text, kernel_name, warp_max_elements, cta_chunk_elements
    )
    # One buffer holds the warp ids, the CTA ids, the partial ranges, the
    # merge records, and the merge of every partial task, in that order.
    records, warp_count, cta_count, partial_count, merge_count = (
        _plan._classification(
            found,
            host_offsets,
            value_count=value_count,
            segment_count=segment_count,
            warp_max_elements=warp_max_elements,
            cta_chunk_elements=cta_chunk_elements,
        )
    )

    def unchanged():
        _validation._require_unchanged_offsets(offsets, offsets_version)
        storage = _validation._storage_binding(values, offsets, output)
        if storage != prepared_storage:
            _validation._refuse_rebound_storage(storage, prepared_storage)

    if segment_count == 0:

        def no_launch():
            unchanged()
            return None

        return _PreparedPersistentSum(no_launch, 0, 0, 0, 0, 0)
    kernel = _execution._compile_once(
        _native_bindings()._compile_persistent_segmented_reduction_ptx,
        module_text,
        kernel_name=kernel_name,
        target=target,
    )

    blocks = _execution._target_description()
    persistent_block = blocks.persistent_block_threads
    warp_slots = persistent_block // blocks.subgroup_width
    work_groups = (
        cta_count + partial_count + (warp_count + warp_slots - 1) // warp_slots
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if persistent_block > properties.max_threads_per_block:
        raise ValueError(
            f"persistent block size {persistent_block} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if resident_blocks is None:
        resident_blocks = properties.multi_processor_count * 2
    active_blocks = min(resident_blocks, work_groups)

    device = offsets.device
    device_index = device.index
    prepared_context = _cuda_backend.ensure_context(torch, device_index)
    lease = _execution._lease(torch, kernel)
    # One upload carries every record. The kernel takes one pointer per
    # list; the classifier wrote the merge of every partial task behind the
    # merge records.
    task_records = torch.tensor(records, dtype=torch.int32, device=device)
    warp_pointer = task_records.data_ptr()
    cta_pointer = warp_pointer + 4 * warp_count
    partial_pointer = cta_pointer + 4 * cta_count
    merge_pointer = partial_pointer + 8 * partial_count
    scratch = torch.empty(partial_count, dtype=torch.float32, device=device)
    # Every launch zeroes the counters before it enqueues the kernel, so
    # they need no initial value.
    counters = torch.empty(3 + merge_count, dtype=torch.int32, device=device)
    values_pointer, offsets_pointer, output_pointer = prepared_storage[::3]
    arguments = _execution._bind(
        kernel,
        _execution._user_arguments(
            module_text,
            values=values_pointer,
            offsets=offsets_pointer,
            output=output_pointer,
            value_count=value_count,
            segment_count=segment_count,
        ),
        {
            "warp_ids": warp_pointer,
            "cta_ids": cta_pointer,
            "partial_ranges": partial_pointer,
            "partial_merge_ids": merge_pointer + 12 * merge_count,
            "merge_records": merge_pointer,
            "scratch": scratch.data_ptr(),
            "counters": counters.data_ptr(),
            "warp_task_count": warp_count,
            "cta_task_count": cta_count,
            "partial_count": partial_count,
            "merge_count": merge_count,
        },
    )
    tasks_ready = torch.cuda.Event()
    tasks_ready.record(torch.cuda.current_stream(device_index))
    tasks_ready_complete = False
    # One prepared object has one counter array and one scratch buffer, so
    # two launches must not run at once. `launching` keeps a second thread
    # out of `launch`; `in_flight` is recorded behind each launch and tells
    # a launch on another stream whether the previous one has finished.
    launching = threading.Lock()
    in_flight = torch.cuda.Event()
    in_flight_stream = None

    def require_idle(stream, capturing):
        # A launch on the stream of the previous one is ordered behind it.
        # CUDA forbids an event query during a capture, so a launch that is
        # being captured is not checked.
        if (
            capturing
            or in_flight_stream is None
            or in_flight_stream == stream.cuda_stream
        ):
            return
        if not in_flight.query():
            raise RuntimeError(
                "prepared persistent sum still has a launch in flight on "
                "another stream; synchronize that stream first"
            )

    def wait_for_tasks(stream):
        nonlocal tasks_ready_complete
        if tasks_ready_complete:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepared persistent sum must launch once after task "
                "initialization before CUDA graph capture; a launch that "
                "only queued the wait does not count, so launch, "
                "synchronize, and launch again"
            )
        if tasks_ready.query():
            tasks_ready_complete = True
        else:
            stream.wait_event(tasks_ready)

    def launch():
        nonlocal in_flight_stream, offsets_version
        unchanged()
        if not launching.acquire(blocking=False):
            raise RuntimeError(
                "prepared persistent sum is already launching on another "
                "thread; one prepared object admits one launch at a time"
            )
        try:
            stream = _prepared_stream(
                torch, device_index, prepared_context, "persistent sum"
            )
            capturing = torch.cuda.is_current_stream_capturing()
            require_idle(stream, capturing)
            wait_for_tasks(stream)
            counters.zero_()
            _execution._enqueue(
                torch,
                lease,
                kernel,
                arguments,
                active_blocks,
                stream,
                capturing=capturing,
                context=prepared_context,
            )
            for tensor in (
                values,
                offsets,
                output,
                task_records,
                scratch,
                counters,
            ):
                tensor.record_stream(stream)
            if not capturing:
                in_flight.record(stream)
                in_flight_stream = stream.cuda_stream
            _runtime._advance_version(torch, output)
            offsets_version += version_step
        finally:
            launching.release()
        return None

    return _PreparedPersistentSum(
        launch,
        active_blocks,
        warp_count,
        cta_count,
        partial_count,
        merge_count,
    )


def launch_softmax_gpu(values, offsets, output, block_size=None):
    """Launch the internally qualified ragged softmax.

    The kernel is compiled once per block size and target, and loaded once
    per CUDA context. An omitted block size is the CTA block of the target
    description. Rank-two values run the column kernel of the softmax,
    which normalizes every column of a segment over its rows.
    """
    torch = _runtime._import_torch()
    if getattr(values, "ndim", 1) == 2:
        return _launch_columns(
            values,
            offsets,
            output,
            module_text=_programs._softmax_text(2),
            kernel_name="ragged_softmax_r2",
            validate_offsets=_validation._validate_softmax_offsets,
            block_size=block_size,
            clamp_rows_to_output=True,
        )
    value_count, segment_count = _validation._validate_softmax_tensors(
        values, offsets, output
    )
    return _enqueue_softmax(
        torch, values, offsets, output, value_count, segment_count, block_size
    )


def _enqueue_softmax(
    torch, values, offsets, output, value_count, segment_count, block_size
):
    """Compile, load, and enqueue the ragged softmax on validated tensors.

    Args:
        torch: The PyTorch module.
        values: Validated CUDA f32 input tensor.
        offsets: Validated CUDA i32 segment offsets.
        output: Validated CUDA f32 output tensor, disjoint from both.
        value_count: Element bound the kernel receives, as
            `_validate_softmax_tensors` returns it.
        segment_count: Number of segments the offsets describe.
        block_size: Threads per CTA, or None for the CTA block of the
            target description.
    """
    block_size = _block_size(torch, block_size)
    if segment_count == 0:
        return None

    kernel = _execution._compile_once(
        _execution._native_swage()._compile_segmented_reduction_ptx,
        _programs._SOFTMAX_MODULE,
        kernel_name="ragged_softmax",
        block_size=block_size,
        target=_execution._target(torch, torch.cuda.current_device()),
    )
    _launch_one(
        torch,
        kernel,
        _programs._SOFTMAX_MODULE,
        (values, offsets, output),
        {"value_count": value_count, "segment_count": segment_count},
        segment_count,
        {},
    )
    return None
