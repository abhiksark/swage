# python/swage/_segmented_qualification.py
"""Private qualification runner for native segmented programs."""

import pathlib
import re
import shutil
import struct
import subprocess
import threading
import weakref
from collections.abc import Callable
from typing import NamedTuple

from . import _runtime

_I32_LIMIT = 1 << 31
_WARP_BLOCK = 32
_CTA_BLOCK = 128
_SPLIT_BLOCK = 512
_PERSISTENT_BLOCK = 512
_CTA_CHUNK_ELEMENTS = 4096
_LOWERING_PIPELINE = (
    "builtin.module(func.func(convert-scf-to-cf,convert-math-to-llvm,"
    "convert-arith-to-llvm),"
    "finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)
# One memo for every kernel this module compiles and loads. PTX is keyed by
# the native compile function, the semantic module text, and every code
# generation option (kernel name, block size, target). Loaded handles are
# keyed per driver by CUDA context, PTX text, and kernel name, so a hit
# never crosses a context or a target. Python keeps the hash of a string, and
# the PTX memo returns the same string for the same kernel, so neither lookup
# reads the PTX text again.
#
# Both memos are bounded: each keeps `_runtime._CACHE_LIMIT` kernels and then
# forgets its oldest. A forgotten kernel is compiled or loaded again on its
# next use. Its module is unloaded once no prepared launch holds its function
# handle any more; see `_runtime._Function`.
#
# A hit takes no lock. A miss takes the process-wide cold-path lock, which
# serializes native compiles; see `_runtime._compile_lock`.
_memo_lock = _runtime._compile_lock
_ptx_memo = _runtime._BoundedCache(_runtime._CACHE_LIMIT)
_load_memo = weakref.WeakKeyDictionary()


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
        offsets: Host offsets. An int32 array, the host copy of a validated
            tensor, is checked with array operations. Any other sequence is
            checked one element at a time, which also rejects a value that
            is not a signed i32 integer; the dtype rules that out for the
            array.
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
    if isinstance(offsets, numpy.ndarray) and offsets.dtype == numpy.int32:
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
    values, offsets, output, validate_offsets, *, require_cuda=True
):
    """Validate tensor shapes against one of the two output ABIs.

    Returns:
        The value count, the segment count, and the offsets on the host as
        one int32 array. Copying a CUDA tensor to the host waits for the
        work queued on it; that copy is the only device synchronization
        here, and no Python integer is created per offset.
    """
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
        _validate_storage(name, tensor)
    for name, tensor in (("values", values), ("output", output)):
        if tensor.requires_grad:
            raise ValueError(
                f"{name} must not require grad; segmented kernels write "
                "through raw pointers"
            )

    value_count = values.numel()
    host_offsets = offsets.detach().cpu().numpy()
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
    warp_count = (block_size + _WARP_BLOCK - 1) // _WARP_BLOCK
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
        pointer, count, dtype = binding[3 * index:3 * index + 3]
        was = prepared[3 * index:3 * index + 3]
        if (pointer, count, dtype) != was:
            raise RuntimeError(
                f"{name} is bound to other storage than at preparation: "
                f"found {count} {dtype} elements at {pointer:#x}, prepared "
                f"with {was[1]} {was[2]} elements at {was[0]:#x}; prepare "
                "again"
            )


# The NVPTX processor of each CUDA device index. A device keeps its compute
# capability for as long as the process runs.
_targets = {}


def _target(torch, device_index):
    """Return the NVPTX processor name of one CUDA device, looked up once."""
    target = _targets.get(device_index)
    if target is None:
        major, minor = torch.cuda.get_device_capability(device_index)
        target = _targets[device_index] = f"sm_{major}{minor}"
    return target


# The segment ids 0, 1, 2, ... of each CUDA device index, which a launch of
# one task per segment reads as its task list. They depend on the segment
# count only, so one tensor per device serves every preparation: it is
# uploaded once, complete when the upload returns, and only read afterwards.
# It holds 4 MiB of device memory per device for as long as the process runs.
_IDENTITY_LIMIT = 1 << 20
_identity_memo = {}


def _identity_ids(torch, device, count):
    """Return a tensor that starts with the segment ids 0 to `count - 1`.

    Args:
        torch: The PyTorch module.
        device: CUDA device of the prepared tensors.
        count: Number of ids a launch reads, the segment count of the
            preparation.

    Returns:
        An int32 tensor on the device with at least `count` ascending ids.
        Up to `_IDENTITY_LIMIT` ids it is the shared tensor of the device,
        which needs no kernel and no wait. A longer list is filled on the
        device for this preparation alone, asynchronously, on the current
        stream.
    """
    if count > _IDENTITY_LIMIT:
        return torch.arange(count, dtype=torch.int32, device=device)
    ids = _identity_memo.get(device.index)
    if ids is None:
        import numpy

        ids = _identity_memo[device.index] = torch.tensor(
            numpy.arange(_IDENTITY_LIMIT, dtype=numpy.int32),
            dtype=torch.int32,
            device=device,
        )
    return ids


def _compile_once(compile_ptx, module_text, *, module=None, **options):
    """Compile one kernel at most once per process and return its PTX.

    Args:
        compile_ptx: Native compile function, looked up by the caller at call
            time. It is part of the key, so a replaced function is called
            instead of being served an earlier result.
        module_text: Semantic module text that identifies the program.
        module: The same program already parsed in an active MLIR context.
            When omitted, the text is parsed only on a miss.
        **options: Keyword arguments of the compile function: the kernel
            name, the target, the block size, and any other code generation
            option. Each one is part of the key.

    Returns:
        The PTX text of the compiled kernel. A failed compile is not kept.
        A kernel compiled before is returned without taking a lock, so it
        never waits for another thread's compile.

    Raises:
        RuntimeError: SWAGE_NO_COMPILE=1 is set and this process does not
            hold the kernel. Nothing is parsed or compiled.
        ValueError: SWAGE_NO_COMPILE has a value other than 0 or 1.
    """
    key = (compile_ptx, module_text, tuple(sorted(options.items())))
    ptx = _ptx_memo.get(key)
    if ptx is not None:
        return ptx
    with _memo_lock:
        ptx = _ptx_memo.get(key)
        if ptx is None:
            if _runtime._switch_on("SWAGE_NO_COMPILE"):
                raise _compile_refusal(options)
            if module is None:
                from mlir_swage import ir
                from mlir_swage.dialects import swage

                with ir.Context() as context:
                    swage.register_dialects(context)
                    _, ptx = compile_ptx(
                        ir.Module.parse(module_text), **options
                    )
            else:
                _, ptx = compile_ptx(module, **options)
            _ptx_memo[key] = ptx
    return ptx


def _compile_refusal(options):
    """Return the error for a kernel this process may not compile.

    The public launch can answer a miss from the persistent cache. This
    path keeps its kernels in the process only, so a kernel it does not
    hold stays unavailable while compiling is switched off.

    Args:
        options: The code generation options of the refused compile.
    """
    held_for = " and ".join(
        f"{label} {options[name]}"
        for name, label in (("block_size", "block size"), ("target", "target"))
        if name in options
    )
    return _runtime._compile_refusal(
        options.get("kernel_name"),
        f"this process does not hold it for {held_for}, and the private "
        "segmented path has no persistent cache",
    )


def _load_once(driver, ptx, kernel_name):
    """Load one kernel at most once per CUDA context.

    Args:
        driver: CUDA driver wrapper that owns the returned handles.
        ptx: PTX text of the compiled kernel.
        kernel_name: Name of the function to resolve in the loaded module.

    Returns:
        The module and function handles, as returned by `driver.load`. A
        driver that cannot name its current context is never memoized,
        because a handle is valid only in the context that loaded it. A
        kernel loaded before is returned without taking a lock.

        The caller must keep the function handle for as long as it may
        launch it: the memo forgets its oldest kernel at its bound, and a
        module is unloaded once nothing holds its function handle.
    """
    current_context = getattr(driver, "current_context", None)
    if current_context is None:
        return driver.load(ptx, kernel_name)
    key = (current_context(), ptx, kernel_name)
    loaded = _load_memo.get(driver)
    if loaded is None:
        with _memo_lock:
            loaded = _load_memo.setdefault(
                driver, _runtime._BoundedCache(_runtime._CACHE_LIMIT)
            )
    handles = loaded.get(key)
    if handles is None:
        handles = _runtime._load_cold(loaded, key, driver, ptx, kernel_name)
    return handles


def _semantic_module(kind):
    """Return the canonical private qualification module."""
    if kind not in {"sum", "max"}:
        raise ValueError("reduction kind must be 'sum' or 'max'")
    return f"""
module {{
  func.func @segmented_{kind}(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {{
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<{kind}>
        : !swage.segment<f32> -> f32 {{
    ^bb0(%value: f32):
      swage.yield %value : f32
    }}
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }}
}}
"""


_SOFTMAX_MODULE = """
module {
  func.func @ragged_softmax(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %shifted = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %m: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }
    %total = swage.reduce %shifted kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    swage.map_store %segment, %output captures(%max, %total : f32, f32)
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32, %m: f32, %t: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      %normalized = arith.divf %exponential, %t : f32
      swage.yield %normalized : f32
    }
    return
  }
}
"""

_SENTINEL = -1.0


class _PreparedReduction(NamedTuple):
    """Prepared pure and classified launch policies.

    Each callable owns what it launches: it keeps its kernels loaded and its
    task storage allocated for as long as it is referenced, also after the
    kernel memo has forgotten them. A CUDA graph that captured a launch
    borrows both, so the callable must outlive the graph.

    A callable raises RuntimeError instead of launching when the current
    CUDA context is not the one it was prepared in.
    """

    warp: Callable[[], None]
    cta: Callable[[], None]
    mixed: Callable[[], None]


class _PreparedPersistentSum(NamedTuple):
    """One prepared persistent launch and its fixed task metadata.

    `launch` owns its kernel, its task storage, and one queue: it keeps them
    for as long as it is referenced, and a CUDA graph that captured a launch
    borrows them, so `launch` must outlive the graph.

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


def launch_gpu(values, offsets, output, kind, block_size=128):
    """Launch one internally qualified segmented reduction.

    The kernel is compiled once per kind, block size, and target, and loaded
    once per CUDA context.
    """
    torch = _runtime._import_torch()
    value_count, segment_count = _validate_tensors(values, offsets, output)
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block size must be a positive integer")
    _validate_warp_count(block_size)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if segment_count == 0:
        return None

    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    target = _target(torch, torch.cuda.current_device())
    kernel_name = f"segmented_{kind}"
    ptx = _compile_once(
        native_swage._compile_segmented_reduction_ptx,
        _semantic_module(kind),
        kernel_name=kernel_name,
        block_size=block_size,
        target=target,
    )

    driver = _runtime._get_driver()
    _, function = _load_once(driver, ptx, kernel_name)
    stream = torch.cuda.current_stream()
    driver.launch_segmented(
        function,
        (segment_count,),
        block_size,
        stream.cuda_stream,
        (
            values.data_ptr(),
            offsets.data_ptr(),
            output.data_ptr(),
            value_count,
            segment_count,
        ),
    )
    for tensor in (values, offsets, output):
        tensor.record_stream(stream)
    return None


def _launch_segmented_sum_tasks(
    values, offsets, output, task_ids, *, block_size
):
    """Launch one internal identity-sum task list with a warp or CTA block."""
    torch = _runtime._import_torch()
    value_count, segment_count = _validate_tensors(values, offsets, output)
    if not isinstance(task_ids, torch.Tensor):
        raise TypeError("task_ids must be a torch.Tensor")
    if task_ids.dtype != torch.int32:
        raise TypeError("task_ids must have dtype torch.int32")
    if task_ids.dim() != 1:
        raise TypeError("task_ids must have rank one")
    if not task_ids.is_contiguous():
        raise ValueError("task_ids must be contiguous")
    _validate_storage("task_ids", task_ids)
    # A kernel loads a task ID after other CTAs have stored through the
    # output, so IDs that share its memory are read after they were
    # overwritten.
    _validate_disjoint("task_ids", task_ids, output)
    if task_ids.device.type != "cuda":
        raise TypeError("task_ids must be a CUDA tensor")
    if task_ids.device.index != torch.cuda.current_device():
        raise ValueError("task_ids must be on the current CUDA device")
    host_task_ids = task_ids.detach().cpu().numpy()
    task_count = len(host_task_ids)
    _validate_counts(value_count, task_count)
    if task_count and not (
        0 <= host_task_ids.min() and host_task_ids.max() < segment_count
    ):
        raise ValueError("task_ids must contain valid segment IDs")
    if block_size not in {_WARP_BLOCK, _CTA_BLOCK}:
        raise ValueError("task block size must be 32 or 128")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if task_count == 0:
        return None

    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    target = _target(torch, torch.cuda.current_device())
    kernel_name = "segmented_sum"
    ptx = _compile_once(
        native_swage._compile_segmented_reduction_ptx,
        _semantic_module("sum"),
        kernel_name=kernel_name,
        block_size=block_size,
        target=target,
        use_task_ids=True,
    )

    driver = _runtime._get_driver()
    _, function = _load_once(driver, ptx, kernel_name)
    stream = torch.cuda.current_stream()
    driver.launch_segmented_tasks(
        function,
        (task_count,),
        block_size,
        stream.cuda_stream,
        (
            values.data_ptr(),
            offsets.data_ptr(),
            output.data_ptr(),
            task_ids.data_ptr(),
            value_count,
            task_count,
            segment_count,
        ),
    )
    for tensor in (values, offsets, output, task_ids):
        tensor.record_stream(stream)
    return None


def _prepare_planned_sum(
    values,
    offsets,
    output,
    *,
    warp_max_elements=32,
    cta_chunk_elements=_CTA_CHUNK_ELEMENTS,
):
    """Prepare the canonical identity sum used by existing qualification."""
    return _prepare_planned_reduction(
        values,
        offsets,
        output,
        module_text=_semantic_module("sum"),
        kernel_name="segmented_sum",
        warp_max_elements=warp_max_elements,
        cta_chunk_elements=cta_chunk_elements,
        select_schedule=False,
    )


def _has_small_element_program(module):
    """Conservatively bound work in already-admitted element regions."""
    from mlir_swage import ir
    from mlir_swage.dialects import arith, math, swage

    cheap_operations = (
        arith.AddFOp, arith.SubFOp, arith.MulFOp,
        arith.MaximumFOp, arith.MinimumFOp,
    )
    work = 0
    eligible = True

    def inspect(operation):
        nonlocal work, eligible
        if not isinstance(operation.opview, (swage.MapOp, swage.ReduceOp)):
            return ir.WalkResult.ADVANCE
        for instruction in operation.regions[0].blocks[0].operations:
            if isinstance(instruction, (arith.ConstantOp, swage.YieldOp)):
                continue
            # Relative work units calibrated on the held-out GPU benchmarks.
            # They are a bounded heuristic, not instruction latency estimates.
            if isinstance(instruction, cheap_operations):
                work += 1
            elif isinstance(instruction, math.Exp2Op):
                work += 8
            elif isinstance(instruction, arith.DivFOp):
                work += 16
            else:
                eligible = False
                return ir.WalkResult.INTERRUPT
            if work > 32:
                eligible = False
                return ir.WalkResult.INTERRUPT
        return ir.WalkResult.SKIP

    module.operation.walk(inspect, ir.WalkOrder.PRE_ORDER)
    return eligible


# What a preparation needs to know about a program, none of which depends on
# the offsets. `_module_memo` keeps one parsed module per semantic module
# text with the result of inspecting its element program. `_admitted` keeps
# that result per program text and pair of planning limits the planning pass
# has accepted, so the pass runs once per program and limits and a later
# preparation only classifies its offsets.
#
# Threads share a parsed module, which is safe because the planning pass
# holds the GIL and leaves the module unchanged. Compiles do not use these
# modules: on a miss `_compile_once` parses the text in a context of its
# own, so a compile never runs on a context shared here.
#
# Both memos are bounded like the kernel memos: each keeps
# `_runtime._CACHE_LIMIT` entries and then forgets its oldest, which is
# parsed or admitted again on its next use. A hit takes no lock.
_module_memo = _runtime._BoundedCache(_runtime._CACHE_LIMIT)
_admitted = _runtime._BoundedCache(_runtime._CACHE_LIMIT)


def _parsed_module(module_text):
    """Parse and inspect one semantic module at most once per process.

    Args:
        module_text: Semantic module text that identifies the program.

    Returns:
        The module, parsed in a context that it keeps alive, and whether
        `_has_small_element_program` holds for it. A text that does not
        parse is not kept.
    """
    entry = _module_memo.get(module_text)
    if entry is None:
        from mlir_swage import ir
        from mlir_swage.dialects import swage

        context = ir.Context()
        swage.register_dialects(context)
        module = ir.Module.parse(module_text, context=context)
        entry = (module, _has_small_element_program(module))
        with _memo_lock:
            _module_memo[module_text] = entry
    return entry


def _admit_program(module_text, warp_max_elements, cta_chunk_elements):
    """Admit one program for planning under one pair of limits, once.

    The planning pass decides whether a program can be classified and
    whether the limits are valid. Neither depends on the offsets, so the
    pass runs at the first preparation of a program with a pair of limits,
    on a layout without segments. Later preparations classify their offsets
    without the module.

    Args:
        module_text: Semantic module text that identifies the program.
        warp_max_elements: Largest segment assigned to direct warp work.
        cta_chunk_elements: Largest input range assigned to one CTA task.

    Returns:
        Whether `_has_small_element_program` holds for the program.

    Raises:
        ValueError: The planning pass rejects the program or the limits. A
            rejection is not kept, so every preparation raises it again.
    """
    key = (module_text, warp_max_elements, cta_chunk_elements)
    small_element_program = _admitted.get(key)
    if small_element_program is None:
        import numpy
        from mlir_swage._mlir_libs._swageDialectsNanobind import (
            swage as native_swage,
        )

        module, small_element_program = _parsed_module(module_text)
        native_swage._materialize_segmented_plan(
            module,
            offsets=numpy.zeros(1, dtype=numpy.int32),
            value_count=0,
            segment_count=0,
            warp_max_elements=warp_max_elements,
            cta_chunk_elements=cta_chunk_elements,
        )
        with _memo_lock:
            _admitted[key] = small_element_program
    return small_element_program


def _classifying_validator(warp_max_elements, cta_chunk_elements):
    """Return an offsets validator that classifies the offsets it admits.

    The native classifier checks everything `_validate_offsets` checks about
    a host int32 array, so a preparation walks valid offsets once, there,
    instead of once to validate and once to classify. The classifier runs at
    the point where `_validate_shapes` validates the offsets, which keeps
    the order of every error.

    When the classifier refuses, `_validate_offsets` runs and raises its own
    message for offsets it refuses too. Offsets it admits were refused for
    the planning limits or for the size of the plan; the classification is
    then left out, and the caller classifies again after it has admitted
    the program, which reports the limits first.

    Args:
        warp_max_elements: Largest segment assigned to direct warp work.
        cta_chunk_elements: Largest input range assigned to one CTA task.

    Returns:
        The validator, with the signature of `_validate_offsets`, and a list
        that holds the result of `_classify_segments` once the validator
        has admitted and classified the offsets.
    """
    classification = []

    def validate(host_offsets, value_count, output_count):
        from mlir_swage._mlir_libs._swageDialectsNanobind import (
            swage as native_swage,
        )

        segment_count = len(host_offsets) - 1
        try:
            classification.append(
                native_swage._classify_segments(
                    host_offsets,
                    value_count=value_count,
                    segment_count=segment_count,
                    warp_max_elements=warp_max_elements,
                    cta_chunk_elements=cta_chunk_elements,
                )
            )
        except (TypeError, ValueError):
            return _validate_offsets(host_offsets, value_count, output_count)
        if type(output_count) is not int or output_count < segment_count:
            classification.clear()
            return _validate_offsets(host_offsets, value_count, output_count)
        return segment_count

    return validate, classification


def _prepare_planned_reduction(
    values,
    offsets,
    output,
    *,
    module_text,
    kernel_name,
    warp_max_elements=32,
    cta_chunk_elements=_CTA_CHUNK_ELEMENTS,
    select_schedule=True,
):
    """Prepare static policies for one private capture-free sum/max module.

    Args:
        values: Contiguous rank-one CUDA f32 input tensor.
        offsets: Contiguous rank-one CUDA i32 segment offsets. They must not
            change in place after preparation; values may.
        output: Disjoint contiguous CUDA f32 output, one value per segment.
        module_text: Native qualification MLIR with the semantic program.
        kernel_name: Name of its single semantic function.
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
        is enqueued. Kernels are compiled once per kernel and target and
        loaded once per CUDA context, so a preparation compiles and loads
        only the kernels the process has not already compiled and loaded.
    """
    if type(select_schedule) is not bool:
        raise TypeError("select_schedule must be a bool")
    torch = _runtime._import_torch()
    validate, classification = _classifying_validator(
        warp_max_elements, cta_chunk_elements
    )
    value_count, segment_count, host_offsets = _validate_shapes(
        values, offsets, output, validate
    )
    offsets_version = _offsets_version(offsets)
    # Each launch builds this record again and compares the two. The record
    # also carries the data pointers that the launch passes to the kernel.
    prepared_storage = _storage_binding(values, offsets, output)

    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    target = _target(torch, torch.cuda.current_device())
    small_element_program = _admit_program(
        module_text, warp_max_elements, cta_chunk_elements
    )
    if not classification:
        # The classifier refused offsets that are valid. The program and
        # the limits are admitted, so this raises its reason.
        classification.append(
            native_swage._classify_segments(
                host_offsets,
                value_count=value_count,
                segment_count=segment_count,
                warp_max_elements=warp_max_elements,
                cta_chunk_elements=cta_chunk_elements,
            )
        )
    # One buffer holds the warp ids, the CTA ids, the partial ranges, the
    # merge records, and the merge of every partial task, in that order.
    (
        records,
        direct_warp_count,
        direct_cta_count,
        partial_count,
        merge_count,
    ) = classification[0]
    direct_count = direct_warp_count + direct_cta_count
    # ponytail: a measured two-chunk rule, not a general cost model.
    # Retain splitting for sparse batches, larger tails, or mixed lengths.
    # Every segment is split when the merge count equals the segment count,
    # so the longest segment is read only then.
    use_direct_cta = (
        select_schedule
        and segment_count > 0
        and cta_chunk_elements == _CTA_CHUNK_ELEMENTS
        and merge_count == segment_count
        and int((host_offsets[1:] - host_offsets[:-1]).max())
        <= 2 * cta_chunk_elements
        and segment_count
        >= torch.cuda.get_device_properties(
            values.device
        ).multi_processor_count
        and small_element_program
    )
    if segment_count == 0:

        def no_launch():
            _require_unchanged_offsets(offsets, offsets_version)
            storage = _storage_binding(values, offsets, output)
            if storage != prepared_storage:
                _refuse_rebound_storage(storage, prepared_storage)
            return None

        return _PreparedReduction(no_launch, no_launch, no_launch)
    warp_ptx = _compile_once(
        native_swage._compile_segmented_reduction_ptx,
        module_text,
        kernel_name=kernel_name,
        block_size=_WARP_BLOCK,
        target=target,
        use_task_ids=True,
    )
    cta_ptx = _compile_once(
        native_swage._compile_segmented_reduction_ptx,
        module_text,
        kernel_name=kernel_name,
        block_size=_CTA_BLOCK,
        target=target,
        use_task_ids=True,
    )
    mixed_ptx = None
    if direct_count:
        mixed_ptx = _compile_once(
            native_swage._compile_fused_segmented_reduction_ptx,
            module_text,
            kernel_name=kernel_name,
            target=target,
        )
    partial_ptx = None
    merge_ptx = None
    if partial_count and not use_direct_cta:
        partial_ptx = _compile_once(
            native_swage._compile_split_partial_reduction_ptx,
            module_text,
            kernel_name=kernel_name,
            target=target,
        )
        merge_ptx = _compile_once(
            native_swage._compile_split_merge_reduction_ptx,
            module_text,
            kernel_name=kernel_name,
            target=target,
        )

    driver = _runtime._get_driver()
    _, warp_function = _load_once(driver, warp_ptx, kernel_name)
    _, cta_function = _load_once(driver, cta_ptx, kernel_name)
    mixed_function = None
    if mixed_ptx is not None:
        _, mixed_function = _load_once(driver, mixed_ptx, kernel_name)
    partial_function = None
    merge_function = None
    if partial_ptx is not None:
        _, partial_function = _load_once(
            driver, partial_ptx, f"{kernel_name}__partial"
        )
        _, merge_function = _load_once(
            driver, merge_ptx, f"{kernel_name}__merge"
        )
    device = offsets.device
    device_index = device.index
    all_tasks = _identity_ids(torch, device, segment_count)
    # One upload carries every record the mixed policy launches. The fused
    # kernel reads the warp ids and then the CTA ids at the start of the
    # buffer; the partial ranges and the merge records follow them. The
    # direct-CTA selection launches none of them: every segment is split
    # then, so there is no direct id either.
    task_records = None
    mixed_pointer = partial_pointer = merge_pointer = scratch = None
    if len(records) and not use_direct_cta:
        task_records = torch.tensor(records, dtype=torch.int32, device=device)
        mixed_pointer = task_records.data_ptr()
        partial_pointer = mixed_pointer + 4 * direct_count
        merge_pointer = partial_pointer + 8 * partial_count
        if partial_count:
            scratch = torch.empty(
                partial_count, dtype=torch.float32, device=device
            )
    tasks_ready = torch.cuda.Event()
    tasks_ready.record(torch.cuda.current_stream(device_index))

    tasks_ready_complete = False

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

    def submit(function, block_size, stream, storage):
        # One task per segment, in segment order: the task list is the
        # first `segment_count` ids of `all_tasks`.
        wait_for_tasks(stream)
        driver.launch_segmented_tasks(
            function,
            (segment_count,),
            block_size,
            stream.cuda_stream,
            (
                storage[0],
                storage[3],
                storage[6],
                all_tasks.data_ptr(),
                value_count,
                segment_count,
                segment_count,
            ),
        )
        for tensor in (values, offsets, output, all_tasks):
            tensor.record_stream(stream)
        return None

    current_context = getattr(driver, "current_context", None)
    prepared_context = None if current_context is None else current_context()

    def current_stream():
        if torch.cuda.current_device() != device_index:
            raise ValueError(
                "prepared reduction must launch on its prepared device"
            )
        # The kernels are loaded in one CUDA context and are not valid in
        # another, also on the same device.
        if current_context is not None:
            try:
                context = current_context()
            except RuntimeError:
                context = _runtime._make_context_current(
                    torch, current_context, device_index
                )
            if context != prepared_context:
                raise RuntimeError(
                    "prepared reduction must launch in its prepared CUDA "
                    "context"
                )
        return torch.cuda.current_stream()

    def warp():
        _require_unchanged_offsets(offsets, offsets_version)
        storage = _storage_binding(values, offsets, output)
        if storage != prepared_storage:
            _refuse_rebound_storage(storage, prepared_storage)
        return submit(warp_function, _WARP_BLOCK, current_stream(), storage)

    def cta():
        _require_unchanged_offsets(offsets, offsets_version)
        storage = _storage_binding(values, offsets, output)
        if storage != prepared_storage:
            _refuse_rebound_storage(storage, prepared_storage)
        return submit(cta_function, _CTA_BLOCK, current_stream(), storage)

    def mixed():
        _require_unchanged_offsets(offsets, offsets_version)
        storage = _storage_binding(values, offsets, output)
        if storage != prepared_storage:
            _refuse_rebound_storage(storage, prepared_storage)
        values_pointer, offsets_pointer, output_pointer = storage[::3]
        stream = current_stream()
        wait_for_tasks(stream)
        if direct_count:
            driver.launch_segmented_mixed(
                mixed_function,
                ((direct_warp_count + 3) // 4 + direct_cta_count,),
                _CTA_BLOCK,
                stream.cuda_stream,
                (
                    values_pointer,
                    offsets_pointer,
                    output_pointer,
                    mixed_pointer,
                    value_count,
                    direct_warp_count,
                    direct_cta_count,
                    segment_count,
                ),
            )
            for tensor in (values, offsets, output, task_records):
                tensor.record_stream(stream)
        if partial_count:
            driver.launch_segmented(
                partial_function,
                (partial_count,),
                _SPLIT_BLOCK,
                stream.cuda_stream,
                (
                    values_pointer,
                    partial_pointer,
                    scratch.data_ptr(),
                    value_count,
                    partial_count,
                ),
            )
            for tensor in (values, offsets, task_records, scratch):
                tensor.record_stream(stream)
            driver.launch_segmented(
                merge_function,
                (merge_count,),
                _SPLIT_BLOCK,
                stream.cuda_stream,
                (
                    scratch.data_ptr(),
                    output_pointer,
                    merge_pointer,
                    partial_count,
                    merge_count,
                    segment_count,
                ),
            )
            for tensor in (offsets, output, task_records, scratch):
                tensor.record_stream(stream)
        return None

    return _PreparedReduction(warp, cta, cta if use_direct_cta else mixed)


def _prepare_persistent_sum(
    values,
    offsets,
    output,
    *,
    warp_max_elements=32,
    cta_chunk_elements=_CTA_CHUNK_ELEMENTS,
    resident_blocks=None,
):
    """Prepare a private resident kernel with split completion handling.

    The offsets must not change in place after preparation; the prepared
    launch raises RuntimeError instead of launching when they did. The
    launch is bound to the storage of values, offsets, and output at
    preparation: in-place writes to values and output are fine, and a
    tensor that was given another data pointer, element count, or dtype
    since, for example through `tensor.data = other`, raises RuntimeError
    before anything is enqueued. The kernel is compiled once per target and
    loaded once per CUDA context.
    """
    if resident_blocks is not None and (
        type(resident_blocks) is not int
        or resident_blocks <= 0
        or resident_blocks > (1 << 32) - 1
    ):
        raise ValueError("resident_blocks must be a positive u32")
    torch = _runtime._import_torch()
    validate, classification = _classifying_validator(
        warp_max_elements, cta_chunk_elements
    )
    value_count, segment_count, host_offsets = _validate_shapes(
        values, offsets, output, validate
    )
    offsets_version = _offsets_version(offsets)
    # Each launch builds this record again and compares the two. The record
    # also carries the data pointers that the launch passes to the kernel.
    prepared_storage = _storage_binding(values, offsets, output)

    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    target = _target(torch, torch.cuda.current_device())
    kernel_name = "segmented_sum"
    module_text = _semantic_module("sum")
    _admit_program(module_text, warp_max_elements, cta_chunk_elements)
    if not classification:
        # The classifier refused offsets that are valid. The program and
        # the limits are admitted, so this raises its reason.
        classification.append(
            native_swage._classify_segments(
                host_offsets,
                value_count=value_count,
                segment_count=segment_count,
                warp_max_elements=warp_max_elements,
                cta_chunk_elements=cta_chunk_elements,
            )
        )
    # One buffer holds the warp ids, the CTA ids, the partial ranges, the
    # merge records, and the merge of every partial task, in that order.
    records, warp_count, cta_count, partial_count, merge_count = (
        classification[0]
    )
    if segment_count == 0:

        def no_launch():
            _require_unchanged_offsets(offsets, offsets_version)
            storage = _storage_binding(values, offsets, output)
            if storage != prepared_storage:
                _refuse_rebound_storage(storage, prepared_storage)
            return None

        return _PreparedPersistentSum(no_launch, 0, 0, 0, 0, 0)
    ptx = _compile_once(
        native_swage._compile_persistent_segmented_reduction_ptx,
        module_text,
        kernel_name=kernel_name,
        target=target,
    )

    warp_slots = _PERSISTENT_BLOCK // _WARP_BLOCK
    work_groups = (
        cta_count + partial_count + (warp_count + warp_slots - 1) // warp_slots
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if _PERSISTENT_BLOCK > properties.max_threads_per_block:
        raise ValueError(
            f"persistent block size {_PERSISTENT_BLOCK} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if resident_blocks is None:
        resident_blocks = properties.multi_processor_count * 2
    active_blocks = min(resident_blocks, work_groups)

    driver = _runtime._get_driver()
    _, function = _load_once(driver, ptx, kernel_name)
    device = offsets.device
    device_index = device.index
    # One upload carries every record. The kernel takes one pointer per
    # list; the classifier wrote the merge of every partial task behind the
    # merge records.
    task_records = torch.tensor(records, dtype=torch.int32, device=device)
    warp_pointer = task_records.data_ptr()
    cta_pointer = warp_pointer + 4 * warp_count
    partial_pointer = cta_pointer + 4 * cta_count
    merge_pointer = partial_pointer + 8 * partial_count
    partial_merges_pointer = merge_pointer + 12 * merge_count
    scratch = torch.empty(partial_count, dtype=torch.float32, device=device)
    # Every launch zeroes the counters before it enqueues the kernel, so
    # they need no initial value.
    counters = torch.empty(3 + merge_count, dtype=torch.int32, device=device)
    tasks_ready = torch.cuda.Event()
    tasks_ready.record(torch.cuda.current_stream(device_index))
    tasks_ready_complete = False
    current_context = getattr(driver, "current_context", None)
    prepared_context = None if current_context is None else current_context()
    # One prepared object has one counter array and one scratch buffer, so
    # two launches must not run at once. `launching` keeps a second thread
    # out of `launch`; `in_flight` is recorded behind each launch and tells
    # a launch on another stream whether the previous one has finished.
    launching = threading.Lock()
    in_flight = torch.cuda.Event()
    in_flight_stream = None

    def current_stream():
        if torch.cuda.current_device() != device_index:
            raise ValueError(
                "prepared persistent sum must launch on its prepared device"
            )
        # The kernel is loaded in one CUDA context and is not valid in
        # another, also on the same device.
        if current_context is not None:
            try:
                context = current_context()
            except RuntimeError:
                context = _runtime._make_context_current(
                    torch, current_context, device_index
                )
            if context != prepared_context:
                raise RuntimeError(
                    "prepared persistent sum must launch in its prepared "
                    "CUDA context"
                )
        return torch.cuda.current_stream()

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
        nonlocal in_flight_stream
        _require_unchanged_offsets(offsets, offsets_version)
        storage = _storage_binding(values, offsets, output)
        if storage != prepared_storage:
            _refuse_rebound_storage(storage, prepared_storage)
        values_pointer, offsets_pointer, output_pointer = storage[::3]
        if not launching.acquire(blocking=False):
            raise RuntimeError(
                "prepared persistent sum is already launching on another "
                "thread; one prepared object admits one launch at a time"
            )
        try:
            stream = current_stream()
            capturing = torch.cuda.is_current_stream_capturing()
            require_idle(stream, capturing)
            wait_for_tasks(stream)
            counters.zero_()
            driver.launch_persistent(
                function,
                (active_blocks,),
                _PERSISTENT_BLOCK,
                stream.cuda_stream,
                (
                    values_pointer,
                    offsets_pointer,
                    output_pointer,
                    warp_pointer,
                    cta_pointer,
                    partial_pointer,
                    partial_merges_pointer,
                    merge_pointer,
                    scratch.data_ptr(),
                    counters.data_ptr(),
                    value_count,
                    warp_count,
                    cta_count,
                    partial_count,
                    merge_count,
                    segment_count,
                ),
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


def launch_softmax_gpu(values, offsets, output, block_size=128):
    """Launch the internally qualified ragged softmax.

    The kernel is compiled once per block size and target, and loaded once
    per CUDA context.
    """
    torch = _runtime._import_torch()
    value_count, segment_count = _validate_softmax_tensors(
        values, offsets, output
    )
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block size must be a positive integer")
    _validate_warp_count(block_size)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if block_size > properties.max_threads_per_block:
        raise ValueError(
            f"block size {block_size} exceeds device limit "
            f"{properties.max_threads_per_block}"
        )
    if segment_count == 0:
        return None

    from mlir_swage._mlir_libs._swageDialectsNanobind import (
        swage as native_swage,
    )

    target = _target(torch, torch.cuda.current_device())
    kernel_name = "ragged_softmax"
    ptx = _compile_once(
        native_swage._compile_segmented_reduction_ptx,
        _SOFTMAX_MODULE,
        kernel_name=kernel_name,
        block_size=block_size,
        target=target,
    )

    driver = _runtime._get_driver()
    _, function = _load_once(driver, ptx, kernel_name)
    stream = torch.cuda.current_stream()
    driver.launch_segmented(
        function,
        (segment_count,),
        block_size,
        stream.cuda_stream,
        (
            values.data_ptr(),
            offsets.data_ptr(),
            output.data_ptr(),
            value_count,
            segment_count,
        ),
    )
    for tensor in (values, offsets, output):
        tensor.record_stream(stream)
    return None


def _float_literal(value):
    """Emit an exact f32 bit pattern accepted by the MLIR parser."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return f"0x{bits:08X}"


def _dense_literal(code, numbers):
    """Emit the raw little-endian bytes of a dense MLIR initializer."""
    payload = struct.pack(f"<{len(numbers)}{code}", *numbers)
    return f'dense<"0x{payload.hex()}">'


def _runner_module(values, offsets, semantic, kernel_name, output_length):
    """Add a no-argument executable wrapper around the semantic kernel.

    The inputs become constant globals initialized from their exact bytes,
    one operation per buffer instead of three per element. The result
    leaves as bit patterns: each f32 output is bitcast to i32 and printed by
    the integer memref printer, because the float printer keeps only six
    significant digits.
    """
    value_count = values.numel()
    segment_count = offsets.numel() - 1
    values_type = f"memref<{value_count}xf32>"
    offsets_type = f"memref<{segment_count + 1}xi32>"
    output_type = f"memref<{output_length}xf32>"
    bits_type = f"memref<{output_length}xi32>"
    lines = [
        semantic.rstrip()[:-1],
        "",
        (
            f'  memref.global "private" constant @runner_offsets : '
            f"{offsets_type} = {_dense_literal('i', offsets.tolist())}"
        ),
    ]
    # A zero-element dense initializer has no bytes to parse, so an empty
    # values buffer stays a plain allocation that nothing reads.
    values_storage = f"memref.alloc() : {values_type}"
    if value_count:
        lines.append(
            f'  memref.global "private" constant @runner_values : '
            f"{values_type} = {_dense_literal('f', values.tolist())}"
        )
        values_storage = f"memref.get_global @runner_values : {values_type}"
    lines.extend(
        [
            "",
            "  func.func @main() {",
            f"    %values_storage = {values_storage}",
            (
                f"    %offsets_storage = memref.get_global @runner_offsets : "
                f"{offsets_type}"
            ),
            f"    %output_storage = memref.alloc() : {output_type}",
            f"    %bits = memref.alloc() : {bits_type}",
            (
                f"    %values = memref.cast %values_storage : {values_type} "
                "to memref<?xf32>"
            ),
            (
                f"    %offsets = memref.cast %offsets_storage : "
                f"{offsets_type} to memref<?xi32>"
            ),
            (
                f"    %output = memref.cast %output_storage : {output_type} "
                "to memref<?xf32>"
            ),
            # Prefill so that a store past the live range is visible in the
            # parsed result instead of being garbage.
            (
                f"    %sentinel = arith.constant "
                f"{_float_literal(_SENTINEL)} : f32"
            ),
            "    %from = arith.constant 0 : index",
            "    %step = arith.constant 1 : index",
            f"    %to = arith.constant {output_length} : index",
            "    scf.for %pi = %from to %to step %step {",
            "      memref.store %sentinel, %output[%pi] : memref<?xf32>",
            "    }",
            f"    %value_count = arith.constant {value_count} : i32",
            f"    %segment_count = arith.constant {segment_count} : i32",
            (
                f"    call @{kernel_name}(%values, %offsets, %output, "
                "%value_count, %segment_count) : (memref<?xf32>, "
                "memref<?xi32>, memref<?xf32>, i32, i32) -> ()"
            ),
            "    scf.for %bi = %from to %to step %step {",
            "      %result = memref.load %output[%bi] : memref<?xf32>",
            "      %pattern = arith.bitcast %result : f32 to i32",
            f"      memref.store %pattern, %bits[%bi] : {bits_type}",
            "    }",
            (
                f"    %unranked = memref.cast %bits : {bits_type} "
                "to memref<*xi32>"
            ),
            "    call @printMemrefI32(%unranked) : (memref<*xi32>) -> ()",
        ]
    )
    if not value_count:
        lines.append(f"    memref.dealloc %values_storage : {values_type}")
    lines.extend(
        [
            f"    memref.dealloc %output_storage : {output_type}",
            f"    memref.dealloc %bits : {bits_type}",
            "    return",
            "  }",
            "",
            (
                "  func.func private @printMemrefI32(memref<*xi32>) "
                "attributes {llvm.emit_c_interface}"
            ),
            "}",
        ]
    )
    return "\n".join(lines)


def _llvm_root(root):
    """Find the pinned install used to configure the current build."""
    cache = root / "build" / "CMakeCache.txt"
    match = re.search(r"^MLIR_DIR:[^=]*=(.+)$", cache.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError("build/CMakeCache.txt does not identify MLIR_DIR")
    return pathlib.Path(match.group(1)).parents[2]


def _llvm_tool(llvm_root, name):
    """Return one LLVM tool, preferring the pinned install over PATH.

    The runner libraries always come from the pinned install. A tool found
    on PATH may belong to a different LLVM, so it is used only when the
    install does not provide that tool.
    """
    pinned = llvm_root / "bin" / name
    if pinned.is_file():
        return pinned
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(
            f"{name} was found neither at {pinned} nor on PATH; the CPU "
            "oracle requires it from the pinned LLVM install"
        )
    return pathlib.Path(found)


def _run(command, source):
    """Run one compiler stage and return its text output."""
    result = subprocess.run(
        [str(argument) for argument in command],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"{' '.join(map(str, command))} failed:\n{result.stderr}"
        )
    return result.stdout


def _execute(module_text):
    """Lower and run one executable module, returning its exact f32 results.

    The module prints one signed i32 bit pattern per output element. Each
    is reinterpreted as the f32 it encodes, so the returned Python floats
    hold the computed values with no decimal rounding in between.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    llvm_root = _llvm_root(root)
    swage_opt = root / "build" / "bin" / "swage-opt"
    mlir_opt = _llvm_tool(llvm_root, "mlir-opt")
    mlir_runner = _llvm_tool(llvm_root, "mlir-runner")
    lowered = _run(
        [swage_opt, "--swage-segmented-reduction-to-scf"], module_text
    )
    llvm = _run([mlir_opt, f"--pass-pipeline={_LOWERING_PIPELINE}"], lowered)
    runner_utils = llvm_root / "lib" / "libmlir_runner_utils.so"
    c_runner_utils = llvm_root / "lib" / "libmlir_c_runner_utils.so"
    printed = _run(
        [
            mlir_runner,
            "-e",
            "main",
            "-entry-point-result=void",
            f"-shared-libs={runner_utils}",
            f"-shared-libs={c_runner_utils}",
        ],
        llvm,
    )
    match = re.search(r"data =\s*\n\[(.*?)\]", printed, re.DOTALL)
    if not match:
        raise RuntimeError(
            f"mlir-runner returned an unreadable result:\n{printed}"
        )
    patterns = [
        int(token) & 0xFFFFFFFF
        for token in match.group(1).split(",")
        if token.strip()
    ]
    payload = struct.pack(f"<{len(patterns)}I", *patterns)
    return list(struct.unpack(f"<{len(patterns)}f", payload))


def _execute_guarded(values, offsets, semantic, kernel_name, live):
    """Run a kernel with one guard slot after its live output range.

    The extra slot keeps the prefilled sentinel. That makes a zero-length
    result printable and turns "the kernel never writes past its live
    range" into a checked invariant of every oracle call.
    """
    results = _execute(
        _runner_module(values, offsets, semantic, kernel_name, live + 1)
    )
    if len(results) != live + 1:
        raise RuntimeError(
            f"oracle printed {len(results)} values for {live + 1} slots"
        )
    if results[-1] != _SENTINEL:
        raise RuntimeError(
            f"{kernel_name} wrote past its live output range: {results[-1]}"
        )
    return results[:-1]


def cpu_oracle(values, offsets, kind):
    """Execute the sequential reduction lowering with the MLIR runner.

    Returns the exact f32 result of each segment, accumulated left to right.
    """
    torch = _runtime._import_torch()
    segment_count = max(offsets.numel() - 1, 0)
    output = torch.empty(segment_count, dtype=torch.float32)
    _validate_tensors(values, offsets, output, require_cuda=False)
    results = _execute_guarded(
        values,
        offsets,
        _semantic_module(kind),
        f"segmented_{kind}",
        segment_count,
    )
    return torch.tensor(results, dtype=torch.float32)


def cpu_softmax_oracle(values, offsets):
    """Execute the sequential softmax lowering with the MLIR runner.

    Returns the exact f32 value the lowering stored for each covered element.
    """
    torch = _runtime._import_torch()
    covered = int(offsets[-1]) if offsets.numel() else 0
    output = torch.empty(covered, dtype=torch.float32)
    _validate_softmax_tensors(values, offsets, output, require_cuda=False)
    results = _execute_guarded(
        values, offsets, _SOFTMAX_MODULE, "ragged_softmax", covered
    )
    return torch.tensor(results, dtype=torch.float32)
